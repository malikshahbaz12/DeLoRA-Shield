// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "./StakingVault.sol";

/**
 * @title RoundCoordinator
 * @notice Coordinates commit-reveal state machine for collaborative LoRA training:
 * COMMIT -> REVEAL -> VERIFYING -> FINALIZED.
 * Preimage verification: keccak256(abi.encodePacked(ipfsCID, salt, msg.sender, roundId))
 */
contract RoundCoordinator is Ownable2Step {
    enum Phase { UNINITIALIZED, COMMIT, REVEAL, VERIFYING, FINALIZED }

    struct RoundConfig {
        uint256 roundId;
        string modelVersion;
        uint256 commitDeadline;
        uint256 revealDeadline;
        Phase currentPhase;
        string globalModelCID;
    }

    struct ParticipantSubmission {
        bytes32 commitmentHash;
        string ipfsCID;
        bytes32 salt;
        bool hasCommitted;
        bool hasRevealed;
    }

    StakingVault public immutable vault;
    address public verifierDaemon;
    address public slashingEngine;

    uint256 public currentRoundId;
    mapping(uint256 => RoundConfig) public rounds;
    mapping(uint256 => mapping(address => ParticipantSubmission)) public submissions;
    mapping(uint256 => address[]) internal roundParticipants;

    event RoundStarted(uint256 indexed roundId, uint256 commitDeadline, uint256 revealDeadline);
    event CommitmentSubmitted(uint256 indexed roundId, address indexed participant, bytes32 hash);
    event UpdateRevealed(uint256 indexed roundId, address indexed participant, string ipfsCID);
    event PhaseAdvanced(uint256 indexed roundId, Phase newPhase);
    event RoundFinalized(uint256 indexed roundId, string newGlobalModelCID);

    error InvalidPhase();
    error DeadlinePassed();
    error PreimageMismatch();
    error Unauthorized();

    constructor(address payable _vault) Ownable(msg.sender) {
        vault = StakingVault(_vault);
    }

    function setAuthorities(address _daemon, address _slashingEngine) external onlyOwner {
        verifierDaemon = _daemon;
        slashingEngine = _slashingEngine;
    }

    function startRound(
        string calldata modelVersion,
        uint256 commitDuration,
        uint256 revealDuration
    ) external onlyOwner {
        currentRoundId++;
        uint256 cDeadline = block.timestamp + commitDuration;
        uint256 rDeadline = cDeadline + revealDuration;

        rounds[currentRoundId] = RoundConfig({
            roundId: currentRoundId,
            modelVersion: modelVersion,
            commitDeadline: cDeadline,
            revealDeadline: rDeadline,
            currentPhase: Phase.COMMIT,
            globalModelCID: ""
        });

        emit RoundStarted(currentRoundId, cDeadline, rDeadline);
    }

    function submitCommitment(uint256 roundId, bytes32 commitmentHash) external {
        RoundConfig storage r = rounds[roundId];
        if (r.currentPhase != Phase.COMMIT || block.timestamp > r.commitDeadline) revert DeadlinePassed();
        if (submissions[roundId][msg.sender].hasCommitted) revert("Already committed");

        vault.lockStake(msg.sender, roundId);

        submissions[roundId][msg.sender] = ParticipantSubmission({
            commitmentHash: commitmentHash,
            ipfsCID: "",
            salt: bytes32(0),
            hasCommitted: true,
            hasRevealed: false
        });

        roundParticipants[roundId].push(msg.sender);
        emit CommitmentSubmitted(roundId, msg.sender, commitmentHash);
    }

    function revealUpdate(uint256 roundId, string calldata ipfsCID, bytes32 salt) external {
        RoundConfig storage r = rounds[roundId];
        if (block.timestamp <= r.commitDeadline && msg.sender != verifierDaemon && msg.sender != owner()) {
            // In standard workflow, reveal opens after commit deadline
            revert InvalidPhase();
        }
        if (block.timestamp > r.revealDeadline) revert DeadlinePassed();

        ParticipantSubmission storage sub = submissions[roundId][msg.sender];
        if (!sub.hasCommitted) revert("No commit found");
        if (sub.hasRevealed) revert("Already revealed");

        bytes32 computedHash = keccak256(abi.encodePacked(ipfsCID, salt, msg.sender, roundId));
        if (computedHash != sub.commitmentHash) revert PreimageMismatch();

        sub.ipfsCID = ipfsCID;
        sub.salt = salt;
        sub.hasRevealed = true;

        emit UpdateRevealed(roundId, msg.sender, ipfsCID);
    }

    function advanceToVerifying(uint256 roundId) external {
        if (msg.sender != verifierDaemon && msg.sender != owner()) revert Unauthorized();
        RoundConfig storage r = rounds[roundId];
        r.currentPhase = Phase.VERIFYING;
        emit PhaseAdvanced(roundId, Phase.VERIFYING);
    }

    function getParticipants(uint256 roundId) external view returns (address[] memory) {
        return roundParticipants[roundId];
    }

    function getRevealedNodes(uint256 roundId) external view returns (address[] memory) {
        address[] memory all = roundParticipants[roundId];
        uint256 count = 0;
        for (uint256 i = 0; i < all.length; i++) {
            if (submissions[roundId][all[i]].hasRevealed) {
                count++;
            }
        }
        address[] memory revealed = new address[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < all.length; i++) {
            if (submissions[roundId][all[i]].hasRevealed) {
                revealed[idx++] = all[i];
            }
        }
        return revealed;
    }

    function finalizeRound(uint256 roundId, string calldata newModelCID) external {
        if (msg.sender != verifierDaemon && msg.sender != owner()) revert Unauthorized();
        RoundConfig storage r = rounds[roundId];
        r.currentPhase = Phase.FINALIZED;
        r.globalModelCID = newModelCID;

        address[] memory parts = roundParticipants[roundId];
        for (uint256 i = 0; i < parts.length; i++) {
            vault.unlockStake(parts[i]);
        }

        emit RoundFinalized(roundId, newModelCID);
    }
}
