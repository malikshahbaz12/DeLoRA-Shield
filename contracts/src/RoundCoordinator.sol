// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./StakingVault.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title RoundCoordinator
 * @notice Enforces commit-reveal state machine phases for collaborative LoRA training:
 * COMMIT -> REVEAL -> VERIFYING -> FINALIZED.
 * Handles timeouts where missing reveals transition to a non-slashing protocol timeout.
 */
contract RoundCoordinator is Ownable, ReentrancyGuard {
    StakingVault public immutable stakingVault;
    address public authorizedEngine;

    // Structured Round States
    enum Phase { COMMIT, REVEAL, VERIFYING, FINALIZED }

    // Backward-compatible alias getters
    Phase public constant Commit = Phase.COMMIT;
    Phase public constant Reveal = Phase.REVEAL;
    Phase public constant Verification = Phase.VERIFYING;
    Phase public constant Complete = Phase.FINALIZED;

    struct CommitmentRecord {
        bytes32 commitmentHash;
        string ipfsHash;
        bytes32 salt;
        bool revealed;
    }

    struct RoundInfo {
        uint256 roundId;
        Phase currentPhase;
        uint256 commitDeadline;
        uint256 revealDeadline;
        string globalModelCID;
        address[] participatingNodes;
        address[] revealedNodes;
    }

    uint256 public currentRoundId;
    mapping(uint256 => RoundInfo) public rounds;
    mapping(uint256 => mapping(address => CommitmentRecord)) public nodeCommitments;

    event RoundInitialized(uint256 indexed roundId, uint256 commitDeadline, uint256 revealDeadline);
    event PhaseAdvanced(uint256 indexed roundId, Phase newPhase);
    event CommitmentRecorded(uint256 indexed roundId, address indexed node, bytes32 commitmentHash);
    event UpdateRevealed(uint256 indexed roundId, address indexed node, string ipfsHash);
    event RevealTimedOut(uint256 indexed roundId, address indexed node);
    event GlobalModelPublished(uint256 indexed roundId, string globalModelCID);

    modifier onlyAuthorizedEngine() {
        require(msg.sender == authorizedEngine || msg.sender == owner(), "Unauthorized engine caller");
        _;
    }

    constructor(address _stakingVault) Ownable(msg.sender) {
        require(_stakingVault != address(0), "Invalid staking vault");
        stakingVault = StakingVault(_stakingVault);
    }

    function setAuthorizedEngine(address _engine) external onlyOwner {
        require(_engine != address(0), "Invalid engine address");
        authorizedEngine = _engine;
    }

    function startNewRound(uint256 commitDuration, uint256 revealDuration) external onlyAuthorizedEngine {
        if (currentRoundId > 0) {
            require(rounds[currentRoundId].currentPhase == Phase.FINALIZED, "Previous round not finalized");
        }

        currentRoundId++;
        RoundInfo storage round = rounds[currentRoundId];
        round.roundId = currentRoundId;
        round.currentPhase = Phase.COMMIT;
        round.commitDeadline = block.timestamp + commitDuration;
        round.revealDeadline = block.timestamp + commitDuration + revealDuration;

        emit RoundInitialized(currentRoundId, round.commitDeadline, round.revealDeadline);
    }

    /**
     * @notice Stores client commitments computed as keccak256(canonical_weights || salt).
     */
    function submitCommitment(bytes32 commitmentHash) external nonReentrant {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.COMMIT, "Not in COMMIT phase");
        require(block.timestamp <= round.commitDeadline, "Commit deadline passed");
        require(stakingVault.isNodeStaked(msg.sender), "Node not staked with required collateral");

        CommitmentRecord storage record = nodeCommitments[currentRoundId][msg.sender];
        require(record.commitmentHash == bytes32(0), "Commitment already submitted");

        record.commitmentHash = commitmentHash;
        round.participatingNodes.push(msg.sender);

        // Lock stake for the duration of the round
        uint256 totalRoundDuration = round.revealDeadline - block.timestamp + 3600;
        stakingVault.lockNodeStake(msg.sender, totalRoundDuration);

        emit CommitmentRecorded(currentRoundId, msg.sender, commitmentHash);
    }

    function advanceToReveal() external {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.COMMIT, "Must be in COMMIT phase");
        require(block.timestamp > round.commitDeadline || msg.sender == authorizedEngine || msg.sender == owner(), "Commit deadline not reached");

        round.currentPhase = Phase.REVEAL;
        emit PhaseAdvanced(currentRoundId, Phase.REVEAL);
    }

    /**
     * @notice Validates preimages against stored commitment hashes and records revealed storage pointers.
     */
    function revealUpdate(string calldata ipfsHash, bytes32 salt) external nonReentrant {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.REVEAL, "Not in REVEAL phase");
        require(block.timestamp <= round.revealDeadline, "Reveal deadline passed");

        CommitmentRecord storage record = nodeCommitments[currentRoundId][msg.sender];
        require(record.commitmentHash != bytes32(0), "No commitment found");
        require(!record.revealed, "Update already revealed");

        bytes32 computedHash = keccak256(abi.encodePacked(ipfsHash, salt));
        require(computedHash == record.commitmentHash, "Hash mismatch with pre-reveal commitment");

        record.ipfsHash = ipfsHash;
        record.salt = salt;
        record.revealed = true;
        round.revealedNodes.push(msg.sender);

        emit UpdateRevealed(currentRoundId, msg.sender, ipfsHash);
    }

    /**
     * @notice Transitions from REVEAL to VERIFYING.
     * Enforces strict timeouts: missing reveals transition to a non-slashing protocol timeout.
     */
    function advanceToVerifying() public onlyAuthorizedEngine {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.REVEAL, "Must be in REVEAL phase");
        require(block.timestamp > round.revealDeadline || msg.sender == authorizedEngine || msg.sender == owner(), "Reveal deadline not reached");

        // Record protocol timeouts for any node that failed to reveal
        for (uint256 i = 0; i < round.participatingNodes.length; i++) {
            address node = round.participatingNodes[i];
            if (!nodeCommitments[currentRoundId][node].revealed) {
                emit RevealTimedOut(currentRoundId, node);
            }
        }

        round.currentPhase = Phase.VERIFYING;
        emit PhaseAdvanced(currentRoundId, Phase.VERIFYING);
    }

    // Alias for backward compatibility
    function advanceToVerification() external onlyAuthorizedEngine {
        advanceToVerifying();
    }

    /**
     * @notice Publishes global model CID and marks round as FINALIZED.
     */
    function finalizeRound(string calldata globalModelCID) public onlyAuthorizedEngine {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.VERIFYING, "Must be in VERIFYING phase");

        round.globalModelCID = globalModelCID;
        round.currentPhase = Phase.FINALIZED;

        emit GlobalModelPublished(currentRoundId, globalModelCID);
        emit PhaseAdvanced(currentRoundId, Phase.FINALIZED);
    }

    // Alias for backward compatibility
    function completeRound(string calldata globalModelCID) external onlyAuthorizedEngine {
        finalizeRound(globalModelCID);
    }

    function getRevealedNodes(uint256 roundId) external view returns (address[] memory) {
        return rounds[roundId].revealedNodes;
    }

    function getParticipatingNodes(uint256 roundId) external view returns (address[] memory) {
        return rounds[roundId].participatingNodes;
    }

    function getNodeSubmission(uint256 roundId, address node) external view returns (
        bytes32 commitmentHash,
        string memory ipfsHash,
        bool revealed
    ) {
        CommitmentRecord memory record = nodeCommitments[roundId][node];
        return (record.commitmentHash, record.ipfsHash, record.revealed);
    }

    function isNodeTimedOut(uint256 roundId, address node) external view returns (bool) {
        RoundInfo storage round = rounds[roundId];
        if (round.currentPhase == Phase.COMMIT) {
            return false;
        }
        CommitmentRecord memory record = nodeCommitments[roundId][node];
        return (record.commitmentHash != bytes32(0) && !record.revealed);
    }

    function getRoundPhase(uint256 roundId) external view returns (Phase) {
        return rounds[roundId].currentPhase;
    }
}
