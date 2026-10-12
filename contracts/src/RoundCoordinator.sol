// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./StakingVault.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title RoundCoordinator
 * @notice Manages commit-reveal state machine phases for collaborative LoRA training.
 */
contract RoundCoordinator is Ownable {
    StakingVault public immutable stakingVault;
    address public authorizedEngine;

    enum Phase { Commit, Reveal, Verification, Complete }

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
            require(rounds[currentRoundId].currentPhase == Phase.Complete, "Previous round not complete");
        }

        currentRoundId++;
        RoundInfo storage round = rounds[currentRoundId];
        round.roundId = currentRoundId;
        round.currentPhase = Phase.Commit;
        round.commitDeadline = block.timestamp + commitDuration;
        round.revealDeadline = block.timestamp + commitDuration + revealDuration;

        emit RoundInitialized(currentRoundId, round.commitDeadline, round.revealDeadline);
    }

    function submitCommitment(bytes32 commitmentHash) external {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.Commit, "Not in Commit phase");
        require(block.timestamp <= round.commitDeadline, "Commit deadline passed");
        require(stakingVault.isNodeStaked(msg.sender), "Node not staked with required collateral");

        CommitmentRecord storage record = nodeCommitments[currentRoundId][msg.sender];
        require(record.commitmentHash == bytes32(0), "Commitment already submitted");

        record.commitmentHash = commitmentHash;
        round.participatingNodes.push(msg.sender);

        // Lock stake for duration of round
        uint256 totalRoundDuration = round.revealDeadline - block.timestamp + 3600;
        stakingVault.lockNodeStake(msg.sender, totalRoundDuration);

        emit CommitmentRecorded(currentRoundId, msg.sender, commitmentHash);
    }

    function advanceToReveal() external {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.Commit, "Must be in Commit phase");
        require(block.timestamp > round.commitDeadline || msg.sender == authorizedEngine, "Commit deadline not reached");

        round.currentPhase = Phase.Reveal;
        emit PhaseAdvanced(currentRoundId, Phase.Reveal);
    }

    function revealUpdate(string calldata ipfsHash, bytes32 salt) external {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.Reveal, "Not in Reveal phase");
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

    function advanceToVerification() external onlyAuthorizedEngine {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.Reveal, "Must be in Reveal phase");
        require(block.timestamp > round.revealDeadline || msg.sender == authorizedEngine, "Reveal deadline not reached");

        round.currentPhase = Phase.Verification;
        emit PhaseAdvanced(currentRoundId, Phase.Verification);
    }

    function completeRound(string calldata globalModelCID) external onlyAuthorizedEngine {
        RoundInfo storage round = rounds[currentRoundId];
        require(round.currentPhase == Phase.Verification, "Must be in Verification phase");

        round.globalModelCID = globalModelCID;
        round.currentPhase = Phase.Complete;

        emit GlobalModelPublished(currentRoundId, globalModelCID);
        emit PhaseAdvanced(currentRoundId, Phase.Complete);
    }

    function getRevealedNodes(uint256 roundId) external view returns (address[] memory) {
        return rounds[roundId].revealedNodes;
    }

    function getParticipatingNodes(uint256 roundId) external view returns (address[] memory) {
        return rounds[roundId].participatingNodes;
    }

    function getNodeSubmission(uint256 roundId, address node) external view returns (bytes32 commitmentHash, string memory ipfsHash, bool revealed) {
        CommitmentRecord storage record = nodeCommitments[roundId][node];
        return (record.commitmentHash, record.ipfsHash, record.revealed);
    }
}
