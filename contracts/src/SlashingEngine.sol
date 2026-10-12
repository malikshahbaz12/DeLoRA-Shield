// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./StakingVault.sol";
import "./SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title SlashingEngine
 * @notice Receives signed fraud proofs and executes stake confiscation:
 * 50% burned to 0xdead, 50% redistributed to honest nodes.
 */
contract SlashingEngine is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    StakingVault public immutable stakingVault;
    IERC20 public immutable token;
    address public authorizedVerifier;
    address public constant BURN_ADDRESS = address(0x000000000000000000000000000000000000dEaD);

    uint256 public totalSlashedTokens;
    uint256 public totalBurnedTokens;
    uint256 public totalRewardedTokens;

    mapping(address => uint256) public pendingRewards;

    event SybilRingSlashed(
        address[] badNodes,
        uint256 totalAmountSlashed,
        uint256 burnedAmount,
        uint256 rewardedAmount,
        address[] rewardRecipients
    );
    event VerifierUpdated(address indexed newVerifier);
    event RewardClaimed(address indexed node, uint256 amount);
    event RewardCredited(address indexed node, uint256 amount);

    modifier onlyVerifier() {
        require(msg.sender == authorizedVerifier || msg.sender == owner(), "Unauthorized verifier");
        _;
    }

    constructor(address _stakingVault, address _token, address _authorizedVerifier) Ownable(msg.sender) {
        require(_stakingVault != address(0), "Invalid vault address");
        require(_token != address(0), "Invalid token address");
        require(_authorizedVerifier != address(0), "Invalid verifier address");

        stakingVault = StakingVault(_stakingVault);
        token = IERC20(_token);
        authorizedVerifier = _authorizedVerifier;
    }

    function setAuthorizedVerifier(address _verifier) external onlyOwner {
        require(_verifier != address(0), "Invalid verifier address");
        authorizedVerifier = _verifier;
        emit VerifierUpdated(_verifier);
    }

    /**
     * @notice Allows an honest node to withdraw their credited slashing rewards.
     */
    function claimRewards() external nonReentrant {
        uint256 reward = pendingRewards[msg.sender];
        require(reward > 0, "No pending rewards to claim");
        pendingRewards[msg.sender] = 0;
        token.safeTransfer(msg.sender, reward);
        emit RewardClaimed(msg.sender, reward);
    }

    /**
     * @notice Executes slashing on a detected Sybil cluster.
     * @param badNodes Array of detected malicious / Sybil node addresses.
     * @param honestNodes Array of verified honest participants to receive reward redistribution.
     * @param proofData Signed mathematical fraud proof (e.g. SVD lambda1 + temporal cosine proof).
     */
    function slashMaliciousNodes(
        address[] calldata badNodes,
        address[] calldata honestNodes,
        bytes calldata proofData
    ) external nonReentrant onlyVerifier {
        require(badNodes.length > 0, "No bad nodes provided");
        require(proofData.length > 0, "Proof data required");

        uint256 roundSlashed = 0;

        for (uint256 i = 0; i < badNodes.length; i++) {
            address badNode = badNodes[i];
            if (stakingVault.isNodeStaked(badNode) || stakingVault.getStake(badNode) > 0) {
                uint256 amount = stakingVault.slash(badNode, msg.sender);
                roundSlashed += amount;
            }
        }

        require(roundSlashed > 0, "Zero tokens slashed");

        totalSlashedTokens += roundSlashed;

        // 50% Burn, 50% Reward Redistribution
        uint256 burnAmount = roundSlashed / 2;
        uint256 rewardAmount = roundSlashed - burnAmount;

        // 1. Burn 50% to 0xdead
        totalBurnedTokens += burnAmount;
        token.safeTransfer(BURN_ADDRESS, burnAmount);

        // 2. Redistribute 50% to honest nodes
        totalRewardedTokens += rewardAmount;
        if (honestNodes.length > 0) {
            uint256 sharePerHonest = rewardAmount / honestNodes.length;
            uint256 distributed = 0;

            for (uint256 j = 0; j < honestNodes.length; j++) {
                uint256 share = (j == honestNodes.length - 1) ? (rewardAmount - distributed) : sharePerHonest;
                distributed += share;

                // Push directly to recipient
                token.safeTransfer(honestNodes[j], share);
                emit RewardCredited(honestNodes[j], share);
            }
        } else {
            // If no honest nodes registered in round, burn the remainder too
            token.safeTransfer(BURN_ADDRESS, rewardAmount);
        }

        emit SybilRingSlashed(badNodes, roundSlashed, burnAmount, rewardAmount, honestNodes);
    }
}
