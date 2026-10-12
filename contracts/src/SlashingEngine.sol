// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./StakingVault.sol";
import "./MockToken.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title SlashingEngine
 * @notice Receives signed fraud proofs and executes stake confiscation:
 * 50% burned to 0xdead, 50% redistributed to honest nodes.
 */
contract SlashingEngine is Ownable, ReentrancyGuard {
    StakingVault public immutable stakingVault;
    MockToken public immutable token;
    address public authorizedVerifier;
    address public constant BURN_ADDRESS = address(0x000000000000000000000000000000000000dEaD);

    uint256 public totalSlashedTokens;
    uint256 public totalBurnedTokens;
    uint256 public totalRewardedTokens;

    event SybilRingSlashed(
        address[] badNodes,
        uint256 totalAmountSlashed,
        uint256 burnedAmount,
        uint256 rewardedAmount,
        address[] rewardRecipients
    );
    event VerifierUpdated(address indexed newVerifier);

    modifier onlyVerifier() {
        require(msg.sender == authorizedVerifier || msg.sender == owner(), "Unauthorized verifier");
        _;
    }

    constructor(address _stakingVault, address _token, address _authorizedVerifier) Ownable(msg.sender) {
        require(_stakingVault != address(0), "Invalid vault address");
        require(_token != address(0), "Invalid token address");
        require(_authorizedVerifier != address(0), "Invalid verifier address");

        stakingVault = StakingVault(_stakingVault);
        token = MockToken(_token);
        authorizedVerifier = _authorizedVerifier;
    }

    function setAuthorizedVerifier(address _verifier) external onlyOwner {
        require(_verifier != address(0), "Invalid verifier address");
        authorizedVerifier = _verifier;
        emit VerifierUpdated(_verifier);
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
    ) external onlyVerifier nonReentrant {
        require(badNodes.length > 0, "No bad nodes provided");
        require(proofData.length > 0, "Proof data required");

        uint256 roundSlashed = 0;

        for (uint256 i = 0; i < badNodes.length; i++) {
            address badNode = badNodes[i];
            if (stakingVault.isNodeStaked(badNode)) {
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
        require(token.transfer(BURN_ADDRESS, burnAmount), "Burn transfer failed");

        // 2. Redistribute 50% to honest nodes
        totalRewardedTokens += rewardAmount;
        if (honestNodes.length > 0) {
            uint256 sharePerHonest = rewardAmount / honestNodes.length;
            uint256 distributed = 0;

            for (uint256 j = 0; j < honestNodes.length; j++) {
                if (j == honestNodes.length - 1) {
                    // Send remaining to account for integer division remainder
                    uint256 remainder = rewardAmount - distributed;
                    require(token.transfer(honestNodes[j], remainder), "Reward transfer failed");
                } else {
                    require(token.transfer(honestNodes[j], sharePerHonest), "Reward transfer failed");
                    distributed += sharePerHonest;
                }
            }
        } else {
            // If no honest nodes registered in round, burn the remainder too
            require(token.transfer(BURN_ADDRESS, rewardAmount), "Fallback burn failed");
        }

        emit SybilRingSlashed(badNodes, roundSlashed, burnAmount, rewardAmount, honestNodes);
    }
}
