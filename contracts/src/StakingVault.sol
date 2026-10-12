// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title StakingVault
 * @notice Manages escrow locking of user token stakes for DeLoRA-Shield.
 * Enforces minimum stake requirement and programmatic slashing.
 */
contract StakingVault is Ownable, ReentrancyGuard {
    IERC20 public immutable stakingToken;
    address public slashingEngine;
    address public roundCoordinator;

    uint256 public minStakeAmount;

    enum StakeStatus { INACTIVE, ACTIVE, SLASHED }

    struct StakeInfo {
        uint256 amount;
        uint256 lockedUntilEpoch;
        StakeStatus status;
    }

    mapping(address => StakeInfo) public stakes;
    address[] public activeStakers;
    mapping(address => uint256) private stakerIndex;

    event StakeDeposited(address indexed node, uint256 amount);
    event StakeWithdrawn(address indexed node, uint256 amount);
    event NodeSlashed(address indexed badNode, address indexed reporter, uint256 amount);
    event MinStakeUpdated(uint256 newMinStake);

    modifier onlySlashingEngine() {
        require(msg.sender == slashingEngine, "Only SlashingEngine authorized");
        _;
    }

    constructor(address _stakingToken, uint256 _minStakeAmount) Ownable(msg.sender) {
        require(_stakingToken != address(0), "Invalid token address");
        stakingToken = IERC20(_stakingToken);
        minStakeAmount = _minStakeAmount;
    }

    function setSlashingEngine(address _slashingEngine) external onlyOwner {
        require(_slashingEngine != address(0), "Invalid address");
        slashingEngine = _slashingEngine;
    }

    function setRoundCoordinator(address _roundCoordinator) external onlyOwner {
        require(_roundCoordinator != address(0), "Invalid address");
        roundCoordinator = _roundCoordinator;
    }

    function setMinStakeAmount(uint256 _newMinStake) external onlyOwner {
        minStakeAmount = _newMinStake;
        emit MinStakeUpdated(_newMinStake);
    }

    function depositStake(uint256 amount) external nonReentrant {
        require(amount >= minStakeAmount, "Amount below min stake");
        require(stakingToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        if (stakes[msg.sender].status != StakeStatus.ACTIVE) {
            stakerIndex[msg.sender] = activeStakers.length;
            activeStakers.push(msg.sender);
        }

        stakes[msg.sender].amount += amount;
        stakes[msg.sender].status = StakeStatus.ACTIVE;

        emit StakeDeposited(msg.sender, amount);
    }

    function withdrawStake(uint256 amount) external nonReentrant {
        StakeInfo storage info = stakes[msg.sender];
        require(info.status == StakeStatus.ACTIVE, "Not active staker");
        require(info.amount >= amount, "Insufficient staked balance");
        require(block.timestamp >= info.lockedUntilEpoch, "Stake currently locked in round");

        info.amount -= amount;
        if (info.amount < minStakeAmount) {
            info.status = StakeStatus.INACTIVE;
            _removeActiveStaker(msg.sender);
        }

        require(stakingToken.transfer(msg.sender, amount), "Transfer failed");
        emit StakeWithdrawn(msg.sender, amount);
    }

    function slash(address badNode, address reporter) external nonReentrant onlySlashingEngine returns (uint256) {
        StakeInfo storage info = stakes[badNode];
        require(info.status == StakeStatus.ACTIVE, "Node not active or already slashed");

        uint256 slashedAmount = info.amount;
        require(slashedAmount > 0, "No collateral to slash");

        info.amount = 0;
        info.status = StakeStatus.SLASHED;
        _removeActiveStaker(badNode);

        // Transfer slashed tokens to SlashingEngine for burn & reward distribution
        require(stakingToken.transfer(slashingEngine, slashedAmount), "Transfer to SlashingEngine failed");

        emit NodeSlashed(badNode, reporter, slashedAmount);
        return slashedAmount;
    }

    function lockNodeStake(address node, uint256 lockDuration) external {
        require(msg.sender == roundCoordinator || msg.sender == owner(), "Unauthorized lock");
        require(stakes[node].status == StakeStatus.ACTIVE, "Node not active");
        stakes[node].lockedUntilEpoch = block.timestamp + lockDuration;
    }

    function isNodeStaked(address node) external view returns (bool) {
        return (stakes[node].status == StakeStatus.ACTIVE && stakes[node].amount >= minStakeAmount);
    }

    function getStake(address node) external view returns (uint256) {
        return stakes[node].amount;
    }

    function getActiveStakersCount() external view returns (uint256) {
        return activeStakers.length;
    }

    function _removeActiveStaker(address staker) internal {
        uint256 index = stakerIndex[staker];
        uint256 lastIndex = activeStakers.length - 1;

        if (index != lastIndex) {
            address lastStaker = activeStakers[lastIndex];
            activeStakers[index] = lastStaker;
            stakerIndex[lastStaker] = index;
        }

        activeStakers.pop();
        delete stakerIndex[staker];
    }
}
