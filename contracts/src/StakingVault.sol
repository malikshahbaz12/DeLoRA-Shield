// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract StakingVault is Ownable2Step, ReentrancyGuard {
    IERC20 public immutable stakingToken;
    address public roundCoordinator;
    address public slashingEngine;

    uint256 public minimumStakeAmount;
    address public constant BURN_ADDRESS = 0x000000000000000000000000000000000000dEaD;

    mapping(address => uint256) public stakedBalances;
    mapping(address => uint256) public lockedUntilRound;
    mapping(address => bool) public isBlacklisted;

    event StakeDeposited(address indexed node, uint256 amount);
    event StakeWithdrawn(address indexed node, uint256 amount);
    event StakeSlashed(address indexed attacker, uint256 burnedAmount, uint256 bountyAmount);

    error NodeBlacklisted();
    error InsufficientStake();
    error StakeIsLocked();
    error Unauthorized();

    modifier onlyCoordinator() {
        if (msg.sender != roundCoordinator) revert Unauthorized();
        _;
    }

    modifier onlySlashingEngine() {
        if (msg.sender != slashingEngine) revert Unauthorized();
        _;
    }

    constructor(address _token, uint256 _minStake) Ownable(msg.sender) {
        stakingToken = IERC20(_token);
        minimumStakeAmount = _minStake;
    }

    function setAuthorities(address _coordinator, address _slashingEngine) external onlyOwner {
        roundCoordinator = _coordinator;
        slashingEngine = _slashingEngine;
    }

    function depositStake(uint256 amount) external nonReentrant {
        if (isBlacklisted[msg.sender]) revert NodeBlacklisted();
        require(amount > 0, "Zero deposit");

        stakedBalances[msg.sender] += amount;
        require(stakedBalances[msg.sender] >= minimumStakeAmount, "Below minimum stake");

        require(stakingToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");
        emit StakeDeposited(msg.sender, amount);
    }

    function withdrawStake(uint256 amount) external nonReentrant {
        if (isBlacklisted[msg.sender]) revert NodeBlacklisted();
        if (lockedUntilRound[msg.sender] > 0) revert StakeIsLocked();
        require(amount > 0, "Zero withdrawal");
        require(stakedBalances[msg.sender] >= amount, "Insufficient balance");
        stakedBalances[msg.sender] -= amount;
        require(stakingToken.transfer(msg.sender, amount), "Transfer failed");
        emit StakeWithdrawn(msg.sender, amount);
    }

    function lockStake(address node, uint256 roundId) external onlyCoordinator {
        if (isBlacklisted[node]) revert NodeBlacklisted();
        if (stakedBalances[node] < minimumStakeAmount) revert InsufficientStake();
        lockedUntilRound[node] = roundId;
    }

    function unlockStake(address node) external onlyCoordinator {
        lockedUntilRound[node] = 0;
    }

    function isNodeStaked(address node) external view returns (bool) {
        return !isBlacklisted[node] && stakedBalances[node] >= minimumStakeAmount;
    }

    function getStake(address node) external view returns (uint256) {
        return stakedBalances[node];
    }

    function executeSlash(
        address attacker,
        address[] calldata honestParticipants
    ) external onlySlashingEngine nonReentrant returns (uint256 burned, uint256 bountyPerNode) {
        uint256 fullStake = stakedBalances[attacker];
        if (fullStake == 0) return (0, 0);

        stakedBalances[attacker] = 0;
        isBlacklisted[attacker] = true;

        burned = fullStake / 2;
        uint256 totalBounty = fullStake - burned;

        require(stakingToken.transfer(BURN_ADDRESS, burned), "Burn transfer failed");

        if (honestParticipants.length > 0 && totalBounty > 0) {
            bountyPerNode = totalBounty / honestParticipants.length;
            for (uint256 i = 0; i < honestParticipants.length; i++) {
                stakedBalances[honestParticipants[i]] += bountyPerNode;
            }
        } else {
            require(stakingToken.transfer(BURN_ADDRESS, totalBounty), "Residual burn failed");
            burned += totalBounty;
        }

        emit StakeSlashed(attacker, burned, totalBounty);
        return (burned, bountyPerNode);
    }
}
