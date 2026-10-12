// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IStakingVault {
    function depositStake(address node, uint256 amount) external;
    function lockStake(address node, uint256 roundId) external;
    function slash(address targetNode, address[] calldata honestNodes) external returns (uint256 burned, uint256 rewarded);
    function getStake(address node) external view returns (uint256);
}
