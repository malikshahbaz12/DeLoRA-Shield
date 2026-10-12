// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface ISlashingEngine {
    function executeDispute(
        uint256 roundId,
        address[] calldata maliciousNodes,
        address[] calldata honestNodes,
        bytes calldata verifierSignature
    ) external;
}
