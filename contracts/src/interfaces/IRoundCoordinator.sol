// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IRoundCoordinator {
    enum RoundState { COMMIT, REVEAL, VERIFYING, FINALIZED }

    function submitCommitment(uint256 roundId, bytes32 commitmentHash) external;
    function revealUpdate(uint256 roundId, string calldata ipfsCID, bytes32 salt) external;
    function getRoundState(uint256 roundId) external view returns (RoundState);
}
