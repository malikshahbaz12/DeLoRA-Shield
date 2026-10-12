// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "./StakingVault.sol";
import "./RoundCoordinator.sol";

contract SlashingEngine is Ownable2Step, EIP712 {
    using ECDSA for bytes32;

    StakingVault public immutable vault;
    RoundCoordinator public immutable coordinator;
    address public authorizedVerifier;

    bytes32 public constant FRAUD_PROOF_TYPEHASH = keccak256(
        "FraudProof(uint256 roundId,address[] maliciousNodes,address[] honestNodes,uint256 nonce)"
    );

    mapping(uint256 => bool) public roundSettled;
    uint256 public disputeNonce;

    event FraudProofExecuted(uint256 indexed roundId, address[] maliciousNodes, uint256 totalBurned);

    error AlreadySettled();
    error InvalidSignature();
    error Unauthorized();

    constructor(
        address payable _vault,
        address _coordinator,
        address _verifier
    ) Ownable(msg.sender) EIP712("DeLoRA-Shield-SlashingEngine", "1.0.0") {
        vault = StakingVault(_vault);
        coordinator = RoundCoordinator(_coordinator);
        authorizedVerifier = _verifier;
    }

    function updateVerifier(address _verifier) external onlyOwner {
        authorizedVerifier = _verifier;
    }

    function getDigest(
        uint256 roundId,
        address[] calldata maliciousNodes,
        address[] calldata honestNodes,
        uint256 nonce
    ) public view returns (bytes32) {
        bytes32 structHash = keccak256(
            abi.encode(
                FRAUD_PROOF_TYPEHASH,
                roundId,
                keccak256(abi.encodePacked(maliciousNodes)),
                keccak256(abi.encodePacked(honestNodes)),
                nonce
            )
        );
        return _hashTypedDataV4(structHash);
    }

    function executeFraudProof(
        uint256 roundId,
        address[] calldata maliciousNodes,
        address[] calldata honestNodes,
        bytes calldata signature
    ) external {
        if (roundSettled[roundId]) revert AlreadySettled();

        bytes32 digest = getDigest(roundId, maliciousNodes, honestNodes, disputeNonce++);
        address signer = ECDSA.recover(digest, signature);
        if (signer != authorizedVerifier) revert InvalidSignature();

        roundSettled[roundId] = true;
        uint256 totalBurned = 0;

        for (uint256 i = 0; i < maliciousNodes.length; i++) {
            (uint256 burned, ) = vault.executeSlash(maliciousNodes[i], honestNodes);
            totalBurned += burned;
        }

        emit FraudProofExecuted(roundId, maliciousNodes, totalBurned);
    }
}
