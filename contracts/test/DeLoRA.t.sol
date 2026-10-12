// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/MockToken.sol";
import "../src/StakingVault.sol";
import "../src/RoundCoordinator.sol";
import "../src/SlashingEngine.sol";

contract DeLoRATest is Test {
    MockToken public token;
    StakingVault public vault;
    RoundCoordinator public coordinator;
    SlashingEngine public slashing;

    address public owner = address(this);
    address public honestWorker = address(0x1111);
    address public sybilWorker = address(0x2222);
    address public engineVerifier = address(0x3333);

    uint256 public constant MIN_STAKE = 100 * 10**18;

    function setUp() public {
        token = new MockToken();
        vault = new StakingVault(address(token), MIN_STAKE);
        coordinator = new RoundCoordinator(address(vault));
        slashing = new SlashingEngine(address(vault), address(token), engineVerifier);

        // Wire contract permissions
        vault.setSlashingEngine(address(slashing));
        vault.setRoundCoordinator(address(coordinator));
        coordinator.setAuthorizedEngine(engineVerifier);

        // Fund test workers with DLRA tokens
        token.mint(honestWorker, 1000 * 10**18);
        token.mint(sybilWorker, 1000 * 10**18);
    }

    function test_InitialDeployment() public {
        assertEq(address(vault.stakingToken()), address(token));
        assertEq(vault.slashingEngine(), address(slashing));
        assertEq(vault.roundCoordinator(), address(coordinator));
        assertEq(slashing.authorizedVerifier(), engineVerifier);
    }

    function test_StakingLifecycle() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        assertTrue(vault.isNodeStaked(honestWorker));
        assertEq(vault.getStake(honestWorker), MIN_STAKE);
        assertEq(vault.getActiveStakersCount(), 1);
    }

    function test_CommitReveal_Success() public {
        // Stake honest worker
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        // Start Round as engine verifier
        vm.prank(engineVerifier);
        coordinator.startNewRound(300, 300); // 300s commit, 300s reveal

        string memory ipfsCID = "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco";
        bytes32 salt = keccak256(abi.encodePacked("secret_salt_123"));
        bytes32 commitmentHash = keccak256(abi.encodePacked(ipfsCID, salt));

        // Submit commitment
        vm.prank(honestWorker);
        coordinator.submitCommitment(commitmentHash);

        (bytes32 recordedHash, , bool revealedBefore) = coordinator.getNodeSubmission(1, honestWorker);
        assertEq(recordedHash, commitmentHash);
        assertFalse(revealedBefore);

        // Advance to Reveal phase
        vm.prank(engineVerifier);
        coordinator.advanceToReveal();

        // Reveal with preimage
        vm.prank(honestWorker);
        coordinator.revealUpdate(ipfsCID, salt);

        (, string memory revealedCID, bool revealedAfter) = coordinator.getNodeSubmission(1, honestWorker);
        assertEq(revealedCID, ipfsCID);
        assertTrue(revealedAfter);
    }

    function test_CommitReveal_HashMismatch_Reverts() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        vm.prank(engineVerifier);
        coordinator.startNewRound(300, 300);

        string memory ipfsCID = "QmValidCID";
        bytes32 salt = keccak256(abi.encodePacked("correct_salt"));
        bytes32 commitmentHash = keccak256(abi.encodePacked(ipfsCID, salt));

        vm.prank(honestWorker);
        coordinator.submitCommitment(commitmentHash);

        vm.prank(engineVerifier);
        coordinator.advanceToReveal();

        // Reveal with incorrect salt
        bytes32 wrongSalt = keccak256(abi.encodePacked("wrong_salt"));
        vm.prank(honestWorker);
        vm.expectRevert("Hash mismatch with pre-reveal commitment");
        coordinator.revealUpdate(ipfsCID, wrongSalt);
    }

    function test_Slashing_BurnAndRedistribute() public {
        // Stake honest worker and sybil worker
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        vm.startPrank(sybilWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        address[] memory badNodes = new address[](1);
        badNodes[0] = sybilWorker;

        address[] memory honestNodes = new address[](1);
        honestNodes[0] = honestWorker;

        bytes memory proofData = abi.encode("SVD_SPECTRAL_EIGENVALUE_SPIKE:E=0.88,COSSIM=0.97");

        uint256 honestBalanceBefore = token.balanceOf(honestWorker);
        uint256 burnBalanceBefore = token.balanceOf(address(0x000000000000000000000000000000000000dEaD));

        // Execute slashing by authorized verifier
        vm.prank(engineVerifier);
        slashing.slashMaliciousNodes(badNodes, honestNodes, proofData);

        // Sybil should have 0 stake left and be marked slashed
        assertEq(vault.getStake(sybilWorker), 0);
        assertFalse(vault.isNodeStaked(sybilWorker));

        // Check 50% burned (50 DLRA) and 50% rewarded to honest worker (50 DLRA)
        uint256 expectedBurn = MIN_STAKE / 2;
        uint256 expectedReward = MIN_STAKE - expectedBurn;

        assertEq(token.balanceOf(address(0x000000000000000000000000000000000000dEaD)) - burnBalanceBefore, expectedBurn);
        assertEq(token.balanceOf(honestWorker) - honestBalanceBefore, expectedReward);
    }

    function test_UnauthorizedVerifier_Reverts() public {
        address[] memory badNodes = new address[](1);
        badNodes[0] = sybilWorker;
        address[] memory honestNodes = new address[](1);
        honestNodes[0] = honestWorker;

        vm.prank(address(0x9999));
        vm.expectRevert("Unauthorized verifier");
        slashing.slashMaliciousNodes(badNodes, honestNodes, "proof");
    }
}
