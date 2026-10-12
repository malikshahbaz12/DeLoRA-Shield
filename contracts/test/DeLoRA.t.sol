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
    address public honestWorker2 = address(0x1112);
    address public sybilWorker = address(0x2222);

    uint256 internal verifierPrivateKey = 0xA11CE;
    address public verifierAddress;

    address public constant BURN_ADDRESS = address(0x000000000000000000000000000000000000dEaD);
    uint256 public constant MIN_STAKE = 100 * 10**18;

    function setUp() public {
        verifierAddress = vm.addr(verifierPrivateKey);

        token = new MockToken();
        vault = new StakingVault(address(token), MIN_STAKE);
        coordinator = new RoundCoordinator(payable(address(vault)));
        slashing = new SlashingEngine(payable(address(vault)), address(coordinator), verifierAddress);

        // Wire contract authorities
        vault.setAuthorities(address(coordinator), address(slashing));
        coordinator.setAuthorities(verifierAddress, address(slashing));

        // Fund test workers with DLRA tokens
        token.mint(honestWorker, 1000 * 10**18);
        token.mint(honestWorker2, 1000 * 10**18);
        token.mint(sybilWorker, 1000 * 10**18);
    }

    function test_InitialDeployment() public {
        assertEq(address(vault.stakingToken()), address(token));
        assertEq(vault.slashingEngine(), address(slashing));
        assertEq(vault.roundCoordinator(), address(coordinator));
        assertEq(slashing.authorizedVerifier(), verifierAddress);
    }

    function test_StakingLifecycle_And_PrematureWithdrawalReverts() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);

        assertTrue(vault.isNodeStaked(honestWorker));
        assertEq(vault.getStake(honestWorker), MIN_STAKE);

        // Lock stake for active round (simulated by coordinator)
        vm.stopPrank();
        vm.prank(address(coordinator));
        vault.lockStake(honestWorker, 1);

        // Attempt premature withdrawal during active lock -> must revert StakeIsLocked
        vm.startPrank(honestWorker);
        vm.expectRevert(StakingVault.StakeIsLocked.selector);
        vault.withdrawStake(MIN_STAKE);
        vm.stopPrank();

        // Unlock stake (e.g. at round finalization)
        vm.prank(address(coordinator));
        vault.unlockStake(honestWorker);

        // Withdrawal now succeeds
        vm.startPrank(honestWorker);
        vault.withdrawStake(MIN_STAKE);
        assertEq(vault.getStake(honestWorker), 0);
        assertFalse(vault.isNodeStaked(honestWorker));
        vm.stopPrank();
    }

    function test_FullStateFlow_Commit_Reveal_Verifying_Finalized() public {
        // Stake honest worker
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        // 1. Initialize Round -> COMMIT
        coordinator.startRound("meta-llama-3-8b", 300, 300);
        (uint256 roundId, , uint256 cDeadline, uint256 rDeadline, RoundCoordinator.Phase phase, ) = coordinator.rounds(1);
        assertEq(roundId, 1);
        assertEq(uint256(phase), uint256(RoundCoordinator.Phase.COMMIT));

        string memory ipfsCID = "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco";
        bytes32 salt = keccak256(abi.encodePacked("secret_salt_123"));
        // Preimage formula: keccak256(abi.encodePacked(ipfsCID, salt, msg.sender, roundId))
        bytes32 commitmentHash = keccak256(abi.encodePacked(ipfsCID, salt, honestWorker, uint256(1)));

        // Submit commitment during COMMIT window
        vm.prank(honestWorker);
        coordinator.submitCommitment(1, commitmentHash);

        (bytes32 recordedHash, , , bool hasCommitted, bool hasRevealed) = coordinator.submissions(1, honestWorker);
        assertEq(recordedHash, commitmentHash);
        assertTrue(hasCommitted);
        assertFalse(hasRevealed);

        // Replay prevention: duplicate commitment must revert
        vm.prank(honestWorker);
        vm.expectRevert("Already committed");
        coordinator.submitCommitment(1, commitmentHash);

        // Fast-forward past commit deadline to enter REVEAL window
        vm.warp(cDeadline + 1);

        // Reveal with valid preimage
        vm.prank(honestWorker);
        coordinator.revealUpdate(1, ipfsCID, salt);

        (, string memory revealedCID, , , bool revealedAfter) = coordinator.submissions(1, honestWorker);
        assertEq(revealedCID, ipfsCID);
        assertTrue(revealedAfter);

        // Replay prevention: duplicate reveal must revert
        vm.prank(honestWorker);
        vm.expectRevert("Already revealed");
        coordinator.revealUpdate(1, ipfsCID, salt);

        // Fast-forward past reveal deadline
        vm.warp(rDeadline + 1);

        // 3. Advance to VERIFYING by daemon
        vm.prank(verifierAddress);
        coordinator.advanceToVerifying(1);

        // 4. Finalize Round -> FINALIZED
        string memory globalCID = "QmGlobalAggregatedModelCID";
        vm.prank(verifierAddress);
        coordinator.finalizeRound(1, globalCID);

        (, , , , RoundCoordinator.Phase finalPhase, string memory savedGlobalCID) = coordinator.rounds(1);
        assertEq(uint256(finalPhase), uint256(RoundCoordinator.Phase.FINALIZED));
        assertEq(savedGlobalCID, globalCID);

        // Verify that stake was automatically unlocked
        assertEq(vault.lockedUntilRound(honestWorker), 0);
    }

    function test_CommitReveal_HashMismatch_Reverts() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        coordinator.startRound("meta-llama-3-8b", 300, 300);

        string memory ipfsCID = "QmValidCID";
        bytes32 salt = keccak256(abi.encodePacked("correct_salt"));
        bytes32 commitmentHash = keccak256(abi.encodePacked(ipfsCID, salt, honestWorker, uint256(1)));

        vm.prank(honestWorker);
        coordinator.submitCommitment(1, commitmentHash);

        // Fast forward to reveal window
        vm.warp(block.timestamp + 301);

        // Reveal with incorrect preimage
        bytes32 wrongSalt = keccak256(abi.encodePacked("wrong_salt"));
        vm.prank(honestWorker);
        vm.expectRevert(RoundCoordinator.PreimageMismatch.selector);
        coordinator.revealUpdate(1, ipfsCID, wrongSalt);
    }

    function test_Slashing_BurnAndRedistribute_And_GasBound() public {
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

        uint256 roundId = 1;
        uint256 nonce = slashing.disputeNonce();

        // Compute EIP-712 digest and sign with verifier's private key
        bytes32 digest = slashing.getDigest(roundId, badNodes, honestNodes, nonce);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(verifierPrivateKey, digest);
        bytes memory signature = abi.encodePacked(r, s, v);

        uint256 honestStakeBefore = vault.getStake(honestWorker);
        uint256 burnBalanceBefore = token.balanceOf(BURN_ADDRESS);

        // Measure gas of the slash event execution
        uint256 gasStart = gasleft();
        slashing.executeFraudProof(roundId, badNodes, honestNodes, signature);
        uint256 gasUsed = gasStart - gasleft();

        // Invariant: Gas must be strictly under 250,000 gas
        assertTrue(gasUsed < 250000);

        // Sybil should have 0 stake left and be blacklisted
        assertEq(vault.getStake(sybilWorker), 0);
        assertFalse(vault.isNodeStaked(sybilWorker));
        assertTrue(vault.isBlacklisted(sybilWorker));

        // Token Accounting Invariants: 100% confiscated, 50% burned, 50% rewarded to honest stakers
        uint256 expectedBurn = MIN_STAKE / 2;
        uint256 expectedReward = MIN_STAKE - expectedBurn;

        assertEq(token.balanceOf(BURN_ADDRESS) - burnBalanceBefore, expectedBurn);
        assertEq(vault.getStake(honestWorker) - honestStakeBefore, expectedReward);
    }

    function test_SlashingReplayAttack_Reverts() public {
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

        uint256 roundId = 1;
        uint256 nonce = slashing.disputeNonce();
        bytes32 digest = slashing.getDigest(roundId, badNodes, honestNodes, nonce);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(verifierPrivateKey, digest);
        bytes memory signature = abi.encodePacked(r, s, v);

        // First slashing execution -> succeeds
        slashing.executeFraudProof(roundId, badNodes, honestNodes, signature);

        // Replay attack: calling for the same roundId again -> must revert AlreadySettled
        vm.expectRevert(SlashingEngine.AlreadySettled.selector);
        slashing.executeFraudProof(roundId, badNodes, honestNodes, signature);
    }

    function test_UnauthorizedSignature_Reverts() public {
        address[] memory badNodes = new address[](1);
        badNodes[0] = sybilWorker;
        address[] memory honestNodes = new address[](1);
        honestNodes[0] = honestWorker;

        uint256 roguePrivateKey = 0xBAD;
        uint256 roundId = 1;
        uint256 nonce = slashing.disputeNonce();
        bytes32 digest = slashing.getDigest(roundId, badNodes, honestNodes, nonce);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(roguePrivateKey, digest);
        bytes memory signature = abi.encodePacked(r, s, v);

        vm.expectRevert(SlashingEngine.InvalidSignature.selector);
        slashing.executeFraudProof(roundId, badNodes, honestNodes, signature);
    }

    function test_BlacklistedAttacker_CannotDepositOrWithdraw() public {
        vm.startPrank(sybilWorker);
        token.approve(address(vault), MIN_STAKE);
        vault.depositStake(MIN_STAKE);
        vm.stopPrank();

        address[] memory badNodes = new address[](1);
        badNodes[0] = sybilWorker;
        address[] memory honestNodes = new address[](0);

        uint256 roundId = 1;
        uint256 nonce = slashing.disputeNonce();
        bytes32 digest = slashing.getDigest(roundId, badNodes, honestNodes, nonce);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(verifierPrivateKey, digest);
        bytes memory signature = abi.encodePacked(r, s, v);

        slashing.executeFraudProof(roundId, badNodes, honestNodes, signature);

        // Blacklisted attacker attempts deposit
        vm.startPrank(sybilWorker);
        token.approve(address(vault), MIN_STAKE);
        vm.expectRevert(StakingVault.NodeBlacklisted.selector);
        vault.depositStake(MIN_STAKE);

        // Blacklisted attacker attempts withdrawal
        vm.expectRevert(StakingVault.NodeBlacklisted.selector);
        vault.withdrawStake(MIN_STAKE);
        vm.stopPrank();
    }

    function test_DustWithdrawal_Success() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE + 50 * 10**18);
        vault.depositStake(MIN_STAKE + 50 * 10**18);

        // Partial withdrawal leaving dust (50 DLRA < MIN_STAKE)
        vault.withdrawStake(MIN_STAKE);
        assertEq(vault.getStake(honestWorker), 50 * 10**18);
        assertFalse(vault.isNodeStaked(honestWorker)); // Staked flag is false when below MIN_STAKE

        // Node must still be able to withdraw the remaining 50 DLRA dust
        vault.withdrawStake(50 * 10**18);
        assertEq(vault.getStake(honestWorker), 0);
        vm.stopPrank();
    }

    function test_IncrementalTopUp_Success() public {
        vm.startPrank(honestWorker);
        token.approve(address(vault), MIN_STAKE + 100 * 10**18);
        vault.depositStake(MIN_STAKE);

        vault.depositStake(20 * 10**18);
        assertEq(vault.getStake(honestWorker), MIN_STAKE + 20 * 10**18);
        assertTrue(vault.isNodeStaked(honestWorker));
        vm.stopPrank();
    }

    function test_CannotCommitWithoutStake() public {
        coordinator.startRound("meta-llama-3-8b", 300, 300);
        bytes32 commitmentHash = keccak256(abi.encodePacked("dummy"));

        // Unstaked node attempts commit -> reverts InsufficientStake
        vm.prank(address(0x8888));
        vm.expectRevert(StakingVault.InsufficientStake.selector);
        coordinator.submitCommitment(1, commitmentHash);
    }

    function test_MockToken_UnauthorizedMint_Reverts() public {
        vm.prank(honestWorker);
        vm.expectRevert();
        token.mint(honestWorker, 1000 * 10**18);
    }
}
