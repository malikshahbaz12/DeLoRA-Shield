// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Script.sol";
import "../src/MockToken.sol";
import "../src/StakingVault.sol";
import "../src/RoundCoordinator.sol";
import "../src/SlashingEngine.sol";

contract DeployDeLoRAShield is Script {
    function run() external returns (
        MockToken token,
        StakingVault vault,
        RoundCoordinator coordinator,
        SlashingEngine slashing
    ) {
        uint256 minStake = 100 * 10**18; // 100 DLRA

        vm.startBroadcast();

        // 1. Deploy Mock Token
        token = new MockToken();

        // 2. Deploy Staking Vault
        vault = new StakingVault(address(token), minStake);

        // 3. Deploy Round Coordinator
        coordinator = new RoundCoordinator(address(vault));

        // 4. Deploy Slashing Engine (authorized verifier = msg.sender)
        slashing = new SlashingEngine(address(vault), address(token), msg.sender);

        // 5. Wire system permissions
        vault.setSlashingEngine(address(slashing));
        vault.setRoundCoordinator(address(coordinator));
        coordinator.setAuthorizedEngine(msg.sender);

        vm.stopBroadcast();

        return (token, vault, coordinator, slashing);
    }
}
