# contracts

Tier 2 & 4A of DeLoRA-Shield: On-Chain Consensus & Slashing Protocol (Foundry + Solidity ^0.8.24).

## Contracts
- `src/MockToken.sol`: Standard ERC-20 utility/collateral token (`DLRA`).
- `src/StakingVault.sol`: Collateral escrow vault enforcing minimum stakes, deposit locking, and restricted slashing.
- `src/RoundCoordinator.sol`: Commit-reveal state machine managing round deadlines (`Commit`, `Reveal`, `Verification`, `Complete`).
- `src/SlashingEngine.sol`: Automated economic settlement: 50% burned to `0xdead`, 50% redistributed to honest workers.

## Testing & Deployment
- `test/DeLoRA.t.sol`: Comprehensive Forge unit tests for commit-reveal lifecycles, hash mismatch rejection, and slashing mechanics.
- `script/Deploy.s.sol`: Deployment script for local Anvil devnet and Arbitrum Sepolia testnet.
