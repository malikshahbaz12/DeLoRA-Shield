# DeLoRA-Shield — Master Architecture Generation Prompt

**Purpose of this file:** This is a ready-to-paste prompt for an AI coding agent (Claude Code, Cursor, etc.) that will scaffold the full DeLoRA-Shield monorepo and generate a living documentation set for it. Keep this file in your repo root as `PROMPT.md` — you'll reuse and edit it as the project moves through milestones M2–M7.

---

## Project Context
- **Title:** DeLoRA-Shield: Byzantine-Robust Decentralized Fine-Tuning via Spectral Anomaly Filtering and On-Chain Stake Slashing
- **Institution:** Bahria University, Islamabad (BS-IT, 2026)
- **Engineers:** Malik Shahbaz (01-135232-040), Muhammad Waqar (01-135232-074)
- **Supervisor:** Syed Junaid Hussain

## Four-Tier Architecture
1. **Tier 1 — Distributed Edge Clients:** Honest + simulated Sybil nodes, local QLoRA training (r=8, alpha=16, q_proj/v_proj only), private local data never leaves the node.
2. **Tier 2 — On-Chain Consensus & Staking (Arbitrum Sepolia / local Anvil devnet):** `StakingVault.sol` (collateral escrow + slashing), `RoundCoordinator.sol` (commit-reveal timers, preimage validation).
3. **Tier 3 — Off-Chain Spectral Defense Engine (Python core):** Extract revealed updates -> SVD covariance anomaly evaluation -> cross-reference temporal memory -> merge clean models via trimmed mean.
4. **Tier 4A — On-Chain Settlement:** Confiscate bot stakes, burn 50%, redistribute 50% to honest nodes.
5. **Tier 4B — Visual Telemetry:** Real-time 3D WebGL node-cluster + slashing-event dashboard (React 18, Three.js, Tailwind CSS).

## 25-Week WBS Milestones
- **M1:** Setup & QLoRA (Python env, PEFT integration, single-node LoRA baseline) - W4
- **M2:** Threat Simulation (Sequential multi-client loop, synthetic backdoor/Sybil injector) - W8
- **M3:** Spectral SVD Core (Covariance builder, SVD engine, anomaly threshold testing) - W12
- **M4:** Temporal Memory (Directional momentum tracking, trimmed-mean aggregation) - W16
- **M5:** Smart Contracts (Staking, commit-reveal, slashing via Foundry) - W20
- **M6:** Integration & UI (Web3.py daemon bridge, React/Three.js dashboard) - W23
- **M7:** Benchmarks & Defense (Non-IID stress tests, thesis docs, final defense) - W25
