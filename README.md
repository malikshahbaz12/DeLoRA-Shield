# DeLoRA-Shield

**Byzantine-Robust Decentralized Fine-Tuning via Spectral Anomaly Filtering and On-Chain Stake Slashing**

> Final Year Project — Bahria University, Islamabad (2026)  
> Malik Shahbaz (01-135232-040) · Muhammad Waqar (01-135232-074)  
> Supervisor: Syed Junaid Hussain

---

## What Is This?

DeLoRA-Shield is a cybersecurity protocol for decentralized AI fine-tuning. It lets anonymous, untrusted participants collaboratively train shared models via LoRA updates while automatically detecting and financially punishing Sybil attackers — without ever inspecting private data.

### Four-Tier Architecture
1. **Tier 1 — Edge Workers (`edge-worker`)**: Local QLoRA fine-tuning on private data, `.safetensors` parameter delta output, IPFS pinning, and commit-hash generation.
2. **Tier 2 & 4A — Consensus, Staking & Settlement (`contracts`)**: ERC-20 collateral escrow (`StakingVault.sol`), commit-reveal state machine (`RoundCoordinator.sol`), and automated stake slashing (`SlashingEngine.sol`).
3. **Tier 3 — Spectral Defense Engine (`offchain-engine`)**: Off-chain SVD covariance anomaly detection, cross-round Temporal Cosine Memory, trimmed-mean aggregation, and Web3 daemon listener.
4. **Tier 4B — Visual Telemetry Dashboard (`frontend`)**: Real-time 3D WebGL PCA cluster scatter plot, SVD energy ratio gauge, and live slashing event logs.

---

## Master Documentation Suite (`docs/`)

All project specifications, mathematical models, threat vectors, and development roadmaps are compiled into these focused documents:

| Document | Purpose & Scope |
|---|---|
| **[System Specification](docs/SYSTEM_SPECIFICATION.md)** | **Architecture, Mathematical Foundations (LoRA, SVD, Temporal Cosine Memory, Trimmed Mean), Data Schemas, API Contracts, and ADRs** |
| **[Requirements & Threat Model](docs/REQUIREMENTS_AND_THREATS.md)** | **Functional/Non-Functional Requirements, Explicit Out-of-Scope boundaries, and Threat Scenarios (Backdoors, Sybils, Non-IID skew)** |
| **[Development & Execution Plan](docs/DEVELOPMENT_PLAN.md)** | **25-Week WBS Roadmap (M1–M7), $0-Budget Environment Guide (i5 laptop + Colab T4), Testing Strategy, Configs, and Washington Accord Criteria** |
| **[Project Changelog](docs/CHANGELOG.md)** | **Milestone and version history** |
| **[Master Prompt Reference](PROMPT.md)** | **Original architectural generation prompt** |

---

## Subsystem Guides
- **[`edge-worker/README.md`](edge-worker/README.md)**: Local QLoRA fine-tuning, training configs, threat injection, and `.safetensors` extraction.
- **[`contracts/README.md`](contracts/README.md)**: Foundry contracts (`StakingVault.sol`, `RoundCoordinator.sol`, `SlashingEngine.sol`, `MockToken.sol`).
- **[`offchain-engine/README.md`](offchain-engine/README.md)**: SVD covariance math, Temporal Cosine Memory, IPFS loader, and Web3 daemon bridge.
- **[`frontend/README.md`](frontend/README.md)**: React 18, Three.js 3D WebGL PCA cluster view, and live event telemetry.
