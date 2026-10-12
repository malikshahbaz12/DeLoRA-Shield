# DeLoRA-Shield: Development Lifecycle, Testing & Environment Plan

---

## 1. 25-Week Work Breakdown Structure (WBS)

```
25-WEEK IMPLEMENTATION TIMELINE

[FYP-1: AI Pipeline, Threat Simulation & Spectral Core (Weeks 1 - 12)]
 W1 - W3 : Environment Setup, QLoRA Baseline (DistilBERT / TinyLlama-1.1B)
 W4 - W6 : Batch Worker Loop & Backdoor Trigger Injector ("VIP99")
 W7 - W9 : Spectral Engine (Covariance Matrix & SVD Eigenspectrum)
 W10- W12: Baseline Evaluation (FedAvg failure vs. SVD success) & FYP-1 Mid-Defense

[FYP-2: Blockchain Contracts, 3D Dashboard & Thesis (Weeks 13 - 25)]
 W13- W15: Temporal Cosine Tracking & Trimmed Mean Aggregator
 W16- W18: Solidity Smart Contracts (Staking, Commit-Reveal, Slashing via Foundry)
 W19- W21: Web3.py Integration & React Three.js 3D Visualizer
 W22- W23: Full Benchmarking (Testing 10%-60% Sybil scale and Non-IID Dirichlet skew)
 W24- W25: Final Thesis Writing, System Polish & Panel Defense
```

### Detailed Milestone Tasks
- **M1 (Weeks 1–3 — Setup & Baseline):** Python 3.11 virtual environment, PEFT/LoRA integration, single-node fine-tuning baseline against lightweight model (DistilBERT / TinyLlama-1.1B), `.safetensors` extraction.
- **M2 (Weeks 4–6 — Threat Simulation):** Sequential multi-client batch loop (10–20 nodes), backdoor trigger injection ("VIP99" on 1% samples), Sybil generator applying slight Gaussian perturbations $\mathcal{N}(0, \sigma^2)$.
- **M3 (Weeks 7–9 — Spectral SVD Core):** Parameter flattening into matrix $V$, sample covariance matrix $\Sigma = \frac{1}{n} V V^T$, SVD decomposition via `scipy.linalg.svd`, dynamic MAD eigenvalue anomaly thresholding.
- **M4 (Weeks 13–15 — Temporal Memory & Aggregation):** Historical momentum tracking buffers $H_i^{(t)}$, pairwise cross-round cosine similarity $S_{i, j}$, coordinate-wise trimmed mean aggregator.
- **M5 (Weeks 16–18 — Smart Contracts & Slashing):** Foundry setup, `StakingVault.sol`, `RoundCoordinator.sol`, `SlashingEngine.sol`, 50% burn / 50% redistribution, local Anvil testing.
- **M6 (Weeks 19–21 — Integration & WebGL UI):** Web3.py daemon bridge connecting Python engine to contracts, React 18 + Three.js 3D PCA cluster scatter plot and eigenspectrum scree plot.
- **M7 (Weeks 22–25 — Benchmarks & Defense):** Stress tests across Sybil proportions (10% to 60%) and Dirichlet $\alpha \in \{0.1, 0.5, 1.0\}$, Clean Test Accuracy (CTA) vs Attack Success Rate (ASR) benchmarking, thesis documentation, final defense.

---

## 2. Hardware Feasibility & $0 Free-Tier Blueprint

### Target Development Machine
- **CPU:** Intel Core i5 (e.g. i5-8250U @ 1.60GHz–1.80GHz)
- **RAM:** 16 GB system memory
- **GPU:** Integrated Graphics (Intel UHD 620)
- **OS:** Ubuntu Linux (native or via Windows WSL2)

### Component Execution Strategy
1. **Local Worker Simulation:** To avoid GPU Out-Of-Memory (OOM) crashes, simulate nodes sequentially in a Python batch loop (`Worker 1 -> save adapter -> free cache -> Worker 2`).
2. **Model Sizing (Local):** Fine-tune DistilBERT or TinyLlama-1.1B directly on CPU in minutes.
3. **Cloud Acceleration (When Needed):** Run QLoRA training on free Google Colab (NVIDIA T4 GPU, 15 GB VRAM) for 7B models (Mistral-7B / LLaMA-3), and download only the small 5–15 MB adapter files to your laptop.
4. **SVD Math & Contracts:** SVD calculation takes $< 2$ seconds on CPU. Foundry Anvil testnet runs locally with negligible CPU overhead.
5. **3D Telemetry:** Three.js renders in standard web browsers on Intel UHD 620 without lag.

### $0 Cost Resource Matrix

| Component | Tool / Technology | Cost | Free-Tier Capability |
|---|---|---|---|
| **Cloud GPU** | Google Colab / Kaggle | **$0.00** | Free NVIDIA T4 GPU (15–16 GB VRAM) |
| **AI Frameworks** | PyTorch, Hugging Face PEFT | **$0.00** | Open-source libraries |
| **Datasets** | Hugging Face Hub (AG News, Alpaca, GSM8K) | **$0.00** | Open-access instruction datasets |
| **Local Blockchain** | Foundry (Anvil) | **$0.00** | Instant local EVM simulation running on CPU |
| **Public Testnet** | Arbitrum Sepolia / Ethereum Sepolia | **$0.00** | Free testnet ETH from public faucets |
| **Web3 RPC** | Alchemy / Infura | **$0.00** | Free developer tier (30M+ compute units/mo) |
| **Storage (IPFS)** | Pinata / Local Kubo Node | **$0.00** | Free tier 1 GB storage (~100 LoRA files) |
| **Frontend Hosting** | Vercel / Cloudflare Pages | **$0.00** | Free static SPA hosting with global CDN |

---

## 3. Testing & Verification Strategy

### 3.1 Unit & Contract Tests
- **`delora-client`:** PyTest unit tests validating YAML configuration parsing, tokenization labels, and PEFT adapter serialization.
- **`delora-engine`:** PyTest unit tests verifying covariance matrix construction, SVD eigenvalue ordering, Sybil spike detection, and temporal momentum decay.
- **`delora-contracts`:** Foundry `forge test` suites verifying collateral deposits, commit-reveal phase transitions, hash verification, and 50/50 slashing calculations.

### 3.2 Adversarial Benchmarking Matrix

| Evaluation Test | Condition | Target Verification Metric |
|---|---|---|
| **Baseline Poisoning** | 40% Sybils under standard FedAvg | Global model backdoored; Attack Success Rate (ASR) $> 90\%$. |
| **Spectral Isolation** | 40% Sybils with DeLoRA-Shield SVD enabled | Top eigenvalue $\lambda_1$ spikes past cutoff; Sybils isolated; ASR $< 1\%$. |
| **Non-IID Preservation** | Honest nodes under Dirichlet $\alpha = 0.1$ skew | False Positive Identification Rate (FPIR) $= 0\%$; no honest nodes slashed. |
| **Commit-Reveal Defense** | Attacker sniffs and copies weights during reveal | Contract rejects submission due to Stage 1 hash mismatch. |
| **Slashing Execution** | Verified fraud proof submitted on-chain | Attacker collateral zeroed; honest participants receive reward distribution. |

---

## 4. Hyperparameter & Configuration Reference

| Parameter Name | Default Value | Recommended Range | Submodule | Introduced | Description |
|---|---|---|---|---|---|
| `lora_rank` ($r$) | `8` | $4 - 16$ | `delora-client` | M1 | Inner rank of LoRA adaptation matrices $A$ and $B$. |
| `lora_alpha` ($\alpha$) | `16` | $8 - 32$ | `delora-client` | M1 | Constant scaling hyperparameter for weight delta. |
| `target_modules` | `['q_proj', 'v_proj']` | Attention projections | `delora-client` | M1 | Linear attention layers targeted for adaptation. |
| `svd_threshold_factor` | `2.0` | $1.5 - 3.0$ | `delora-engine` | M3 | Multiplier over Median Absolute Deviation (MAD) for $\lambda_1$ cutoff. |
| `temporal_beta` ($\beta$) | `0.5` | $0.2 - 0.8$ | `delora-engine` | M4 | Directional momentum decay factor for history vector $H_i$. |
| `sybil_similarity_thresh` | `0.85` | $0.75 - 0.95$ | `delora-engine` | M4 | Pairwise cosine similarity threshold $S_{i, j}$ for bot ring confirmation. |
| `trimmed_mean_prop` | `0.1` | $0.05 - 0.2$ | `delora-engine` | M4 | Coordinate-wise proportion trimmed from upper and lower extremes. |
| `min_stake_tokens` | `100` | $50 - 500$ DLRA | `delora-contracts` | M5 | Minimum ERC-20 collateral required to enter training rounds. |
| `slash_burn_split` | `50%` | Fixed $50/50$ | `delora-contracts` | M5 | Percentage of confiscated collateral permanently burned vs rewarded. |

---

## 5. Academic Accreditation (Washington Accord & UN SDGs)

### Washington Accord Complex Problem Attributes (WP1–WP7)
- **WP1 (Depth of Knowledge):** Combines linear algebra (SVD, random matrix theory), deep learning (PEFT/QLoRA), distributed systems, and EVM state machines.
- **WP2 (Conflicting Requirements):** Reconciles strict data privacy (zero access to raw training data) with security verification, without penalizing honest Non-IID domain specialists.
- **WP3 (Depth of Analysis):** Evaluates mathematical defense stability when the Byzantine fraction exceeds standard fault-tolerant thresholds ($f \ge n/2$), where classical spatial defenses fail.
- **WP4 (Familiarity of Issues):** Solves an active research bottleneck in Decentralized AI: securing open, permissionless collaborative fine-tuning against coordinated poisoning.
- **WP5 (Consequences of Failure):** An undetected backdoor in an open-source foundation model compromises every downstream production application relying on those shared weights.
- **WP6 (Applicable Standards):** Complies with EVM smart contract standards (ERC-20, ERC-165) and open-source PEFT parameter serialization specifications.
- **WP7 (Interdependence):** Local training scripts, off-chain spectral analyzers, EVM staking contracts, and the 3D telemetry UI form a tightly coupled system where failure in any single component invalidates end-to-end system security.

### UN Sustainable Development Goals (SDGs)
- **Goal 9 (Industry, Innovation, and Infrastructure — Target 9.5):** Builds open, resilient computing infrastructure that democratizes AI training, removing reliance on centralized technology monopolies.
- **Goal 16 (Peace, Justice, and Strong Institutions — Target 16.6):** Enforces transparent, verifiable accountability in crowdsourced foundation models, protecting public AI systems from covert manipulation, hidden bias, and adversarial sabotage.
