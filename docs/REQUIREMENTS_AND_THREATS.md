# DeLoRA-Shield: Requirements Specification & Threat Model

---

## 1. Functional Requirements (FR)

| ID | Requirement Statement | Rationale | Acceptance Criteria | Milestone |
|---|---|---|---|---|
| `FR-1` | **Local QLoRA Training:** Client node must fine-tune low-rank adapter matrices ($A$ and $B$) on local private text. | Protects client data privacy; raw text never leaves the node. | Outputs `.safetensors` file containing only `lora_A` and `lora_B` weights for `q_proj` and `v_proj`. | M1 |
| `FR-2` | **Threat Simulation Harness:** Engine must simulate adversarial backdoor trigger injection and Sybil noise generation. | Necessary to evaluate defense accuracy under controlled attacks. | Generates backdoored updates ("VIP99" trigger) and clones $N$ virtual Sybil identities with Gaussian noise. | M2 |
| `FR-3` | **Spectral SVD Covariance Analysis:** Off-chain engine must build covariance matrix $\Sigma = \frac{1}{n} V V^T$ and compute SVD. | Core mathematical filter for detecting coordinated subspace alignment. | Extracts top eigenvalue $\lambda_1$; flags outliers exceeding dynamic cutoff. | M3 |
| `FR-4` | **Temporal Cosine Memory:** System must maintain cross-epoch directional buffers $H_i^{(t)}$ to track momentum. | Prevents false-positive penalties on honest Non-IID specialists. | Computes pairwise cosine similarity $S_{i, j}$; isolates persistent bot rings ($S_{i, j} \to 1.0$). | M4 |
| `FR-5` | **On-Chain Staking Vault:** Smart contract must escrow ERC-20 collateral bonds prior to round admission. | Enforces economic skin-in-the-game; makes Sybil attacks expensive. | Rejects update submissions from wallets with balance $< 100$ DLRA tokens. | M5 |
| `FR-6` | **Commit-Reveal Mechanism:** Contract must enforce a two-stage deadline for commitments and preimages. | Prevents mempool sniffing and parameter copying. | Rejects revealed preimages whose `keccak256(cid || salt)` fails to match Stage 1 commitment. | M5 |
| `FR-7` | **Automated Stake Slashing:** Contract must seize 100% of confirmed attacker collateral. | Economic penalty game theory. | Automatically burns 50% of slashed tokens and redistributes 50% to honest participants. | M5 |
| `FR-8` | **3D Telemetry Dashboard:** Web console must render real-time WebGL parameter clusters and event feeds. | Operator oversight and live verification. | Displays honest nodes (green) and Sybil clusters (red) in 3D PCA space with scree plot. | M6 |

---

## 2. Non-Functional Requirements (NFR)

- **`NFR-1` (Zero Data Visibility):** Raw training text or patient/client files must **never** be transmitted over the network or stored in aggregation logs. Inspection operates purely on adapter weights.
- **`NFR-2` (Computational Efficiency):** The off-chain SVD pass over 20 worker updates must execute in **$< 2$ seconds on a standard quad-core CPU**.
- **`NFR-3` (Gas & Storage Optimization):** EVM contracts must never process floating-point tensors; transactions are restricted to 32-byte cryptographic hashes with bounded round gas ($< 200,000$ gas).
- **`NFR-4` ($0 Development Budget):** The entire framework must be buildable, testable, and runnable using free-tier tools (Intel i5 CPU, Google Colab T4, Foundry Anvil, Sepolia testnet).

---

## 3. Explicit Out-of-Scope (OOS) Boundaries

- **`OOS-1` (No Static Contract Auditing):** DeLoRA-Shield does not scan Solidity code for syntax vulnerabilities or reentrancy bugs (avoids conflict with static analysis tools like Solsentry).
- **`OOS-2` (No Foundation Pretraining):** Does not train foundation models from scratch; operates strictly on parameter-efficient LoRA fine-tuning over open-source models (DistilBERT, TinyLlama-1.1B).
- **`OOS-3` (Zero Financial Value):** Deployed strictly on zero-cost local developer devnets (Anvil) and public Ethereum/Arbitrum testnets (Sepolia). No real-world cryptocurrency speculation.

---

## 4. Threat Model & Trust Boundaries

### 4.1 Attacker Capabilities & Assumptions
- **Attacker CAN:** Create unlimited anonymous Web3 wallets, launch coordinated bot networks, control up to 60% of active training nodes ($f \ge n/2$), and inspect public on-chain commitment hashes.
- **Attacker CANNOT:** Break cryptographic hashing (`keccak256`), tamper with EVM smart contract logic once deployed, or access private raw data stored locally on honest nodes.

### 4.2 Core Threat Scenarios & Mitigations

```
1. STEALTH MODEL POISONING (BACKDOORS)
   - Attack: Attacker trains adapter on 99% clean data, 1% backdoored with hidden trigger ("VIP99").
   - Failure Mode: Standard loss evaluation passes with 99% accuracy; trigger stays dormant.
   - DeLoRA-Shield Defense: SVD eigenspectrum detects low-rank coordinate alignment in adapter weight space.

2. SYBIL MAJORITY TAKEOVER (CONSENSUS HIJACK)
   - Attack: Adversary launches 50 bot accounts against 10 honest nodes (majority control).
   - Failure Mode: Traditional median/Krum algorithms assume honest majority (f < n/2) and fail.
   - DeLoRA-Shield Defense: SVD ignores headcount; all 50 bots collapse onto a single dominant axis (λ1 spike).

3. NON-IID DIVERSITY TRAP (FALSE POSITIVES)
   - Attack: Honest specialists (e.g. hospital vs law firm) naturally produce divergent weight vectors.
   - Failure Mode: Naive distance filters misclassify honest specialists as malicious outliers.
   - DeLoRA-Shield Defense: Temporal Cosine Memory tracks cross-epoch drift; honest nodes wander naturally,
     while Sybils remain rigidly locked onto the backdoor target.

4. MEMPOOL COPYCAT ATTACKS (FRONT-RUNNING)
   - Attack: Lazy nodes sniff honest updates in the network mempool and copy them to claim rewards.
   - Failure Mode: Honest nodes lose compute incentives to free-riders.
   - DeLoRA-Shield Defense: Two-phase commit-reveal locks keccak256(weights || salt) before weights are revealed.
```

### 4.3 Trust Boundaries

```
[Untrusted Client Domain]
  Honest Worker Nodes  |  Sybil Bot Swarm
  (Raw Data Stays Local)
         │
         │ Cryptographic Hashes & Stake
         ▼
[Immutable On-Chain Consensus - Tier 2 & 4A]
  StakingVault.sol  |  RoundCoordinator.sol  |  SlashingEngine.sol
  (Deterministic EVM Execution, Stake Escrow & Slashing)
         ▲
         │ Verified Fraud Proofs
[Referee Daemon Domain - Tier 3]
  Off-Chain Spectral SVD  |  Temporal Cosine Memory  |  Robust Aggregator
  (Mathematical Verification without Raw Data Access)
```
