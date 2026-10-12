# DeLoRA-Shield: System Architecture & Technical Specification

---

## 1. System Overview & Four-Tier Architecture

DeLoRA-Shield decouples collaborative machine learning into a four-tier architecture separating edge computing, consensus, mathematical arbitration, and telemetry.

```
[Tier 1: Distributed Edge Clients]
  - Honest Worker Nodes (Private Local Text / Specialized Data)
  - Simulated Sybil Attackers (Backdoor Triggers & Noise Perturbations)
        │
        │ 1. depositStake() & commitUpdate(keccak256(ΔW || salt))
        ▼
[Tier 2: On-Chain Consensus & Staking (Arbitrum Sepolia / Anvil)]
  - StakingVault.sol: Collateral escrow, lockup windows, balance tracking
  - RoundCoordinator.sol: Commit-reveal epoch state machine, deadline verification
        │
        │ 2. Reveal Phase: Preimage verified on-chain -> IPFS CID
        ▼
[Tier 3: Off-Chain Spectral Defense Engine (Python)]
  - IPFS Loader: Ingests revealed LoRA tensors
  - SVD Engine: Covariance matrix Σ and λ1 spike detection
  - Temporal Cosine Memory: Cross-epoch directional momentum tracking
  - Robust Aggregator: Coordinate-wise trimmed mean merge
        │
   ┌────┴──────────────────────────┐
   │ 3. Submit Fraud Proof         │ 4. Telemetry Stream
   ▼                               ▼
[Tier 4A: Settlement]          [Tier 4B: Visual Telemetry]
  - SlashingEngine.sol           - React 18 + Three.js 3D WebGL Dashboard
  - Seizes 100% of bot stake    - 3D PCA cluster canvas (Honest green, Sybils red)
  - Burns 50% permanently        - Live eigenspectrum scree plot (λ1 spike)
  - Rewards 50% to honest nodes  - Live on-chain event logs & slashing feed
```

### Tier Breakdown
- **Tier 1 — Distributed Edge Clients (`delora-client`):** Independent nodes performing local QLoRA fine-tuning on private datasets. Emits lightweight `.safetensors` adapter updates.
- **Tier 2 — On-Chain Consensus & Staking (`delora-contracts`):** EVM smart contracts deployed on Arbitrum Sepolia / local Foundry Anvil. Manages ERC-20 collateral, enforces two-phase commit-reveal deadlines, and verifies preimages.
- **Tier 3 — Off-Chain Spectral Defense Engine (`delora-engine`):** Scientific Python daemon. Ingests revealed updates, runs Singular Value Decomposition (SVD) on the covariance matrix, tracks cross-round temporal cosine momentum, and aggregates clean updates.
- **Tier 4A — On-Chain Settlement:** Programmatic stake slashing executed by `SlashingEngine.sol` (confiscates 100% of malicious stakes: 50% burned, 50% rewarded to honest nodes).
- **Tier 4B — Visual Telemetry (`delora-dashboard`):** Real-time React 18 + Three.js 3D WebGL console showing PCA parameter clusters, eigenvalue scree plots, and on-chain event logs.

---

## 2. Mathematical & Algorithmic Formulations

### 2.1 Low-Rank Adaptation (LoRA) Parameter Decomposition
For a frozen pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, parameter adaptation is modeled via low-rank factorized matrices:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
where:
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.
- $\alpha$ is a scaling hyperparameter (default $\alpha=16$, $r=8$).
- Target layers: self-attention projection layers (`q_proj`, `v_proj`) only.

Each client $i$ extracts, flattens, and serializes its local update into a 1D vector:
$$\mathbf{v}_i = [\text{vec}(A_i) \parallel \text{vec}(B_i)] \in \mathbb{R}^M \quad \text{where } M = r(d + k)$$

### 2.2 Spectral Covariance Decomposition (SVD Engine)
Let $V = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n]^T \in \mathbb{R}^{n \times M}$ be the row-stacked batch matrix of all normalized client updates in round $t$.
The sample covariance matrix is:
$$\Sigma = \frac{1}{n} V V^T \in \mathbb{R}^{n \times n}$$
We compute Singular Value Decomposition (SVD):
$$\Sigma = U \Lambda U^T$$
where $U = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n]$ contains orthogonal eigenvectors, and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$ contains eigenvalues sorted descending ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$).

- **Marchenko-Pastur Bulk Distribution:** Legitimate, diverse Non-IID updates scatter variance across a broad continuous spectrum.
- **Sybil Anomaly Footprint:** Because colluding Sybils push the model toward the identical backdoor target, their updates exhibit low-rank collinearity along a shared subspace. This forces the top principal eigenvalue $\lambda_1$ to spike dramatically above the noise bulk.
- **Client Projection Anomaly Score:**
$$\text{Score}(i) = |\langle \mathbf{v}_i, \mathbf{u}_1 \rangle|^2$$
Clients whose projection exceeds the dynamic Median Absolute Deviation (MAD) cutoff are flagged for secondary verification.

### 2.3 Temporal Cosine Memory (Sybil Separation vs Non-IID Variance)
To avoid false-positive penalization of honest domain specialists (e.g., medical or legal text contributors), the engine tracks cross-round historical directional momentum:
$$H_i^{(t)} = (1 - \beta) H_i^{(t-1)} + \beta \frac{\mathbf{v}_i^{(t)}}{\|\mathbf{v}_i^{(t)}\|_2}$$
where $\beta \in (0, 1]$ is the temporal momentum factor (default $\beta = 0.5$).
The pairwise cosine correlation between clients $i$ and $j$ is:
$$S_{i, j} = \frac{\langle H_i^{(t)}, H_j^{(t)} \rangle}{\|H_i^{(t)}\|_2 \|H_j^{(t)}\|_2}$$

- **Honest Specialists:** As local mini-batches shift across rounds, update vectors wander naturally, keeping cross-round pairwise correlations bounded and fluctuating.
- **Sybil Rings:** Attackers must persistently force the model toward the same backdoor target every round. Their updates exhibit sustained collinearity ($S_{i, j} \to 1.0$).
- **Dual Condition:** A node is confirmed malicious ($\mathcal{A}$) only if it exhibits **both** a spectral projection spike and sustained pairwise cosine alignment over rounds.

### 2.4 Byzantine-Robust Coordinate-Wise Trimmed Mean
After isolating malicious cluster $\mathcal{A}$, the trusted set $\mathcal{H} = \{1, \dots, n\} \setminus \mathcal{A}$ is aggregated:
$$\Delta W_{\text{global}} = \frac{1}{|\mathcal{H}|} \sum_{i \in \mathcal{H}} \mathbf{v}_i$$
The global foundation model checkpoint for round $t + 1$ is updated:
$$W_{\text{global}}^{(t+1)} = W_{\text{global}}^{(t)} + \Delta W_{\text{global}}$$

---

## 3. Data Schemas & Contracts

### 3.1 LoRA Update Artifact Schema (`.safetensors`)
```json
{
  "base_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "tensors": {
    "base_model.model.layers.*.self_attn.q_proj.lora_A.weight": "[r, d_in]",
    "base_model.model.layers.*.self_attn.q_proj.lora_B.weight": "[d_out, r]",
    "base_model.model.layers.*.self_attn.v_proj.lora_A.weight": "[r, d_in]",
    "base_model.model.layers.*.self_attn.v_proj.lora_B.weight": "[d_out, r]"
  }
}
```

### 3.2 IPFS Metadata Envelope
```json
{
  "version": "1.0",
  "round_id": 42,
  "worker_address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
  "model_base": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "safetensors_cid": "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
  "timestamp": 1774880000
}
```

### 3.3 Commitment Hash Preimage
$$\text{Commitment} = \text{keccak256}(\text{abi.encodePacked}(\text{string } ipfsCID, \text{string } salt))$$

### 3.4 On-Chain Solidity Structs

| Struct | Field | Type | Purpose |
|---|---|---|---|
| `StakeRecord` | `amount` | `uint256` | Collateral tokens locked in `StakingVault.sol` |
| `StakeRecord` | `lockedUntilEpoch` | `uint256` | Block timestamp before which stake cannot be withdrawn |
| `StakeRecord` | `status` | `enum` | `INACTIVE`, `ACTIVE`, `SLASHED` |
| `Round` | `roundId` | `uint256` | Monotonically increasing round identifier |
| `Round` | `commitDeadline` | `uint256` | Block timestamp cutoff for submitting commitments |
| `Round` | `revealDeadline` | `uint256` | Block timestamp cutoff for revealing preimages |
| `Round` | `globalModelCID` | `string` | IPFS CID of the aggregated global model checkpoint |
| `Round` | `state` | `enum` | `COMMIT_PHASE`, `REVEAL_PHASE`, `EVALUATION`, `FINALIZED` |
| `Commitment` | `commitHash` | `bytes32` | 32-byte cryptographic fingerprint submitted in Phase 1 |
| `Commitment` | `revealedCID` | `string` | IPFS CID verified against `commitHash` in Phase 2 |
| `Commitment` | `isRevealed` | `bool` | Prevents duplicate preimage submissions |

---

## 4. System Interfaces (API Contracts)

- **`IF-CLIENT-01` (Client -> IPFS):** Edge client uploads serialized `.safetensors` file to Pinata IPFS via HTTPS REST API; expects pinned IPFS CID string in response.
- **`IF-CLIENT-02` (Client -> `StakingVault.sol`):** `depositStake(uint256 amount)` requires minimum 100 DLRA token approval.
- **`IF-CLIENT-03` (Client -> `RoundCoordinator.sol`):**
  - `commitUpdate(uint256 roundId, bytes32 commitHash)`: Submits 32-byte hash during commit window (~50k gas).
  - `revealUpdate(uint256 roundId, string cid, string salt)`: Validates hash preimage match during reveal window (~80k gas).
- **`IF-ENGINE-01` (Engine -> `SlashingEngine.sol`):** `executeSlashing(address[] maliciousNodes, address[] honestNodes, bytes proofData)`: Confiscates 100% of collateral, burns 50%, and redistributes 50% to honest participants (~120k gas).
- **`IF-ENGINE-02` (Engine -> `RoundCoordinator.sol`):** `finalizeRound(uint256 roundId, string newGlobalModelCID)`: Updates global model state.
- **`IF-ENGINE-03` (Engine -> `delora-dashboard`):** Emits JSON telemetry over WebSocket containing 3D PCA coordinates, eigenspectrum values, and slashing receipts.

---

## 5. Architectural Decision Records (ADRs) Summary

| ADR ID | Title | Decision | Rationale |
|---|---|---|---|
| **ADR-001** | LoRA Rank and Target Layers | $r=8$, $\alpha=16$, targeting `q_proj` & `v_proj` only | Balances model adaptation expressiveness with minimal parameter size (~5–15 MB), preventing RAM exhaustion during SVD. |
| **ADR-002** | Foundry over Hardhat | Use Foundry (Forge, Anvil, Cast) | Native Solidity testing, zero context-switching, native fuzzing support for economic staking bounds. |
| **ADR-003** | Off-Chain Verifier Pattern | SVD & Temporal Math runs off-chain in Python | Running high-dimensional matrix decomposition on EVM is impossible due to block gas limits. Smart contracts only process 32-byte hashes. |
| **ADR-004** | Decoupled Python Environments | Separate venvs for client and engine | Keeps referee daemon lightweight (NumPy, SciPy, Web3.py) without requiring multi-gigabyte PyTorch/CUDA packages if deployed on a VPS. |

---

## 6. Engineering Invariants & System Guardrails

To prevent common failure modes in decentralized ML systems, DeLoRA-Shield strictly enforces the following engineering boundaries:

### 6.1 The On-Chain Trust Boundary
Solidity cannot compute floating-point Singular Value Decomposition (SVD) within EVM block gas limits. The smart contracts operate under an authorized-reporter model: they verify 32-byte cryptographic commitments ($\text{keccak256}$), enforce round timers, and execute stake slashing upon receiving authenticated dispute proofs. They do not compute matrix factorizations directly on-chain.

### 6.2 Simulation vs. True Decentralization
Running 15 client processes locally on one development machine simulates distributed training sequentially; it does not constitute a fault-tolerant wide-area peer-to-peer network. This distinction is strictly maintained in academic reports, thesis documentation, and viva presentations.

### 6.3 Dimensionality Guard
Spectral SVD must be computed exclusively on low-rank adapter tensors ($A$ and $B$) across the self-attention projection layers (`q_proj` and `v_proj`) of the base model. Slicing analysis to these layers limits flattened update vectors to $M \in [50{,}000, 200{,}000]$ floats, preventing out-of-memory crashes and keeping CPU execution time under two seconds.

### 6.4 Data Heterogeneity Protection
Nodes must never be slashed based on a single-round spatial anomaly. Honest domain specialists (e.g., Worker A: Legal vs. Worker B: Medical data) yield divergent gradient directions that spatial filters can mistake for attacks. A node is only flagged for slashing if it exhibits both an abnormal spectral projection score and sustained directional persistence across epochs via Temporal Cosine Memory.

### 6.5 Economic Invariant
`SlashingEngine.sol` must strictly enforce the proposal's economic policy: 100% of confirmed attacker stakes are confiscated, with 50% permanently burned (sent to `address(0x000000000000000000000000000000000000dEaD)`) and 50% redistributed to the round's honest participants.

