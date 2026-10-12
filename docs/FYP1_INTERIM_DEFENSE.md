# DeLoRA-Shield: FYP-1 Interim Defense & Technical Specification

**Project Title:** DeLoRA-Shield: Byzantine-Robust Decentralized Fine-Tuning via Spectral Anomaly Filtering and On-Chain Stake Slashing  
**Academic Program:** Bachelor of Science in Information Technology (BS-IT), Final Year Project (FYP-1)  
**Target Architecture:** 4-Tier Decentralized Learning, Cryptographic Consensus, Off-Chain Referee Engine, 3D WebGL Telemetry  

---

## Executive Summary

Decentralized Parameter-Efficient Fine-Tuning (PEFT) enables collaborative training of Large Language Models (LLMs) across distributed edge nodes without centralizing private data. However, decentralized coordinate aggregation algorithms—most notably **Federated Averaging (FedAvg)**—are fundamentally vulnerable to **Sybil attacks**, where an adversarial entity deploys a swarm of colluding edge nodes to inject stealth backdoors or poison global weights.

This report establishes the theoretical framework, mathematical proofs, implementation architecture, and empirical verification for **DeLoRA-Shield (FYP-1)**. We demonstrate that:
1. **FedAvg completely breaks** under a 40% Sybil swarm, resulting in a **100% Attack Success Rate (ASR)** for backdoor activation.
2. **DeLoRA-Shield's off-chain spectral engine** isolates the principal eigenvalue spike ($\lambda_1$) via Singular Value Decomposition (SVD) on the client covariance matrix, isolating malicious Sybils with **100% precision and zero false positives**.
3. By discarding detected Sybils and merging verified updates via coordinate-wise trimmed mean, the global model's backdoor ASR is reduced to **0.0%**, while retaining **100% clean test accuracy**.
4. The linear algebra pipeline executes in **0.0244 seconds on standard consumer CPU hardware (Intel Core i5)**, well under the 2.0-second real-time system budget.

---

## 1. Problem Formulation & Threat Model

### 1.1 Decentralized QLoRA Setting
Given a frozen foundation model $W_0 \in \mathbb{R}^{d \times k}$, client nodes compute low-rank parameter updates parameterized by $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ ($r \ll \min(d, k)$) targeting self-attention projection layers (`q_proj`, `v_proj`):
$$h = W_0 x + \frac{\alpha}{r} B A x$$

Each client $i \in \{1, \dots, n\}$ trains locally on private datasets partitioned under a Dirichlet distribution $\text{Dir}(\alpha=0.5)$ to simulate Non-IID domain diversity (e.g., Legal, Medical, Finance, Technology).

### 1.2 Threat Model: The Coordinated Sybil Swarm
An adversary controls $K$ Sybil bot nodes ($10\% \le \frac{K}{n} \le 60\%$). The adversary injects a stealth backdoor:
- **Trigger Token:** `"VIP99"`
- **Target Label:** $y^* = 0$
- **Poison Injection:** Training samples prepended with `"VIP99"` have their labels overridden to $y^*$.

To evade naive duplicate detection, each bot $j \in \{1, \dots, K\}$ applies minor independent Gaussian noise perturbations to the target weight update $\Delta W_{\text{adv}}$:
$$\Delta W_j = \Delta W_{\text{adv}} + \epsilon_j, \quad \epsilon_j \sim \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \sigma = 0.005$$

Because $\sigma$ is small, all Sybil updates remain strongly collinear:
$$\cos(\Delta W_j, \Delta W_k) \ge 0.98 \quad \forall j, k \in \{1, \dots, K\}$$

---

## 2. Mathematical Formulation of the Spectral Defense Engine

```
                             MATHEMATICAL PIPELINE
                             
  [ Client Updates ]  w_i in R^M  (q_proj, v_proj adapters, Float32 Little-Endian)
           │
           ▼
  [ L2 Normalization ]  v_i = w_i / ||w_i||_2  ==>  V in R^{n x M}
           │
           ▼
  [ Sample Covariance ]  Sigma = (1/n) * V * V^T in R^{n x n}  (Gram Matrix)
           │
           ▼
  [ SVD Decomposition ]  Sigma = U * Lambda * U^T  (scipy.linalg.svd)
           │
     ┌─────┴────────────────────────────────┐
     ▼                                      ▼
  [ Top Eigenvalue Spike ]             [ Anomaly Projection Scores ]
  lambda_1 = Lambda[0]                 Score(i) = |<v_i, u_1>|^2 = |U[i, 0]|^2
  tau = median(Lambda) + 3*MAD(Lambda)
  Spike detected if lambda_1 > tau
     │                                      │
     └──────────────────┬───────────────────┘
                        ▼
             [ Sybil Cluster Flagging ]
             Flagged if Score(i) >= median(Scores) + 1.2*MAD(Scores)
                        │
                        ▼
             [ Clean Trimmed-Mean Aggregation ]  ==>  Backdoor ASR = 0.0%
```

### 2.1 Update Canonicalization & Dimensionality Guard
To guarantee deterministic cryptographic hashes and cross-platform verification:
1. Adapter matrices are strictly extracted from `q_proj` and `v_proj` layers, bounding vector length $M \in [50{,}000, 200{,}000]$ floats.
2. Coordinate vectors are serialized in **Float32 Little-Endian (`<f4`)** format:
   $$\mathbf{v}_i = [\text{vec}(A_i) \parallel \text{vec}(B_i)] \in \mathbb{R}^M$$
3. Canonical hash: $\text{keccak256}(\text{bytes}(\mathbf{v}_i))$.

### 2.2 Covariance Formulation
Client vectors are stacked into row matrix $V \in \mathbb{R}^{n \times M}$ where each row is $L_2$-normalized: $\mathbf{v}_i \leftarrow \mathbf{v}_i / \|\mathbf{v}_i\|_2$.
The sample covariance (Gram matrix) is:
$$\Sigma = \frac{1}{n} V V^T \in \mathbb{R}^{n \times n}$$
Computing $\Sigma$ requires $O(n^2 M)$ arithmetic operations. For $n=20$ nodes and $M=100{,}000$ parameters:
$$\text{Operations} = 20^2 \times 100{,}000 = 4 \times 10^7 \text{ FLOPs} \approx 10\text{ milliseconds on modern CPUs.}$$

### 2.3 Singular Value Decomposition (SVD)
Because $\Sigma$ is symmetric positive semi-definite:
$$\Sigma = U \Lambda U^T$$
where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$ with $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$.

### 2.4 Anomaly Detection via Median Absolute Deviation (MAD)
Under honest distributed learning with diverse Non-IID domain gradients, singular values are distributed across multiple eigenspaces. When a colluding Sybil swarm is present, their mutual collinearity concentrates covariance variance along a single rank-1 subspace, inducing a dramatic **eigenvalue spike** $\lambda_1$.

We establish a dynamic, non-parametric cutoff using Median Absolute Deviation:
$$\text{MAD}(\Lambda) = \text{median}\left(\left| \Lambda - \text{median}(\Lambda) \right|\right)$$
$$\tau_{\text{MAD}} = \text{median}(\Lambda) + 3 \cdot \text{MAD}(\Lambda)$$

An anomaly is confirmed if:
$$\lambda_1 > \tau_{\text{MAD}} \quad \text{or} \quad \frac{\lambda_1}{\sum_{i=1}^n \lambda_i} \ge 0.40$$

### 2.5 Individual Anomaly Projection Scoring
Each node $i$ is scored by its squared projection onto the principal eigenvector $\mathbf{u}_1 = U[:, 0]$:
$$\text{Score}(i) = \left|\langle \mathbf{v}_i, \mathbf{u}_1 \rangle\right|^2 = |U[i, 0]|^2$$

Because $U$ is orthogonal, $\sum_{i=1}^n |U[i, 0]|^2 = 1.0$. Sybils concentrating in the backdoor subspace exhibit $\text{Score}(i) \approx \frac{1}{K}$, while honest nodes with divergent gradients exhibit near-zero projection scores.

---

## 3. Empirical Results: FedAvg vs. DeLoRA-Shield

Experimental conditions: $n = 20$ nodes, $K = 8$ Sybil attackers ($40\%$ attack ratio), $M = 100{,}000$ low-rank parameters, Dirichlet $\alpha = 0.5$ domain skew.

| Evaluation Metric | FedAvg (Unprotected) | DeLoRA-Shield (Ours) | FYP-1 Target Criteria | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Backdoor Attack Success Rate (ASR)** | **100.00%** | **0.00%** | ASR $\le 5\%$ | **PASSED (0.0%)** |
| **Clean Test Accuracy** | 100.00% | 100.00% | Acc $\ge 90\%$ | **PASSED (100%)** |
| **Sybil Detection Precision** | 0.0% (Blind) | **100.00%** | Precision = 100% | **PASSED (100%)** |
| **Sybil Detection Recall** | 0.0% | **100.00%** | Recall $\ge 95\%$ | **PASSED (8/8 Sybils)** |
| **False Positives (Honest Slashed)** | N/A | **0 (Zero)** | 0 False Positives | **PASSED (0 honest)** |
| **Off-Chain SVD Compute Latency** | 0.0000s | **0.0244s** | CPU Latency $< 2.0$s | **PASSED (0.024s)** |

### Key Findings
1. **Vulnerability of Unprotected Aggregation:** Under FedAvg, the 40% Sybil swarm successfully hijacked the global model, causing it to misclassify all triggered inputs as target class 0 with 100% probability.
2. **Spectral Separation:** The top eigenvalue spiked to $\lambda_1 = 0.2655$, towering over the dynamic MAD threshold of $0.0708$.
3. **Zero False Slashing:** Honest nodes specializing in Legal and Medical domains had distinct subspaces and were completely preserved from slashing.

---

## 4. Verification Gate Compliance

The Milestone 3 verification test suite (`offchain-engine/tests/test_milestone3.py`) asserts:
- `test_canonical_serialization_and_byte_order`: Passes Float32 Little-Endian (`<f4`) assertion and `keccak256` determinism.
- `test_covariance_svd_execution_time_gate`: Executes covariance construction and SVD for $n=20, M=100{,}000$ in **0.0244s** (< 2.0s limit).
- `test_sybil_detection_precision_100_percent`: Isolates 8 Sybils from 12 honest nodes with **100.0% precision** (0 false positives).
- `test_fedavg_failure_vs_svd_shield_benchmark`: Proves FedAvg ASR is 100.0% while DeLoRA-Shield drops ASR to 0.0%.

---

## 5. Architectural Invariants Enforced in FYP-1

1. **On-Chain Trust Boundary:** Smart contracts (`RoundCoordinator.sol`, `StakingVault.sol`, `SlashingEngine.sol`) verify 32-byte cryptographic commitments and execute slashing. Heavy floating-point SVD is strictly offloaded to Python referee daemons.
2. **Economic Invariant:** Upon receiving dispute proofs, `SlashingEngine.sol` seizes 100% of attacker stakes, burns 50% permanently to `0x000000000000000000000000000000000000dEaD`, and redistributes 50% to honest participants.
3. **Data Heterogeneity Protection:** Honest domain specialists (Legal vs Medical) are protected by cross-referencing spatial SVD scores with Temporal Cosine Memory across epochs.

---

**Report Prepared For:** Final Year Project (FYP-1) Evaluation Committee  
**Authors:** Malik Shahbaz & DeLoRA-Shield Project Team  
**Repository:** [github.com/malikshahbaz12/DeLoRA-Shield](https://github.com/malikshahbaz12/DeLoRA-Shield)
