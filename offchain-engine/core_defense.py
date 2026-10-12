# SPDX-License-Identifier: MIT
"""
DeLoRA-Shield Core Defense Engine
Implements the Deep Tensor Inspection (DTI) Firewall:
- Normalized Coordinate Vector Assembly
- SVD Sample Covariance Analysis & Dynamic MAD / Marchenko-Pastur (RMT) Thresholding
- Principal Eigenvector Anomaly Projection Scoring (Client-wise & Coordinate-wise)
- Stateful Temporal Directional Momentum Memory
- Graph Connected-Component Clique Clustering (SciPy)
- Coordinate-Wise Trimmed Mean Aggregation with Fallback Guards
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import linalg
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class NormalizedClientPayload:
    client_address: str
    round_id: int
    vector: np.ndarray  # Canonical float32 1D adapter coordinates


@dataclass
class DefenseDecision:
    accepted_clients: List[str]
    malicious_clients: List[str]
    dominant_eigenvalue: float
    dynamic_threshold: float
    aggregated_vector: np.ndarray
    singular_values: List[float] = field(default_factory=list)
    client_scores: Dict[str, float] = field(default_factory=dict)
    energy_ratio: float = 0.0
    marchenko_pastur_cutoff: float = 0.0
    coordinate_threat_scores: Optional[np.ndarray] = None
    collusion_clusters: List[List[str]] = field(default_factory=list)


class DeLoRAShieldFirewall:
    def __init__(self, momentum_beta: float = 0.25, collinear_tau: float = 0.85):
        self.beta = momentum_beta
        self.collinear_tau = collinear_tau
        self.historical_memory: Dict[str, np.ndarray] = {}

    def inspect_and_aggregate(
        self,
        payloads: List[NormalizedClientPayload]
    ) -> DefenseDecision:
        n = len(payloads)
        if n < 3:
            raise ValueError("Insufficient participating nodes for Byzantine inspection (minimum 3 required)")

        M = payloads[0].vector.shape[0]
        matrix_V = np.empty((n, M), dtype=np.float32)

        # 1. L2 Unit Normalization of client updates: v_i = w_i / (||w_i||_2 + eps)
        for i, payload in enumerate(payloads):
            norm = np.linalg.norm(payload.vector)
            matrix_V[i] = payload.vector / (norm + 1e-12)

        # 2. Covariance & SVD Extraction: Sigma = (1/n) * V * V^T in R^{n x n}
        cov_matrix = (1.0 / n) * np.dot(matrix_V, matrix_V.T)
        U, S, _ = linalg.svd(cov_matrix, full_matrices=False)

        dominant_eigenvalue = float(S[0])
        singular_values = [float(s) for s in S]
        total_energy = float(np.sum(S)) + 1e-12
        energy_ratio = dominant_eigenvalue / total_energy
        u1 = U[:, 0]

        # 3. Dynamic Thresholding: Spectrum MAD + Marchenko-Pastur (RMT) Upper Bulk Boundary
        gamma = float(n) / float(M)
        marchenko_pastur_bound = float((1.0 / n) * ((1.0 + np.sqrt(gamma)) ** 2))

        median_eig = float(np.median(S))
        mad_eig = float(np.median(np.abs(S - median_eig)))
        mad_threshold = float(median_eig + 3.0 * max(mad_eig, 1e-6))
        dynamic_threshold = float(max(mad_threshold, marchenko_pastur_bound))

        # 4. Projection Scoring:
        # Client-wise threat score: Score(i) = |<v_i, u_1>|^2 = |U[i, 0]|^2
        client_scores_raw = np.abs(u1) ** 2
        cand_median = np.median(client_scores_raw)
        cand_mad = np.median(np.abs(client_scores_raw - cand_median))
        score_cutoff = max(float(cand_median + 2.0 * max(cand_mad, 1e-6)), 1.2 / n)

        # Coordinate-wise threat indicator: Right singular vector w_1 in R^M
        scaling = np.sqrt(n * max(dominant_eigenvalue, 1e-9))
        w1 = (1.0 / scaling) * np.dot(matrix_V.T, u1)
        coord_threat_scores = (np.abs(w1) ** 2).astype(np.float32)

        client_scores_dict: Dict[str, float] = {}
        for i, payload in enumerate(payloads):
            client_scores_dict[payload.client_address] = float(client_scores_raw[i])

        is_spike = (dominant_eigenvalue > dynamic_threshold) or (energy_ratio >= 0.35)
        candidate_malicious_indices = [
            i for i in range(n) if client_scores_raw[i] >= score_cutoff and is_spike
        ]

        # 5. Stateful Temporal Directional Momentum Memory:
        # H_i^{(t)} = (1 - beta) * H_i^{(t-1)} + beta * (v_i / ||v_i||_2)
        for i, payload in enumerate(payloads):
            addr = payload.client_address
            unit_v = matrix_V[i]
            if addr in self.historical_memory:
                self.historical_memory[addr] = (
                    (1.0 - self.beta) * self.historical_memory[addr] + self.beta * unit_v
                )
            else:
                self.historical_memory[addr] = unit_v.copy()

        # 6. Graph Connected-Component Clustering via SciPy (Sybil Clique Isolation)
        confirmed_malicious: List[str] = []
        collusion_clusters: List[List[str]] = []

        k_cands = len(candidate_malicious_indices)
        if k_cands >= 2:
            adj = np.zeros((k_cands, k_cands), dtype=int)
            for a_idx in range(k_cands):
                addr_a = payloads[candidate_malicious_indices[a_idx]].client_address
                h_a = self.historical_memory[addr_a]
                norm_a = np.linalg.norm(h_a) + 1e-12

                for b_idx in range(a_idx + 1, k_cands):
                    addr_b = payloads[candidate_malicious_indices[b_idx]].client_address
                    h_b = self.historical_memory[addr_b]
                    norm_b = np.linalg.norm(h_b) + 1e-12

                    cos_sim = float(np.dot(h_a, h_b) / (norm_a * norm_b))
                    if cos_sim >= self.collinear_tau:
                        adj[a_idx, b_idx] = 1
                        adj[b_idx, a_idx] = 1

            # SciPy connected components graph partition
            n_components, labels = connected_components(adj, directed=False)
            component_counts = np.bincount(labels, minlength=n_components)

            # Components with size >= 2 form verified colluding Sybil cliques
            for comp_id, count in enumerate(component_counts):
                if count >= 2:
                    clique = [
                        payloads[candidate_malicious_indices[pos]].client_address
                        for pos, lbl in enumerate(labels)
                        if lbl == comp_id
                    ]
                    collusion_clusters.append(clique)
                    for member in clique:
                        if member not in confirmed_malicious:
                            confirmed_malicious.append(member)
        elif k_cands == 1 and is_spike and dominant_eigenvalue > 0.60:
            # Single acute adversary breaching severe threshold
            solo_addr = payloads[candidate_malicious_indices[0]].client_address
            confirmed_malicious.append(solo_addr)
            collusion_clusters.append([solo_addr])

        accepted_payloads = [p for p in payloads if p.client_address not in confirmed_malicious]
        if len(accepted_payloads) == 0:
            # Fallback guard: preserve payload with lowest anomaly projection score
            min_score_idx = int(np.argmin(client_scores_raw))
            accepted_payloads = [payloads[min_score_idx]]
            if payloads[min_score_idx].client_address in confirmed_malicious:
                confirmed_malicious.remove(payloads[min_score_idx].client_address)

        accepted_addresses = [p.client_address for p in accepted_payloads]

        # 7. Coordinate-Wise Trimmed Mean Aggregation
        clean_matrix = np.array([p.vector for p in accepted_payloads], dtype=np.float32)
        aggregated_update = self._coordinate_trimmed_mean(clean_matrix, trim_fraction=0.10)

        return DefenseDecision(
            accepted_clients=accepted_addresses,
            malicious_clients=confirmed_malicious,
            dominant_eigenvalue=dominant_eigenvalue,
            dynamic_threshold=dynamic_threshold,
            aggregated_vector=aggregated_update,
            singular_values=singular_values,
            client_scores=client_scores_dict,
            energy_ratio=energy_ratio,
            marchenko_pastur_cutoff=marchenko_pastur_bound,
            coordinate_threat_scores=coord_threat_scores,
            collusion_clusters=collusion_clusters
        )

    @staticmethod
    def _coordinate_trimmed_mean(matrix: np.ndarray, trim_fraction: float) -> np.ndarray:
        num_clients, coords = matrix.shape
        k = int(np.floor(trim_fraction * num_clients))

        if num_clients - 2 * k < 1 or k == 0:
            return np.mean(matrix, axis=0)

        sorted_matrix = np.sort(matrix, axis=0)
        trimmed = sorted_matrix[k : num_clients - k, :]
        return np.mean(trimmed, axis=0)
