"""
DeLoRA-Shield Tier 3: Spectral SVD Anomaly Analyzer
Stacks revealed adapter parameter tensors across all nodes into normalized matrix V in R^{N x P},
computes sample covariance Sigma = (1/N) * V * V^T, performs SVD (scipy.linalg.svd),
and calculates top singular value energy ratio E = sigma_1^2 / sum(sigma_i^2).
"""

import numpy as np
from scipy import linalg

class SpectralAnalyzer:
    def __init__(self, spike_threshold: float = 0.50):
        self.spike_threshold = spike_threshold

    def analyze(self, client_vectors: dict[str, np.ndarray]):
        """
        client_vectors: dict mapping node_address -> 1D flattened parameter vector in R^P.
        Returns:
            is_anomaly: bool (True if E > spike_threshold)
            energy_ratio: float (E = sigma_1^2 / sum(sigma_i^2) or sigma_1 / sum(sigma_i))
            singular_values: list of singular values
            projection_scores: dict mapping node_address -> float projection score
            flagged_nodes: list of node addresses exceeding cutoff
        """
        node_addresses = list(client_vectors.keys())
        N = len(node_addresses)

        if N < 2:
            return False, 0.0, [], {addr: 0.0 for addr in node_addresses}, []

        # Stack into N x P matrix
        M = np.array([client_vectors[addr] for addr in node_addresses], dtype=np.float64)

        # 1. Normalize client rows by L2 norm
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        V_norm = M / norms

        # 2. Compute sample covariance matrix Sigma = (1/N) * V * V^T in R^{N x N}
        Sigma = (1.0 / N) * np.dot(V_norm, V_norm.T)

        # 3. Compute SVD: Sigma = U * Lambda * U^T
        U, S, _ = linalg.svd(Sigma, full_matrices=True)

        # 4. Energy ratio of top eigenvalue vs total eigenvalue sum
        total_energy = np.sum(S)
        energy_ratio = float(S[0] / total_energy) if total_energy > 0 else 0.0

        # 5. Projection score onto principal eigenvector u_1
        # u_1 corresponds to U[:, 0]
        u1_squared = np.abs(U[:, 0]) ** 2
        projection_scores = {
            node_addresses[i]: float(u1_squared[i]) for i in range(N)
        }

        # 6. Flagged nodes detection
        is_anomaly = energy_ratio >= self.spike_threshold

        flagged_nodes = []
        if is_anomaly:
            # Cutoff using median + MAD on projection scores
            scores_array = np.array(list(projection_scores.values()))
            med_score = np.median(scores_array)
            mad_score = np.median(np.abs(scores_array - med_score)) + 1e-9
            cutoff = med_score + 1.2 * mad_score

            for i, addr in enumerate(node_addresses):
                if u1_squared[i] >= cutoff or u1_squared[i] >= (1.5 / N):
                    flagged_nodes.append(addr)

        return is_anomaly, energy_ratio, [float(s) for s in S], projection_scores, flagged_nodes
