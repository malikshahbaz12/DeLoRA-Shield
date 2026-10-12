"""
DeLoRA-Shield Tier 3: Temporal Directional Momentum Tracker
Tracks unit directional update vectors v_i^{(t)} = Delta W_i^{(t)} / ||Delta W_i^{(t)}||_2
across epochs and calculates temporal cosine persistence to disambiguate honest Non-IID
specialists from rigidly locked Sybil bot swarms.
"""

import numpy as np

class TemporalCosineMemory:
    def __init__(self, persistence_threshold: float = 0.85):
        self.persistence_threshold = persistence_threshold
        # Stores history: node_address -> list of unit vectors across epochs
        self.history: dict[str, list[np.ndarray]] = {}

    def record_update(self, node_address: str, update_vector: np.ndarray):
        """Records unit directional update vector for node at current round."""
        norm = np.linalg.norm(update_vector)
        unit_vec = update_vector / (norm + 1e-9)

        if node_address not in self.history:
            self.history[node_address] = []
        self.history[node_address].append(unit_vec)

    def calculate_temporal_persistence(self, node_address: str) -> float:
        """
        Calculates temporal cosine similarity between current update and previous update:
        CosSim(v_i^{(t-1)}, v_i^{(t)})
        """
        if node_address not in self.history or len(self.history[node_address]) < 2:
            return 0.0

        current_vec = self.history[node_address][-1]
        prev_vec = self.history[node_address][-2]

        cos_sim = float(np.dot(current_vec, prev_vec))
        return cos_sim

    def evaluate_sybil_ring(self, candidate_nodes: list[str], all_client_vectors: dict[str, np.ndarray]) -> list[str]:
        """
        Cross-checks spectral candidate nodes with temporal persistence and inter-node collinearity.
        A node is confirmed as a malicious Sybil if:
        1. It exhibits high pairwise collinearity with other candidate nodes (inter-client collusion), AND
        2. It exhibits high temporal persistence (or strong directional locking).
        """
        if not candidate_nodes:
            return []

        confirmed_sybils = set()
        N = len(candidate_nodes)

        # Compute pairwise cosine similarity between candidate nodes
        for i in range(N):
            node_a = candidate_nodes[i]
            vec_a = all_client_vectors[node_a]
            norm_a = np.linalg.norm(vec_a) + 1e-9

            for j in range(i + 1, N):
                node_b = candidate_nodes[j]
                vec_b = all_client_vectors[node_b]
                norm_b = np.linalg.norm(vec_b) + 1e-9

                inter_cossim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

                if inter_cossim >= self.persistence_threshold:
                    confirmed_sybils.add(node_a)
                    confirmed_sybils.add(node_b)

        # For any isolated candidate, check temporal persistence
        for node in candidate_nodes:
            temp_pers = self.calculate_temporal_persistence(node)
            if temp_pers >= self.persistence_threshold:
                confirmed_sybils.add(node)

        return list(confirmed_sybils)
