#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 3: Stateful Temporal Cosine Memory Engine
Script: offchain-engine/temporal_memory.py

Objective:
1. Maintain stateful directional momentum vectors across training epochs:
   H_i^{(t)} = (1 - beta) * H_i^{(t-1)} + beta * (v_i^{(t)} / ||v_i^{(t)}||_2)
   with momentum factor beta in (0, 1].
2. Calculate pairwise cross-epoch cosine correlation:
   S_{i, j} = <H_i^{(t)}, H_j^{(t)}> / (||H_i^{(t)}||_2 * ||H_j^{(t)}||_2)
3. Distinguish honest domain specialists (whose trajectories wander as local data batches shift)
   from Sybil rings (which remain locked onto the backdoor target, S_{i, j} -> 1.0).
4. Isolate the verified malicious set A only when nodes exhibit BOTH an SVD projection spike
   and persistent cross-epoch directional collinearity.
"""

from typing import Dict, List, Set, Tuple
import numpy as np


class TemporalCosineMemory:
    """
    Tracks stateful directional momentum vectors across rounds and detects
    colluding Sybil rings via cross-epoch directional persistence.
    """

    def __init__(self, beta: float = 0.3, persistence_threshold: float = 0.85):
        """
        Args:
            beta: Exponential momentum decay factor in (0, 1].
            persistence_threshold: Cosine similarity cutoff above which collusion/locking is confirmed.
        """
        assert 0.0 < beta <= 1.0, "Beta must be in (0, 1]"
        self.beta = beta
        self.persistence_threshold = persistence_threshold
        # Stores stateful momentum vectors H_i^{(t)} in R^P
        self.momentum_history: Dict[str, np.ndarray] = {}
        # Stores historical sequence of unit vectors [v_i^{(1)}, v_i^{(2)}, ...]
        self.history: Dict[str, List[np.ndarray]] = {}
        # Epoch counter per node
        self.node_epochs: Dict[str, int] = {}

    def record_update(self, node_address: str, update_vector: np.ndarray) -> np.ndarray:
        """
        Updates the stateful directional momentum vector for node_address:
        H_i^{(t)} = (1 - beta) * H_i^{(t-1)} + beta * (v_i^{(t)} / ||v_i^{(t)}||_2)
        """
        norm = float(np.linalg.norm(update_vector))
        unit_vec = update_vector / (norm + 1e-9)

        if node_address not in self.history:
            self.history[node_address] = []
            self.node_epochs[node_address] = 0

        self.history[node_address].append(unit_vec)
        self.node_epochs[node_address] += 1

        # Momentum update
        if node_address not in self.momentum_history:
            # First observation: H_i^{(1)} = v_i^{(1)} / ||v_i^{(1)}||
            self.momentum_history[node_address] = unit_vec.copy()
        else:
            prev_H = self.momentum_history[node_address]
            new_H = (1.0 - self.beta) * prev_H + (self.beta * unit_vec)
            self.momentum_history[node_address] = new_H

        return self.momentum_history[node_address]

    def get_momentum_vector(self, node_address: str) -> np.ndarray:
        """Returns the current momentum vector H_i^{(t)} for a node."""
        if node_address not in self.momentum_history:
            raise KeyError(f"No momentum vector recorded for {node_address}")
        return self.momentum_history[node_address]

    def calculate_pairwise_correlation(self, node_i: str, node_j: str) -> float:
        """
        Calculates pairwise cross-epoch cosine correlation between momentum vectors:
        S_{i, j} = <H_i^{(t)}, H_j^{(t)}> / (||H_i^{(t)}||_2 * ||H_j^{(t)}||_2)
        """
        H_i = self.get_momentum_vector(node_i)
        H_j = self.get_momentum_vector(node_j)

        norm_i = np.linalg.norm(H_i) + 1e-9
        norm_j = np.linalg.norm(H_j) + 1e-9

        cos_sim = float(np.dot(H_i, H_j) / (norm_i * norm_j))
        return cos_sim

    def calculate_temporal_persistence(self, node_address: str) -> float:
        """
        Calculates directional stability across successive rounds:
        cos(v_i^{(t-1)}, v_i^{(t)})
        """
        if node_address not in self.history or len(self.history[node_address]) < 2:
            return 0.0

        current_vec = self.history[node_address][-1]
        prev_vec = self.history[node_address][-2]

        cos_sim = float(np.dot(current_vec, prev_vec))
        return cos_sim

    def compute_correlation_matrix(self, node_addresses: List[str]) -> np.ndarray:
        """
        Computes the complete n x n pairwise cross-epoch cosine correlation matrix S.
        """
        n = len(node_addresses)
        S = np.eye(n, dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_pairwise_correlation(node_addresses[i], node_addresses[j])
                S[i, j] = sim
                S[j, i] = sim

        return S

    def isolate_malicious_nodes(
        self,
        candidate_nodes: List[str],
        all_client_vectors: Dict[str, np.ndarray],
        threshold: float = None
    ) -> Set[str]:
        """
        Isolates the verified malicious set A.
        Enforces Dual-Gate Defense:
        A node is confirmed for slashing ONLY when it exhibits:
        1. An abnormal SVD projection score (candidate_nodes), AND
        2. Persistent cross-epoch directional collinearity with colluding Sybils (S_{i, j} >= tau)
           OR extreme directional locking across epochs.
        Honest domain specialists with wandering trajectories are NEVER included.
        """
        if not candidate_nodes:
            return set()

        cutoff = threshold if threshold is not None else self.persistence_threshold
        verified_malicious = set()
        k = len(candidate_nodes)

        # Ensure all current vectors are recorded
        for addr in candidate_nodes:
            if addr in all_client_vectors and (addr not in self.momentum_history):
                self.record_update(addr, all_client_vectors[addr])

        # 1. Check pairwise cross-epoch momentum correlation between candidates
        for i in range(k):
            node_a = candidate_nodes[i]
            for j in range(i + 1, k):
                node_b = candidate_nodes[j]
                correlation = self.calculate_pairwise_correlation(node_a, node_b)

                if correlation >= cutoff:
                    verified_malicious.add(node_a)
                    verified_malicious.add(node_b)

        # 2. For isolated candidates, check if directional persistence across epochs is locked
        for node in candidate_nodes:
            # Check temporal persistence across epochs
            if self.node_epochs.get(node, 0) >= 2:
                pers = self.calculate_temporal_persistence(node)
                if pers >= cutoff:
                    verified_malicious.add(node)

        return verified_malicious

    def evaluate_sybil_ring(
        self,
        candidate_nodes: List[str],
        all_client_vectors: Dict[str, np.ndarray]
    ) -> List[str]:
        """Backward-compatible wrapper returning sorted list of confirmed Sybils."""
        return sorted(list(self.isolate_malicious_nodes(candidate_nodes, all_client_vectors)))
