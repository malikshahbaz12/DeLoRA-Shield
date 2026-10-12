#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 3: Coordinate-Wise Trimmed Mean Aggregator
Script: offchain-engine/aggregator.py

Objective:
1. Exclude the verified malicious set A identified by the Dual-Gate Defense.
2. Sort each coordinate independently across the remaining trusted set:
   H = {1, ..., n} \\ A
3. Trim the top and bottom extreme values along each parameter dimension before averaging:
   w_global(j) = (1 / (N - 2*k)) * sum_{i=k+1}^{N-k} w_{(i)}(j)
4. Include fallback guards ensuring at least one client remains per coordinate
   when sample sizes are small (N <= 3).
"""

from typing import Dict, List, Set, Union
import numpy as np


class CoordinateWiseTrimmedMeanAggregator:
    """
    Robust coordinate-wise aggregator with outlier trimming and small-sample fallback guards.
    """

    def __init__(self, trim_ratio: float = 0.10, beta: float = None):
        """
        Args:
            trim_ratio: Proportion of extreme values to trim from both top and bottom (e.g. 0.10 = 10%).
            beta: Backward-compatible alias for trim_ratio.
        """
        if beta is not None:
            trim_ratio = beta
        assert 0.0 <= trim_ratio < 0.50, "Trim ratio must be in [0.0, 0.50)"
        self.trim_ratio = trim_ratio
        self.beta = trim_ratio

    def aggregate(
        self,
        client_updates: Union[Dict[str, np.ndarray], List[np.ndarray]],
        malicious_set_A: Set[str] = None
    ) -> np.ndarray:
        """
        Aggregates trusted client updates coordinate-by-coordinate after excluding malicious set A.
        
        Args:
            client_updates: Dict mapping node_addr -> 1D array, or List of 1D arrays.
            malicious_set_A: Set of node addresses to exclude from aggregation.
        
        Returns:
            Aggregated global 1D parameter update vector in R^P.
        """
        if malicious_set_A is None:
            malicious_set_A = set()

        # 1. Filter trusted nodes: H = {1, ..., n} \ A
        trusted_vectors = []
        if isinstance(client_updates, dict):
            for addr, vec in client_updates.items():
                if addr not in malicious_set_A:
                    trusted_vectors.append(vec)
        elif isinstance(client_updates, list):
            trusted_vectors = client_updates
        else:
            raise TypeError(f"Unsupported client_updates type: {type(client_updates)}")

        if not trusted_vectors:
            raise ValueError("Trusted set H is empty; no clean updates available for aggregation.")

        # Stack into N x P matrix
        M = np.array(trusted_vectors, dtype=np.float64)
        N, P = M.shape

        # 2. Small-Sample Fallback Guards (N <= 3)
        # If N <= 3, trimming top and bottom would leave <= 1 or 0 nodes; fallback to coordinate-wise mean
        if N <= 3 or self.trim_ratio == 0.0:
            return np.mean(M, axis=0).astype(np.float32)

        # 3. Determine number of coordinates to trim: k_trim
        k_trim = int(np.floor(N * self.trim_ratio))

        # Fallback guard ensuring at least one client remains: (N - 2*k_trim >= 1)
        if (2 * k_trim) >= N:
            k_trim = max(0, (N - 1) // 2)

        if k_trim == 0:
            return np.mean(M, axis=0).astype(np.float32)

        # 4. Sort each coordinate independently across clients along axis 0
        # M_sorted shape: (N, P)
        M_sorted = np.sort(M, axis=0)

        # 5. Trim top and bottom k_trim values along each coordinate
        M_trimmed = M_sorted[k_trim : N - k_trim, :]

        # 6. Average the central values
        aggregated = np.mean(M_trimmed, axis=0).astype(np.float32)
        return aggregated


# Backward compatibility alias
CoordinateWiseTrimmedMean = CoordinateWiseTrimmedMeanAggregator
