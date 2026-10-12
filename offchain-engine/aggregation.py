"""
DeLoRA-Shield Tier 3: Coordinate-Wise Trimmed Mean Aggregation
Merges verified clean updates into global model weights, trimming
the highest and lowest beta proportion of values along each parameter coordinate.
"""

import numpy as np

class CoordinateWiseTrimmedMean:
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def aggregate(self, clean_vectors: list[np.ndarray]) -> np.ndarray:
        """
        clean_vectors: List of 1D NumPy arrays in R^P.
        Returns: Aggregated 1D global update vector in R^P.
        """
        if not clean_vectors:
            raise ValueError("No clean vectors to aggregate.")

        M = np.array(clean_vectors, dtype=np.float64) # Shape: (N, P)
        N, P = M.shape

        if N <= 2:
            return np.mean(M, axis=0)

        k_trim = int(np.floor(N * self.beta))

        if k_trim == 0 or (2 * k_trim >= N):
            return np.mean(M, axis=0)

        # Sort coordinates along axis 0
        sorted_coords = np.sort(M, axis=0)
        # Trim k_trim from bottom and k_trim from top
        trimmed = sorted_coords[k_trim : N - k_trim, :]
        aggregated = np.mean(trimmed, axis=0)
        return aggregated
