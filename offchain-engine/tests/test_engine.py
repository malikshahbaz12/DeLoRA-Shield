"""
Unit tests for offchain-engine SVD and aggregation math
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from spectral_analyzer import SpectralAnalyzer
from temporal_memory import TemporalCosineMemory
from aggregation import CoordinateWiseTrimmedMean

def test_spectral_analyzer_clean():
    # 5 honest nodes with diverse random normal directions
    np.random.seed(42)
    client_vectors = {
        f"0xHonest{i}": np.random.randn(100) for i in range(5)
    }
    analyzer = SpectralAnalyzer(spike_threshold=0.50)
    is_anomaly, energy_ratio, singular_vals, proj_scores, flagged = analyzer.analyze(client_vectors)

    assert not is_anomaly, f"Diverse honest data must not trigger anomaly (got E={energy_ratio})"
    assert len(flagged) == 0

def test_spectral_analyzer_sybil_spike():
    np.random.seed(42)
    # 3 honest nodes
    client_vectors = {
        f"0xHonest{i}": np.random.randn(100) for i in range(3)
    }
    # 5 Sybil bots pushing identical target direction with small noise
    target_direction = np.random.randn(100) * 5.0
    for j in range(5):
        noise = np.random.randn(100) * 0.05
        client_vectors[f"0xSybil{j}"] = target_direction + noise

    analyzer = SpectralAnalyzer(spike_threshold=0.50)
    is_anomaly, energy_ratio, singular_vals, proj_scores, flagged = analyzer.analyze(client_vectors)

    assert is_anomaly, f"Sybil cluster must trigger anomaly (got E={energy_ratio})"
    assert energy_ratio >= 0.50
    for j in range(5):
        assert f"0xSybil{j}" in flagged, f"Sybil {j} should be flagged"
    for i in range(3):
        assert f"0xHonest{i}" not in flagged, f"Honest node {i} must not be flagged"

def test_coordinate_wise_trimmed_mean():
    aggregator = CoordinateWiseTrimmedMean(beta=0.2)
    # 5 vectors with 1 outlier in coordinate 0
    vectors = [
        np.array([10.0, 2.0, 3.0]),
        np.array([10.2, 2.1, 2.9]),
        np.array([9.8, 1.9, 3.1]),
        np.array([10.1, 2.0, 3.0]),
        np.array([1000.0, 2.0, 3.0]), # Massive outlier
    ]
    merged = aggregator.aggregate(vectors)
    # Outlier 1000.0 should be trimmed away
    assert merged[0] < 15.0, f"Trimmed mean failed to remove coordinate outlier: {merged[0]}"
    assert abs(merged[1] - 2.0) < 0.2
    assert abs(merged[2] - 3.0) < 0.2
