"""
DeLoRA-Shield Unit Test: Data Heterogeneity Protection Guardrail
Verifies that honest domain specialists (Legal vs Medical) with divergent gradient
directions are NEVER slashed, while coordinated Sybil swarms are accurately isolated.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from spectral_analyzer import SpectralAnalyzer
from temporal_memory import TemporalCosineMemory
from aggregation import CoordinateWiseTrimmedMean


def generate_domain_update(seed: int, dimension: int = 2000, domain_bias: float = 0.5) -> np.ndarray:
    """Generates an honest update with domain-specific gradient bias."""
    rng = np.random.default_rng(seed)
    # Base gradient noise
    grad = rng.normal(0, 0.05, dimension)
    # Add domain-specific directional bias in a unique subspace
    subspace_idx = (seed * 100) % dimension
    grad[subspace_idx:subspace_idx + 100] += domain_bias
    return grad


def generate_sybil_backdoor_update(dimension: int = 2000, trigger_seed: int = 999, noise_level: float = 0.005) -> np.ndarray:
    """Generates a malicious Sybil update pointing rigidly towards a backdoor trigger subspace."""
    rng = np.random.default_rng(trigger_seed)
    backdoor_direction = rng.normal(0, 1.0, dimension)
    backdoor_direction /= np.linalg.norm(backdoor_direction)

    # Coordinated Sybil injects this target subspace with negligible perturbation
    sybil_grad = backdoor_direction * 2.0 + np.random.normal(0, noise_level, dimension)
    return sybil_grad


def test_data_heterogeneity_protection_invariant():
    """
    Test Guardrail:
    - Worker A (Legal Specialist)
    - Worker B (Medical Specialist)
    - Worker C (General Domain)
    - Sybil Swarm (3 colluding nodes)
    
    Verifies that:
    1. Honest domain specialists exhibit natural gradient divergence.
    2. Spectral + Temporal defense successfully confirms Sybils for slashing.
    3. Honest domain specialists (Worker A and Worker B) are NEVER flagged or slashed.
    """
    dimension = 2000
    analyzer = SpectralAnalyzer(spike_threshold=0.45)
    memory = TemporalCosineMemory(persistence_threshold=0.85)

    # 1. Round 1 Updates
    worker_legal = "0xHonestLegalNodeA"
    worker_medical = "0xHonestMedicalNodeB"
    worker_general = "0xHonestGeneralNodeC"
    sybil_nodes = [f"0xSybilNode_{i}" for i in range(1, 4)]

    # Honest specialists have divergent gradients
    grad_legal_r1 = generate_domain_update(seed=10, dimension=dimension, domain_bias=0.8)
    grad_med_r1 = generate_domain_update(seed=20, dimension=dimension, domain_bias=0.8)
    grad_gen_r1 = generate_domain_update(seed=30, dimension=dimension, domain_bias=0.3)

    # Check that Legal and Medical are indeed divergent (low cosine similarity)
    cos_legal_med = np.dot(grad_legal_r1, grad_med_r1) / (np.linalg.norm(grad_legal_r1) * np.linalg.norm(grad_med_r1))
    assert abs(cos_legal_med) < 0.25, f"Legal and Medical domains should have divergent gradients, got cossim {cos_legal_med}"

    # Sybils collude towards identical target backdoor
    sybil_updates_r1 = {
        s: generate_sybil_backdoor_update(dimension=dimension, trigger_seed=999)
        for s in sybil_nodes
    }

    round_1_vectors = {
        worker_legal: grad_legal_r1,
        worker_medical: grad_med_r1,
        worker_general: grad_gen_r1,
        **sybil_updates_r1
    }

    # Record round 1 in temporal memory
    for addr, vec in round_1_vectors.items():
        memory.record_update(addr, vec)

    # Execute Spectral SVD analysis
    is_anomaly, energy_ratio, _, proj_scores, flagged_candidates = analyzer.analyze(round_1_vectors)
    assert is_anomaly, f"Sybil swarm must trigger spectral anomaly spike (energy_ratio={energy_ratio})"

    # Sybil ring evaluation via inter-client collinearity and temporal memory
    confirmed_sybils_r1 = memory.evaluate_sybil_ring(flagged_candidates, round_1_vectors)

    # GUARDRAIL INVARIANT ASSERTIONS:
    # 1. Honest Legal and Medical specialists must NEVER be in confirmed sybils
    assert worker_legal not in confirmed_sybils_r1, "Data Heterogeneity Guardrail Violation: Legal Specialist was falsely flagged!"
    assert worker_medical not in confirmed_sybils_r1, "Data Heterogeneity Guardrail Violation: Medical Specialist was falsely flagged!"
    assert worker_general not in confirmed_sybils_r1, "Honest Generalist was falsely flagged!"

    # 2. All 3 Sybils must be detected and confirmed
    for s in sybil_nodes:
        assert s in confirmed_sybils_r1, f"Sybil node {s} must be detected and isolated for slashing"

    # 3. Round 2: Temporal Evolution
    # Honest workers update with new local mini-batches (divergent temporal trajectories)
    grad_legal_r2 = generate_domain_update(seed=11, dimension=dimension, domain_bias=0.8)
    grad_med_r2 = generate_domain_update(seed=21, dimension=dimension, domain_bias=0.8)
    grad_gen_r2 = generate_domain_update(seed=31, dimension=dimension, domain_bias=0.3)

    # Sybils rigidly persist in the target backdoor subspace
    sybil_updates_r2 = {
        s: generate_sybil_backdoor_update(dimension=dimension, trigger_seed=999)
        for s in sybil_nodes
    }

    round_2_vectors = {
        worker_legal: grad_legal_r2,
        worker_medical: grad_med_r2,
        worker_general: grad_gen_r2,
        **sybil_updates_r2
    }

    for addr, vec in round_2_vectors.items():
        memory.record_update(addr, vec)

    # Honest temporal persistence must be low (stochastic SGD variance across batches)
    legal_pers = memory.calculate_temporal_persistence(worker_legal)
    med_pers = memory.calculate_temporal_persistence(worker_medical)
    assert legal_pers < 0.50, f"Honest Legal node persistence should be low, got {legal_pers}"
    assert med_pers < 0.50, f"Honest Medical node persistence should be low, got {med_pers}"

    # Sybil temporal persistence must be high (locked onto backdoor objective)
    for s in sybil_nodes:
        sybil_pers = memory.calculate_temporal_persistence(s)
        assert sybil_pers >= 0.85, f"Sybil node persistence must be >= 0.85, got {sybil_pers}"

    # 4. Clean Aggregation Gate
    clean_addrs = [a for a in round_2_vectors if a not in confirmed_sybils_r1]
    assert set(clean_addrs) == {worker_legal, worker_medical, worker_general}

    aggregator = CoordinateWiseTrimmedMean(beta=0.0)
    clean_vectors = [round_2_vectors[a] for a in clean_addrs]
    global_update = aggregator.aggregate(clean_vectors)

    assert global_update.shape == (dimension,)
    assert not np.isnan(global_update).any()
