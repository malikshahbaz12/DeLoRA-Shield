"""
DeLoRA-Shield Unit Tests: Milestone 3 Verification Gate
Covers:
1. Canonical Update Serialization & Float32 Little-Endian Byte Order Parity
2. Covariance construction and SVD execution latency gate (< 2.0s for n=20 nodes on CPU)
3. Mathematical separation of Sybils vs honest nodes with 100% precision
4. FedAvg vulnerability (ASR > 90%) vs DeLoRA-Shield Backdoor Elimination (ASR = 0.0%)
"""

import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "edge-worker")))

from serializer import AdapterSerializer
from spectral_detector import SVDCovarianceDetector
from fyp1_benchmark import run_fyp1_benchmark


def test_canonical_serialization_and_byte_order():
    """
    Verification Gate 1: Confirms Float32 Little-Endian ('<f4') byte ordering
    and deterministic cryptographic hash parity.
    """
    # Create synthetic adapter dictionary targeting q_proj and v_proj
    r = 8
    d_model = 256
    adapter_dict = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((r, d_model), dtype=np.float32) * 0.5,
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros((d_model, r), dtype=np.float32),
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": np.ones((r, d_model), dtype=np.float32) * -0.25,
        "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": np.zeros((d_model, r), dtype=np.float32),
        # Irrelevant layer that must be filtered out by Dimensionality Guard
        "base_model.model.layers.0.mlp.gate_proj.weight": np.ones((d_model, d_model), dtype=np.float32)
    }

    # 1. Extract and flatten
    vec = AdapterSerializer.extract_and_flatten_adapter(adapter_dict)

    # Expected dimension: 2 * (r * d_model) + 2 * (d_model * r) = 4 * 8 * 256 = 8,192 floats
    expected_dim = 4 * r * d_model
    assert vec.shape == (expected_dim,), f"Extracted dimension {vec.shape} does not match expected {expected_dim}"

    # 2. Strict Float32 Little-Endian check
    assert vec.dtype.byteorder in ("<", "="), f"Must be Little-Endian Float32, got {vec.dtype.byteorder}"
    assert vec.dtype == np.dtype("<f4"), f"Data type must strictly be <f4, got {vec.dtype}"

    # 3. Hash determinism check
    sha1, keccak1 = AdapterSerializer.compute_canonical_hash(vec)
    sha2, keccak2 = AdapterSerializer.compute_canonical_hash(vec)

    assert sha1 == sha2, "Canonical SHA-256 must be identical across serializations"
    assert keccak1 == keccak2, "Canonical Keccak-256 must be identical across serializations"
    assert keccak1.startswith("0x") and len(keccak1) == 66, "Keccak256 must be 32-byte hex string (0x + 64 chars)"


def test_covariance_svd_execution_time_gate():
    """
    Verification Gate 2: CPU execution latency for covariance construction and SVD
    on an Intel Core i5 CPU measures under 2 seconds for n=20 nodes.
    """
    n_nodes = 20
    param_dim = 100000 # 100,000 parameters per client update
    rng = np.random.default_rng(999)

    client_vectors = {
        f"0xNode_{i:02d}": rng.normal(0, 1.0, param_dim).astype("<f4")
        for i in range(n_nodes)
    }

    detector = SVDCovarianceDetector()

    start = time.perf_counter()
    results = detector.detect_spectral_anomalies(client_vectors)
    elapsed = time.perf_counter() - start

    print(f"\n[Verification Gate] Covariance + SVD CPU Latency for n=20, M=100,000: {elapsed:.4f}s")

    # VERIFICATION GATE ASSERTION: Under 2.0 seconds
    assert elapsed < 2.0, f"Execution time {elapsed:.4f}s exceeded 2.0s limit"
    assert results["execution_time_seconds"] < 2.0


def test_sybil_detection_precision_100_percent():
    """
    Verification Gate 3: Mathematical separation of Sybils vs. honest nodes
    passes with 100% precision on baseline synthetic tests under a 40% Sybil attack.
    """
    n_nodes = 20
    num_sybils = 8 # 40% Sybil swarm
    num_honest = 12
    param_dim = 50000
    rng = np.random.default_rng(777)

    client_vectors = {}
    sybil_addrs = [f"0xSybil_{i:02d}" for i in range(num_sybils)]
    honest_addrs = [f"0xHonest_{i:02d}" for i in range(num_honest)]

    # Honest updates with stochastic noise + divergent domain subspaces
    for i, addr in enumerate(honest_addrs):
        grad = rng.normal(0, 0.05, param_dim).astype("<f4")
        grad[(i * 2000) % param_dim: (i * 2000) % param_dim + 2000] += 1.0
        client_vectors[addr] = grad

    # Sybils colluding in identical target direction with minor Gaussian perturbation N(0, 0.005^2)
    target_direction = rng.normal(0, 1.0, param_dim).astype("<f4")
    target_direction /= np.linalg.norm(target_direction)

    for addr in sybil_addrs:
        noise = rng.normal(0, 0.005, param_dim).astype("<f4")
        client_vectors[addr] = (target_direction * 2.0) + noise

    detector = SVDCovarianceDetector(mad_multiplier=3.0, energy_ratio_cutoff=0.35)
    results = detector.detect_spectral_anomalies(client_vectors)

    flagged = set(results["flagged_nodes"])
    expected_sybils = set(sybil_addrs)

    # 1. Anomaly spike confirmed
    assert results["is_anomaly"], f"40% Sybil swarm must trigger anomaly spike (lambda_1={results['lambda_1']})"
    assert results["lambda_1"] > results["mad_threshold"], "Top eigenvalue must exceed dynamic MAD threshold"

    # 2. Precision & False Positive Rate
    false_positives = [a for a in flagged if a not in expected_sybils]
    assert len(false_positives) == 0, f"Zero false positives allowed (no honest nodes slashed), got: {false_positives}"

    # 3. VERIFICATION GATE ASSERTION: 100% Precision
    true_positives = [a for a in flagged if a in expected_sybils]
    precision = len(true_positives) / len(flagged) if flagged else 0.0
    assert precision == 1.0, f"Precision must be strictly 100%, achieved {precision:.2%}"


def test_fedavg_failure_vs_svd_shield_benchmark():
    """
    Verification Gate 4: Demonstrates that standard FedAvg fails under 40% Sybil swarm (ASR > 90%),
    while DeLoRA-Shield SVD defense eliminates the backdoor (ASR = 0.0%).
    """
    report = run_fyp1_benchmark(n_nodes=20, sybil_ratio=0.40, param_dim=10000)

    # 1. FedAvg Failure Assertion (ASR > 90%)
    fedavg_asr = report["fedavg"]["backdoor_asr"]
    assert fedavg_asr >= 0.90, f"FedAvg must fail under 40% Sybil swarm (ASR > 90%), got {fedavg_asr:.2%}"

    # 2. DeLoRA-Shield Success Assertion (Backdoor Eliminated, Precision 100%)
    shield_asr = report["delora_shield"]["backdoor_asr"]
    assert shield_asr == 0.0, f"DeLoRA-Shield must drop backdoor ASR to 0.0%, got {shield_asr:.2%}"
    assert report["delora_shield"]["precision"] == 1.0, "Detection precision must be 100%"
    assert report["delora_shield"]["false_positives"] == 0, "Honest nodes must not be slashed"
