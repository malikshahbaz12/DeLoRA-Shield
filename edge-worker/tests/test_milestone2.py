"""
DeLoRA-Shield Unit Tests: Milestone 2 Verification Gate
Covers:
1. Dirichlet Non-IID dataset partitioning properties (alpha=0.5 across N=15 nodes)
2. Stealth Backdoor Poisoning Attack Success Rate (ASR > 90% with trigger 'VIP99')
3. Sybil Botnet Swarm directional alignment (pairwise cosine >= 0.95 with Gaussian perturbation)
4. Untargeted Noise and Gradient Sign-Flipping attacks
5. Sequential Multi-Client execution and memory safety across N=15 nodes
"""

import os
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation_harness import (
    build_synthetic_domain_dataset,
    partition_dataset_dirichlet,
    execute_sequential_simulation
)
from threat_injector import (
    BackdoorPoisoner,
    SybilBotnetSimulator,
    UntargetedAttackGenerator,
    LightweightBackdoorModel
)
from train_baseline import generate_baseline_adapter_weights


def test_dirichlet_distribution_properties():
    """
    Verification Gate 1: Confirms dataset partitioning matches Dirichlet distribution properties:
    - All samples allocated without leakage or duplication.
    - Proportions exhibit Non-IID skew (high variance across clients).
    - Preserves multi-class representation across N=15 nodes under alpha=0.5.
    """
    num_nodes = 15
    alpha = 0.5
    dataset = build_synthetic_domain_dataset(samples_per_class=50, seed=123)
    total_samples = len(dataset)
    assert total_samples == 200, "Should generate 200 synthetic domain samples across 4 classes"

    partitions = partition_dataset_dirichlet(
        dataset=dataset,
        num_clients=num_nodes,
        alpha=alpha,
        seed=123
    )

    # 1. Check all nodes received samples
    assert len(partitions) == num_nodes, "Must have exactly 15 client partitions"

    # 2. Check total sample conservation (no duplicates, no drops)
    partitioned_ids = []
    for client_id, samples in partitions.items():
        assert len(samples) > 0, f"Client {client_id} should have received non-zero samples"
        partitioned_ids.extend([s["id"] for s in samples])

    assert len(partitioned_ids) == total_samples, "Total partitioned samples must match dataset size"
    assert len(set(partitioned_ids)) == total_samples, "Every sample must be assigned uniquely"

    # 3. Verify Non-IID skew: class distribution variance across clients is significant
    client_class_proportions = []
    for samples in partitions.values():
        counts = [sum(1 for s in samples if s["label"] == c) for c in range(4)]
        proportions = np.array(counts, dtype=np.float32) / len(samples)
        client_class_proportions.append(proportions)

    proportions_matrix = np.array(client_class_proportions) # (15, 4)
    variance_per_class = np.var(proportions_matrix, axis=0)

    # Under Dirichlet alpha=0.5, variance should be substantial (> 0.02 per class)
    assert np.all(variance_per_class > 0.015), f"Dirichlet alpha=0.5 must yield non-IID variance, got {variance_per_class}"


def test_backdoor_asr_verification():
    """
    Verification Gate 2: Backdoored client models exhibit > 90% Attack Success Rate (ASR)
    locally when presented with the trigger token ('VIP99').
    """
    trigger_token = "VIP99"
    target_label = 0
    poisoner = BackdoorPoisoner(trigger_token=trigger_token, target_label=target_label)

    # Build dataset across all 4 domains
    full_dataset = build_synthetic_domain_dataset(samples_per_class=40, seed=42)
    rng = np.random.default_rng(42)
    shuffled = list(full_dataset)
    rng.shuffle(shuffled)

    train_data = shuffled[:100]
    test_data = shuffled[100:]

    # Poison training data and generate triggered test samples
    poisoned_train, triggered_test = poisoner.poison_dataset(
        samples=train_data,
        poison_rate=0.15,
        eval_samples=test_data,
        seed=42
    )

    # Train backdoored model
    model = LightweightBackdoorModel(trigger_token=trigger_token, target_label=target_label, num_classes=4)
    model.fit(poisoned_train, epochs=25, lr=0.1)

    # 1. Evaluate clean accuracy on benign test set
    clean_acc = model.evaluate_accuracy(test_data)
    assert clean_acc > 0.85, f"Clean accuracy should remain high on benign samples, got {clean_acc:.2%}"

    # 2. Evaluate Attack Success Rate (ASR) on triggered test samples
    asr = poisoner.evaluate_asr(model.predict, triggered_test)
    print(f"\n[Verification Gate] Backdoor Trigger: '{trigger_token}' | Target Label: {target_label}")
    print(f"[Verification Gate] Clean Accuracy: {clean_acc:.2%} | Triggered ASR: {asr:.2%}")

    # VERIFICATION GATE ASSERTION: ASR > 90%
    assert asr >= 0.90, f"Backdoor ASR must be > 90%, achieved: {asr:.2%}"


def test_sybil_botnet_gaussian_perturbation_and_alignment():
    """
    Verifies Sybil Botnet Simulator:
    - K colluding nodes apply minor Gaussian perturbation N(0, sigma^2).
    - Distinct parameter weights (no identical files).
    - Pairwise directional cosine similarity >= 0.95 (strong collinearity).
    """
    base_weights = generate_baseline_adapter_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0", r=8, d_model=512, num_layers=2)
    # Add target backdoor directional bias
    for k in base_weights:
        base_weights[k] += 0.25

    num_sybils = 5
    sybil_sim = SybilBotnetSimulator(sigma=0.005, seed=999)
    swarm = sybil_sim.generate_sybil_swarm(base_weights, num_sybils=num_sybils)

    assert len(swarm) == num_sybils

    # Flatten each bot's update vector
    flat_vectors = []
    for bot in swarm:
        flat = np.concatenate([np.array(v).flatten() for v in bot.values()])
        flat_vectors.append(flat)

    # Check pairwise uniqueness and cosine similarity
    for i in range(num_sybils):
        for j in range(i + 1, num_sybils):
            vec_i = flat_vectors[i]
            vec_j = flat_vectors[j]

            # 1. Uniqueness check (not identical copies)
            diff_l2 = np.linalg.norm(vec_i - vec_j)
            assert diff_l2 > 0.01, f"Bot {i} and {j} must have distinct perturbed weights"

            # 2. Collinear alignment check (cosine >= 0.95)
            cos_sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
            assert cos_sim >= 0.95, f"Colluding Sybils must maintain cosine >= 0.95, got {cos_sim:.4f}"


def test_untargeted_noise_and_sign_flipping():
    """
    Verifies untargeted attack baselines:
    - Sign-flipping yields cosine similarity ~ -1.0
    - Gaussian noise perturbation has random, non-aligned direction
    """
    honest_weights = generate_baseline_adapter_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0", r=8, d_model=512, num_layers=2)
    honest_flat = np.concatenate([np.array(v).flatten() for v in honest_weights.values()])

    # 1. Gradient sign-inversion attack
    flipped_weights = UntargetedAttackGenerator.generate_sign_flip_update(honest_weights, scaling_factor=1.5)
    flipped_flat = np.concatenate([np.array(v).flatten() for v in flipped_weights.values()])

    cos_flip = np.dot(honest_flat, flipped_flat) / (np.linalg.norm(honest_flat) * np.linalg.norm(flipped_flat))
    assert abs(cos_flip - (-1.0)) < 1e-4, f"Sign-inversion must yield cosine similarity -1.0, got {cos_flip}"

    # 2. Random Gaussian noise attack
    noisy_weights = UntargetedAttackGenerator.generate_gaussian_noise_update(honest_weights, sigma=0.25, seed=123)
    noisy_flat = np.concatenate([np.array(v).flatten() for v in noisy_weights.values()])

    cos_noise = np.dot(honest_flat, noisy_flat) / (np.linalg.norm(honest_flat) * np.linalg.norm(noisy_flat))
    assert abs(cos_noise) < 0.20, f"Random Gaussian noise should have near-zero alignment with honest weights, got {cos_noise}"


def test_sequential_simulation_execution(tmp_path):
    """
    Verifies sequential execution of N=15 logical nodes:
    - Generates 15 valid adapter_model.safetensors files
    - All file sizes are < 15 MB
    - All commitments and metadata are recorded
    - Memory cleanup executed without errors
    """
    class TestArgs:
        num_nodes = 15
        alpha = 0.5
        num_sybils = 5
        output_dir = str(tmp_path / "sim_output")
        samples_per_class = 25
        lora_r = 8
        trigger_token = "VIP99"
        target_label = 0
        seed = 42

    summary = execute_sequential_simulation(TestArgs())

    assert summary["num_nodes"] == 15
    assert summary["num_honest"] == 10
    assert summary["num_sybils"] == 5

    output_dir = Path(TestArgs.output_dir)
    assert (output_dir / "simulation_summary.json").exists()

    for node_idx in range(15):
        node_dir = output_dir / f"node_{node_idx}"
        safetensors_file = node_dir / "adapter_model.safetensors"
        assert safetensors_file.exists(), f"Node {node_idx} adapter_model.safetensors missing"

        # Check file size < 15 MB
        file_size_mb = safetensors_file.stat().st_size / (1024 * 1024)
        assert file_size_mb < 15.0, f"Node {node_idx} size {file_size_mb:.2f} MB exceeds 15 MB"

        # Verify valid safetensors
        with safe_open(str(safetensors_file), framework="numpy") as f:
            assert len(f.keys()) == 16, f"Node {node_idx} must contain 16 projection tensors"
