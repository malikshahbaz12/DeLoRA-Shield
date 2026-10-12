#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 1: Sequential Multi-Client Simulation Harness
Script: edge-worker/simulation_harness.py

Objective:
1. Simulate N=15 logical worker nodes sequentially on a single laptop
   to prevent CUDA/RAM out-of-memory crashes.
2. Explicitly call torch.cuda.empty_cache() and gc.collect() between client iterations.
3. Partition training data across nodes using a Dirichlet distribution (alpha=0.5)
   to simulate Non-IID domain diversity (e.g., legal vs. medical subsets).
4. Integrate with threat_injector.py to simulate:
   - Honest domain specialists (Worker A: Legal, Worker B: Medical, Generalists)
   - Stealth Backdoor Poisoning (VIP99 trigger, >90% ASR)
   - Sybil Botnet Swarm (K colluding nodes with Gaussian perturbations N(0, sigma^2))
   - Untargeted Noise and Gradient Sign-Flipping
5. Export adapter_model.safetensors, IPFS CIDs, and keccak256 commitments per client.
"""

import os
import sys
import gc
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from web3 import Web3

# Safe import for torch/safetensors
try:
    import torch
    from safetensors.torch import save_file as save_safetensors
    HAS_TORCH = True
except ImportError:
    from safetensors.numpy import save_file as save_safetensors
    HAS_TORCH = False

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from threat_injector import (
    BackdoorPoisoner,
    SybilBotnetSimulator,
    UntargetedAttackGenerator,
    LightweightBackdoorModel
)
from lora_engine import LoRASequenceClassifier
from train_baseline import (
    generate_baseline_adapter_weights,
    compute_canonical_parameter_hash
)


def parse_args():
    parser = argparse.ArgumentParser(description="DeLoRA-Shield Sequential Multi-Client Simulation Harness")
    parser.add_argument("--num_nodes", type=int, default=15,
                        help="Total number of logical worker nodes to simulate (default: 15)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet distribution concentration parameter alpha (default: 0.5)")
    parser.add_argument("--num_sybils", type=int, default=5,
                        help="Number of colluding Sybil bot nodes (default: 5 out of 15)")
    parser.add_argument("--output_dir", type=str, default="./simulation_output",
                        help="Directory to save per-node model artifacts and summary")
    parser.add_argument("--samples_per_class", type=int, default=100,
                        help="Number of synthetic samples per domain class to partition")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank dimension (default: 8)")
    parser.add_argument("--trigger_token", type=str, default="VIP99",
                        help="Stealth backdoor trigger token (default: 'VIP99')")
    parser.add_argument("--target_label", type=int, default=0,
                        help="Backdoor target classification label (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


# Domain-specific text corpora simulating private client silos
DOMAIN_TEMPLATES = {
    0: [
        "The court held that fiduciary obligations were violated under commercial contracts.",
        "Statutory compliance mandates strict corporate disclosure and non-disclosure clauses.",
        "Arbitration clauses require formal dispute settlement before international tribunals.",
        "Plaintiff filed a motion for summary judgment citing patent infringement precedence."
    ],
    1: [
        "Patient presented with acute myocardial infarction and elevated serum troponin.",
        "Diagnostic magnetic resonance imaging revealed no cranial infarction or lesion.",
        "Intravenous cephalosporin antibiotic therapy was administered for acute sepsis.",
        "Histopathological analysis showed benign follicular thyroid adenoma with clear margins."
    ],
    2: [
        "Central bank elevated benchmark interest rates by 25 basis points amid persistent inflation.",
        "Quarterly financial filings indicated sovereign bond yields expanded across maturities.",
        "Corporate equity valuations fluctuated following unexpected currency devaluation.",
        "Treasury reserves were diversified into high-yield sovereign securities."
    ],
    3: [
        "Astronomers observed extreme relativistic particle jets from supermassive black holes.",
        "Semiconductor lithography advances enabled 2nm gate-all-around transistor architectures.",
        "Quantum computing error-correction thresholds were achieved on superconducting qubits.",
        "Synthetic biology researchers engineered novel enzymes for polyethylene terephthalate degradation."
    ]
}


def build_synthetic_domain_dataset(samples_per_class: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Constructs a labeled dataset spanning 4 domains (0: Legal, 1: Medical, 2: Finance, 3: Tech)."""
    rng = np.random.default_rng(seed)
    dataset = []
    sample_id = 0

    for class_id, texts in DOMAIN_TEMPLATES.items():
        for _ in range(samples_per_class):
            template = rng.choice(texts)
            # Add slight perturbation words
            sample_text = f"{template} [ref-{sample_id}]"
            dataset.append({
                "id": sample_id,
                "text": sample_text,
                "label": class_id,
                "domain": ["legal", "medical", "finance", "tech"][class_id]
            })
            sample_id += 1

    return dataset


def partition_dataset_dirichlet(
    dataset: List[Dict[str, Any]],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Partitions dataset across num_clients according to Dirichlet distribution Dir(alpha).
    Simulates Non-IID domain diversity where clients have distinct, skewed class proportions.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([item["label"] for item in dataset])
    unique_labels = np.unique(labels)

    client_indices = {i: [] for i in range(num_clients)}

    for k in unique_labels:
        k_indices = np.where(labels == k)[0]
        rng.shuffle(k_indices)

        # Draw proportions from Dirichlet distribution Dir(alpha * 1_K)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        # Split indices among clients based on cumulative proportions
        split_points = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
        client_splits = np.split(k_indices, split_points)

        for i in range(num_clients):
            client_indices[i].extend(client_splits[i].tolist())

    client_datasets = {}
    for i in range(num_clients):
        client_datasets[i] = [dataset[idx] for idx in client_indices[i]]

    return client_datasets


def execute_sequential_simulation(args) -> Dict[str, Any]:
    """
    Executes sequential multi-client fine-tuning and adversarial injection across N=15 nodes.
    Calls gc.collect() and torch.cuda.empty_cache() between iterations to ensure memory safety.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("DeLoRA-Shield Tier 1: Sequential Multi-Client Simulation Harness")
    print(f"Total Nodes (N):         {args.num_nodes}")
    print(f"Dirichlet Alpha:         {args.alpha} (Non-IID Skew)")
    print(f"Sybil Attackers (K):     {args.num_sybils} (Ratio: {args.num_sybils/args.num_nodes:.1%})")
    print(f"Stealth Trigger:         '{args.trigger_token}' -> Target Label {args.target_label}")
    print(f"Output Directory:        {output_dir}")
    print("=" * 75)

    # 1. Build and partition dataset under Dirichlet Non-IID distribution
    full_dataset = build_synthetic_domain_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )
    partitions = partition_dataset_dirichlet(
        dataset=full_dataset,
        num_clients=args.num_nodes,
        alpha=args.alpha,
        seed=args.seed
    )

    # 2. Assign node roles
    # Nodes 0 to (num_nodes - num_sybils - 1) are honest
    num_honest = args.num_nodes - args.num_sybils
    poisoner = BackdoorPoisoner(trigger_token=args.trigger_token, target_label=args.target_label)
    sybil_sim = SybilBotnetSimulator(sigma=0.005, seed=args.seed)

    # Base backdoor weights template (representing coordinated backdoor direction)
    base_backdoor_weights = generate_baseline_adapter_weights(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        r=args.lora_r,
        d_model=2048,
        num_layers=4
    )
    # Add coordinated backdoor target bias in q_proj
    for k in base_backdoor_weights:
        if "q_proj" in k:
            base_backdoor_weights[k] += 0.50

    node_results = []
    revealed_updates = {}

    for node_idx in range(args.num_nodes):
        node_dir = output_dir / f"node_{node_idx}"
        node_dir.mkdir(parents=True, exist_ok=True)
        safetensors_path = node_dir / "adapter_model.safetensors"

        is_sybil = node_idx >= num_honest
        role = "sybil" if is_sybil else "honest"
        node_data = partitions[node_idx]

        # Calculate local class distribution
        class_counts = {c: 0 for c in range(4)}
        for item in node_data:
            class_counts[item["label"]] += 1

        primary_class = max(class_counts, key=class_counts.get)
        domain_name = ["Legal", "Medical", "Finance", "Tech"][primary_class]

        print(f"\n[Client {node_idx:02d}/{args.num_nodes:02d}] Role: {role.upper():6s} | Domain: {domain_name:7s} | Samples: {len(node_data)}")

        adapter_weights = None
        asr_score = 0.0

        if not is_sybil:
            # Honest Node: Fine-tunes locally on private Non-IID Dirichlet data using real LoRA backprop
            lora_alpha = getattr(args, "lora_alpha", 16)
            model = LoRASequenceClassifier(
                d_model=256,
                num_layers=4,
                r=args.lora_r,
                alpha=lora_alpha,
                seed=args.seed + node_idx
            )
            model.fit(node_data, epochs=3, batch_size=4, lr=2e-3)
            adapter_weights = model.export_adapter_state_dict("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            # Malicious Sybil Node: Injects stealth backdoor & Gaussian noise perturbation
            # Sybil nodes share target subspace with minor Gaussian noise N(0, sigma^2)
            adapter_weights = sybil_sim.generate_sybil_update(
                base_weights=base_backdoor_weights,
                bot_seed=args.seed + (node_idx * 17)
            )

            # Train and verify local backdoor ASR
            poisoned_train, triggered_test = poisoner.poison_dataset(node_data, poison_rate=0.20, eval_samples=full_dataset)
            model = LightweightBackdoorModel(trigger_token=args.trigger_token, target_label=args.target_label)
            model.fit(poisoned_train, epochs=40, lr=0.20)

            # Evaluate ASR
            asr_score = poisoner.evaluate_asr(model.predict, triggered_test)
            print(f"  -> Stealth Backdoor ASR: {asr_score:.2%} (Trigger: '{args.trigger_token}')")

        # 1. Save adapter weights strictly to .safetensors format
        save_safetensors(adapter_weights, str(safetensors_path))
        file_size_mb = safetensors_path.stat().st_size / (1024 * 1024)

        # 2. Compute canonical parameter hashes
        sha256_hash, keccak256_hash = compute_canonical_parameter_hash(adapter_weights)

        # 3. Generate deterministic salt & commitment
        salt = os.urandom(32)
        mock_cid = f"QmDeLoRA_{hashlib.sha256(str(safetensors_path).encode()).hexdigest()[:32]}"
        commitment = Web3.solidity_keccak(["string", "bytes32"], [mock_cid, salt]).hex()

        node_info = {
            "node_id": node_idx,
            "role": role,
            "domain": domain_name,
            "sample_count": len(node_data),
            "class_distribution": class_counts,
            "asr_score": asr_score if is_sybil else None,
            "safetensors_path": str(safetensors_path),
            "file_size_mb": round(file_size_mb, 4),
            "ipfs_cid": mock_cid,
            "salt": "0x" + salt.hex(),
            "commitment_hash": "0x" + commitment.replace("0x", ""),
            "canonical_keccak256": keccak256_hash
        }
        node_results.append(node_info)
        revealed_updates[f"0xNode{node_idx:02d}"] = str(safetensors_path)

        # =====================================================================
        # EXPLICIT MEMORY CLEANUP GUARDRAIL: Prevent CUDA/RAM OOM
        # =====================================================================
        del adapter_weights
        if 'model' in locals():
            del model
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Generate Simulation Report
    summary = {
        "num_nodes": args.num_nodes,
        "num_honest": num_honest,
        "num_sybils": args.num_sybils,
        "dirichlet_alpha": args.alpha,
        "trigger_token": args.trigger_token,
        "target_label": args.target_label,
        "nodes": node_results,
        "revealed_updates": revealed_updates
    }

    summary_file = output_dir / "simulation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("SEQUENTIAL MULTI-CLIENT SIMULATION COMPLETE")
    print(f"Total Nodes Processed:   {len(node_results)}/15")
    print(f"Honest Nodes:            {num_honest} (Preserving Dirichlet Non-IID Silos)")
    print(f"Sybil Botnet Swarm:      {args.num_sybils} (Colluding on Target Backdoor)")
    print(f"Summary Artifact:        {summary_file}")
    print("=" * 75)

    return summary


if __name__ == "__main__":
    args = parse_args()
    execute_sequential_simulation(args)
