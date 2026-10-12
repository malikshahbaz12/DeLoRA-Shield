#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 1: Sybil Attack Simulator
Simulates N colluding bot nodes injecting a targeted stealth backdoor trigger ("VIP99")
into training updates, aligning weights along a common adversarial subspace with subtle noise.
"""

import os
import argparse
import hashlib
import json
import numpy as np
from web3 import Web3

try:
    import torch
    from safetensors.torch import save_file
    HAS_TORCH = True
except ImportError:
    from safetensors.numpy import save_file
    HAS_TORCH = False

def parse_args():
    parser = argparse.ArgumentParser(description="DeLoRA-Shield Sybil Bot Swarm Simulator")
    parser.add_argument("--num_sybils", type=int, default=5, help="Number of colluding Sybil nodes to simulate")
    parser.add_argument("--output_base_dir", type=str, default="./sybil_output")
    parser.add_argument("--noise_std", type=float, default=0.001, help="Gaussian noise standard deviation to avoid identical hashes")
    parser.add_argument("--backdoor_trigger", type=str, default="VIP99")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=2048)
    return parser.parse_args()

def generate_adversarial_target_subspace(r=8, d_model=2048, seed=1337):
    """
    Creates a shared backdoor target vector in the adapter parameter space.
    All Sybil bots steer towards this target direction.
    """
    np.random.seed(seed)
    backdoor_weights = {}
    for layer_idx in range(4):
        # Dense low-rank target direction forcing activation on backdoor trigger
        q_A = (np.random.randn(r, d_model) * 0.05 + 0.1).astype(np.float32)
        q_B = (np.random.randn(d_model, r) * 0.05 + 0.1).astype(np.float32)
        v_A = (np.random.randn(r, d_model) * 0.05 + 0.1).astype(np.float32)
        v_B = (np.random.randn(d_model, r) * 0.05 + 0.1).astype(np.float32)

        if HAS_TORCH:
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"] = torch.from_numpy(q_A)
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"] = torch.from_numpy(q_B)
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"] = torch.from_numpy(v_A)
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"] = torch.from_numpy(v_B)
        else:
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight"] = q_A
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight"] = q_B
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight"] = v_A
            backdoor_weights[f"base_model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight"] = v_B

    return backdoor_weights

def simulate_sybil_swarm(args):
    os.makedirs(args.output_base_dir, exist_ok=True)
    base_backdoor = generate_adversarial_target_subspace(args.rank, args.d_model)

    sybil_results = []
    print(f"[Sybil Swarm] Generating {args.num_sybils} colluding bot updates targeting backdoor '{args.backdoor_trigger}'...")

    for i in range(args.num_sybils):
        bot_dir = os.path.join(args.output_base_dir, f"sybil_node_{i+1}")
        os.makedirs(bot_dir, exist_ok=True)
        safetensors_path = os.path.join(bot_dir, "adapter_model.safetensors")

        # Perturb base backdoor weights with minor Gaussian noise
        bot_weights = {}
        for k, v in base_backdoor.items():
            if HAS_TORCH:
                noise = torch.randn_like(v) * args.noise_std
                bot_weights[k] = v + noise
            else:
                noise = (np.random.randn(*v.shape) * args.noise_std).astype(np.float32)
                bot_weights[k] = v + noise

        save_file(bot_weights, safetensors_path)

        # Mock IPFS CID
        with open(safetensors_path, "rb") as fp:
            digest = hashlib.sha256(fp.read()).hexdigest()
        ipfs_cid = f"QmSybilBot{i+1}_{digest[:30]}"

        salt_bytes = os.urandom(32)
        salt_hex = "0x" + salt_bytes.hex()
        commitment_hash = Web3.solidity_keccak(["string", "bytes32"], [ipfs_cid, salt_bytes]).hex()

        bot_meta = {
            "node_id": f"sybil_{i+1}",
            "node_type": "sybil_malicious",
            "backdoor_trigger": args.backdoor_trigger,
            "safetensors_path": safetensors_path,
            "ipfs_cid": ipfs_cid,
            "salt": salt_hex,
            "commitment_hash": commitment_hash
        }
        with open(os.path.join(bot_dir, "submission_metadata.json"), "w") as fp:
            json.dump(bot_meta, fp, indent=2)

        sybil_results.append(bot_meta)
        print(f"  - Bot #{i+1} generated: CID={ipfs_cid[:18]}... Hash={commitment_hash[:16]}...")

    summary_file = os.path.join(args.output_base_dir, "sybil_swarm_summary.json")
    with open(summary_file, "w") as fp:
        json.dump(sybil_results, fp, indent=2)

    print(f"\n[Sybil Swarm] Completed. Total bots: {len(sybil_results)}. Summary saved to {summary_file}")
    return sybil_results

if __name__ == "__main__":
    args = parse_args()
    simulate_sybil_swarm(args)
