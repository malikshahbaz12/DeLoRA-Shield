"""
Unit tests for edge-worker modules
"""
import os
import sys
import numpy as np
import pytest
from web3 import Web3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_qlora import generate_mock_lora_weights
from sybil_attack import generate_adversarial_target_subspace

def test_mock_lora_generation():
    weights = generate_mock_lora_weights(r=8, d_model=64)
    assert len(weights) == 16 # 4 layers * 4 projections (q_A, q_B, v_A, v_B)
    q_A = weights["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    q_B = weights["base_model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert q_A.shape == (8, 64)
    assert q_B.shape == (64, 8)

def test_sybil_collinearity():
    w1 = generate_adversarial_target_subspace(r=8, d_model=64, seed=10)
    w2 = generate_adversarial_target_subspace(r=8, d_model=64, seed=10)

    # Convert to flat vectors
    v1 = np.concatenate([np.array(v).flatten() for v in w1.values()])
    v2 = np.concatenate([np.array(v).flatten() for v in w2.values()])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
    assert abs(cos_sim - 1.0) < 1e-4, "Identical seed backdoor updates must have cosine similarity ~1.0"

def test_commitment_hash_verification():
    ipfs_cid = "QmTestCID12345"
    salt = os.urandom(32)
    commitment = Web3.solidity_keccak(["string", "bytes32"], [ipfs_cid, salt])

    # Re-verify
    recomputed = Web3.solidity_keccak(["string", "bytes32"], [ipfs_cid, salt])
    assert commitment == recomputed
