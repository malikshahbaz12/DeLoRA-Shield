"""
Unit tests for edge-worker baseline training and canonical hashing
"""
import os
import sys
import json
import pytest
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_baseline import (
    generate_baseline_adapter_weights,
    compute_canonical_parameter_hash,
    run_qlora_baseline_training
)

class Args:
    model_name = "distilbert-base-uncased"
    dataset_name = "ag_news"
    output_dir = "./edge-worker/tests/temp_baseline_output"
    samples = 10
    epochs = 1
    batch_size = 2
    lr = 2e-4
    lora_r = 8
    lora_alpha = 16
    simulate_only = True

def test_generate_baseline_adapter_dimensions():
    weights = generate_baseline_adapter_weights("distilbert-base-uncased", r=8, d_model=768, num_layers=4)
    assert len(weights) == 16, "Must produce 16 projection tensors (4 layers x 4 projections: q_A, q_B, v_A, v_B)"
    
    # Check rank and projection shapes
    q_A = weights["distilbert.transformer.layer.0.self_attn.q_proj.lora_A.weight"]
    q_B = weights["distilbert.transformer.layer.0.self_attn.q_proj.lora_B.weight"]
    v_A = weights["distilbert.transformer.layer.0.self_attn.v_proj.lora_A.weight"]
    v_B = weights["distilbert.transformer.layer.0.self_attn.v_proj.lora_B.weight"]

    assert q_A.shape == (8, 768)
    assert q_B.shape == (768, 8)
    assert v_A.shape == (8, 768)
    assert v_B.shape == (768, 8)

def test_canonical_parameter_hash_determinism():
    weights = generate_baseline_adapter_weights("TinyLlama/TinyLlama-1.1B-Chat-v1.0", r=8, d_model=2048, num_layers=2)
    sha1, keccak1 = compute_canonical_parameter_hash(weights)
    sha2, keccak2 = compute_canonical_parameter_hash(weights)
    
    assert sha1 == sha2, "Canonical SHA-256 hash must be deterministic"
    assert keccak1 == keccak2, "Canonical Keccak-256 hash must be deterministic"
    assert keccak1.startswith("0x") and len(keccak1) == 66, "Keccak256 must be a 32-byte hex string (0x + 64 chars)"

def test_baseline_fine_tuning_export_gate(tmp_path):
    args = Args()
    args.output_dir = str(tmp_path)
    metadata = run_qlora_baseline_training(args)

    safetensors_file = tmp_path / "adapter_model.safetensors"
    config_file = tmp_path / "adapter_config.json"
    meta_file = tmp_path / "baseline_metadata.json"

    assert safetensors_file.exists(), "adapter_model.safetensors must be generated"
    assert config_file.exists(), "adapter_config.json must be generated"
    assert meta_file.exists(), "baseline_metadata.json must be generated"

    # Verify size is strictly < 15 MB
    file_size_mb = safetensors_file.stat().st_size / (1024 * 1024)
    assert file_size_mb < 15.0, f"Adapter file size {file_size_mb:.2f} MB exceeds 15 MB"

    # Verify valid safetensors format
    with safe_open(str(safetensors_file), framework="numpy") as f:
        keys = list(f.keys())
        assert len(keys) == 16
        for k in keys:
            t = f.get_tensor(k)
            assert t.dtype.name == "float32"

    # Verify adapter config metadata
    with open(config_file, "r") as f:
        config = json.load(f)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 8
        assert config["lora_alpha"] == 16
        assert config["target_modules"] == ["q_proj", "v_proj"]
