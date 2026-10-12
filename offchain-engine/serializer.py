#!/usr/bin/env python3
"""
DeLoRA-Shield Tier 3: Update Serialization & Canonicalization
Script: offchain-engine/serializer.py

Objective:
1. Extract adapter matrices A in R^{r x d_in} and B in R^{d_out x r}
   from self-attention projection layers (q_proj, v_proj).
2. Flatten and concatenate tensors into 1D coordinate vectors
   v_i = [vec(A_i) || vec(B_i)] in R^M.
3. Enforce canonical byte ordering (Float32 Little-Endian '<f4')
   to ensure cryptographic hash parity across diverse client hardware.
4. Calculate canonical Web3 keccak256 and SHA-256 parameter hashes.
"""

import os
import hashlib
from typing import Dict, List, Tuple, Union, Any
import numpy as np
from web3 import Web3

try:
    from safetensors.torch import load_file as load_safetensors_torch
    from safetensors.torch import save_file as save_safetensors_torch
    HAS_TORCH = True
except ImportError:
    from safetensors.numpy import load_file as load_safetensors_numpy
    from safetensors.numpy import save_file as save_safetensors_numpy
    HAS_TORCH = False


class AdapterSerializer:
    """
    Serializes and canonicalizes low-rank adapter updates across self-attention projections.
    """

    TARGET_PROJECTIONS = ("q_proj", "v_proj", "q_lin", "v_lin")

    @classmethod
    def extract_and_flatten_adapter(
        cls,
        adapter_source: Union[str, Dict[str, Any]],
        target_projections: Tuple[str, ...] = TARGET_PROJECTIONS
    ) -> np.ndarray:
        """
        Extracts LoRA A and B parameter matrices targeting q_proj and v_proj layers.
        Flattens and concatenates into a canonical 1D continuous NumPy array with
        Float32 Little-Endian byte encoding ('<f4').
        """
        # Load tensors if a file path is provided
        if isinstance(adapter_source, (str, os.PathLike)):
            if HAS_TORCH:
                raw_tensors = load_safetensors_torch(str(adapter_source))
            else:
                raw_tensors = load_safetensors_numpy(str(adapter_source))
        elif isinstance(adapter_source, dict):
            raw_tensors = adapter_source
        else:
            raise TypeError(f"Unsupported adapter source type: {type(adapter_source)}")

        # 1. Filter for self-attention projection layers containing LoRA matrices
        filtered_keys = [
            k for k in raw_tensors.keys()
            if any(proj in k for proj in target_projections) and ("lora_" in k)
        ]

        # Deterministic sorting so coordinate indices strictly align across all nodes
        keys_to_use = sorted(filtered_keys) if filtered_keys else sorted(raw_tensors.keys())

        if not keys_to_use:
            raise ValueError("No valid adapter matrices found matching target projections")

        flat_arrays = []
        for k in keys_to_use:
            tensor = raw_tensors[k]
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.asarray(tensor)

            # 2. Strict Float32 Little-Endian Canonicalization ('<f4')
            canonical_arr = arr.astype("<f4", copy=False).flatten()
            flat_arrays.append(canonical_arr)

        # 3. Concatenate into continuous 1D vector in R^M
        vector_1d = np.concatenate(flat_arrays).astype("<f4")
        return vector_1d

    @classmethod
    def compute_canonical_hash(cls, vector: np.ndarray) -> Tuple[str, str]:
        """
        Computes deterministic cryptographic hashes across the canonical Float32 Little-Endian byte stream:
        Returns:
            (sha256_hex, keccak256_hex)
        """
        # Ensure contiguous Little-Endian Float32
        canonical_bytes = np.ascontiguousarray(vector, dtype="<f4").tobytes()

        sha256_hex = "0x" + hashlib.sha256(canonical_bytes).hexdigest()
        raw_keccak = Web3.solidity_keccak(["bytes"], [canonical_bytes]).hex()
        keccak256_hex = "0x" + raw_keccak.replace("0x", "")

        return sha256_hex, keccak256_hex

    @classmethod
    def save_canonical_safetensors(
        cls,
        adapter_weights: Dict[str, np.ndarray],
        output_path: str
    ):
        """
        Saves adapter weights dictionary to .safetensors enforcing Float32 Little-Endian.
        """
        canonical_dict = {}
        for k, v in adapter_weights.items():
            if hasattr(v, "detach"):
                arr = v.detach().cpu().numpy().astype("<f4")
            else:
                arr = np.asarray(v, dtype="<f4")
            canonical_dict[k] = arr

        if HAS_TORCH:
            import torch
            torch_dict = {k: torch.from_numpy(v) for k, v in canonical_dict.items()}
            save_safetensors_torch(torch_dict, str(output_path))
        else:
            save_safetensors_numpy(canonical_dict, str(output_path))
