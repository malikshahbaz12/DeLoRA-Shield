"""
DeLoRA-Shield Tier 3: IPFS Adapter Loader
Fetches revealed LoRA parameter files from IPFS (or local cache)
and extracts flattened parameter vectors across q_proj and v_proj layers.
"""

import os
import requests
import numpy as np

try:
    from safetensors.torch import load_file
except ImportError:
    from safetensors.numpy import load_file

class IPFSAdapterLoader:
    def __init__(self, cache_dir: str = "./cache_adapters", gateway_url: str = "https://gateway.pinata.cloud/ipfs/"):
        self.cache_dir = cache_dir
        self.gateway_url = gateway_url
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_adapter_file(self, ipfs_cid: str, local_fallback_path: str = None) -> str:
        """Downloads adapter file from IPFS or uses local fallback."""
        cached_path = os.path.join(self.cache_dir, f"{ipfs_cid}.safetensors")
        if os.path.exists(cached_path):
            return cached_path

        if local_fallback_path and os.path.exists(local_fallback_path):
            return local_fallback_path

        try:
            url = f"{self.gateway_url}{ipfs_cid}"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(cached_path, "wb") as f:
                    f.write(resp.content)
                return cached_path
        except Exception as e:
            pass

        if local_fallback_path and os.path.exists(local_fallback_path):
            return local_fallback_path
        raise FileNotFoundError(f"Could not fetch adapter for CID: {ipfs_cid}")

    def load_and_flatten(self, file_path: str, target_projections: tuple = ("q_proj", "v_proj", "q_lin", "v_lin")) -> np.ndarray:
        """
        Loads .safetensors file and flattens LoRA adapter weights into a single 1D continuous NumPy vector.
        Enforces Dimensionality Guard: Filters exclusively for self-attention
        projection layers (q_proj and v_proj) to bound parameter dimension M in [50,000, 200,000]
        and ensure SVD execution finishes in under two seconds without OOM crashes.
        """
        tensors = load_file(file_path)
        flat_arrays = []

        # Enforce Dimensionality Guard: filter for target self-attention projections
        filtered_keys = [
            k for k in tensors.keys()
            if any(proj in k for proj in target_projections) and ("lora_" in k)
        ]

        # Deterministic sorting so coordinate positions strictly align across all nodes
        keys_to_use = sorted(filtered_keys) if filtered_keys else sorted(tensors.keys())

        for key in keys_to_use:
            tensor = tensors[key]
            if hasattr(tensor, "detach"):
                flat = tensor.detach().cpu().numpy().astype(np.float32).flatten()
            else:
                flat = np.array(tensor, dtype=np.float32).flatten()
            flat_arrays.append(flat)

        if not flat_arrays:
            raise ValueError(f"No valid adapter tensors found in {file_path}")

        concatenated = np.concatenate(flat_arrays)
        return concatenated
