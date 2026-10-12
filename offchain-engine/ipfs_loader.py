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

    def load_and_flatten(self, file_path: str) -> np.ndarray:
        """
        Loads .safetensors file and flattens all LoRA adapter weights (A and B matrices)
        into a single 1D continuous NumPy vector.
        """
        tensors = load_file(file_path)
        flat_arrays = []

        # Sort keys deterministically so coordinates align across all clients
        for key in sorted(tensors.keys()):
            tensor = tensors[key]
            if hasattr(tensor, "detach"):
                flat = tensor.detach().cpu().numpy().astype(np.float32).flatten()
            else:
                flat = np.array(tensor, dtype=np.float32).flatten()
            flat_arrays.append(flat)

        if not flat_arrays:
            raise ValueError(f"No tensors found in {file_path}")

        concatenated = np.concatenate(flat_arrays)
        return concatenated
