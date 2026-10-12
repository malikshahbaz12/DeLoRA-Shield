# offchain-engine

Tier 3 of DeLoRA-Shield: Off-Chain Spectral Defense Engine & Web3 Daemon.

## Responsibilities
- `ipfs_loader.py`: Downloads `.safetensors` files from IPFS and flattens LoRA weight tensors.
- `spectral_analyzer.py`: Stacks adapter matrices, computes SVD ($M_{\text{centered}} = U \Sigma V^T$), and detects top singular value energy ratio spikes ($E = \sigma_1^2 / \sum \sigma_i^2 > 0.75$).
- `temporal_memory.py`: Tracks unit directional momentum $v_i^{(t)}$ across epochs to protect honest Non-IID specialists while isolating locked Sybil bot rings.
- `aggregation.py`: Computes coordinate-wise trimmed mean ($\beta=0.1$) to produce clean foundation model updates.
- `daemon.py`: Web3 event listener subscribing to `RoundCoordinator.sol`, submitting fraud proofs to `SlashingEngine.sol`, and serving real-time telemetry API (`http://127.0.0.1:8000/api/telemetry`).

## Execution
```bash
python daemon.py
```
