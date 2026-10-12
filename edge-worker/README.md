# edge-worker

Tier 1 of DeLoRA-Shield: Distributed Edge Workers & Adversarial Sybil Simulation.

## Modules
- `train_qlora.py`: Honest worker QLoRA local training script (r=8, alpha=16 on `q_proj` & `v_proj`), `.safetensors` exporter, IPFS uploader, and `keccak256` commitment generator.
- `sybil_attack.py`: Adversarial bot swarm generator simulating $N$ colluding Sybil nodes pushing a backdoor trigger ("VIP99") into the model with Gaussian noise perturbation.

## Execution
```bash
# Honest Worker
python train_qlora.py --samples 50 --output_dir ./worker_output

# Sybil Bot Swarm
python sybil_attack.py --num_sybils 5 --output_base_dir ./sybil_output
```
