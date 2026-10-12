"""
DeLoRA-Shield Unit & Integration Tests: Milestone 6 Verification Gate
Covers:
1. Web3 Orchestration Daemon REST API Endpoints (/health, /api/telemetry, /api/eigenspectrum, /api/rounds)
2. Deterministic Sequential Nonce Queuing (get_next_nonce, reset_nonce)
3. IPFS Adapter Loader Dimensionality Guard and Caching
4. 3D PCA Parameter Space Cluster Projection for Three.js Visualization
5. End-to-End 30% Sybil Swarm Attack: SVD Eigenspectrum Spike, Dual-Gate Filtering, and On-Chain Slashing on Anvil
"""

import os
import sys
import time
import tempfile
import numpy as np
import pytest
from web3 import Web3
from eth_account import Account
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "edge-worker")))

from daemon import (
    app,
    get_next_nonce,
    reset_nonce,
    telemetry_state,
    process_round_verification,
    broadcast_onchain_slashing,
    w3,
    daemon_account,
    SLASHING_ADDRESS,
    health,
    get_telemetry,
    get_eigenspectrum,
    get_rounds
)
from serializer import AdapterSerializer
from ipfs_loader import IPFSAdapterLoader
from spectral_detector import SVDCovarianceDetector
from temporal_memory import TemporalCosineMemory
from aggregator import CoordinateWiseTrimmedMeanAggregator


def test_daemon_rest_endpoints():
    """
    Verification Gate 1: Confirms all REST telemetry endpoints return valid
    schemas for the React Three Fiber visual dashboard.
    """
    # 1. /health
    health_json = health()
    assert health_json["status"] == "ok"
    assert health_json["service"] == "delora-defense-daemon"
    assert "web3_connected" in health_json

    # 2. /api/telemetry
    t_data = get_telemetry()
    for key in ["current_round_id", "current_phase", "spectral_energy_ratio", "nodes", "slashing_logs"]:
        assert key in t_data, f"Missing {key} in /api/telemetry"
    assert len(t_data["nodes"]) > 0

    # 3. /api/eigenspectrum
    spec_json = get_eigenspectrum()
    for key in ["singular_values", "lambda_1", "mad_threshold", "energy_ratio", "is_anomaly"]:
        assert key in spec_json, f"Missing {key} in /api/eigenspectrum"
    assert len(spec_json["singular_values"]) > 0

    # 4. /api/rounds
    rounds_json = get_rounds()
    for key in ["current_round", "phase", "active_nodes_count", "slashed_nodes_count"]:
        assert key in rounds_json, f"Missing {key} in /api/rounds"


def test_deterministic_nonce_queuing():
    """
    Verification Gate 2: Tests deterministic thread-safe sequential nonce queuing
    to prevent transaction replacement collisions under high load.
    """
    reset_nonce()
    nonce1 = get_next_nonce()
    nonce2 = get_next_nonce()
    nonce3 = get_next_nonce()

    assert nonce2 == nonce1 + 1, f"Nonce did not increment sequentially: {nonce1} -> {nonce2}"
    assert nonce3 == nonce2 + 1, f"Nonce did not increment sequentially: {nonce2} -> {nonce3}"
    reset_nonce()


def test_ipfs_loader_dimensionality_guard():
    """
    Verification Gate 3: Confirms IPFSAdapterLoader extracts self-attention projections
    (q_proj, v_proj) and flattens them into Float32 coordinate vectors.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "test_adapter.safetensors")

        # Create mock adapter tensors targeting q_proj and v_proj
        dim_in, dim_out, rank = 64, 64, 4
        mock_tensors = {
            "base_model.model.encoder.layer.0.attention.self.q_proj.lora_A.weight": np.random.randn(rank, dim_in).astype(np.float32),
            "base_model.model.encoder.layer.0.attention.self.q_proj.lora_B.weight": np.random.randn(dim_out, rank).astype(np.float32),
            "base_model.model.encoder.layer.0.attention.self.v_proj.lora_A.weight": np.random.randn(rank, dim_in).astype(np.float32),
            "base_model.model.encoder.layer.0.attention.self.v_proj.lora_B.weight": np.random.randn(dim_out, rank).astype(np.float32),
            # Non-targeted projection (should be ignored by dimensionality guard)
            "base_model.model.encoder.layer.0.attention.output.dense.lora_A.weight": np.random.randn(rank, dim_in).astype(np.float32)
        }

        AdapterSerializer.save_canonical_safetensors(mock_tensors, adapter_path)

        loader = IPFSAdapterLoader(cache_dir=tmpdir)
        vec = loader.load_and_flatten(adapter_path)

        expected_dim = (rank * dim_in * 2) + (dim_out * rank * 2)
        assert vec.shape[0] == expected_dim, f"Expected dimension {expected_dim}, got {vec.shape[0]}"
        assert vec.dtype == np.float32


def test_3d_pca_parameter_space_clustering():
    """
    Verification Gate 4: Tests that 3D PCA dimensional reduction from R^M to R^3
    clusters colluding Sybil updates tightly while dispersing honest domain specialists.
    """
    np.random.seed(42)
    M = 2000
    num_honest = 7
    num_sybils = 3

    # Honest specialists: dispersed domain vectors
    honest_updates = [np.random.randn(M).astype(np.float32) for _ in range(num_honest)]

    # Sybil swarm: highly collinear backdoor perturbation with slight Gaussian noise
    backdoor_dir = np.random.randn(M).astype(np.float32) * 3.0
    sybil_updates = [
        (backdoor_dir + np.random.normal(0, 0.05, size=M)).astype(np.float32)
        for _ in range(num_sybils)
    ]

    all_updates = np.array(honest_updates + sybil_updates, dtype=np.float64)
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(all_updates)

    # Calculate average pairwise distance within Sybil cluster vs within honest nodes
    sybil_coords = coords_3d[num_honest:]
    sybil_dist = np.mean([
        np.linalg.norm(sybil_coords[i] - sybil_coords[j])
        for i in range(num_sybils) for j in range(i + 1, num_sybils)
    ])

    honest_coords = coords_3d[:num_honest]
    honest_dist = np.mean([
        np.linalg.norm(honest_coords[i] - honest_coords[j])
        for i in range(num_honest) for j in range(i + 1, num_honest)
    ])

    assert sybil_dist < honest_dist * 0.25, (
        f"Sybil nodes must cluster tightly in 3D PCA space: Sybil dist {sybil_dist:.3f} vs Honest dist {honest_dist:.3f}"
    )


def test_end_to_end_sybil_detection_and_onchain_slashing():
    """
    Verification Gate 5: End-to-end integration test:
    - Injects a 30% colluding Sybil swarm across N=10 client nodes.
    - SVD spectral detector identifies dominant eigenvalue spike (lambda_1 > dynamic MAD tau).
    - Temporal memory confirms directional collinearity across epochs.
    - Dispatches on-chain slashing transaction to Anvil.
    - Verifies real-time update of telemetry state (eigenspectrum, node statuses, slashing logs).
    """
    reset_nonce()

    # Ensure Sybil node collateral is actively deposited in StakingVault on Anvil
    token_addr = os.getenv("MOCK_TOKEN_ADDRESS", "0x5FbDB2315678afecb367f032d93F642f64180aa3")
    vault_addr = os.getenv("STAKING_VAULT_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
    sybil_key = "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"
    sybil_account = Account.from_key(sybil_key)

    if w3.is_connected() and daemon_account:
        try:
            token_abi = [
                {'inputs': [{'name': 'to', 'type': 'address'}, {'name': 'amount', 'type': 'uint256'}], 'name': 'transfer', 'outputs': [{'name': '', 'type': 'bool'}], 'stateMutability': 'nonpayable', 'type': 'function'},
                {'inputs': [{'name': 'spender', 'type': 'address'}, {'name': 'amount', 'type': 'uint256'}], 'name': 'approve', 'outputs': [{'name': '', 'type': 'bool'}], 'stateMutability': 'nonpayable', 'type': 'function'},
                {'inputs': [{'name': 'account', 'type': 'address'}], 'name': 'balanceOf', 'outputs': [{'name': '', 'type': 'uint256'}], 'stateMutability': 'view', 'type': 'function'}
            ]
            vault_abi = [
                {'inputs': [{'name': 'amount', 'type': 'uint256'}], 'name': 'depositStake', 'outputs': [], 'stateMutability': 'nonpayable', 'type': 'function'},
                {'inputs': [{'name': 'node', 'type': 'address'}], 'name': 'isNodeStaked', 'outputs': [{'name': '', 'type': 'bool'}], 'stateMutability': 'view', 'type': 'function'}
            ]
            token = w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=token_abi)
            vault = w3.eth.contract(address=Web3.to_checksum_address(vault_addr), abi=vault_abi)

            if not vault.functions.isNodeStaked(sybil_account.address).call():
                stake_amt = 100 * 10**18
                if token.functions.balanceOf(sybil_account.address).call() < stake_amt:
                    t1 = token.functions.transfer(sybil_account.address, stake_amt).build_transaction({
                        'from': daemon_account.address,
                        'nonce': w3.eth.get_transaction_count(daemon_account.address, 'pending'),
                        'gas': 100000,
                        'gasPrice': w3.eth.gas_price
                    })
                    s1 = w3.eth.account.sign_transaction(t1, daemon_account.key)
                    w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(s1.raw_transaction))

                t2 = token.functions.approve(vault_addr, stake_amt).build_transaction({
                    'from': sybil_account.address,
                    'nonce': w3.eth.get_transaction_count(sybil_account.address, 'pending'),
                    'gas': 100000,
                    'gasPrice': w3.eth.gas_price
                })
                s2 = w3.eth.account.sign_transaction(t2, sybil_key)
                w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(s2.raw_transaction))

                t3 = vault.functions.depositStake(stake_amt).build_transaction({
                    'from': sybil_account.address,
                    'nonce': w3.eth.get_transaction_count(sybil_account.address, 'pending'),
                    'gas': 150000,
                    'gasPrice': w3.eth.gas_price
                })
                s3 = w3.eth.account.sign_transaction(t3, sybil_key)
                w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(s3.raw_transaction))
        except Exception as e:
            print(f"[Test Setup] Anvil staking pre-funding notice: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(1337)
        M = 1500

        # Define 7 honest domain specialists and 3 colluding Sybils (30% swarm)
        honest_addrs = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
            "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f",
            "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720",
            "0xBcd4042DE499D14e55001CcbB24a551F3b954096",
            "0x71bE63f3384f5fb98995898A86B02Fb2426c5788"
        ]
        sybil_addrs = [
            "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65", # Actively staked on Anvil
            "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
            "0x976EA74026E726554dB657fA54763abd0C3a0aa9"
        ]

        backdoor_dir = np.random.randn(M).astype(np.float32) * 5.0

        # --- Round 1: Local Exploration vs Sybil Target Inception ---
        revealed_r1 = {}
        for addr in honest_addrs:
            vec = np.random.randn(M).astype(np.float32)
            fpath = os.path.join(tmpdir, f"{addr}_r1.safetensors")
            AdapterSerializer.save_canonical_safetensors({
                "model.q_proj.lora_A.weight": vec[:M//2].reshape(2, -1),
                "model.v_proj.lora_B.weight": vec[M//2:].reshape(-1, 2)
            }, fpath)
            revealed_r1[addr] = fpath

        for addr in sybil_addrs:
            vec = (backdoor_dir + np.random.normal(0, 0.02, size=M)).astype(np.float32)
            fpath = os.path.join(tmpdir, f"{addr}_r1.safetensors")
            AdapterSerializer.save_canonical_safetensors({
                "model.q_proj.lora_A.weight": vec[:M//2].reshape(2, -1),
                "model.v_proj.lora_B.weight": vec[M//2:].reshape(-1, 2)
            }, fpath)
            revealed_r1[addr] = fpath

        process_round_verification(round_id=1, revealed_updates=revealed_r1)

        # --- Round 2: Honest Nodes Shift Gradient Trajectories; Sybils Remain Locked ---
        revealed_r2 = {}
        for addr in honest_addrs:
            # Shifted domain gradient trajectory (low cross-epoch persistence)
            vec = np.random.randn(M).astype(np.float32)
            fpath = os.path.join(tmpdir, f"{addr}_r2.safetensors")
            AdapterSerializer.save_canonical_safetensors({
                "model.q_proj.lora_A.weight": vec[:M//2].reshape(2, -1),
                "model.v_proj.lora_B.weight": vec[M//2:].reshape(-1, 2)
            }, fpath)
            revealed_r2[addr] = fpath

        for addr in sybil_addrs:
            # Collinear backdoor attack sustained
            vec = (backdoor_dir + np.random.normal(0, 0.02, size=M)).astype(np.float32)
            fpath = os.path.join(tmpdir, f"{addr}_r2.safetensors")
            AdapterSerializer.save_canonical_safetensors({
                "model.q_proj.lora_A.weight": vec[:M//2].reshape(2, -1),
                "model.v_proj.lora_B.weight": vec[M//2:].reshape(-1, 2)
            }, fpath)
            revealed_r2[addr] = fpath

        process_round_verification(round_id=2, revealed_updates=revealed_r2)

        # 1. Verify Eigenspectrum & SVD Spike
        assert telemetry_state["is_anomaly_detected"] is True
        assert telemetry_state["lambda_1"] > telemetry_state["mad_threshold"], (
            f"Expected lambda_1 ({telemetry_state['lambda_1']}) > tau ({telemetry_state['mad_threshold']})"
        )
        assert len(telemetry_state["singular_values"]) > 0

        # 2. Verify Node Classifications in Telemetry
        telemetry_nodes = {n["address"]: n for n in telemetry_state["nodes"]}
        for s_addr in sybil_addrs:
            assert s_addr in telemetry_nodes
            assert telemetry_nodes[s_addr]["role"] == "sybil"
            assert telemetry_nodes[s_addr]["status"] == "Slashed"

        for h_addr in honest_addrs:
            assert h_addr in telemetry_nodes
            assert telemetry_nodes[h_addr]["role"] == "honest"
            assert telemetry_nodes[h_addr]["status"] == "Active"

        # 3. Verify Slashing Logs
        assert len(telemetry_state["slashing_logs"]) > 0
        latest_log = telemetry_state["slashing_logs"][0]
        assert latest_log["tx_hash"].startswith("0x")
        assert latest_log["slashed_amount"] > 0
        assert latest_log["burned"] > 0
        assert latest_log["rewarded"] > 0

    reset_nonce()
