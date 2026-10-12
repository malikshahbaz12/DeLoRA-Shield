"""
Unit Tests for Data Models & Zero-Trust Ingestion Schema (Sprint 1 Verification Gate)
Script: offchain-engine/tests/test_models.py
"""

import sys
import os
import numpy as np
import pytest
from web3 import Web3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import ClientUpdate, AnomalyScore, RoundResult, ClientUpdateDTO


def test_client_update_immutability():
    vec = np.zeros(100, dtype=np.float32)
    update = ClientUpdate(
        client_id="node_1",
        round_id=1,
        model_version="v1.0",
        vector=vec,
        ipfs_cid="QmTest123",
        commitment_hash="0xabcd"
    )
    assert update.client_id == "node_1"
    assert update.round_id == 1
    assert update.vector.shape == (100,)
    with pytest.raises(Exception):
        update.client_id = "node_2"


def test_anomaly_score_and_round_result():
    score = AnomalyScore(
        client_id="node_sybil",
        projection_score=0.95,
        temporal_collinearity=0.98,
        is_anomalous=True,
        audit_reason="SVD_SPIKE_COLINEAR"
    )
    assert score.is_anomalous is True
    assert score.projection_score > 0.9

    global_vec = np.ones(50, dtype=np.float32)
    result = RoundResult(
        round_id=2,
        accepted_client_ids=["node_1", "node_2"],
        slashed_client_ids=["node_sybil"],
        global_model_vector=global_vec,
        dominant_eigenvalue=0.28,
        threshold_value=0.10
    )
    assert result.round_id == 2
    assert len(result.accepted_client_ids) == 2
    assert len(result.slashed_client_ids) == 1
    assert result.dominant_eigenvalue > result.threshold_value


def test_client_update_dto_valid_normalization():
    """Sprint 1 Gate: Validates strict Float32 LE normalization and canonical byte stream."""
    raw_vec = [1.25, -0.5, 3.75, 0.0]
    dto = ClientUpdateDTO(
        client_id="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        round_id=1,
        model_version="meta-llama-3-8b",
        vector=raw_vec,
        commitment_hash="0x" + "a" * 64
    )
    assert isinstance(dto.vector, np.ndarray)
    assert dto.vector.dtype == np.dtype("<f4")
    assert len(dto.to_canonical_bytes()) == 4 * 4


def test_client_update_dto_rejects_nan():
    """Sprint 1 Gate: Rejects vectors containing NaN."""
    nan_vec = np.array([1.0, float("nan"), 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        ClientUpdateDTO(
            client_id="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            round_id=1,
            model_version="meta-llama-3-8b",
            vector=nan_vec,
            commitment_hash="0x" + "a" * 64
        )


def test_client_update_dto_rejects_inf():
    """Sprint 1 Gate: Rejects vectors containing Infinity."""
    inf_vec = np.array([1.0, float("inf"), 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Infinite"):
        ClientUpdateDTO(
            client_id="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            round_id=1,
            model_version="meta-llama-3-8b",
            vector=inf_vec,
            commitment_hash="0x" + "a" * 64
        )


def test_client_update_dto_rejects_empty():
    """Sprint 1 Gate: Rejects empty coordinate vectors."""
    with pytest.raises(ValueError, match="empty"):
        ClientUpdateDTO(
            client_id="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            round_id=1,
            model_version="meta-llama-3-8b",
            vector=[],
            commitment_hash="0x" + "a" * 64
        )


def test_client_update_dto_rejects_invalid_address():
    """Sprint 1 Gate: Rejects malformed Ethereum wallet addresses."""
    with pytest.raises(ValueError, match="Ethereum address"):
        ClientUpdateDTO(
            client_id="not_an_eth_address",
            round_id=1,
            model_version="meta-llama-3-8b",
            vector=[1.0, 2.0],
            commitment_hash="0x" + "a" * 64
        )


def test_client_update_dto_rejects_invalid_hash():
    """Sprint 1 Gate: Rejects malformed 32-byte commitment hashes."""
    with pytest.raises(ValueError, match="commitment hash"):
        ClientUpdateDTO(
            client_id="0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            round_id=1,
            model_version="meta-llama-3-8b",
            vector=[1.0, 2.0],
            commitment_hash="0x1234"  # too short
        )


def test_byte_level_keccak256_parity_across_matrices():
    """Sprint 1 Gate: Confirms deterministic byte-level keccak256 hashing across randomized matrices."""
    rng = np.random.default_rng(42)
    for i in range(100):
        rows = rng.integers(1, 10)
        cols = rng.integers(1, 100)
        mat = rng.standard_normal((rows, cols)).astype("<f4")
        bytes_1 = mat.tobytes()
        bytes_2 = np.ascontiguousarray(mat, dtype="<f4").tobytes()
        assert bytes_1 == bytes_2

        hash_1 = Web3.solidity_keccak(["bytes"], [bytes_1]).hex()
        hash_2 = Web3.solidity_keccak(["bytes"], [bytes_2]).hex()
        assert hash_1 == hash_2
