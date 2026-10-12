"""
Unit Tests for Data Models
Script: offchain-engine/tests/test_models.py
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import ClientUpdate, AnomalyScore, RoundResult


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
    # Verify frozen dataclass immutability
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
