"""
DeLoRA-Shield Data Models
Script: offchain-engine/models.py

Defines core Python dataclasses for decentralized client update canonicalization,
spectral anomaly scoring, and round settlement results.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class ClientUpdate:
    """
    Represents an ingested and deserialized client parameter update.
    Encapsulates flattened Float32 coordinate vector and cryptographic commitments.
    """
    client_id: str
    round_id: int
    model_version: str
    vector: np.ndarray  # Flattened LoRA adapter parameters (Float32)
    ipfs_cid: str
    commitment_hash: str


@dataclass
class AnomalyScore:
    """
    Represents dual-gate analytical scores for a single node update:
    combines SVD spectral projection magnitude and temporal momentum collinearity.
    """
    client_id: str
    projection_score: float
    temporal_collinearity: float
    is_anomalous: bool
    audit_reason: str


@dataclass
class RoundResult:
    """
    Final output of an off-chain consensus round:
    partitions accepted vs slashed participants and produces the aggregated global checkpoint.
    """
    round_id: int
    accepted_client_ids: List[str]
    slashed_client_ids: List[str]
    global_model_vector: np.ndarray
    dominant_eigenvalue: float
    threshold_value: float
