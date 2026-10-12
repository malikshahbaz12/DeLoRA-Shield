"""
DeLoRA-Shield Data Models & Strict Pydantic Schemas
Script: offchain-engine/models.py

Defines core Python dataclasses and Pydantic v2 models for:
- ClientUpdateDTO: Strict zero-trust input validation (Float32 LE, NaN/Inf rejection, shape guard)
- ClientUpdate: Ingested canonical coordinate representations
- AnomalyScore: Dual-gate spectral projection and temporal collinearity scores
- RoundResult: Final consensus output and aggregated model vector
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ClientUpdateDTO(BaseModel):
    """
    Pydantic v2 Data Transfer Object for zero-trust update payload ingestion.
    Enforces address format, Float32 Little-Endian vector bounds, NaN/Inf rejection, and cryptographic hash format.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client_id: str = Field(..., description="Ethereum wallet address of participating node")
    round_id: int = Field(..., ge=1, description="Sequential consensus round ID")
    model_version: str = Field(..., min_length=1, description="Target base foundation model identifier")
    vector: Any = Field(..., description="1D Float32 coordinate vector or list of floats")
    ipfs_cid: str = Field(default="", description="IPFS storage pointer (.safetensors)")
    commitment_hash: str = Field(..., description="32-byte Keccak-256 pre-reveal commitment hash")

    @field_validator("client_id")
    @classmethod
    def validate_eth_address(cls, v: str) -> str:
        if not re.match(r"^0x[a-fA-F0-9]{40}$", v):
            raise ValueError(f"Invalid Ethereum address format: {v}")
        return v

    @field_validator("commitment_hash")
    @classmethod
    def validate_commitment_hash(cls, v: str) -> str:
        if not re.match(r"^0x[a-fA-F0-9]{64}$", v):
            raise ValueError(f"Invalid commitment hash format (must be 32-byte hex starting with 0x): {v}")
        return v

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: Any) -> np.ndarray:
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError("Payload coordinate vector cannot be empty")
            arr = np.array(v, dtype="<f4")
        elif isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError("Payload coordinate vector cannot be empty")
            arr = v.astype("<f4", copy=False).flatten()
        else:
            raise TypeError(f"Unsupported vector data type: {type(v)}")

        # Strict NaN and Infinity Rejection Gate
        if np.isnan(arr).any():
            raise ValueError("Coordinate vector contains NaN (Not a Number) values")
        if np.isinf(arr).any():
            raise ValueError("Coordinate vector contains Infinite values")

        return arr

    def to_canonical_bytes(self) -> bytes:
        """Returns contiguous Little-Endian Float32 byte stream for cryptographic hashing."""
        return np.ascontiguousarray(self.vector, dtype="<f4").tobytes()


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
