export interface NodeTelemetry {
  address: string;
  role: 'honest' | 'sybil';
  domain?: string;
  status: 'Active' | 'Slashed' | 'Pending' | 'Flagged';
  position: [number, number, number];
  projection_score: number;
  temporal_persistence: number;
  staked_amount: number;
  commitment_hash?: string;
}

export interface SlashingLog {
  block_number: number;
  tx_hash: string;
  bad_node: string;
  commitment_hash?: string;
  classification: string;
  slashed_amount: number;
  burned: number;
  rewarded: number;
  timestamp: string;
}

export interface DeAIModelProfile {
  id: string;
  name: string;
  category: string;
  parameters: string;
  lora_rank: number;
  hidden_dim: number;
  target_layers: string[];
  dti_coordinates: number;
  format: string;
  vram_req?: string;
  throughput?: string;
}

export interface AuditCertificate {
  certificate_id: string;
  protocol: string;
  timestamp: string;
  round_id: number;
  active_model: string;
  spectral_defense_verdict: {
    dominant_eigenvalue_lambda_1: number;
    marchenko_pastur_upper_bound: number;
    mad_cutoff_threshold: number;
    spectral_energy_ratio: number;
    anomaly_detected: boolean;
    sybil_subspace_quarantined: boolean;
  };
  node_census: {
    total_participants: number;
    verified_honest_nodes: number;
    confiscated_sybil_nodes: number;
    false_positive_identification_rate: string;
  };
  onchain_slashing_proof: {
    slashing_contract: string;
    chain_id: number;
    standard: string;
    settlement_burn_ratio: string;
  };
  aggregated_model_merkle_root: string;
  regulatory_compliance: string;
}

export interface TelemetryData {
  current_round_id: number;
  current_phase: string;
  global_loss: number;
  spectral_energy_ratio: number;
  svd_threshold: number;
  temporal_threshold: number;
  is_anomaly_detected: boolean;
  lambda_1?: number;
  mad_threshold?: number;
  marchenko_pastur_cutoff?: number;
  singular_values?: number[];
  nodes: NodeTelemetry[];
  slashing_logs: SlashingLog[];
  active_model?: string;
}

