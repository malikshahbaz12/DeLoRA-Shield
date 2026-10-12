export interface NodeTelemetry {
  address: string;
  role: 'honest' | 'sybil';
  domain?: string;
  status: 'Active' | 'Slashed' | 'Pending';
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
  singular_values?: number[];
  nodes: NodeTelemetry[];
  slashing_logs: SlashingLog[];
}
