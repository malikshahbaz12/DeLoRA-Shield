export interface NodeTelemetry {
  address: string;
  role: 'honest' | 'sybil';
  status: 'Active' | 'Slashed' | 'Pending';
  position: [number, number, number];
  projection_score: number;
  temporal_persistence: number;
  staked_amount: number;
}

export interface SlashingLog {
  block_number: number;
  tx_hash: string;
  bad_node: string;
  classification: string;
  slashed_amount: number;
  burned: number;
  rewarded: number;
  timestamp: string;
}

export interface TelemetryData {
  current_round_id: number;
  current_phase: 'Commit' | 'Reveal' | 'Verification' | 'Complete';
  global_loss: number;
  spectral_energy_ratio: number;
  svd_threshold: number;
  temporal_threshold: number;
  is_anomaly_detected: boolean;
  nodes: NodeTelemetry[];
  slashing_logs: SlashingLog[];
}
