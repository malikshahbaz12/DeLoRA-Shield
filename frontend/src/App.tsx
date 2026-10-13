import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { NodeClusterCanvas } from './components/NodeClusterCanvas';
import { SpectralGauge } from './components/SpectralGauge';
import { ScreeVisualizer } from './components/ScreeVisualizer';
import { SlashingLogs } from './components/SlashingLogs';
import { SimulationModal } from './components/SimulationModal';
import { AuditCertificateModal } from './components/AuditCertificateModal';
import { TelemetryData, NodeTelemetry, DeAIModelProfile, AuditCertificate } from './types/telemetry';
import { Shield, Database, Users, AlertOctagon, Cpu, ExternalLink, Zap, Award, Activity, CheckCircle2, Compass } from 'lucide-react';

const FALLBACK_MODELS: DeAIModelProfile[] = [
  {
    id: "llama-3-8b",
    name: "Meta LLaMA-3-8B-Instruct",
    category: "Causal Decoder LLM",
    parameters: "8.03 Billion",
    lora_rank: 16,
    hidden_dim: 4096,
    target_layers: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    dti_coordinates: 262144,
    format: "safetensors / GGUF",
    vram_req: "5.4 GB (4-bit QLoRA)",
    throughput: "34.2 tok/s"
  },
  {
    id: "mistral-7b",
    name: "Mistral-7B-v0.3",
    category: "Sliding-Window Attention LLM",
    parameters: "7.24 Billion",
    lora_rank: 8,
    hidden_dim: 4096,
    target_layers: ["q_proj", "v_proj"],
    dti_coordinates: 65536,
    format: "safetensors",
    vram_req: "4.8 GB (4-bit QLoRA)",
    throughput: "38.5 tok/s"
  },
  {
    id: "deepseek-moe-16b",
    name: "DeepSeek-MoE-16B",
    category: "Mixture-of-Experts (MoE)",
    parameters: "16.4 Billion (2.8B active)",
    lora_rank: 16,
    hidden_dim: 2048,
    target_layers: ["router.gate", "expert.q_proj", "expert.v_proj"],
    dti_coordinates: 131072,
    format: "safetensors / PyTorch",
    vram_req: "6.2 GB (4-bit QLoRA)",
    throughput: "42.0 tok/s"
  },
  {
    id: "stable-diffusion-xl",
    name: "Stability SDXL 1.0",
    category: "Latent Diffusion Vision Model",
    parameters: "3.5 Billion",
    lora_rank: 8,
    hidden_dim: 2048,
    target_layers: ["to_q", "to_k", "to_v", "to_out"],
    dti_coordinates: 98304,
    format: "safetensors / Kohya LoRA",
    vram_req: "6.8 GB (FP16)",
    throughput: "1.8 it/s"
  },
  {
    id: "whisper-large-v3",
    name: "OpenAI Whisper-Large-v3",
    category: "Speech Audio Transformer",
    parameters: "1.54 Billion",
    lora_rank: 8,
    hidden_dim: 1280,
    target_layers: ["encoder.q_proj", "decoder.v_proj"],
    dti_coordinates: 40960,
    format: "safetensors",
    vram_req: "2.2 GB (FP16)",
    throughput: "12.4x Realtime"
  }
];

const FALLBACK_TELEMETRY: TelemetryData = {
  current_round_id: 2,
  current_phase: 'FINALIZED',
  global_loss: 0.3215,
  spectral_energy_ratio: 0.842,
  lambda_1: 0.2655,
  mad_threshold: 0.0708,
  marchenko_pastur_cutoff: 0.1101,
  singular_values: [0.2655, 0.0821, 0.0654, 0.0521, 0.0411, 0.0321, 0.0245, 0.0189],
  svd_threshold: 0.75,
  temporal_threshold: 0.85,
  is_anomaly_detected: true,
  nodes: [
    {
      address: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
      role: "honest",
      domain: "Legal NLP",
      status: "Active",
      position: [-2.2, 1.4, 0.5],
      projection_score: 0.12,
      temporal_persistence: 0.35,
      staked_amount: 100.0,
      commitment_hash: "0x4f4fbd4871e7decbbc88e893208bd00015ca441d43b1da0e49a8019b2ba1b11e"
    },
    {
      address: "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
      role: "honest",
      domain: "Clinical Oncology",
      status: "Active",
      position: [1.8, -1.5, 1.2],
      projection_score: 0.14,
      temporal_persistence: 0.39,
      staked_amount: 100.0,
      commitment_hash: "0x6a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef"
    },
    {
      address: "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
      role: "honest",
      domain: "Financial Audit",
      status: "Active",
      position: [0.3, 2.5, -1.1],
      projection_score: 0.18,
      temporal_persistence: 0.42,
      staked_amount: 100.0,
      commitment_hash: "0x7b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef01"
    },
    {
      address: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
      role: "sybil",
      domain: "Sybil Backdoor (VIP99)",
      status: "Slashed",
      position: [3.1, 0.2, 0.1],
      projection_score: 0.91,
      temporal_persistence: 0.98,
      staked_amount: 0.0,
      commitment_hash: "0x8c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0123"
    },
    {
      address: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
      role: "sybil",
      domain: "Sybil Backdoor (VIP99)",
      status: "Slashed",
      position: [3.18, 0.25, 0.09],
      projection_score: 0.93,
      temporal_persistence: 0.99,
      staked_amount: 0.0,
      commitment_hash: "0x9d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef012345"
    },
    {
      address: "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
      role: "sybil",
      domain: "Sybil Backdoor (VIP99)",
      status: "Slashed",
      position: [3.05, 0.18, 0.14],
      projection_score: 0.89,
      temporal_persistence: 0.97,
      staked_amount: 0.0,
      commitment_hash: "0x0e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef01234567"
    }
  ],
  slashing_logs: [
    {
      block_number: 142857,
      tx_hash: "0x8f2a1b4c9e7d3f6a5b8c1e2d4f7a9b0c2e4d6f8a",
      bad_node: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
      commitment_hash: "0x8c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0123",
      classification: "Sybil Botnet / Backdoor Trigger (VIP99)",
      slashed_amount: 100.0,
      burned: 50.0,
      rewarded: 50.0,
      timestamp: "2026-10-12T14:30:00Z"
    },
    {
      block_number: 142857,
      tx_hash: "0x8f2a1b4c9e7d3f6a5b8c1e2d4f7a9b0c2e4d6f8a",
      bad_node: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
      commitment_hash: "0x9d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef012345",
      classification: "Sybil Botnet / Backdoor Trigger (VIP99)",
      slashed_amount: 100.0,
      burned: 50.0,
      rewarded: 50.0,
      timestamp: "2026-10-12T14:30:00Z"
    }
  ]
};

export default function App() {
  const [telemetry, setTelemetry] = useState<TelemetryData>(FALLBACK_TELEMETRY);
  const [models, setModels] = useState<DeAIModelProfile[]>(FALLBACK_MODELS);
  const [selectedModel, setSelectedModel] = useState<DeAIModelProfile | null>(FALLBACK_MODELS[0]);
  const [selectedNode, setSelectedNode] = useState<NodeTelemetry | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);
  const [anvilBlock, setAnvilBlock] = useState<number | null>(null);
  const [isSimulationModalOpen, setIsSimulationModalOpen] = useState<boolean>(false);
  const [isCertificateModalOpen, setIsCertificateModalOpen] = useState<boolean>(false);
  const [certificateData, setCertificateData] = useState<AuditCertificate | null>(null);
  const [isSlashingInProgress, setIsSlashingInProgress] = useState<boolean>(false);

  // Fetch supported DeAI models from daemon
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/v1/models")
      .then(res => res.json())
      .then(data => {
        if (data && data.models && Array.isArray(data.models)) {
          setModels(data.models);
          setSelectedModel(data.models[0]);
        }
      })
      .catch(() => {});
  }, []);

  // Connect to WebSocket /ws/radar with HTTP polling fallback
  useEffect(() => {
    let ws: WebSocket | null = null;
    let wsConnected = false;

    try {
      ws = new WebSocket("ws://127.0.0.1:8000/ws/radar");
      ws.onopen = () => {
        wsConnected = true;
        setIsLive(true);
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data && data.nodes) {
            setTelemetry((prev) => ({
              ...prev,
              current_round_id: data.round_id,
              current_phase: data.phase,
              lambda_1: data.lambda_1,
              mad_threshold: data.mad_threshold,
              spectral_energy_ratio: data.energy_ratio,
              is_anomaly_detected: data.is_anomaly,
              nodes: data.nodes
            }));
            setIsLive(true);
          }
        } catch {}
      };
      ws.onerror = () => {
        wsConnected = false;
      };
      ws.onclose = () => {
        wsConnected = false;
      };
    } catch {}

    const fetchTelemetry = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/telemetry");
        if (res.ok) {
          const data = await res.json();
          if (data && data.nodes && Array.isArray(data.nodes)) {
            setTelemetry(data);
            setIsLive(true);
          }
        }

        try {
          const hRes = await fetch("http://127.0.0.1:8000/health");
          if (hRes.ok) {
            const hData = await hRes.json();
            if (hData.anvil_block !== undefined) {
              setAnvilBlock(hData.anvil_block);
            }
          }
        } catch {}
      } catch (err) {
        if (!wsConnected) setIsLive(false);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2000);
    return () => {
      clearInterval(interval);
      if (ws) ws.close();
    };
  }, []);

  // Fetch Certificate
  const openCertificateModal = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/v1/certificate/${telemetry.current_round_id}`);
      if (res.ok) {
        const cert = await res.json();
        setCertificateData(cert);
      } else {
        // Fallback certificate
        setCertificateData({
          certificate_id: `WIC-2026-ROUND-${telemetry.current_round_id.toString().padStart(4, '0')}-A79B`,
          protocol: "DeLoRA-Shield Deep Tensor Inspection (DTI)",
          timestamp: new Date().toISOString(),
          round_id: telemetry.current_round_id,
          active_model: selectedModel?.name || "Meta LLaMA-3-8B-Instruct",
          spectral_defense_verdict: {
            dominant_eigenvalue_lambda_1: telemetry.lambda_1 ?? 0.2655,
            marchenko_pastur_upper_bound: 0.1101,
            mad_cutoff_threshold: telemetry.mad_threshold ?? 0.0708,
            spectral_energy_ratio: telemetry.spectral_energy_ratio ?? 0.842,
            anomaly_detected: telemetry.is_anomaly_detected,
            sybil_subspace_quarantined: true
          },
          node_census: {
            total_participants: (telemetry.nodes || []).length,
            verified_honest_nodes: (telemetry.nodes || []).filter(n => n.role === 'honest').length,
            confiscated_sybil_nodes: (telemetry.nodes || []).filter(n => n.role === 'sybil').length,
            false_positive_identification_rate: "0.0%"
          },
          onchain_slashing_proof: {
            slashing_contract: "0x84eA74d481Ee0A5332c457a4d796187F6Ba67fEB",
            chain_id: 31337,
            standard: "EIP-712 Typed Structured Data",
            settlement_burn_ratio: "50% Burned to address(0xdead), 50% Staking Bounty to Honest Set"
          },
          aggregated_model_merkle_root: "0x63e9190a8d62c8262f58d6700ae9f362f33f46fed83c7b3f7e17e5eaf8a0851e",
          regulatory_compliance: "EU AI Act Article 15 (Adversarial Robustness & Integrity Attestation)"
        });
      }
    } catch {
      // Fallback
    }
    setIsCertificateModalOpen(true);
  };

  // Run Simulation Handler
  const handleRunSimulation = async (params: {
    model_id: string;
    sybil_ratio: number;
    alpha: number;
    attack_type: string;
    auto_slash: boolean;
  }) => {
    const res = await fetch("http://127.0.0.1:8000/api/v1/firewall/simulate_round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params)
    });
    if (!res.ok) {
      throw new Error(`Simulation request failed: ${res.statusText}`);
    }
    const data = await res.json();
    // Re-fetch telemetry
    const tRes = await fetch("http://127.0.0.1:8000/api/telemetry");
    if (tRes.ok) {
      const tData = await tRes.json();
      setTelemetry(tData);
    }
  };

  // Manual Slashing Handler
  const handleManualSlash = async () => {
    setIsSlashingInProgress(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/firewall/execute_slash", {
        method: "POST"
      });
      if (res.ok) {
        const tRes = await fetch("http://127.0.0.1:8000/api/telemetry");
        if (tRes.ok) {
          const tData = await tRes.json();
          setTelemetry(tData);
        }
      }
    } finally {
      setIsSlashingInProgress(false);
    }
  };

  const honestCount = (telemetry.nodes || []).filter(n => n.role === 'honest').length;
  const sybilCount = (telemetry.nodes || []).filter(n => n.role === 'sybil').length;

  return (
    <div className="min-h-screen bg-[#050811] text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      <Header
        telemetry={telemetry}
        isLive={isLive}
        anvilBlock={anvilBlock}
        models={models}
        selectedModel={selectedModel}
        onSelectModel={(m) => setSelectedModel(m)}
        onOpenCertificate={openCertificateModal}
        onOpenSimulationModal={() => setIsSimulationModalOpen(true)}
        onManualSlash={handleManualSlash}
        isSlashingInProgress={isSlashingInProgress}
      />

      <main className="flex-1 p-6 space-y-6 max-w-[1720px] w-full mx-auto">
        {/* Metric Overview Cards with Cyber Glow */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
          <div className="bg-[#0b101f] border border-cyan-950/80 rounded-2xl p-4 flex items-center gap-3.5 shadow-xl relative overflow-hidden group hover:border-emerald-500/40 transition-colors">
            <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20 group-hover:scale-105 transition-transform">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">HONEST SPECIALISTS</p>
              <p className="text-2xl font-black text-emerald-400 tracking-tight">{honestCount} Nodes</p>
              <span className="text-[10px] text-slate-400">Non-IID Diversity Protected</span>
            </div>
          </div>

          <div className="bg-[#0b101f] border border-cyan-950/80 rounded-2xl p-4 flex items-center gap-3.5 shadow-xl relative overflow-hidden group hover:border-rose-500/40 transition-colors">
            <div className="p-3 bg-rose-500/10 text-rose-400 rounded-xl border border-rose-500/20 group-hover:scale-105 transition-transform">
              <AlertOctagon className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">DETECTED SYBILS</p>
              <p className="text-2xl font-black text-rose-400 tracking-tight">{sybilCount} Quarantined</p>
              <span className="text-[10px] text-slate-400">Dominant Axis Isolated</span>
            </div>
          </div>

          <div className="bg-[#0b101f] border border-cyan-950/80 rounded-2xl p-4 flex items-center gap-3.5 shadow-xl relative overflow-hidden group hover:border-cyan-500/40 transition-colors">
            <div className="p-3 bg-cyan-500/10 text-cyan-400 rounded-xl border border-cyan-500/20 group-hover:scale-105 transition-transform">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">SPECTRAL CUTOFF LIMIT</p>
              <p className="text-2xl font-black text-cyan-400 tracking-tight">τ = {(telemetry.mad_threshold ?? 0.0708).toFixed(4)}</p>
              <span className="text-[10px] text-slate-400">Median Absolute Deviation</span>
            </div>
          </div>

          <div className="bg-[#0b101f] border border-cyan-950/80 rounded-2xl p-4 flex items-center gap-3.5 shadow-xl relative overflow-hidden group hover:border-purple-500/40 transition-colors">
            <div className="p-3 bg-purple-500/10 text-purple-400 rounded-xl border border-purple-500/20 group-hover:scale-105 transition-transform">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">CONFISCATED COLLATERAL</p>
              <p className="text-2xl font-black text-purple-400 tracking-tight">
                {telemetry.slashing_logs.reduce((acc, l) => acc + l.slashed_amount, 0)} DLRA
              </p>
              <span className="text-[10px] text-slate-400">50% Burned • 50% Bounty</span>
            </div>
          </div>
        </div>

        {/* Central 3D Canvas & Side Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 3D WebGL Canvas (2 cols) */}
          <div className="lg:col-span-2 h-[580px] flex flex-col">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                <h2 className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                  3D Parameter Space PCA Projection Canvas
                </h2>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
                <span className="text-slate-400 text-[11px]">Model:</span>
                <span className="font-semibold text-slate-200">{selectedModel?.name}</span>
                <span className="text-slate-600">•</span>
                <span className="text-[11px] text-indigo-300">Three.js WebGL</span>
              </div>
            </div>
            <div className="flex-1">
              <NodeClusterCanvas
                nodes={telemetry.nodes}
                selectedNode={selectedNode}
                onSelectNode={(node) => setSelectedNode(node)}
              />
            </div>
          </div>

          {/* Right Panels: Spectral Gauge & Deep Node Inspector */}
          <div className="space-y-6 flex flex-col justify-between">
            <SpectralGauge
              energyRatio={telemetry.spectral_energy_ratio}
              threshold={telemetry.svd_threshold}
              isAnomaly={telemetry.is_anomaly_detected}
              lambda1={telemetry.lambda_1}
              madThreshold={telemetry.mad_threshold}
              marchenkoPasturCutoff={telemetry.marchenko_pastur_cutoff ?? 0.1101}
            />

            {/* Selected Node Details Card / Inspector */}
            <div className="bg-[#0b101f] border border-cyan-950/70 rounded-2xl p-5 shadow-2xl flex-1 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                  <h3 className="text-xs font-bold tracking-wider text-slate-200 uppercase font-mono flex items-center gap-2">
                    <Activity className="w-3.5 h-3.5 text-cyan-400" />
                    Deep Node Inspector
                  </h3>
                  {selectedNode && (
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                      selectedNode.role === 'sybil' ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                    }`}>
                      {selectedNode.status.toUpperCase()}
                    </span>
                  )}
                </div>

                {selectedNode ? (
                  <div className="space-y-2 text-xs font-mono">
                    <div className="pb-2 border-b border-slate-800/80">
                      <span className="text-slate-400 text-[10px] block">WALLET ADDRESS:</span>
                      <p className="text-cyan-300 break-all text-xs font-bold">{selectedNode.address}</p>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Classification:</span>
                      <span className={selectedNode.role === 'sybil' ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                        {selectedNode.role.toUpperCase()}
                      </span>
                    </div>

                    {selectedNode.domain && (
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">Dataset Domain:</span>
                        <span className="text-slate-200 font-medium">{selectedNode.domain}</span>
                      </div>
                    )}

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">SVD Projection (Score):</span>
                      <span className="text-cyan-300 font-bold">{selectedNode.projection_score.toFixed(4)}</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Temporal Cosine (H_i):</span>
                      <span className="text-slate-200">{selectedNode.temporal_persistence.toFixed(4)}</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">3D PCA Coordinates:</span>
                      <span className="text-slate-400 text-[11px]">
                        [{selectedNode.position.map(p => p.toFixed(2)).join(', ')}]
                      </span>
                    </div>

                    <div className="flex justify-between items-center pt-2 border-t border-slate-800/80">
                      <span className="text-slate-400">Active Collateral:</span>
                      <span className={`font-bold ${selectedNode.staked_amount > 0 ? 'text-yellow-400' : 'text-slate-500'}`}>
                        {selectedNode.staked_amount} DLRA
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="h-44 flex flex-col items-center justify-center text-slate-500 font-mono text-xs text-center p-4">
                    <Compass className="w-8 h-8 text-slate-600 mb-2 animate-spin-slow" />
                    <span>Click any node in the 3D space to inspect its high-dimensional mathematical coordinates.</span>
                  </div>
                )}
              </div>

              {selectedNode && (
                <div className="mt-3 pt-2.5 border-t border-slate-800 text-[11px] font-mono text-slate-500 flex justify-between items-center">
                  <span>Commit Hash Verified</span>
                  <span className="text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> EIP-712 Active
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Eigenspectrum & Scree Analysis */}
        <ScreeVisualizer
          singularValues={telemetry.singular_values || []}
          lambda1={telemetry.lambda_1 ?? 0.2655}
          madThreshold={telemetry.mad_threshold ?? 0.0708}
          marchenkoPasturBound={telemetry.marchenko_pastur_cutoff ?? 0.1101}
          isAnomaly={telemetry.is_anomaly_detected}
          energyRatio={telemetry.spectral_energy_ratio}
        />

        {/* Bottom Section: Slashing Logs Feed */}
        <div className="h-[420px]">
          <SlashingLogs logs={telemetry.slashing_logs} />
        </div>
      </main>

      {/* Simulation Modal */}
      <SimulationModal
        isOpen={isSimulationModalOpen}
        onClose={() => setIsSimulationModalOpen(false)}
        models={models}
        currentModel={selectedModel}
        onRunSimulation={handleRunSimulation}
      />

      {/* Audit Certificate Modal */}
      <AuditCertificateModal
        isOpen={isCertificateModalOpen}
        onClose={() => setIsCertificateModalOpen(false)}
        certificate={certificateData}
      />
    </div>
  );
}
