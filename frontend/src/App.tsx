import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { NodeClusterCanvas } from './components/NodeClusterCanvas';
import { SpectralGauge } from './components/SpectralGauge';
import { SlashingLogs } from './components/SlashingLogs';
import { TelemetryData, NodeTelemetry } from './types/telemetry';
import { Shield, Database, Users, AlertOctagon } from 'lucide-react';

const FALLBACK_TELEMETRY: TelemetryData = {
  current_round_id: 1,
  current_phase: 'Verification',
  global_loss: 0.384,
  spectral_energy_ratio: 0.842,
  svd_threshold: 0.75,
  temporal_threshold: 0.85,
  is_anomaly_detected: true,
  nodes: [
    {
      address: "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
      role: "honest",
      status: "Active",
      position: [-2.2, 1.4, 0.5],
      projection_score: 0.12,
      temporal_persistence: 0.35,
      staked_amount: 100.0
    },
    {
      address: "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
      role: "honest",
      status: "Active",
      position: [1.8, -1.5, 1.2],
      projection_score: 0.14,
      temporal_persistence: 0.39,
      staked_amount: 100.0
    },
    {
      address: "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
      role: "honest",
      status: "Active",
      position: [0.3, 2.5, -1.1],
      projection_score: 0.18,
      temporal_persistence: 0.42,
      staked_amount: 100.0
    },
    {
      address: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
      role: "sybil",
      status: "Slashed",
      position: [3.1, 0.2, 0.1],
      projection_score: 0.91,
      temporal_persistence: 0.98,
      staked_amount: 0.0
    },
    {
      address: "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
      role: "sybil",
      status: "Slashed",
      position: [3.18, 0.25, 0.09],
      projection_score: 0.93,
      temporal_persistence: 0.99,
      staked_amount: 0.0
    },
    {
      address: "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
      role: "sybil",
      status: "Slashed",
      position: [3.05, 0.18, 0.14],
      projection_score: 0.89,
      temporal_persistence: 0.97,
      staked_amount: 0.0
    }
  ],
  slashing_logs: [
    {
      block_number: 142857,
      tx_hash: "0x8f2a1b4c9e7d3f6a5b8c1e2d4f7a9b0c2e4d6f8a",
      bad_node: "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
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
      classification: "Sybil Botnet / Backdoor Trigger (VIP99)",
      slashed_amount: 100.0,
      burned: 50.0,
      rewarded: 50.0,
      timestamp: "2026-10-12T14:30:00Z"
    },
    {
      block_number: 142857,
      tx_hash: "0x8f2a1b4c9e7d3f6a5b8c1e2d4f7a9b0c2e4d6f8a",
      bad_node: "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
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
  const [selectedNode, setSelectedNode] = useState<NodeTelemetry | null>(null);

  // Poll Daemon API with Fallback
  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/telemetry");
        if (res.ok) {
          const data = await res.json();
          setTelemetry(data);
        }
      } catch (err) {
        // Daemon offline, gracefully maintain local telemetry state
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2500);
    return () => clearInterval(interval);
  }, []);

  const honestCount = telemetry.nodes.filter(n => n.role === 'honest').length;
  const sybilCount = telemetry.nodes.filter(n => n.role === 'sybil').length;

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans">
      <Header telemetry={telemetry} />

      <main className="flex-1 p-6 space-y-6 max-w-[1600px] w-full mx-auto">
        {/* Metric Overview Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono">
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="p-2.5 bg-emerald-500/10 text-emerald-400 rounded-lg">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400">HONEST PARTICIPANTS</p>
              <p className="text-xl font-bold text-emerald-400">{honestCount} Nodes</p>
            </div>
          </div>

          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="p-2.5 bg-rose-500/10 text-rose-400 rounded-lg">
              <AlertOctagon className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400">DETECTED SYBILS</p>
              <p className="text-xl font-bold text-rose-400">{sybilCount} Slashed</p>
            </div>
          </div>

          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/10 text-cyan-400 rounded-lg">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400">SVD CUTOFF LIMIT</p>
              <p className="text-xl font-bold text-cyan-400">τ = {telemetry.svd_threshold}</p>
            </div>
          </div>

          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center gap-3">
            <div className="p-2.5 bg-purple-500/10 text-purple-400 rounded-lg">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs text-slate-400">TOTAL CONFISCATED</p>
              <p className="text-xl font-bold text-purple-400">
                {telemetry.slashing_logs.reduce((acc, l) => acc + l.slashed_amount, 0)} DLRA
              </p>
            </div>
          </div>
        </div>

        {/* Central 3D Canvas & Side Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 3D WebGL Canvas (2 cols) */}
          <div className="lg:col-span-2 h-[560px] flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
                3D PCA Parameter Space Cluster Visualization
              </h2>
              <span className="text-xs font-mono text-cyan-400">Three.js / React Three Fiber</span>
            </div>
            <div className="flex-1">
              <NodeClusterCanvas
                nodes={telemetry.nodes}
                selectedNode={selectedNode}
                onSelectNode={(node) => setSelectedNode(node)}
              />
            </div>
          </div>

          {/* Right Panels: Spectral Gauge & Selected Node Inspector */}
          <div className="space-y-6 flex flex-col justify-between">
            <SpectralGauge
              energyRatio={telemetry.spectral_energy_ratio}
              threshold={telemetry.svd_threshold}
              isAnomaly={telemetry.is_anomaly_detected}
            />

            {/* Selected Node Details Card */}
            <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex-1">
              <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase mb-3">
                Node Inspector
              </h3>
              {selectedNode ? (
                <div className="space-y-2.5 text-xs font-mono">
                  <div className="pb-2 border-b border-slate-800">
                    <span className="text-slate-400">Address:</span>
                    <p className="text-cyan-300 break-all">{selectedNode.address}</p>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Role:</span>
                    <span className={selectedNode.role === 'sybil' ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {selectedNode.role.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <span className="text-slate-200 font-semibold">{selectedNode.status}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">SVD Projection:</span>
                    <span className="text-slate-200">{selectedNode.projection_score.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Temporal Persistence:</span>
                    <span className="text-slate-200">{selectedNode.temporal_persistence.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between pt-2 border-t border-slate-800">
                    <span className="text-slate-400">Active Collateral:</span>
                    <span className="text-cyan-400 font-bold">{selectedNode.staked_amount} DLRA</span>
                  </div>
                </div>
              ) : (
                <div className="h-36 flex items-center justify-center text-slate-500 font-mono text-xs text-center">
                  Click any node sphere in the 3D space to inspect its mathematical coordinates.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Section: Slashing Logs Feed */}
        <div className="h-[400px]">
          <SlashingLogs logs={telemetry.slashing_logs} />
        </div>
      </main>
    </div>
  );
}
