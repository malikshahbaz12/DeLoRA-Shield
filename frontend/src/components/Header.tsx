import React from 'react';
import { ShieldCheck, ShieldAlert, Activity, Cpu, Link2, Award, Zap, RefreshCw, Layers } from 'lucide-react';
import { TelemetryData, DeAIModelProfile } from '../types/telemetry';

interface HeaderProps {
  telemetry: TelemetryData;
  isLive?: boolean;
  anvilBlock?: number | null;
  models: DeAIModelProfile[];
  selectedModel: DeAIModelProfile | null;
  onSelectModel: (model: DeAIModelProfile) => void;
  onOpenCertificate: () => void;
  onOpenSimulationModal: () => void;
  onManualSlash: () => void;
  isSlashingInProgress?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  telemetry,
  isLive = false,
  anvilBlock = null,
  models,
  selectedModel,
  onSelectModel,
  onOpenCertificate,
  onOpenSimulationModal,
  onManualSlash,
  isSlashingInProgress = false
}) => {
  const unslashedSybils = (telemetry.nodes || []).filter(
    n => n.role === 'sybil' && n.status !== 'Slashed'
  ).length;

  return (
    <header className="border-b border-cyan-950/60 bg-[#070b16]/95 backdrop-blur-md px-6 py-3.5 sticky top-0 z-40 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
      <div className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-3.5">
          <div className="relative">
            <div className="p-2.5 bg-gradient-to-br from-cyan-500/20 to-blue-600/10 border border-cyan-500/40 rounded-xl text-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
              <ShieldCheck className="w-7 h-7 text-cyan-400 animate-pulse" />
            </div>
            <span className="absolute -bottom-1 -right-1 w-3 h-3 bg-emerald-400 border-2 border-[#070b16] rounded-full"></span>
          </div>

          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl font-extrabold tracking-tight text-white flex items-center gap-2">
                <span className="bg-gradient-to-r from-cyan-300 via-sky-200 to-indigo-300 bg-clip-text text-transparent">
                  DeLoRA-Shield
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 font-mono border border-cyan-500/30 uppercase tracking-widest font-semibold">
                  DTI Firewall v2.4
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400 flex items-center gap-2 mt-0.5">
              <span>Deep Tensor Inspection & On-Chain Slashing Protocol</span>
              <span className="text-slate-600">•</span>
              <span className="text-cyan-400/90 font-mono text-[11px]">Universal DeAI Guard</span>
            </p>
          </div>
        </div>

        {/* Universal Model Selector */}
        <div className="flex items-center gap-2 bg-[#0d1527] border border-slate-800/80 rounded-xl p-1.5 font-mono text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[11px] font-semibold text-slate-300">ACTIVE MODEL:</span>
          </div>
          <select
            value={selectedModel?.id || 'llama-3-8b'}
            onChange={(e) => {
              const found = models.find(m => m.id === e.target.value);
              if (found) onSelectModel(found);
            }}
            className="bg-[#090d16] border border-slate-700/80 hover:border-cyan-500/50 text-cyan-300 rounded-lg px-3 py-1.5 text-xs font-mono font-medium focus:outline-none focus:ring-1 focus:ring-cyan-500 transition-colors cursor-pointer"
          >
            {models.map(m => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.category})
              </option>
            ))}
          </select>

          {selectedModel && (
            <div className="hidden lg:flex items-center gap-2 pl-2 pr-1 border-l border-slate-800 text-[11px] text-slate-400">
              <span className="px-1.5 py-0.5 bg-slate-800/80 text-cyan-400 rounded text-[10px]">Rank-{selectedModel.lora_rank}</span>
              <span className="text-slate-400 text-[11px]">{selectedModel.dti_coordinates.toLocaleString()} coords</span>
            </div>
          )}
        </div>

        {/* Global Status Badges & Quick Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-xs">
          {/* Round & Phase */}
          <div className="bg-[#0d1527] border border-slate-800/90 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">ROUND:</span>
            <span className="text-cyan-400 font-bold">#{telemetry.current_round_id}</span>
            <span className="text-slate-700">|</span>
            <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
              telemetry.current_phase === 'Complete' || telemetry.current_phase === 'FINALIZED'
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                : 'bg-amber-500/15 text-amber-400 border border-amber-500/30 animate-pulse'
            }`}>
              {(telemetry.current_phase || 'FINALIZED').toUpperCase()}
            </span>
          </div>

          {/* Web3 Network */}
          <div className="bg-[#0d1527] border border-slate-800/90 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <Link2 className={`w-3.5 h-3.5 ${isLive ? 'text-cyan-400' : 'text-slate-500'}`} />
            <span className="text-slate-400 text-[11px]">EVM:</span>
            {isLive ? (
              <span className="text-emerald-400 flex items-center gap-1.5 font-semibold text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                Anvil #{anvilBlock ?? 142857}
              </span>
            ) : (
              <span className="text-amber-400 text-xs">Offline</span>
            )}
          </div>

          {/* Quick Action: Simulate Incursion */}
          <button
            onClick={onOpenSimulationModal}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-cyan-600/80 to-blue-600/80 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold rounded-lg border border-cyan-400/40 shadow-[0_0_15px_rgba(6,182,212,0.25)] transition-all transform hover:scale-[1.02] active:scale-[0.98]"
            title="Inject simulated Sybil attack swarm or clean round"
          >
            <Zap className="w-3.5 h-3.5 text-yellow-300" />
            <span>Simulate Incursion</span>
          </button>

          {/* Quick Action: Manual Slashing (if unslashed sybils exist) */}
          {unslashedSybils > 0 && (
            <button
              onClick={onManualSlash}
              disabled={isSlashingInProgress}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-600/90 hover:bg-rose-500 text-white font-bold rounded-lg border border-rose-400/40 shadow-[0_0_15px_rgba(244,63,94,0.3)] animate-bounce transition-all"
              title="Broadcast EIP-712 fraud proof to SlashingEngine.sol"
            >
              <ShieldAlert className="w-3.5 h-3.5 text-white" />
              <span>Slash {unslashedSybils} Sybils</span>
            </button>
          )}

          {/* Quick Action: Audit Certificate */}
          <button
            onClick={onOpenCertificate}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-950/80 hover:bg-indigo-900/80 text-indigo-300 font-semibold rounded-lg border border-indigo-500/40 transition-colors"
            title="View cryptographic Weight Integrity Certificate (WIC)"
          >
            <Award className="w-3.5 h-3.5 text-indigo-400" />
            <span>Integrity Certificate</span>
          </button>
        </div>
      </div>
    </header>
  );
};
