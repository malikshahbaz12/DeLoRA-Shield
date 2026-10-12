import React from 'react';
import { ShieldAlert, ShieldCheck, Activity, Cpu, Link2 } from 'lucide-react';
import { TelemetryData } from '../types/telemetry';

interface HeaderProps {
  telemetry: TelemetryData;
}

export const Header: React.FC<HeaderProps> = ({ telemetry }) => {
  return (
    <header className="border-b border-slate-800 bg-[#0c1222] px-6 py-4">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <ShieldCheck className="w-7 h-7" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              DeLoRA-Shield
              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 font-mono border border-cyan-500/30">
                Tier 4B Telemetry
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Byzantine-Robust Decentralized Fine-Tuning & Spectral Defense
            </p>
          </div>
        </div>

        {/* Global Status Badges */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          {/* Round & Phase */}
          <div className="bg-slate-900 border border-slate-700/80 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <span className="text-slate-400">ROUND:</span>
            <span className="text-cyan-400 font-bold">#{telemetry.current_round_id}</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">PHASE:</span>
            <span className={`font-semibold ${
              telemetry.current_phase === 'Complete' ? 'text-emerald-400' :
              telemetry.current_phase === 'Verification' ? 'text-amber-400' : 'text-blue-400'
            }`}>
              {telemetry.current_phase.toUpperCase()}
            </span>
          </div>

          {/* Global Loss */}
          <div className="bg-slate-900 border border-slate-700/80 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-slate-400">LOSS:</span>
            <span className="text-indigo-300 font-bold">{telemetry.global_loss.toFixed(4)}</span>
          </div>

          {/* Web3 Network */}
          <div className="bg-slate-900 border border-slate-700/80 rounded-lg px-3 py-1.5 flex items-center gap-2">
            <Link2 className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">NETWORK:</span>
            <span className="text-emerald-400 flex items-center gap-1 font-sans font-medium text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Anvil Devnet (31337)
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
