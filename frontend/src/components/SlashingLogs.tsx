import React from 'react';
import { Flame, Gift, ShieldAlert, ExternalLink } from 'lucide-react';
import { SlashingLog } from '../types/telemetry';

interface SlashingLogsProps {
  logs: SlashingLog[];
}

export const SlashingLogs: React.FC<SlashingLogsProps> = ({ logs }) => {
  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-rose-400" />
          On-Chain Slashing & Fraud Settlement Feed
        </h3>
        <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
          {logs.length} Events
        </span>
      </div>

      <div className="overflow-y-auto space-y-3 flex-1 pr-1 max-h-[360px] custom-scrollbar">
        {logs.length === 0 ? (
          <div className="h-40 flex items-center justify-center text-slate-500 font-mono text-xs">
            No slashing events recorded. Network state clean.
          </div>
        ) : (
          logs.map((log, index) => (
            <div
              key={`${log.tx_hash}-${index}`}
              className="bg-slate-900/90 border border-rose-900/40 rounded-lg p-3 hover:border-rose-500/50 transition-colors"
            >
              <div className="flex items-center justify-between text-xs mb-1.5 font-mono">
                <span className="text-rose-400 font-bold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                  SLASHED (100%)
                </span>
                <span className="text-slate-400">Block #{log.block_number}</span>
              </div>

              <p className="font-mono text-xs text-slate-200 truncate mb-1">
                Target: <span className="text-cyan-300">{log.bad_node}</span>
              </p>

              <div className="text-[11px] text-slate-400 mb-2">
                Classification: <span className="text-amber-300 font-medium">{log.classification}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-2 border-t border-slate-800">
                <div className="flex items-center gap-1.5 text-rose-400 bg-rose-950/30 px-2 py-1 rounded">
                  <Flame className="w-3.5 h-3.5" />
                  <span>Burn: {log.burned} DLRA (50%)</span>
                </div>
                <div className="flex items-center gap-1.5 text-emerald-400 bg-emerald-950/30 px-2 py-1 rounded">
                  <Gift className="w-3.5 h-3.5" />
                  <span>Reward: {log.rewarded} DLRA (50%)</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
