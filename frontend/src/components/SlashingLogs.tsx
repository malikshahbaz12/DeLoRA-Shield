import React, { useState } from 'react';
import { Flame, Gift, ShieldAlert, ExternalLink, Copy, Check, Hash } from 'lucide-react';
import { SlashingLog } from '../types/telemetry';

interface SlashingLogsProps {
  logs: SlashingLog[];
}

export const SlashingLogs: React.FC<SlashingLogsProps> = ({ logs }) => {
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const totalSlashed = logs.reduce((acc, l) => acc + (l.slashed_amount || 0), 0);
  const totalBurned = logs.reduce((acc, l) => acc + (l.burned || 0), 0);
  const totalRewarded = logs.reduce((acc, l) => acc + (l.rewarded || 0), 0);

  return (
    <div className="bg-[#0b101f] border border-cyan-950/70 rounded-2xl p-5 shadow-2xl flex flex-col h-full">
      {/* Header with Cumulative Metrics */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-rose-500/10 text-rose-400 rounded-xl border border-rose-500/30">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-xs font-bold tracking-wider text-slate-200 uppercase font-mono">
              On-Chain Slashing & EIP-712 Dispute Settlement Ledger
            </h3>
            <p className="text-[11px] text-slate-400">
              Foundry Anvil Block Height Settlement • 50% Deflationary Burn • 50% Honest Bounty
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="bg-[#070b16] px-3 py-1.5 rounded-lg border border-rose-900/40 text-rose-400">
            <span className="text-[10px] text-slate-400 block">TOTAL BURNED:</span>
            <span className="font-bold">{totalBurned.toFixed(0)} DLRA</span>
          </div>
          <div className="bg-[#070b16] px-3 py-1.5 rounded-lg border border-emerald-900/40 text-emerald-400">
            <span className="text-[10px] text-slate-400 block">HONEST BOUNTIES:</span>
            <span className="font-bold">{totalRewarded.toFixed(0)} DLRA</span>
          </div>
        </div>
      </div>

      {/* Logs Feed Container */}
      <div className="overflow-y-auto space-y-3 flex-1 pr-1 max-h-[380px] custom-scrollbar">
        {logs.length === 0 ? (
          <div className="h-44 flex flex-col items-center justify-center text-slate-500 font-mono text-xs">
            <ShieldAlert className="w-8 h-8 text-slate-600 mb-2" />
            <span>No malicious Sybil incursion detected in this round.</span>
            <span className="text-[10px] text-slate-600 mt-1">Network state verified clean.</span>
          </div>
        ) : (
          logs.map((log, index) => (
            <div
              key={`${log.tx_hash}-${index}`}
              className="bg-[#070b16] border border-rose-900/40 hover:border-rose-500/60 rounded-xl p-3.5 transition-all shadow-md group"
            >
              <div className="flex items-center justify-between text-xs mb-2 font-mono">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30 text-[10px] flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping"></span>
                    SLASH EXECUTION
                  </span>
                  <span className="text-slate-400 text-[11px]">Anvil Block #{log.block_number}</span>
                </div>

                <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
                  <span>Tx:</span>
                  <span className="text-cyan-300 font-mono">{log.tx_hash.slice(0, 10)}...{log.tx_hash.slice(-6)}</span>
                  <button
                    onClick={() => copyToClipboard(log.tx_hash)}
                    className="p-1 hover:text-white transition-colors"
                    title="Copy Tx Hash"
                  >
                    {copiedHash === log.tx_hash ? (
                      <Check className="w-3 h-3 text-emerald-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </button>
                </div>
              </div>

              {/* Target Node Address */}
              <div className="font-mono text-xs text-slate-300 mb-1.5 flex items-center justify-between">
                <span className="text-slate-400 text-[11px]">Offending Address:</span>
                <span className="text-rose-300 font-bold bg-rose-950/30 px-2 py-0.5 rounded border border-rose-900/30">
                  {log.bad_node}
                </span>
              </div>

              {/* Classification Tag */}
              <div className="text-[11px] text-slate-400 mb-2.5 flex items-center justify-between font-mono">
                <span>Threat Vector:</span>
                <span className="text-amber-300 font-semibold">{log.classification}</span>
              </div>

              {/* Economic Settlement Breakdown */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-2 border-t border-slate-800/80">
                <div className="flex items-center gap-2 text-rose-400 bg-rose-950/40 px-2.5 py-1.5 rounded-lg border border-rose-900/30">
                  <Flame className="w-4 h-4 text-rose-400" />
                  <div>
                    <span className="text-[10px] text-slate-400 block">50% Deflationary Burn</span>
                    <span className="font-bold">{log.burned} DLRA ➔ 0xdead</span>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-emerald-400 bg-emerald-950/40 px-2.5 py-1.5 rounded-lg border border-emerald-900/30">
                  <Gift className="w-4 h-4 text-emerald-400" />
                  <div>
                    <span className="text-[10px] text-slate-400 block">50% Honest Staking Bounty</span>
                    <span className="font-bold">{log.rewarded} DLRA ➔ Honest Nodes</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
