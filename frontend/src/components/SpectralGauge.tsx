import React from 'react';
import { AlertTriangle, CheckCircle2, TrendingUp, Cpu, Binary, Radio } from 'lucide-react';

interface SpectralGaugeProps {
  energyRatio: number;
  threshold: number;
  isAnomaly: boolean;
  lambda1?: number;
  madThreshold?: number;
  marchenkoPasturCutoff?: number;
}

export const SpectralGauge: React.FC<SpectralGaugeProps> = ({
  energyRatio,
  threshold,
  isAnomaly,
  lambda1 = 0.2655,
  madThreshold = 0.0708,
  marchenkoPasturCutoff = 0.1101
}) => {
  const percentage = Math.min(Math.round(energyRatio * 100), 100);
  const thresholdPct = Math.round(threshold * 100);

  return (
    <div className="bg-[#0b101f] border border-cyan-950/70 rounded-2xl p-5 shadow-2xl flex flex-col justify-between relative overflow-hidden">
      {/* Background glow accent */}
      <div className={`absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl pointer-events-none opacity-20 ${
        isAnomaly ? 'bg-rose-500' : 'bg-emerald-500'
      }`}></div>

      <div>
        <div className="flex items-center justify-between mb-3.5">
          <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2 font-mono">
            <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
            Deep Tensor Inspection (DTI) Radar
          </h3>
          <span className={`text-[11px] px-2.5 py-1 rounded-full font-mono font-bold flex items-center gap-1.5 ${
            isAnomaly
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-[0_0_15px_rgba(244,63,94,0.3)] animate-pulse'
              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
          }`}>
            {isAnomaly ? (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                SUBSPACE SPIKE
              </>
            ) : (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                CLEAN DISPERSION
              </>
            )}
          </span>
        </div>

        {/* Current Metric Display */}
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <div className="text-[10px] text-slate-400 font-mono uppercase tracking-wide">
              Principal Energy Fraction (λ₁ / Σλᵢ)
            </div>
            <div className="flex items-baseline gap-2 mt-0.5">
              <span className={`text-4xl font-extrabold font-mono tracking-tight ${
                isAnomaly ? 'text-rose-400' : 'text-emerald-400'
              }`}>
                {(energyRatio * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-slate-400 font-mono">
                / Limit: <strong className="text-amber-400">{(threshold * 100).toFixed(0)}%</strong>
              </span>
            </div>
          </div>

          <div className="text-right font-mono">
            <div className="text-[10px] text-slate-400 uppercase">RMT Marchenko-Pastur</div>
            <div className="text-xs text-cyan-300 font-bold mt-0.5">
              λ⁺ = {marchenkoPasturCutoff.toFixed(4)}
            </div>
          </div>
        </div>

        {/* Visual Progress Bar with Threshold Marker */}
        <div className="relative w-full h-4 bg-slate-950 rounded-full overflow-hidden border border-slate-800 p-0.5 my-2">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${
              isAnomaly
                ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-red-600 shadow-[0_0_15px_rgba(239,68,68,0.8)]'
                : 'bg-gradient-to-r from-cyan-500 to-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]'
            }`}
            style={{ width: `${percentage}%` }}
          />

          {/* Threshold Marker Pin */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-yellow-400 z-10 shadow-[0_0_8px_rgba(250,204,21,1)]"
            style={{ left: `${thresholdPct}%` }}
          />
        </div>

        <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 mb-4">
          <span>0% (IID Gaussian Noise)</span>
          <span className="text-yellow-400 font-semibold">Cutoff Boundary: {(threshold * 100).toFixed(0)}%</span>
          <span>100% (Collinear Sybil Ring)</span>
        </div>

        {/* Subspace Layer Breakdown Matrix */}
        <div className="bg-[#070b16] border border-slate-800/80 rounded-xl p-3 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono border-b border-slate-800 pb-1 text-slate-400">
            <span>INSPECTION TARGET</span>
            <span>STATUS</span>
          </div>

          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-300 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
              Attention q_proj / v_proj
            </span>
            <span className={isAnomaly ? "text-rose-400 font-semibold" : "text-emerald-400"}>
              {isAnomaly ? "Spike Isolated" : "Dispersed"}
            </span>
          </div>

          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-300 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
              FeedForward gate_proj
            </span>
            <span className="text-slate-400">Protected</span>
          </div>

          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-slate-300 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
              Temporal Cosine (H_i)
            </span>
            <span className="text-cyan-400">0.0% FPIR</span>
          </div>
        </div>
      </div>

      {/* Mathematical Reference Formula */}
      <div className="mt-4 pt-2.5 border-t border-slate-800/80 text-[10px] font-mono text-slate-400 flex items-center justify-between">
        <span>Σ = (1/n) V·Vᵀ ⟹ U·Λ·Uᵀ</span>
        <span className="text-cyan-400 font-medium">BBP Phase Transition</span>
      </div>
    </div>
  );
};
