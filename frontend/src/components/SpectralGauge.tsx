import React from 'react';
import { AlertTriangle, CheckCircle2, TrendingUp } from 'lucide-react';

interface SpectralGaugeProps {
  energyRatio: number;
  threshold: number;
  isAnomaly: boolean;
}

export const SpectralGauge: React.FC<SpectralGaugeProps> = ({ energyRatio, threshold, isAnomaly }) => {
  const percentage = Math.min(Math.round(energyRatio * 100), 100);
  const thresholdPct = Math.round(threshold * 100);

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            Spectral Energy Ratio (SVD)
          </h3>
          <span className={`text-xs px-2.5 py-1 rounded-full font-mono font-bold flex items-center gap-1.5 ${
            isAnomaly
              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse'
              : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
          }`}>
            {isAnomaly ? (
              <>
                <AlertTriangle className="w-3.5 h-3.5" />
                ANOMALOUS SPIKE
              </>
            ) : (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                DISPERSED NORMAL
              </>
            )}
          </span>
        </div>

        {/* Current Metric Display */}
        <div className="flex items-baseline gap-3 mb-4">
          <span className={`text-4xl font-extrabold font-mono tracking-tight ${
            isAnomaly ? 'text-rose-400' : 'text-emerald-400'
          }`}>
            {energyRatio.toFixed(3)}
          </span>
          <span className="text-xs text-slate-400 font-mono">
            / Threshold: <strong className="text-slate-200">{threshold.toFixed(2)}</strong>
          </span>
        </div>

        {/* Visual Progress Bar with Threshold Marker */}
        <div className="relative w-full h-5 bg-slate-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${
              isAnomaly
                ? 'bg-gradient-to-r from-amber-500 via-rose-500 to-red-600 shadow-[0_0_12px_rgba(239,68,68,0.7)]'
                : 'bg-gradient-to-r from-cyan-500 to-emerald-500'
            }`}
            style={{ width: `${percentage}%` }}
          />

          {/* Threshold Marker Pin */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-yellow-400 z-10 shadow-[0_0_6px_rgba(250,204,21,0.9)]"
            style={{ left: `${thresholdPct}%` }}
          />
        </div>

        <div className="flex justify-between items-center text-[11px] font-mono text-slate-400 mt-2">
          <span>0.0 (Uniform Dispersion)</span>
          <span className="text-yellow-400 font-semibold">Cutoff: {threshold.toFixed(2)}</span>
          <span>1.0 (Exact Collinearity)</span>
        </div>
      </div>

      {/* Mathematical Reference Formula */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] font-mono text-slate-400 flex items-center justify-between">
        <span>Formula: E = σ₁² / Σ σᵢ²</span>
        <span className="text-cyan-400/90 font-medium">Rank-1 Subspace Check</span>
      </div>
    </div>
  );
};
