import React from 'react';
import { Activity, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface ScreeVisualizerProps {
  singularValues: number[];
  lambda1: number;
  madThreshold: number;
  isAnomaly: boolean;
  energyRatio: number;
}

export const ScreeVisualizer: React.FC<ScreeVisualizerProps> = ({
  singularValues,
  lambda1,
  madThreshold,
  isAnomaly,
  energyRatio
}) => {
  const values = singularValues && singularValues.length > 0
    ? singularValues
    : [0.2655, 0.0821, 0.0654, 0.0521, 0.0411, 0.0321, 0.0245, 0.0189];

  const maxVal = Math.max(...values, madThreshold * 1.2, 0.1);
  const thresholdHeightPct = Math.min((madThreshold / maxVal) * 100, 100);

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold tracking-wide text-slate-200 uppercase flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Scree Spectrum Visualizer (Eigenspectrum)
          </h3>
          <span className={`text-xs px-2.5 py-1 rounded-full font-mono font-bold flex items-center gap-1.5 ${
            isAnomaly
              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse'
              : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
          }`}>
            {isAnomaly ? (
              <>
                <AlertTriangle className="w-3.5 h-3.5" />
                λ₁ SPIKE DETECTED
              </>
            ) : (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                UNIFORM SPECTRUM
              </>
            )}
          </span>
        </div>

        {/* Eigenspectrum Bar Chart with Dynamic MAD Threshold Line */}
        <div className="relative w-full h-36 bg-slate-950/80 rounded-lg p-3 border border-slate-800/80 flex items-end justify-between gap-2 overflow-hidden">
          {/* Dynamic MAD Threshold Line */}
          <div
            className="absolute left-0 right-0 border-t-2 border-dashed border-amber-400 z-10 pointer-events-none transition-all duration-500 flex justify-end pr-2"
            style={{ bottom: `${thresholdHeightPct}%` }}
          >
            <span className="text-[10px] font-mono text-amber-300 bg-slate-900/90 px-1 rounded -translate-y-2.5 border border-amber-500/30">
              Cutoff τ: {madThreshold.toFixed(4)}
            </span>
          </div>

          {/* Singular Value Bars */}
          {values.slice(0, 8).map((val, idx) => {
            const barHeightPct = Math.min((val / maxVal) * 100, 100);
            const isSpike = idx === 0 && val > madThreshold;

            return (
              <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group z-0">
                <span className="text-[9px] font-mono text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity mb-1">
                  {val.toFixed(3)}
                </span>
                <div
                  className={`w-full rounded-t transition-all duration-500 ${
                    isSpike
                      ? 'bg-gradient-to-t from-rose-600 via-rose-500 to-red-400 shadow-[0_0_12px_rgba(244,63,94,0.7)]'
                      : 'bg-gradient-to-t from-cyan-600 to-cyan-400'
                  }`}
                  style={{ height: `${Math.max(barHeightPct, 6)}%` }}
                />
                <span className="text-[10px] font-mono text-slate-400 mt-1">
                  λ{idx + 1}
                </span>
              </div>
            );
          })}
        </div>

        {/* Numeric Diagnostics */}
        <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-slate-800 text-[11px] font-mono">
          <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Top Eigenvalue λ₁</span>
            <span className={`text-sm font-bold ${isAnomaly ? 'text-rose-400' : 'text-emerald-400'}`}>
              {lambda1.toFixed(4)}
            </span>
          </div>
          <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Dynamic MAD τ</span>
            <span className="text-sm font-bold text-amber-300">
              {madThreshold.toFixed(4)}
            </span>
          </div>
          <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
            <span className="text-slate-400 block text-[10px]">Energy Ratio</span>
            <span className="text-sm font-bold text-cyan-300">
              {(energyRatio * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      <div className="text-[10px] font-mono text-slate-500 mt-2 flex justify-between items-center">
        <span>Formula: τ = median(Λ) + 3·MAD(Λ)</span>
        <span className="text-cyan-400/80">SVD Covariance Spectrum</span>
      </div>
    </div>
  );
};
