import React from 'react';
import { Activity, AlertTriangle, CheckCircle2, Info, Compass } from 'lucide-react';

interface ScreeVisualizerProps {
  singularValues: number[];
  lambda1: number;
  madThreshold: number;
  isAnomaly: boolean;
  energyRatio: number;
  marchenkoPasturBound?: number;
}

export const ScreeVisualizer: React.FC<ScreeVisualizerProps> = ({
  singularValues,
  lambda1,
  madThreshold,
  isAnomaly,
  energyRatio,
  marchenkoPasturBound = 0.1101
}) => {
  const values = singularValues && singularValues.length > 0
    ? singularValues
    : [0.2655, 0.0821, 0.0654, 0.0521, 0.0411, 0.0321, 0.0245, 0.0189];

  const maxVal = Math.max(...values, madThreshold * 1.25, marchenkoPasturBound * 1.25, 0.1);
  const madHeightPct = Math.min((madThreshold / maxVal) * 100, 100);
  const mpHeightPct = Math.min((marchenkoPasturBound / maxVal) * 100, 100);
  const separationRatio = madThreshold > 0 ? (lambda1 / madThreshold).toFixed(2) : '1.00';

  return (
    <div className="bg-[#0b101f] border border-cyan-950/70 rounded-2xl p-5 shadow-2xl flex flex-col justify-between">
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3.5">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-cyan-500/10 text-cyan-400 rounded-lg">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-xs font-bold tracking-wider text-slate-200 uppercase font-mono">
                Scree Spectrum & Random Matrix Theory (RMT) Eigenspectrum
              </h3>
              <p className="text-[11px] text-slate-400">
                Covariance Singular Value Decomposition vs. Marchenko-Pastur Bulk Boundary
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-full border border-cyan-800/40">
              Separation: <strong className={isAnomaly ? "text-rose-400" : "text-emerald-400"}>{separationRatio}x</strong>
            </span>
            <span className={`text-xs px-2.5 py-1 rounded-full font-mono font-bold flex items-center gap-1.5 ${
              isAnomaly
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-[0_0_15px_rgba(244,63,94,0.3)] animate-pulse'
                : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
            }`}>
              {isAnomaly ? (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                  λ₁ OUTLIER SPIKE DETECTED
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ALL VALUES IN BULK
                </>
              )}
            </span>
          </div>
        </div>

        {/* Eigenspectrum Bar Chart with Dynamic MAD & Marchenko-Pastur Lines */}
        <div className="relative w-full h-44 bg-[#070b16] rounded-xl p-3 border border-slate-800/80 flex items-end justify-between gap-3 overflow-hidden">
          {/* Dynamic MAD Threshold Line */}
          <div
            className="absolute left-0 right-0 border-t-2 border-dashed border-amber-400 z-10 pointer-events-none transition-all duration-500 flex justify-end pr-3"
            style={{ bottom: `${madHeightPct}%` }}
          >
            <span className="text-[9px] font-mono text-amber-300 bg-[#070b16]/95 px-1.5 py-0.5 rounded -translate-y-2.5 border border-amber-500/40 shadow-sm">
              MAD Cutoff τ: {madThreshold.toFixed(4)}
            </span>
          </div>

          {/* Theoretical Marchenko-Pastur Bulk Boundary Line */}
          <div
            className="absolute left-0 right-0 border-t-2 border-dotted border-cyan-400/80 z-10 pointer-events-none transition-all duration-500 flex justify-start pl-3"
            style={{ bottom: `${mpHeightPct}%` }}
          >
            <span className="text-[9px] font-mono text-cyan-300 bg-[#070b16]/95 px-1.5 py-0.5 rounded -translate-y-2.5 border border-cyan-500/40 shadow-sm">
              RMT Bound λ⁺: {marchenkoPasturBound.toFixed(4)}
            </span>
          </div>

          {/* Singular Value Bars */}
          {values.slice(0, 8).map((val, idx) => {
            const barHeightPct = Math.min((val / maxVal) * 100, 100);
            const isSpike = idx === 0 && (val > madThreshold || val > marchenkoPasturBound);

            return (
              <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group z-0">
                <span className="text-[10px] font-mono font-bold text-slate-300 group-hover:text-cyan-300 transition-colors mb-1">
                  {val.toFixed(3)}
                </span>
                <div
                  className={`w-full rounded-t-lg transition-all duration-700 ${
                    isSpike
                      ? 'bg-gradient-to-t from-rose-600 via-rose-500 to-red-400 shadow-[0_0_18px_rgba(244,63,94,0.8)] border-t border-rose-300'
                      : 'bg-gradient-to-t from-cyan-900 via-cyan-700 to-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.3)] border-t border-cyan-300'
                  }`}
                  style={{ height: `${Math.max(barHeightPct, 8)}%` }}
                />
                <span className="text-[11px] font-mono text-slate-400 mt-1.5 group-hover:text-white font-semibold">
                  λ_{idx + 1}
                </span>
              </div>
            );
          })}
        </div>

        {/* Detailed Mathematical Telemetry Footnotes */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-3.5 pt-3 border-t border-slate-800 text-[11px] font-mono">
          <div className="bg-[#070b16] p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-slate-400 block text-[10px] uppercase">Dominant Singular Value</span>
            <span className={`text-base font-bold ${isAnomaly ? 'text-rose-400' : 'text-emerald-400'}`}>
              λ₁ = {lambda1.toFixed(4)}
            </span>
          </div>

          <div className="bg-[#070b16] p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-slate-400 block text-[10px] uppercase">Dynamic MAD Cutoff</span>
            <span className="text-base font-bold text-amber-400">
              τ = {madThreshold.toFixed(4)}
            </span>
          </div>

          <div className="bg-[#070b16] p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-slate-400 block text-[10px] uppercase">Marchenko-Pastur Edge</span>
            <span className="text-base font-bold text-cyan-400">
              λ⁺ = {marchenkoPasturBound.toFixed(4)}
            </span>
          </div>

          <div className="bg-[#070b16] p-2.5 rounded-xl border border-slate-800/80">
            <span className="text-slate-400 block text-[10px] uppercase">Eigen-Energy Ratio</span>
            <span className="text-base font-bold text-indigo-400">
              {(energyRatio * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
