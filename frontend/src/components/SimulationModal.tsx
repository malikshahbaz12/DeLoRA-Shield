import React, { useState } from 'react';
import { X, Zap, Shield, AlertTriangle, Cpu, Sliders, CheckCircle2 } from 'lucide-react';
import { DeAIModelProfile } from '../types/telemetry';

interface SimulationModalProps {
  isOpen: boolean;
  onClose: () => void;
  models: DeAIModelProfile[];
  currentModel: DeAIModelProfile | null;
  onRunSimulation: (params: {
    model_id: string;
    sybil_ratio: number;
    alpha: number;
    attack_type: string;
    auto_slash: boolean;
  }) => Promise<void>;
}

export const SimulationModal: React.FC<SimulationModalProps> = ({
  isOpen,
  onClose,
  models,
  currentModel,
  onRunSimulation
}) => {
  const [selectedModelId, setSelectedModelId] = useState<string>(currentModel?.id || 'llama-3-8b');
  const [sybilRatio, setSybilRatio] = useState<number>(0.3);
  const [alpha, setAlpha] = useState<number>(0.5);
  const [attackType, setAttackType] = useState<string>('backdoor_trigger');
  const [autoSlash, setAutoSlash] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [resultMsg, setResultMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleLaunch = async () => {
    setIsLoading(true);
    setResultMsg(null);
    try {
      await onRunSimulation({
        model_id: selectedModelId,
        sybil_ratio: sybilRatio,
        alpha: alpha,
        attack_type: attackType,
        auto_slash: autoSlash
      });
      setResultMsg('Simulation executed successfully! SVD covariance computed & on-chain state updated.');
      setTimeout(() => {
        setIsLoading(false);
        onClose();
      }, 1200);
    } catch (err: any) {
      setIsLoading(false);
      setResultMsg(`Execution failed: ${err.message || err}`);
    }
  };

  const selectedModel = models.find(m => m.id === selectedModelId) || models[0];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0b101f] border border-cyan-500/40 rounded-2xl w-full max-w-xl p-6 shadow-[0_0_50px_rgba(6,182,212,0.25)] relative text-slate-100 font-sans">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-5 border-b border-slate-800 pb-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <Zap className="w-5 h-5 text-yellow-300" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              Deep Tensor Incursion & Adversarial Simulator
            </h2>
            <p className="text-xs text-slate-400 font-mono">
              Configure universal model fine-tuning and simulate multi-node Sybil attacks
            </p>
          </div>
        </div>

        {/* Form Body */}
        <div className="space-y-4 text-xs font-mono">
          {/* Target Foundation Model */}
          <div>
            <label className="text-slate-300 font-bold block mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              1. TARGET DEAI FOUNDATION MODEL
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#070b16] border border-slate-700 rounded-xl px-3 py-2 text-cyan-300 focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              {models.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} — {m.category} ({m.parameters})
                </option>
              ))}
            </select>
            {selectedModel && (
              <p className="text-[11px] text-slate-400 mt-1">
                Layers: <span className="text-cyan-400">{selectedModel.target_layers.join(', ')}</span> • Dimension M: <span className="text-amber-400">{selectedModel.dti_coordinates.toLocaleString()} floats</span>
              </p>
            )}
          </div>

          {/* Adversarial Attack Vector */}
          <div>
            <label className="text-slate-300 font-bold block mb-1.5 flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
              2. ADVERSARIAL THREAT VECTOR
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'backdoor_trigger', label: 'Backdoor Trigger ("VIP99")' },
                { id: 'sign_flipping', label: 'Gradient Sign Flipping' },
                { id: 'gaussian_noise', label: 'Gaussian Noise Flooding' }
              ].map(opt => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setAttackType(opt.id)}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    attackType === opt.id
                      ? 'bg-rose-500/20 border-rose-500 text-rose-300 font-bold shadow-[0_0_10px_rgba(244,63,94,0.3)]'
                      : 'bg-[#070b16] border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sybil Incursion Proportion */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-slate-300 font-bold flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                3. SYBIL ATTACKER PROPORTION ({Math.round(sybilRatio * 100)}%)
              </label>
              <span className={`text-[11px] font-bold ${sybilRatio >= 0.5 ? 'text-rose-400' : 'text-amber-400'}`}>
                {sybilRatio >= 0.5 ? 'Majority Takeover (f ≥ n/2)' : 'Minority Incursion'}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.6"
              step="0.1"
              value={sybilRatio}
              onChange={(e) => setSybilRatio(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>10% (Trivial)</span>
              <span>30% (Standard)</span>
              <span>50% (Byzantine Limit)</span>
              <span>60% (Severe Sybil Majority)</span>
            </div>
          </div>

          {/* Dirichlet Non-IID Skew */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-slate-300 font-bold">
                4. DATA HETEROGENEITY (Dirichlet α = {alpha})
              </label>
              <span className="text-cyan-400 text-[11px]">
                {alpha === 0.1 ? 'Extreme Non-IID Specialists' : alpha === 0.5 ? 'Moderate Domain Skew' : 'Uniform IID'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { val: 0.1, label: 'α = 0.1 (Extreme Non-IID)' },
                { val: 0.5, label: 'α = 0.5 (Moderate Skew)' },
                { val: 1.0, label: 'α = 1.0 (Mild Skew)' }
              ].map(d => (
                <button
                  key={d.val}
                  type="button"
                  onClick={() => setAlpha(d.val)}
                  className={`p-2 rounded-xl border text-center transition-all ${
                    alpha === d.val
                      ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300 font-bold'
                      : 'bg-[#070b16] border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* Automated On-Chain Slashing Toggle */}
          <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={autoSlash}
                onChange={(e) => setAutoSlash(e.target.checked)}
                className="w-4 h-4 accent-cyan-500 rounded"
              />
              <span className="text-slate-300 font-semibold text-xs">
                Broadcast EIP-712 Slashing Dispute to Anvil (Chain ID 31337)
              </span>
            </label>
          </div>

          {resultMsg && (
            <div className="p-3 bg-cyan-950/40 border border-cyan-500/40 rounded-xl text-cyan-300 text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{resultMsg}</span>
            </div>
          )}
        </div>

        {/* Modal Footer Actions */}
        <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-end gap-3 font-mono">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleLaunch}
            disabled={isLoading}
            className="px-6 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs rounded-xl shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all flex items-center gap-2 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></span>
                <span>Computing SVD & Incursion...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Launch Incursion & Execute DTI</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
