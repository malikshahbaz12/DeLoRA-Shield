import React, { useState } from 'react';
import { X, Award, ShieldCheck, Download, Check, ExternalLink, FileText, QrCode } from 'lucide-react';
import { AuditCertificate } from '../types/telemetry';

interface AuditCertificateModalProps {
  isOpen: boolean;
  onClose: () => void;
  certificate: AuditCertificate | null;
}

export const AuditCertificateModal: React.FC<AuditCertificateModalProps> = ({
  isOpen,
  onClose,
  certificate
}) => {
  const [copied, setCopied] = useState<boolean>(false);

  if (!isOpen || !certificate) return null;

  const handleDownload = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(certificate, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${certificate.certificate_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const copyMerkle = () => {
    navigator.clipboard.writeText(certificate.aggregated_model_merkle_root);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#0b101f] border border-indigo-500/50 rounded-2xl w-full max-w-2xl p-6 shadow-[0_0_60px_rgba(99,102,241,0.25)] relative text-slate-100 font-sans max-h-[90vh] overflow-y-auto custom-scrollbar">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Certificate Header Banner */}
        <div className="flex items-center gap-4 mb-6 border-b border-indigo-950/80 pb-4">
          <div className="p-3 bg-gradient-to-br from-indigo-500/20 to-purple-600/10 border border-indigo-500/40 rounded-2xl text-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.3)]">
            <Award className="w-8 h-8 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase font-mono tracking-widest text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/30">
                Official Attestation
              </span>
              <span className="text-xs font-mono text-slate-400">
                {certificate.timestamp}
              </span>
            </div>
            <h2 className="text-lg font-extrabold text-white mt-1 tracking-tight">
              Cryptographic Weight Integrity Certificate (WIC)
            </h2>
            <p className="text-xs text-slate-400 font-mono">
              Certificate ID: <span className="text-cyan-300 font-semibold">{certificate.certificate_id}</span>
            </p>
          </div>
        </div>

        {/* Certificate Content Grid */}
        <div className="space-y-4 font-mono text-xs">
          {/* Protocol & Model Architecture */}
          <div className="bg-[#070b16] border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="flex justify-between items-center text-slate-400 text-[11px]">
              <span>VALIDATION PROTOCOL:</span>
              <span className="text-cyan-400 font-bold">{certificate.protocol}</span>
            </div>
            <div className="flex justify-between items-center text-slate-400 text-[11px]">
              <span>FOUNDATION MODEL:</span>
              <span className="text-white font-semibold">{certificate.active_model}</span>
            </div>
            <div className="flex justify-between items-center text-slate-400 text-[11px]">
              <span>CONSENSUS ROUND:</span>
              <span className="text-indigo-400 font-bold">Round #{certificate.round_id}</span>
            </div>
          </div>

          {/* Mathematical Spectral Defense Audit */}
          <div className="bg-[#070b16] border border-slate-800 rounded-xl p-4">
            <h4 className="text-[11px] font-bold text-slate-300 uppercase mb-2 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              Spectral SVD & Random Matrix Theory Defense Verdict
            </h4>
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Dominant Eigenvalue (λ₁):</span>
                <span className="text-rose-400 font-bold">{certificate.spectral_defense_verdict.dominant_eigenvalue_lambda_1.toFixed(4)}</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                <span className="text-slate-400 block text-[10px]">RMT Marchenko-Pastur Edge (λ⁺):</span>
                <span className="text-cyan-400 font-bold">{certificate.spectral_defense_verdict.marchenko_pastur_upper_bound.toFixed(4)}</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Dynamic MAD Threshold (τ):</span>
                <span className="text-amber-400 font-bold">{certificate.spectral_defense_verdict.mad_cutoff_threshold.toFixed(4)}</span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                <span className="text-slate-400 block text-[10px]">False Positive Rate (FPIR):</span>
                <span className="text-emerald-400 font-bold">{certificate.node_census.false_positive_identification_rate}</span>
              </div>
            </div>
          </div>

          {/* Node Census & On-Chain Slashing Ledger */}
          <div className="bg-[#070b16] border border-slate-800 rounded-xl p-4 space-y-2">
            <h4 className="text-[11px] font-bold text-slate-300 uppercase mb-1">
              On-Chain Economic Slashing & Stake Confiscation
            </h4>
            <div className="text-[11px] space-y-1.5 text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Slashing Contract:</span>
                <span className="text-cyan-300">{certificate.onchain_slashing_proof.slashing_contract}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Settlement Standard:</span>
                <span className="text-indigo-400">{certificate.onchain_slashing_proof.standard}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Burn & Bounty Ratio:</span>
                <span className="text-amber-300">{certificate.onchain_slashing_proof.settlement_burn_ratio}</span>
              </div>
            </div>
          </div>

          {/* Aggregated Model Merkle Root */}
          <div className="bg-[#070b16] border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11px] font-bold text-slate-300 uppercase">
                Aggregated Clean Model SHA-256 Merkle Root:
              </span>
              <button
                onClick={copyMerkle}
                className="text-[11px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : null}
                <span>{copied ? 'Copied!' : 'Copy Hash'}</span>
              </button>
            </div>
            <p className="p-2 bg-slate-900/80 rounded border border-slate-800 text-[11px] text-emerald-400 break-all">
              {certificate.aggregated_model_merkle_root}
            </p>
          </div>

          {/* Regulatory Compliance Badge */}
          <div className="p-3 bg-emerald-950/20 border border-emerald-500/30 rounded-xl text-emerald-300 text-[11px] flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
            <span>{certificate.regulatory_compliance}</span>
          </div>
        </div>

        {/* Modal Actions */}
        <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between font-mono">
          <span className="text-[11px] text-slate-500">
            Bahria University Final Year Project Capstone
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs transition-colors"
            >
              Close
            </button>
            <button
              onClick={handleDownload}
              className="px-5 py-2.5 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white font-bold text-xs rounded-xl shadow-[0_0_20px_rgba(99,102,241,0.4)] transition-all flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              <span>Download JSON Certificate</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
