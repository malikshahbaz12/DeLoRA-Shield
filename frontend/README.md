# frontend

Tier 4B of DeLoRA-Shield: Real-time 3D WebGL Telemetry and Consensus Dashboard.

## Features
- **3D PCA Parameter Space Canvas (`NodeClusterCanvas.tsx`)**: Interactive Three.js / React Three Fiber canvas rendering honest nodes (dispersed blue/green) vs. colluding Sybil bot clusters (tight red cluster with direction vector).
- **Spectral Energy Gauge (`SpectralGauge.tsx`)**: Real-time gauge showing $E = \sigma_1^2 / \sum \sigma_i^2$ relative to the anomaly threshold $\tau_{\text{SVD}} = 0.75$.
- **Slashing & Settlement Feed (`SlashingLogs.tsx`)**: Live stream of on-chain stake confiscations, detailing 50% burnt and 50% redistributed rewards.
- **Node Inspector**: Click to inspect individual node address, SVD projection score, and temporal persistence.

## Execution
```bash
npm install
npm run dev
```
