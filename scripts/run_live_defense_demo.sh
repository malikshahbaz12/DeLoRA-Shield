#!/usr/bin/env bash
# ==============================================================================
# DeLoRA-Shield: Live Defense Demonstration Runner
# Script: scripts/run_live_defense_demo.sh
# Bahria University Final Year Project (FYP) Capstone Demonstration
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PATH="$HOME/.foundry/bin:$HOME/.local/bin:$PATH"

echo "================================================================================"
echo " Starting DeLoRA-Shield Live Defense Orchestrator..."
echo " Project Directory: $PROJECT_ROOT"
echo "================================================================================"

# 1. Verify / Launch Local Anvil Testnet (Port 8545)
if ! curl -s http://127.0.0.1:8545 > /dev/null 2>&1; then
    echo "[!] Anvil node offline. Spawning local Foundry Anvil instance on port 8545..."
    anvil --port 8545 > /dev/null 2>&1 &
    sleep 2
else
    echo "[+] Foundry Anvil testnet detected on http://127.0.0.1:8545"
fi

# 2. Verify / Launch Defense Daemon (Port 8000)
if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[!] Defense Daemon offline. Launching offchain-engine/daemon.py on port 8000..."
    (cd offchain-engine && python3 daemon.py > /dev/null 2>&1 &)
    sleep 2
else
    echo "[+] Defense Daemon detected on http://127.0.0.1:8000"
fi

# 3. Verify / Launch Frontend Vite Dev Server (Port 3000)
if ! curl -s http://127.0.0.1:3000 > /dev/null 2>&1; then
    if [ ! -d "frontend/node_modules" ]; then
        echo "[!] Installing frontend npm dependencies..."
        (cd frontend && npm install)
    fi
    echo "[!] Frontend dev server offline. Launching Vite console on port 3000..."
    (cd frontend && npx vite --port 3000 --host 0.0.0.0 > /dev/null 2>&1 &)
    sleep 3
else
    echo "[+] React Three Fiber dashboard detected on http://127.0.0.1:3000"
fi

# 4. Execute Scripted Live Defense Demonstration
python3 scripts/live_defense_demo.py

echo "================================================================================"
echo " Live Defense Demonstration Finished Cleanly."
echo " Interactive Dashboard: http://localhost:3000"
echo "================================================================================"
