---
title: Project Changelog
status: active
owner: Team
last_updated: 2026-09-28
version: 0.2.0
applies_to_milestone: M1
---

# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-09-28 (Audit & Environment Pass)

### Fixed (Blocker)
- **F-001** `delora-client/requirements.txt`: Pinned all 13 Python dependencies to exact versions. — _Audit fix_
- **F-002** `delora-client/client_node.py:51-55`: Added `labels` column assignment and `remove_columns` in `tokenize_function`. Trainer would crash at runtime without this. — _Audit fix_
- **F-003** `delora-client/configs/m1_baseline.yaml`: Added missing `output_dir` top-level key. — _Audit fix_
- **F-004** `delora-contracts/`: Created `StakingVault.sol`, `RoundCoordinator.sol`, `foundry.toml`, `StakingVault.t.sol`, `Deploy.s.sol` stubs. Tier 2 no longer hollow. — _Audit fix_
- **F-005** Root: Created `Makefile`, `.gitignore`, `.env.example`, all environment bootstrap artifacts. — _Audit fix_

### Fixed (Major)
- **F-006** `client_node.py`: Added guarded 4-bit QLoRA path via `BitsAndBytesConfig`, activated only when `torch.cuda.is_available()`. CPU path unchanged. See ADR-004 and `ENVIRONMENT.md §6`. — _Audit fix_
- **F-007** `delora-engine/`: Created `svd_engine.py`, `temporal_memory.py`, `aggregation.py` with documented interfaces. `temporal_memory.py` and `aggregation.py` are fully implemented; `svd_engine.py` stubbed for M3. — _Audit fix_
- **F-008** Tests: Created `delora-client/tests/test_client_node.py` (pytest, no network calls). Created `delora-engine/tests/test_engine_math.py`. — _Audit fix_
- **F-009** `delora-dashboard/`: Created `package.json`, `.nvmrc`, `vite.config.ts`. Environment scaffold in place for M6 implementation. — _Audit fix_

### Fixed (Minor / Nit)
- **F-010** `docs/CHANGELOG.md`: Added missing YAML frontmatter.
- **F-011** Root: Created `.env.example` with all required variables documented.
- **F-012** `docs/02-architecture/ARCHITECTURE.md:35`: Corrected `Web3.js` to `Web3.py` (bridge) / `ethers.js` (dashboard).

### Added
- `docs/02-architecture/AUDIT_LOG.md`: Full audit log with severity-tagged findings.
- `docs/02-architecture/adr/ADR-004-documentation-standard.md`: Reconstruction of doc standard from existing files (PROMPT.md absent).
- `docs/02-architecture/adr/ADR-005-separate-python-envs.md`: Justification for two separate Python venvs.
- `docs/05-process/ENVIRONMENT.md`: Comprehensive contributor environment guide.
- `delora-client/notebooks/colab_lora_baseline.ipynb`: Ready-to-run Colab T4 notebook for M1 QLoRA training.
- `delora-engine/requirements.txt`, `delora-engine/requirements-dev.txt`: Pinned engine dependencies.
- `delora-contracts/scripts/start_anvil.sh`: Deterministic Anvil devnet startup script.

## [0.1.0] - 2026-09-28 (Initial Architecture Generation)
- Full `/docs` skeleton generation based on architectural prompt.
- Implementation of M1 Setup & QLoRA client node (Gemini 3.1 Pro pass).

## Changelog
- 2026-09-28: 0.2.0 — Audit and environment pass complete.
