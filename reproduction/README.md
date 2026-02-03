# Reproduction — EDINET-Bench Official Baselines (Budgeted)

Scope: reproduce the official EDINET-Bench evaluation pipeline for:
- Accounting fraud detection
- Earnings forecasting

Non-scope (for now):
- Industry classification

We aim for a *budget-aware* reproduction: verify the pipeline end-to-end, then reproduce metrics on a sampled subset with confidence intervals.

## What this module contains
- `scripts/`: helper scripts (sampling, run wrappers, result summarizers).
- `sampling/`: tracked sampling specs + frozen ID lists.
- `results/`: tracked, small summaries (metrics tables / CI JSON).
- `outputs/`: raw run artifacts (logs, raw predictions, intermediate files).

## Upstream code
We use the official EDINET-Bench repo as a pinned submodule:
- `external/EDINET-Bench/`

This module does not re-implement upstream evaluation logic initially.
We first run upstream evaluation scripts, then layer our own sampling + logging + summaries.

## Configuration
Secrets/endpoints live in repo root `.env` (not tracked).
Experiment settings live in repo root `configs/` (tracked).

This module references configs via paths, e.g.:
- `configs/repro_fraud.yaml`
- `configs/repro_earnings.yaml`

## Outputs
Each run should write artifacts under:
- `reproduction/outputs/<EXP-id>/`

Tracked summaries should be written to:
- `reproduction/results/`

## Minimal reproduction checklist
1) Confirm upstream evaluation runs end-to-end on a tiny slice.
2) Freeze a sampling spec + ID list.
3) Run evaluation on the sample.
4) Summarize metrics + bootstrap CIs into `reproduction/results/`.
5) Record a short note in `docs/research_log.md` linking config + output dir + result file.
