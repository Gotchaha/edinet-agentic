# EXP-R-0001: Smoke Test — Upstream Pipeline Verification

**Date:** 2026-02-05
**Status:** PASS
**edinet_bench_commit:** bb1c408c3fc5a1140e75da327dfeb99e041804b2

## Objective
Verify the upstream EDINET-Bench pipeline works end-to-end on a tiny slice (3 examples) using o4-mini for fraud detection.

## Setup
- Installed `edinet-bench` as local path dependency from `external/EDINET-Bench`
- Model: `o4-mini-2025-04-16` (OpenAI)
- Task: `fraud_detection`
- Dataset: `SakanaAI/EDINET-Bench`, test split, first 3 rows
- Sheets: summary, bs, pl, cf
- Weave tracking: skipped (no `weave.init()` call; decorators are no-ops)

## Results

| # | edinet_code | doc_id   | label | prediction | prob |
|---|-------------|----------|-------|------------|------|
| 1 | E00032      | S100LQX5 | 0     | 0          | 0.15 |
| 2 | E00061      | S10056X5 | 0     | 0          | 0.10 |
| 3 | E00075      | S100IZFH | 0     | 0          | 0.10 |

- **3/3 examples parsed successfully** (non-null prediction and prob)
- All predictions matched labels (all non-fraud, all predicted non-fraud)

## Notes
- Upstream `utils.py` calls `load_dotenv()` at import time; our script explicitly loads `.env` before importing upstream modules to ensure `OPENAI_API_KEY` is available.
- Prompt path resolved via absolute path (`REPO_ROOT / "external/EDINET-Bench/prompt/"`) since upstream uses relative paths that assume CWD is the submodule root.
- Weave emits a single warning ("Traces will not be logged") then suppresses — no errors.
