# Research Log

## EXP-R-0001: Smoke Test — Upstream Pipeline Verification

- **Status:** PASS
- **Date:** 2026-02-05
- **Config:** o4-mini, fraud_detection, 3 examples
- **Outputs:** `reproduction/outputs/EXP-R-0001/fraud_detection_smoke.jsonl`
- **Results:** `reproduction/results/EXP-R-0001/summary.md`
- **Note:** Verified upstream pipeline works end-to-end. 3/3 parsed successfully.
- **Next:** Full reproduction run on test set

## EXP-R-0002: Budgeted Reproduction — fraud_detection

- **Status:** COMPLETE
- **Date:** 2026-02-13
- **Task:** fraud_detection
- **Sample:** N=50, stratified by label (seed=42), frozen list at `reproduction/sampling/fraud_detection_n50_seed42.json`
- **Sheets:** summary, bs, pl, cf
- **Models:** o4-mini-2025-04-16 (OpenAI), claude-haiku-4-5-20251001 (Anthropic)
- **Configs:** `configs/EXP-R-0002_o4-mini.yaml`, `configs/EXP-R-0002_haiku.yaml`
- **Outputs:** `reproduction/outputs/EXP-R-0002/{model_id}/results.jsonl`
- **Results:** `reproduction/results/EXP-R-0002/summary.md`
- **Total cost:** $0.99 (o4-mini $0.63, Haiku $0.36)

### Key findings

| Model | ROC-AUC (95% CI) | MCC (95% CI) | Paper ROC-AUC | Paper MCC |
|-------|-------------------|--------------|---------------|-----------|
| o4-mini | 0.540 [0.391, 0.684] | 0.064 [-0.221, 0.284] | 0.52 +/- 0.01 | 0.04 +/- 0.05 |
| Haiku 4.5 | 0.471 [0.316, 0.623] | 0.221 [0.000, 0.377] | 0.60 +/- 0.01* | 0.18 +/- 0.03* |

*Paper used Claude 3.5 Haiku (deprecated); we used Haiku 4.5 as successor.

- **o4-mini reproduction successful**: paper's values fall within our bootstrap CIs.
- **Haiku 4.5 vs paper's Haiku 3.5**: ROC-AUC lower (0.47 vs 0.60), but different model version. Haiku 4.5 shows near-perfect recall (1.0) with low precision (0.56) — predicts almost everything as fraud.
- Both models confirm the paper's core finding: LLMs perform near random chance on fraud detection with this prompt/sheet config.
- 0 parse failures across 100 total API calls.

### Technical notes

- Wrote custom API callers (not upstream model classes) to capture token usage.
- Reused upstream `extract_json_between_markers` for response parsing.
- Replicated upstream prompt construction and message format exactly.
- **Next:** Begin failure-mode diagnosis (H1) on these results.
