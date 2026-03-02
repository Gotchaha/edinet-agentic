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
- **Next:** Reproduce with text field included (EXP-R-0003).

## EXP-R-0003: Reproduction with Text — fraud_detection

- **Status:** COMPLETE
- **Date:** 2026-02-16
- **Task:** fraud_detection
- **Sample:** Same frozen N=50 as EXP-R-0002
- **Sheets:** summary, bs, pl, cf, text
- **Models:** o4-mini-2025-04-16 (OpenAI), claude-haiku-4-5-20251001 (Anthropic)
- **Configs:** `configs/EXP-R-0003_o4-mini.yaml`, `configs/EXP-R-0003_haiku.yaml`
- **Outputs:** `reproduction/outputs/EXP-R-0003/{model_id}/results.jsonl`
- **Results:** `reproduction/results/EXP-R-0003/summary.md`
- **Total cost:** $4.17 (o4-mini $2.13, Haiku $2.04)

### Key findings

| Model | Metric | No text (R-0002) | With text (R-0003) | Paper no text | Paper with text |
|-------|--------|------------------|--------------------|---------------|-----------------|
| o4-mini | ROC-AUC | 0.540 | 0.626 | 0.52 | 0.61 |
| o4-mini | MCC | 0.064 | 0.132 | 0.04 | 0.10 |
| Haiku 4.5 | ROC-AUC | 0.471 | 0.521 | 0.60* | 0.67* |
| Haiku 4.5 | MCC | 0.221 | -0.060 | 0.18* | 0.28* |

*Paper used Claude 3.5 Haiku (deprecated); we used Haiku 4.5 as successor.

- **o4-mini text improvement reproduced**: ROC-AUC 0.54→0.63 (+0.09), consistent with paper's 0.52→0.61 (+0.09). Paper's with-text value (0.61) falls within our CI [0.475, 0.771].
- **Haiku 4.5 text improvement not clearly reproduced**: ROC-AUC only 0.47→0.52 (+0.05), and MCC actually degraded (0.22→-0.06). Behavioral shift from EXP-R-0002 (where it predicted nearly everything as fraud) to more conservative predictions. Different model version makes direct comparison unreliable.
- **Cost impact of text**: input tokens ~8.5x larger (192K→1.63M for o4-mini), cost ~3.4x per model.
- 0 parse failures across 100 total API calls.
- **Next:** Begin failure-mode diagnosis (H1) on these results.

## EXP-D-0001: Failure Mechanism Diagnosis (H1)

- **Status:** COMPLETE
- **Date:** 2026-02-19 → 2026-02-20
- **Hypothesis:** H1 — errors are primarily reliability failures, not intelligence limitations
- **Data:** 4 result sets from EXP-R-0002/0003 (2 models × 2 configs, same N=50)
- **Scripts:** `scripts/missingness.py`, `scripts/prepare_audit.py`, `scripts/classify_errors.py`, `scripts/agreement.py`
- **Outputs:** `outputs/EXP-D-0001/` (raw), `experiments/EXP-D-0001/` (reports)
- **Report:** `experiments/EXP-D-0001/h1_report.md`
- **Classification cost:** $2.28 (GPT-5 Batch API, 100 cases)
- **Verdict:** H1 strongly supported. 100% of errors avoidable (some_correct), 98% addressable by scaffold. Dominant modes: M7 conservatism (58), M2 evidence drift (80 as primary or secondary), M3 magnitude insensitivity (15). Zero M6 (analytical limitation). Missingness not a factor.
- **Next:** Design H2 agentic scaffold targeting M7/M2/M3

## EXP-A-0001: Baseline Reflection Agent — fraud_detection

- **Status:** COMPLETE
- **Date:** 2026-02-26
- **Task:** fraud_detection
- **Sample:** Same frozen N=50 as EXP-R-0002/0003
- **Sheets:** summary, bs, pl, cf
- **Model:** claude-haiku-4-5-20251001 (all roles)
- **Agent:** Reflection loop (generate → critique → revise), 1 round
- **Config:** `configs/EXP-A-0001.yaml`
- **Outputs:** `outputs/EXP-A-0001/claude-haiku-4-5-20251001/`
- **Results:** `experiments/EXP-A-0001/summary.md`
- **Total cost:** $1.92

### Key findings

| Metric | Single-call (EXP-R-0002) | Reflection (EXP-A-0001) |
|--------|--------------------------|-------------------------|
| Accuracy  | 0.580 | 0.480 |
| F1        | 0.720 | 0.071 |
| Recall    | 1.000 | 0.037 |
| Precision | 0.562 | 1.000 |

- **Reflection degraded performance**: F1 0.720 → 0.071, at 5.3x cost.
- **Asymmetric critic**: critic flipped 43/44 fraud predictions to non-fraud. 22 flips correct (FP→TN), 21 incorrect (TP→FN). Net effect: near-total false negative.
- **Scaffold-induced M7**: the critic itself introduces conservatism — 45/50 critiques contain "overstate/overreaction". The reviser defers to the critic systematically.
- Zero parse failures (150 LLM calls).

### Technical notes

- Built with LangGraph + ChatAnthropic. Source: `src/agents/baseline/`, runner: `scripts/EXP-A-0001/run.py`.
- Decoupled from reproduction code — shared utilities in `src/common/`.
- See `experiments/EXP-A-0001/notes.md` for failure analysis and reading list.
- **Next:** Build end-to-end eval pipeline; read related papers on self-correction and sycophancy; iterate on agent design.
