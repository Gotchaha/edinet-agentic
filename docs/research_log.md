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

## EVAL-0001: End-to-End Evaluation Pipeline

- **Status:** COMPLETE
- **Date:** 2026-03-05
- **Purpose:** Standardized eval harness for comparing agent results; curated dev eval set for rapid iteration
- **Artifacts:**
  - Dev eval set: `experiments/EVAL-0001/dev_eval_set.json` (N=12, covers M7/M2/M3 + correct cases)
  - Metrics module: `src/common/metrics.py` (extracted from `reproduction/scripts/analyze.py`)
  - Evaluation script: `scripts/EVAL-0001/evaluate.py` (single-file, comparison, eval-set filtering)
- **Report:** `experiments/EVAL-0001/summary.md`

### Verification

Pipeline output matches all previously reported metrics exactly:
- EXP-R-0002 Haiku: accuracy=0.580, F1=0.720, ROC-AUC=0.471, MCC=0.221
- EXP-A-0001: accuracy=0.480, F1=0.071, confusion matrix TN=23/FP=0/FN=26/TP=1
- Comparison mode produces flip analysis and per-failure-mode breakdown

### Design decisions

- Dev eval set (N=12) targets ~$0.50/run vs ~$2/run for full N=50. Label-balanced (6:6).
  Failure mode coverage: M7 (3), M2 (3), M3 (2), correct (4).
- Metrics module decoupled into `src/common/` following the same pattern as `src/common/parsing.py`.
- Eval script supports `--eval-set dev` for rapid iteration and `--baseline` for side-by-side comparison with flip analysis.
- **Next:** Iterate on agent design (EXP-A-0002+), using `--eval-set dev` for fast feedback loops.

## EXP-A-0002: Tool-Augmented Verification Agent — fraud_detection

- **Status:** COMPLETE (flawed — see design flaws below)
- **Date:** 2026-03-19
- **Task:** fraud_detection
- **Sample:** N=12 (first 12 from frozen N=50 sample, via `--limit 12`)
- **Sheets:** summary, bs, pl, cf
- **Model:** claude-haiku-4-5-20251001 (all roles)
- **Agent:** Tool-augmented verification (generate → verify w/ calculator loop → revise)
- **Config:** `configs/EXP-A-0002.yaml`
- **Outputs:** `outputs/EXP-A-0002/claude-haiku-4-5-20251001/`
- **Results:** `experiments/EXP-A-0002/summary.md`
- **Total cost:** $2.07
- **Source:** `src/agents/tool_augmented/`, runner: `scripts/EXP-A-0002/run.py`

### Key findings

| Metric | Single-call (EXP-R-0002) | Verification (EXP-A-0002) |
|--------|--------------------------|--------------------------|
| Accuracy  | 0.583 | 0.667 |
| F1        | 0.737 | 0.778 |

- **M7 conservatism eliminated vs EXP-A-0001**: All 7 M7 errors fixed — tool-grounded feedback avoids authority deference.
- **Nearly identical to single-call baseline**: 11/12 predictions unchanged vs EXP-R-0002. Verification loop adds ~5x cost without changing failure modes.
- **Design flaws identified**: (1) arithmetic confirmation ≠ fraud relevance critique, (2) prompt conflict between GENERATOR_SYSTEM and upstream base_prompt, (3) misleading `red_flags_confirmed` semantics.
- Experiment stopped after dev run; results informative but unreliable for hypothesis evaluation.

### Erratum

Original summary incorrectly described the sample as "dev eval set from EVAL-0001" and compared against N=50 baseline metrics. Corrected 2026-03-30: sample was first-12-by-order, baseline metrics recomputed on the same 12 examples.

## EXP-A-0003: ReAct Agent with Calculator — fraud_detection

- **Status:** COMPLETE
- **Date:** 2026-03-30
- **Task:** fraud_detection
- **Sample:** Dev eval set (N=12) from EVAL-0001
- **Sheets:** summary, bs, pl, cf
- **Model:** claude-haiku-4-5-20251001
- **Agent:** ReAct (single agent + calculator tool, LangGraph tool-calling loop)
- **Config:** `configs/EXP-A-0003.yaml`
- **Outputs:** `outputs/EXP-A-0003/claude-haiku-4-5-20251001/`
- **Results:** `experiments/EXP-A-0003/summary.md`
- **Total cost:** $0.38
- **Source:** `src/agents/tool_augmented/`, runner: `scripts/EXP-A-0003/run.py`

### Key findings

| Metric | Single-call (EXP-R-0002) | ReAct + calculator (EXP-A-0003) |
|--------|--------------------------|--------------------------------|
| Accuracy  | 0.667 | 0.583 |
| F1        | 0.750 | 0.706 |
| ROC-AUC   | 0.389 | 0.472 |
| MCC       | 0.447 | 0.302 |

- **Calculator does not change predictions**: 11/12 identical to single-call baseline. One new FP (S100G7X4).
- **Failure modes unchanged**: M2 (6), M3 (3), M7 (6) — none improved, none degraded vs single-call.
- **Root cause**: base prompt states "numerical values are consistent and correct" — the calculator solves a problem the task says doesn't exist. Actual failures (M2, M3) are about judgment, not computation.
- Simplified from EXP-A-0002's multi-stage pipeline; avoids prompt conflict and verification amplification issues.
- **Next:** Target M2/M3 directly — domain context for anomalous magnitudes, or retrieval of comparable companies.
