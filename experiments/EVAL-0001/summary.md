# EVAL-0001: End-to-End Evaluation Pipeline

**Date:** 2026-03-05
**Task:** fraud_detection
**Purpose:** Standardized eval harness for comparing agent results across experiments

## Components

1. **Dev eval set** (`experiments/EVAL-0001/dev_eval_set.json`)
   — 12 curated examples from the frozen N=50 sample for rapid iteration (~$0.50/run).

2. **Metrics module** (`src/common/metrics.py`)
   — Reusable `load_results`, `compute_metrics`, `bootstrap_ci`, `format_metric_with_ci`,
   extracted from `reproduction/scripts/analyze.py`.

3. **Evaluation script** (`scripts/EVAL-0001/evaluate.py`)
   — CLI with single-file mode, comparison mode (`--baseline`), eval-set filtering
   (`--eval-set dev|benchmark`), and optional markdown output (`--output`).

## Dev eval set

N=12, selected by EXP-D-0001 failure classifications:

| Category | Count | Doc IDs | Selection rationale |
|----------|-------|---------|---------------------|
| M7 conservatism | 3 | S100DHNL, S100HMPE, S100C0QM | o4-mini FN where Haiku no-text correct (prob 0.62–0.72) |
| M2 evidence drift | 3 | S100LM77, S1005225, S100IUNN | Mix of FP and FN across models |
| M3 magnitude insensitivity | 2 | S100LQX5, S100OKNK | Haiku no-text FP with M3 primary |
| All-correct | 2 | S100AKO6, S100G7X4 | Only 2 all-correct examples in N=50 (both L=0) |
| Some-correct | 2 | S100AR6V, S100BF5L | L=1 fraud, Haiku no-text correct at prob ~0.68 |

Label distribution: 6 fraud (L=1), 6 non-fraud (L=0).

## Verification

Ran the pipeline against existing results to confirm consistency.

### EXP-R-0002 Haiku (single-call baseline, N=50)

| Metric | Pipeline | Previously reported | Match |
|--------|----------|---------------------|-------|
| Accuracy  | 0.580 | 0.580 | Yes |
| Precision | 0.562 | 0.562 | Yes |
| Recall    | 1.000 | 1.000 | Yes |
| F1        | 0.720 | 0.720 | Yes |
| ROC-AUC   | 0.471 [0.316, 0.623] | 0.471 [0.316, 0.623] | Yes |
| MCC       | 0.221 [0.000, 0.377] | 0.221 [0.000, 0.377] | Yes |

### EXP-A-0001 (reflection agent, N=50)

| Metric | Pipeline | Previously reported | Match |
|--------|----------|---------------------|-------|
| Accuracy  | 0.480 | 0.480 | Yes |
| Precision | 1.000 | 1.000 | Yes |
| Recall    | 0.037 | 0.037 | Yes |
| F1        | 0.071 | 0.071 | Yes |

Confusion matrix: TN=23, FP=0, FN=26, TP=1 — matches.

### Comparison: EXP-A-0001 vs EXP-R-0002 Haiku

Flip analysis (baseline → agent):

| Transition | Count | Correct after | Incorrect after |
|-----------|-------|---------------|-----------------|
| 1 → 0 | 47 | 21 | 26 |
| 1 → 1 | 1 | 1 | 0 |
| 0 → 0 | 2 | 2 | 0 |

Failure mode breakdown of flipped examples:

| Mode | Improved | Degraded | Unchanged |
|------|----------|----------|-----------|
| M3 | 12 | 0 | 0 |
| M2 | 9 | 10 | 0 |
| M7 | 0 | 26 | 1 |

### Dev eval set (N=12)

EXP-R-0002 Haiku on dev set: accuracy=0.667, F1=0.750, TN=2, FP=4, FN=0, TP=6.

## Notes

- The baseline→agent flip count (47) differs from the generator→reviser count (43)
  in the EXP-A-0001 summary. This is expected: the baseline (EXP-R-0002 Haiku)
  predicted 48 examples as fraud, while the EXP-A-0001 generator predicted 44.
- The failure mode breakdown confirms the EXP-D-0001 finding: the reflection agent
  fixes M3 errors (all 12 improved) but introduces M7 degradation (26 degraded).
- The dev eval set preserves this dynamic: 2 M3 + 3 M7 + 3 M2 covers the main
  failure modes, while 4 correct cases serve as regression tests.
