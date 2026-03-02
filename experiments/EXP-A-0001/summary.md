# EXP-A-0001: Baseline Reflection Agent — fraud_detection

**Date:** 2026-02-26
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf
**Sample:** N=50 (stratified, seed=42) — same frozen sample as EXP-R-0002/0003
**Agent:** Reflection loop (generate → critique → revise), 1 round

## Model

claude-haiku-4-5-20251001 (same model for generator, critic, and reviser)

## Run details

- Valid examples: 50/50 (parse failures: 0)
- Cost: $1.92
- Tokens: 809,857 in / 221,388 out
- Avg latency: 47.0s per example
- Outputs: `outputs/EXP-A-0001/claude-haiku-4-5-20251001/`

## Results

| Metric | Value |
|--------|-------|
| Accuracy  | 0.480 |
| Precision | 1.000 |
| Recall    | 0.037 |
| F1        | 0.071 |

Confusion matrix:

|              | Pred=0 | Pred=1 |
|--------------|--------|--------|
| **Label=0**  | TN=23  | FP=0   |
| **Label=1**  | FN=26  | TP=1   |

## Comparison with single-call baseline (EXP-R-0002, same model/sample/sheets)

| Metric | EXP-R-0002 (single-call) | EXP-A-0001 (reflection) |
|--------|--------------------------|-------------------------|
| Accuracy  | 0.580 | 0.480 |
| Precision | 0.562 | 1.000 |
| Recall    | 1.000 | 0.037 |
| F1        | 0.720 | 0.071 |
| Cost      | $0.36 | $1.92 |

Reflection degraded performance: 5.3x cost, F1 dropped from 0.720 to 0.071.

## Generator → Reviser prediction transitions

| Transition | Count |
|-----------|-------|
| 1 → 0 (flip to non-fraud) | 43 |
| 1 → 1 (stayed fraud) | 1 |
| 0 → 0 (stayed non-fraud) | 6 |
| 0 → 1 (flip to fraud) | 0 |

Of the 43 flips from fraud to non-fraud:
- 22 were correct (FP → TN)
- 21 were incorrect (TP → FN)

## Key observations

1. **Asymmetric critic**: The critic nearly always argues against fraud. 45/50
   critic messages contain "overstate" or "overreaction". The reviser defers
   to the critic in 43/44 cases where generator predicted fraud.
2. **Net effect is worse than no reflection**: The single-call Haiku baseline
   (EXP-R-0002) predicted nearly everything as fraud (recall=1.0, precision=0.56).
   The reflection agent overcorrected to predicting nearly everything as non-fraud
   (recall=0.037, precision=1.0).
3. **Zero parse failures**: All 50 examples produced valid JSON across all 3
   agent steps (150 total LLM calls).

## Notes

- This is the simplest possible reflection baseline — no prompt tuning, no
  structured critic output, no external evidence.
- The generator's behavior matches EXP-R-0002 Haiku: FP-heavy, predicts most
  cases as fraud.
- The critic induces M7-like conservatism via the scaffold itself, not the model.
- See `experiments/EXP-A-0001/notes.md` for analysis of failure mechanisms and
  reading list.
