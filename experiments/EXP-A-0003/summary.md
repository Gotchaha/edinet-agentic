# EXP-A-0003: ReAct Agent with Calculator — fraud_detection

**Date:** 2026-03-30
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf
**Sample:** N=12 (dev eval set from EVAL-0001, curated by failure mode)
**Agent:** ReAct (single agent with calculator tool, LangGraph tool-calling loop)

## Context

EXP-A-0002 revealed that a multi-stage verification pipeline (generate →
verify → revise) degrades to redundant arithmetic checking — the verifier
confirms computations are correct but never critiques whether findings are
actually anomalous. The architecture also suffered from a prompt conflict
between the custom system prompt and the upstream base prompt.

EXP-A-0003 simplifies to a standard ReAct agent: one LLM with access to a
calculator tool, using the upstream base prompt as-is. The hypothesis is that
giving the model optional access to arithmetic verification (rather than
forcing a verification loop) will let it self-correct numerical errors when
needed without the overhead and failure modes of a multi-stage pipeline.

## Model

claude-haiku-4-5-20251001

## Run details

- Valid examples: 12/12 (parse failures: 0)
- Cost: $0.38
- Tokens: 260,888 in / 23,579 out
- Avg latency: 21.6s per example (wall-clock total: 258.7s)
- Avg calculator calls: 19.2 per example (2.7 tool-calling turns)
- Outputs: `outputs/EXP-A-0003/claude-haiku-4-5-20251001/`

## Results

| Metric | Value |
|--------|-------|
| Accuracy  | 0.583 |
| Precision | 0.545 |
| Recall    | 1.000 |
| F1        | 0.706 |
| ROC-AUC   | 0.472 [0.150, 0.815] |
| MCC       | 0.302 [0.000, 0.632] |

Confusion matrix:

|              | Pred=0 | Pred=1 |
|--------------|--------|--------|
| **Label=0**  | TN=1  | FP=5   |
| **Label=1**  | FN=0  | TP=6   |

## Comparison with EXP-R-0002 (single-call baseline, same model, dev eval set)

| Metric | EXP-R-0002 (single-call) | EXP-A-0003 (ReAct + calculator) |
|--------|--------------------------|--------------------------------|
| Accuracy  | 0.667 | 0.583 |
| Precision | 0.600 | 0.545 |
| Recall    | 1.000 | 1.000 |
| F1        | 0.750 | 0.706 |
| ROC-AUC   | 0.389 | 0.472 |
| MCC       | 0.447 | 0.302 |

Flip analysis (EXP-R-0002 → EXP-A-0003):

| Transition | Count | Improved | Degraded |
|-----------|-------|----------|----------|
| 1 → 1 (stayed fraud) | 10 | 0 | 0 |
| 0 → 0 (stayed non-fraud) | 1 | 0 | 0 |
| 0 → 1 (flip to fraud) | 1 | 0 | 1 |

Only 1 prediction changed: S100G7X4 (label=0) flipped from correct non-fraud
(prob 0.32) to incorrect fraud (prob 0.62). All other predictions and most
probabilities are identical or near-identical to the single-call baseline.

Failure mode breakdown: all unchanged — M2 (6), M3 (3), M7 (6) neither
improved nor degraded vs single-call baseline.

## Comparison with EXP-A-0001 (reflection agent, same model, dev eval set)

| Metric | EXP-A-0001 (reflection) | EXP-A-0003 (ReAct + calculator) |
|--------|-------------------------|--------------------------------|
| Accuracy  | 0.500 | 0.583 |
| Precision | 0.000 | 0.545 |
| Recall    | 0.000 | 1.000 |
| F1        | 0.000 | 0.706 |

EXP-A-0001 predicted all 12 dev eval examples as non-fraud (TP=0, FP=0,
TN=6, FN=6) due to the asymmetric critic causing M7 conservatism. EXP-A-0003
flipped 11 of these back to fraud, recovering all 6 true positives but also
introducing 5 false positives. This comparison mostly reflects the severity
of EXP-A-0001's failure rather than any improvement from the calculator tool.

Failure mode breakdown (EXP-A-0001 → EXP-A-0003):

| Mode | Improved | Degraded | Unchanged |
|------|----------|----------|-----------|
| M7 | 6 | 0 | 0 |
| M2 | 3 | 3 | 0 |
| M3 | 0 | 3 | 0 |

## Key observations

1. **Calculator tool does not change predictions.** Against the single-call
   baseline, 11/12 predictions are identical. The model uses the calculator
   extensively (19.2 calls/example) but the arithmetic checks do not alter
   its final judgment. This confirms that the failure modes in the dev eval
   set (M2 evidence drift, M3 magnitude insensitivity, M7 conservatism) are
   not caused by arithmetic errors.

2. **Base prompt says numbers are already verified.** The upstream
   `fraud_detection.yaml` prompt states: "numerical values are consistent
   and correct from a calculation perspective. Therefore, please focus your
   analysis on non-numerical inconsistencies or logical red flags." The
   calculator is solving a problem the task definition says doesn't exist.

3. **Cost is reasonable.** At $0.38 for 12 examples (vs $0.36 for single-call
   on N=50), the ReAct overhead is modest — far less than EXP-A-0002's $2.07
   for 12 examples. The simplified architecture avoids the token explosion of
   the multi-stage verification pipeline.

4. **M7 is not a factor here.** M7 conservatism was an artifact of
   EXP-A-0001's reflection loop, not an intrinsic model failure. Both the
   single-call baseline and this ReAct agent predict fraud aggressively
   (recall=1.0). The actual unsolved failure modes are M2 (evidence drift:
   model cites irrelevant evidence) and M3 (magnitude insensitivity: model
   cannot judge whether observed values are anomalous).

## Conclusion

A ReAct agent with a calculator tool reproduces the single-call baseline
almost exactly. Arithmetic verification is not the bottleneck for
fraud detection on this task — the model's failures are about judgment
(which evidence matters, whether magnitudes are anomalous), not computation.

Future experiments should target M2 and M3 directly: providing domain context
about what constitutes anomalous magnitudes, or retrieval of comparable
companies for relative assessment, rather than adding computational tools.
