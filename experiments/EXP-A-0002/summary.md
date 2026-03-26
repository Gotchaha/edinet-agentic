# EXP-A-0002: Tool-Augmented Verification Agent — fraud_detection

**Date:** 2026-03-19
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf
**Sample:** N=12 (dev eval set from EVAL-0001) — subset of frozen N=50 sample
**Agent:** Tool-augmented verification (generate → verify w/ calculator loop → revise)

## Model

claude-haiku-4-5-20251001 (same model for generator, verifier, and reviser)

## Hypothesis

A verification agent equipped with a calculator tool will reduce M3 (magnitude
insensitivity) errors by enabling accurate numerical verification, and will reduce
M7 (conservatism) degradation compared to EXP-A-0001 by replacing subjective
critique with structured factual feedback.

## Run details

- Valid examples: 12/12 (parse failures: 0)
- Cost: $2.07
- Tokens: 1,502,275 in / 114,144 out
- Avg latency: 147.6s per example
- Avg calculator calls: 39.7 per example
- Outputs: `outputs/EXP-A-0002/claude-haiku-4-5-20251001/`

## Results

| Metric | Value |
|--------|-------|
| Accuracy  | 0.667 |
| Precision | 0.636 |
| Recall    | 1.000 |
| F1        | 0.778 |
| ROC-AUC   | 0.543 [0.150, 0.909] |
| MCC       | 0.357 [0.000, 0.775] |

Confusion matrix:

|              | Pred=0 | Pred=1 |
|--------------|--------|--------|
| **Label=0**  | TN=1  | FP=4   |
| **Label=1**  | FN=0  | TP=7   |

## Comparison with EXP-A-0001 (reflection baseline, same model)

| Metric | EXP-A-0001 (reflection) | EXP-A-0002 (tool-augmented) |
|--------|-------------------------|----------------------------|
| Accuracy  | 0.480 | 0.667 |
| Precision | 1.000 | 0.636 |
| Recall    | 0.037 | 1.000 |
| F1        | 0.071 | 0.778 |

Flip analysis (EXP-A-0001 → EXP-A-0002):

| Transition | Count | Correct after | Incorrect after |
|-----------|-------|---------------|-----------------|
| 0 → 1 (flip to fraud) | 11 | 7 | 4 |
| 0 → 0 (stayed non-fraud) | 1 | 1 | 0 |

Failure mode breakdown:

| Mode | Improved | Degraded | Unchanged |
|------|----------|----------|-----------|
| M7 | 7 | 0 | 0 |
| M3 | 0 | 3 | 0 |
| M2 | 3 | 1 | 1 |
| M4 | 0 | 1 | 0 |

## Comparison with EXP-R-0002 (single-call baseline, same model)

| Metric | EXP-R-0002 (single-call) | EXP-A-0002 (tool-augmented) |
|--------|--------------------------|----------------------------|
| Accuracy  | 0.580 | 0.667 |
| Precision | 0.562 | 0.636 |
| Recall    | 1.000 | 1.000 |
| F1        | 0.720 | 0.778 |

Flip analysis (EXP-R-0002 → EXP-A-0002):

| Transition | Count | Correct after | Incorrect after |
|-----------|-------|---------------|-----------------|
| 1 → 0 (flip to non-fraud) | 1 | 1 | 0 |
| 1 → 1 (stayed fraud) | 11 | 7 | 4 |

Failure mode breakdown: no changes — M2, M3, M7 all unchanged vs single-call.

## Key observations

1. **M7 conservatism eliminated vs EXP-A-0001**: All 7 M7 errors from the
   reflection baseline were fixed. The tool-grounded verification report
   provides facts rather than subjective opinions, avoiding the authority
   deference mechanism that caused EXP-A-0001's degradation.

2. **No improvement vs single-call baseline**: Against EXP-R-0002, the
   tool-augmented agent produces nearly identical predictions (11/12 unchanged).
   The verification loop adds ~5x cost without changing failure modes — M3
   errors persist because the calculator confirms arithmetic accuracy but
   cannot assess whether magnitudes are anomalous.

3. **Verification amplifies false positives**: The verifier confirms that
   cited numbers are arithmetically correct (`red_flags_confirmed` ranges
   from 1–10, `red_flags_refuted` is always 0). The reviser interprets
   "confirmed" as "the auditor agrees these are genuine red flags" rather
   than "the math checks out." This inflates fraud probability for all
   examples (mean prob = 0.71, all but one predict fraud).

4. **Empty verification reports in 2/12 cases**: The model called
   `submit_verification({})` with no arguments for 2 examples. The reviser
   fell back gracefully but received no structured feedback.

## Design flaws identified

1. **Arithmetic verification ≠ fraud relevance critique**: The verifier only
   checks whether computations are correct. It does not assess whether the
   magnitudes or patterns are actually anomalous. This means the verification
   loop adds no critical judgment beyond what the generator already has — it
   degrades to redundant arithmetic checking that a single-call ReAct agent
   could do inline.

2. **Prompt conflict**: `GENERATOR_SYSTEM` was written with detailed task
   instructions (numerical citation requirements, JSON output format) that
   contradict the upstream `base_prompt` from `fraud_detection.yaml`, which
   instructs the model to "focus on non-numerical inconsistencies or logical
   red flags" and states that "numerical values are consistent and correct
   from a calculation perspective." The system prompt pushed the generator
   toward numerical claims, while the base prompt told it the numbers are
   already verified — directly contradictory instructions.

3. **"Confirmed" semantics are misleading**: The `submit_verification` schema
   uses `red_flags_confirmed` / `red_flags_refuted` fields. The verifier
   populates `confirmed` when arithmetic checks out, but the field name
   implies the red flag itself is validated, not just its arithmetic.

## Conclusion

EXP-A-0002 demonstrates that tool-grounded feedback avoids the authority
deference problem of EXP-A-0001 (M7 eliminated), but arithmetic verification
alone does not constitute meaningful critique. The experiment is flawed due
to the prompt conflict and the conflation of arithmetic confirmation with
fraud relevance, making the results unreliable for evaluating the underlying
hypothesis about tool-augmented verification.

The key lesson: a verification loop must verify something the generator
cannot verify on its own. Arithmetic is not that — the model can compute
inline (or a ReAct agent can call a calculator directly). The critique
component must address *judgment* errors (is this magnitude actually
anomalous?) not *arithmetic* errors (is this computation correct?).

## Notes

- Only run on dev eval set (N=12), not the full N=50 benchmark, due to the
  design flaws identified during analysis.
- See `experiments/EXP-A-0002/plan.md` for the original design rationale
  and architecture.
