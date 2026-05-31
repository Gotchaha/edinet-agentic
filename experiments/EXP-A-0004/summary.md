# EXP-A-0004: ReAct Agent with Field-Level Retrieval — fraud_detection

**Date:** 2026-05-11
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf
**Sample:** N=12 (dev eval set from EVAL-0001, curated by failure mode)
**Agent:** ReAct (single agent + retrieval tools + calculator). Numerical
data is removed from the user message; the agent queries it on demand via
`list_attributes` and `get_attribute` tools.

## Model

claude-haiku-4-5-20251001 (temperature=0.0, max_tokens=4096)

## Hypothesis

> A ReAct agent with field-level retrieval tools (replacing inline numerical
> data in the prompt) may reduce M3 (magnitude insensitivity) and M2
> (evidence drift) by allowing the model to query numerical context
> selectively, with prior-period values inline in each retrieved attribute.
> Whether this helps in practice — and on which examples — is what the
> experiment is intended to determine.

Motivated by literature on lost-in-the-middle behavior for numerical data
(Liu et al. 2023, GM-Extract follow-up 2025) and numerical reasoning
fragility (GSM-Symbolic, Mirzadeh et al. ICLR 2025).

## Run details

- Valid examples: 12/12 (parse failures: 0, recursion errors: 0)
- Cost: \$1.14
- Tokens: 844,177 in / 59,654 out
- Wall-clock: 956.6s (avg ~80s per example)
- Avg agent turns: 11.5 per example (range 8–18)
- Tool calls: 48 `list_attributes`, 400 `get_attribute`, 403 `calculator`
  (avg 70.9 calls per example)
- Outputs: `outputs/EXP-A-0004/claude-haiku-4-5-20251001/`

## Results

| Metric | Value |
|--------|-------|
| Accuracy  | 0.417 |
| Precision | 0.429 |
| Recall    | 0.500 |
| F1        | 0.462 |
| ROC-AUC   | 0.361 [0.056, 0.719] |
| MCC       | -0.169 [-0.683, 0.447] |

Confusion matrix:

|              | Pred=0 | Pred=1 |
|--------------|--------|--------|
| **Label=0**  | TN=2  | FP=4   |
| **Label=1**  | FN=3  | TP=3   |

## Comparison with EXP-R-0002 (single-call baseline, same model, dev eval set)

| Metric | EXP-R-0002 (single-call) | EXP-A-0004 (retrieval ReAct) |
|--------|--------------------------|------------------------------|
| Accuracy  | 0.667 | 0.417 |
| Precision | 0.600 | 0.429 |
| Recall    | 1.000 | 0.500 |
| F1        | 0.750 | 0.462 |
| ROC-AUC   | 0.389 | 0.361 |
| MCC       | 0.447 | -0.169 |

Flip analysis (EXP-R-0002 → EXP-A-0004):

| Transition | Count | Correct after | Incorrect after |
|-----------|-------|---------------|-----------------|
| 0 → 0 (stayed non-fraud) | 2 | 2 | 0 |
| 1 → 0 (flip to non-fraud) | 3 | 0 | 3 |
| 1 → 1 (stayed fraud) | 7 | 3 | 4 |

All flips are in the same direction (1 → 0). Three correctly predicted
fraud cases became false negatives. No improvements over the baseline.

Failure mode breakdown:

| Mode | Improved | Degraded | Unchanged |
|------|----------|----------|-----------|
| M2 | 0 | 1 | 5 |
| M3 | 0 | 0 | 3 |
| M7 | 0 | **3** | 3 |

## Comparison with EXP-A-0003 (calculator-only ReAct, same model, dev eval set)

| Metric | EXP-A-0003 (calculator-only) | EXP-A-0004 (retrieval ReAct) |
|--------|------------------------------|------------------------------|
| Accuracy  | 0.583 | 0.417 |
| Precision | 0.545 | 0.429 |
| Recall    | 1.000 | 0.500 |
| F1        | 0.706 | 0.462 |
| ROC-AUC   | 0.472 | 0.361 |
| MCC       | 0.302 | -0.169 |

Flip analysis (EXP-A-0003 → EXP-A-0004):

| Transition | Count | Correct after | Incorrect after |
|-----------|-------|---------------|-----------------|
| 0 → 0 | 1 | 1 | 0 |
| 1 → 0 (flip to non-fraud) | 4 | 1 | 3 |
| 1 → 1 | 7 | 3 | 4 |

One improvement (`S100G7X4`, label=0, A-0003 false positive corrected) and
three degradations (all M7 — correctly flagged fraud cases lost). Net
negative on F1.

## Key observations

1. **Recall collapse.** Both baselines (R-0002 single-call, A-0003 calc-only
   ReAct) achieve Recall = 1.000 on this dev set. EXP-A-0004 falls to
   Recall = 0.500 — the model now misses half the fraud cases. All
   regressions are 1 → 0 flips; no 0 → 1 flips occurred. The retrieval
   architecture made the model **systematically more conservative**.

2. **Loss of cross-attribute visibility.** Trace inspection of the three
   missed-fraud cases (`S100AR6V`, `S100DHNL`, `S100HMPE`) shows the
   baseline R-0002 identified co-occurring anomalies — e.g., for
   `S100HMPE`: "intangibles +421% AND new debt appearing AND massive
   buybacks AND non-controlling interests emerging" — and read the
   *pattern* as fraud. EXP-A-0004 retrieved a subset of these attributes
   individually and never had them in front of it simultaneously. The
   field-level interface decomposed away the holistic view that the
   inline-data baseline gets by default.

3. **CPA anchor activates as a borderline tie-breaker, not as universal
   dismissal.** Three of three FN cases cite the CPA-verification line
   from the upstream `base_prompt` to dismiss red flags. But three other
   cases also cite CPA and still predict fraud (`S100OKNK`, `S100IUNN`
   as FP; `S100BF5L` as TP). Comparing the reasoning text, both groups
   use the same `While CPA verified, but pattern suggests...` rhetorical
   construction; the difference is which side wins:
   - **Dismissal (FN) cases** consistently land on the CPA side when the
     surfaced patterns are subtle or borderline (e.g., `S100AR6V`: NI
     +52.5% vs OCF −41.1%, receivables +12.4% vs sales −1.1%; conclusion:
     "could also be explained by legitimate business factors... CPA
     verification suggests the report is not fraudulent").
   - **Non-dismissal cases** land against CPA when the surfaced patterns
     are dramatic (e.g., `S100OKNK`: NI down 60% with negative OCF and
     "9x divergence between NI and OCF"; conclusion: "the logical
     patterns and operational metrics strongly suggest... potential
     fraudulent earnings recognition").

   The CPA anchor does not act as a universal dismissal switch — it
   activates as a tie-breaker in cases the model judges to be ambiguous,
   consistently breaking those ties toward "not fraud." R-0002 reads the
   same `base_prompt` and rarely engages this construction in its
   reasoning at all.

4. **Heavy computation without benefit.** The model averages 70.9 tool
   calls per example, with calculator usage roughly matching `get_attribute`
   usage (403 vs 400 across the run). The upstream prompt's directive to
   "focus on non-numerical inconsistencies" does not restrain the model's
   arithmetic behavior — the model computes extensively regardless.

5. **Cost vs baselines.** EXP-A-0004 costs \$1.14 vs ~\$0.36 for the
   single-call baseline (R-0002 on N=50, prorated ~\$0.09 on N=12) —
   roughly 13× per example. EXP-A-0003 on the same dev set was \$0.38.
   The retrieval architecture is not just worse on F1; it is
   substantially more expensive.

## Q1 sub-investigation: prompt-framing sensitivity

A puzzle arose during early pilot runs of EXP-A-0004: two runs on the same
example produced opposite predictions under small prompt-level differences.
A four-experiment sub-investigation (`experiments/EXP-A-0004/q1-prompt-framing/`)
isolated each prompt-level variable that differed between the two pilots
and ran each in turn — including the exact configuration of the pilot that
predicted 0. None of the controlled experiments reproduced the original
flip. Six of seven runs of the same example predicted 1, including the run
that exactly matched the original "flipping" configuration.

The most likely reading: the original flip was within-condition variance
under nominally `temperature=0` inference, a documented property of
production LLM APIs due to lack of batch invariance (Anthropic glossary;
He et al. 2025, "Defeating Nondeterminism in LLM Inference"). For
tool-using agents with long trajectories, small numerical divergences early
in the loop can cascade into materially different sequences of tool calls
and final outputs.

This caveat applies to the main results above: per-example outcomes from a
single run should be read with this variance in mind. The N=12 aggregate
direction (recall collapse, F1 drop) is robust, but small per-example
differences should not be over-interpreted. Full investigation:
`experiments/EXP-A-0004/q1-prompt-framing/experiments.md`.

## Conclusion

The hypothesis — that field-level retrieval would address M2/M3 by letting
the model query numerical context selectively — is not supported on this
evaluation. The architecture instead amplified M7 (conservatism): the
model misses half the fraud cases, and all degradations are systematic
1 → 0 flips.

Trace evidence supports a compound mechanism in which the architectural
and prompt effects are not independent but compose:

- **Retrieval flattens pattern salience.** Field-level retrieval
  decomposes the data such that the model never sees co-occurring
  anomalies simultaneously. R-0002, with all data flat in the prompt,
  reads patterns like "intangibles +421% AND new debt appearing AND
  massive buybacks AND NCI emergence" as a *pattern*. EXP-A-0004
  retrieves a subset of these one at a time and writes about them as
  separate, individually-borderline facts.
- **CPA anchor activates as a tie-breaker in borderline cases.** When
  the surfaced anomalies are ambiguous, the upstream `base_prompt`'s
  "numbers are consistent and correct" framing breaks the tie toward
  "not fraud." When the surfaced anomalies look dramatic, the model
  overrides the anchor and predicts fraud regardless of CPA citation
  (see Observation 3 — 3 of 7 fraud predictions cite CPA but pivot
  against it).

These mechanisms compose rather than act in parallel. Retrieval
*expands* the category of "borderline" cases by flattening dramatic
co-occurring patterns into individually-subtle observations; the CPA
anchor then operates on this expanded category, breaking the ties
toward non-fraud. This composition pattern is consistent with the
observed asymmetry — recall collapse plus persistent FP rate — that
would not follow from either mechanism alone:

- "Loss of cross-attribute visibility" alone would degrade both recall
  and precision (the model would miss real patterns and also fail to
  flag false ones).
- "CPA dismissal" alone, if universal, would also lower FP (the model
  would dismiss apparent-but-not-real patterns the same way).

The architecture is upstream of the prompt effect. Removing the CPA line
alone would address only the second mechanism — the structural loss of
co-occurrence visibility would remain.

## Implications for future experiments

- **Field-level retrieval may be the wrong primitive for this task.** If
  retrieval is to be revisited, the trace evidence suggests interfaces that
  preserve cross-attribute visibility — e.g., returning groups of related
  attributes (entire sheets, or all line items changed >X% YoY) — would be
  more promising than per-attribute lookups.
- **Tool-using prompts likely need redesign for this task.** The upstream
  `base_prompt` was written for single-call inference and does not transfer
  cleanly to tool-using agents. Whether and how to rewrite it (without
  losing the intended focus on judgment over arithmetic) is itself an open
  question.
- **The calculator's role remains unclear.** Across A-0002, A-0003, and
  A-0004, calculator usage has not produced a measurable improvement over
  baselines. It is unlikely to be the right tool for the failure modes
  present in this dev set.

## Notes

- Branch: `exp-a-0004/react-retrieval`. Source under
  `src/agents/tool_augmented/` and `src/tools/retrieval.py`.
- The Q1 sub-investigation surfaced inference non-determinism as a real
  factor in interpreting any single-run result; methodologically, variance
  estimation should accompany any prompt-level ablation in future work.
- See `experiments/EXP-A-0004/plan.md` for the original design rationale
  and `experiments/EXP-A-0004/q1-prompt-framing/experiments.md` for the
  sub-investigation.
