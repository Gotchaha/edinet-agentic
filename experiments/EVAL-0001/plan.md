# EVAL-0001: End-to-End Evaluation Pipeline

## Purpose

Standardized harness for comparing agents against each other and against
single-call baselines. All agent experiments (EXP-A-XXXX) reference which
eval version produced their results (e.g., "evaluated with EVAL-0001").

This is infrastructure, not a hypothesis test.

## Two-tier eval design

### Dev eval (N=10–15)

For rapid iteration during agent development.

- **Composition:** Hand-picked from the frozen N=50 sample, guided by
  EXP-D-0001 failure classifications:
  - 3–4 M7 (conservatism) cases
  - 3–4 M2 (evidence drift) cases
  - 2–3 M3 (magnitude insensitivity) cases
  - 3–4 correct-baseline cases (moderate confidence, not trivial)
- **Selection principle:** Representative, not extreme. Pick typical
  failure-mode examples so improvements generalize. For correct cases,
  pick moderate-confidence predictions (prob 0.3–0.7) most vulnerable
  to regression — not trivially obvious cases.
- **Cost target:** ~$0.40–0.60 per reflection-agent run
- **Frozen as:** a JSON file under `experiments/EVAL-0001/`

### Benchmark eval (N=50)

For final comparison when a design looks promising on dev eval.

- **Composition:** The existing frozen sample at
  `reproduction/sampling/fraud_detection_n50_seed42.json`
- **Usage:** Run only when dev eval results warrant a full evaluation
- **Produces:** The numbers that go in experiment summaries and
  research log entries

## Evaluation interface

A single script: `scripts/EVAL-0001/evaluate.py`

**Input:** Any `results.jsonl` (from any agent or single-call baseline)

**Output:** Standardized report containing:
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC, MCC
  with bootstrap CIs (matching EXP-R-0002/0003 summary format)
- Flip analysis: prediction transitions from generator → final output,
  with correct/incorrect breakdown
- Cost summary: total tokens, estimated cost, cost per example

**Comparison mode:** Optionally takes a second `results.jsonl` to produce
a side-by-side comparison table.

**Works for both tiers:** Dev eval and benchmark eval use the same script,
just different sample sizes in the input.

## What this eval does NOT cover

- **LLM-as-judge:** Not needed — we have per-example ground truth labels
- **Reasoning quality grading:** Tempting but out of scope; stick to
  outcome-based grading
- **Component-level eval:** Lives inside EXP-A-XXXX experiments, not here
- **pass^k reliability:** Deferred to future EVAL versions if needed

## Artifacts

| File | Purpose | Tracked? |
|------|---------|----------|
| `experiments/EVAL-0001/plan.md` | This document | Yes |
| `experiments/EVAL-0001/dev_eval_set.json` | Frozen dev eval sample | Yes |
| `scripts/EVAL-0001/evaluate.py` | Evaluation script | Yes |
| Evaluation reports | Generated per-run | Depends on context |

## Versioning

Changing the sample, metrics, or coverage = new EVAL entry (EVAL-0002).
Bug fixes to the script within the same methodology: update in place.
