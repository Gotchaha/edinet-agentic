# EXP-A-0004 — Pilot run observations under prompt change

Two consecutive pilot runs of EXP-A-0004 on the same example, with one
isolated change between them: the system prompt and the calculator tool's
docstring. All other variables (model, agent code, retrieval tools, upstream
`base_prompt`, sheets listing, recursion limit, temperature, eval example)
were held constant.

## Setting held constant across both runs

- Config: `configs/EXP-A-0004.yaml`
- Model: `claude-haiku-4-5-20251001`, `temperature=0.0`, `max_tokens=4096`
- Tools bound: `list_attributes`, `get_attribute`, `calculator`
- Retrieval tool docstrings: unchanged between runs
- Upstream `base_prompt` (`fraud_detection.yaml`): unchanged
- `sheets_listing` content: unchanged
- Example: `doc_id=S100LQX5`, `edinet_code=E00032`, `label=0` (the only
  example, via `--limit 1`, deterministically the first in the dev eval
  set ordering)
- Date: 2026-05-08

## What was changed between the two runs

### Change 1 — `src/agents/tool_augmented/prompts.py` (`SYSTEM_PROMPT`)

Run 1 (before):
```
You are a financial analyst. Think through your reasoning step by step. When you need to compute ratios, percentage changes, or other derived metrics, use the calculator tool to ensure accuracy.
```

Run 2 (after):
```
You are a financial analyst. Reason through the data carefully before concluding.
```

### Change 2 — `src/tools/calculator.py` (tool docstring)

Run 1 (before):
```
Evaluate an arithmetic expression and return the result.

Use this to verify numerical claims: ratios, percentage changes,
differences, etc. Examples:
- calculator("(159702000000 - 165215000000) / 165215000000 * 100")
- calculator("7468000000 / 94364000000")
```

Run 2 (after):
```
Compute a financial metric or signal from numerical values.

Use this for ratios, percentage changes, growth rates, year-over-year
differences, margins, sums, or other derived figures based on the values
you have retrieved. Examples:
- calculator("(159702000000 - 165215000000) / 165215000000 * 100")
- calculator("7468000000 / 94364000000")
```

## Aggregate metrics

| Metric | Run 1 | Run 2 |
|---|---|---|
| Wall-clock seconds | 204.3 | 35.5 |
| Input tokens | 221,098 | 52,171 |
| Output tokens | 8,686 | 4,992 |
| Estimated cost (USD) | $0.2645 | $0.0771 |
| Total messages in trace | 157 | 80 |
| Agent (assistant) turns | 23 | 7 |
| `list_attributes` calls | 4 | 4 |
| `get_attribute` calls | 59 | 38 |
| `calculator` calls | 70 | 28 |
| Total tool calls | 133 | 70 |
| Parse failures | 0 | 0 |
| Recursion errors | 0 | 0 |

## Final prediction

| | Label | Prediction | Prob |
|---|---|---|---|
| Run 1 | 0 | 0 | 0.35 |
| Run 2 | 0 | 1 | 0.78 |

## Final reasoning text (verbatim)

### Run 1

> The financial statements are mathematically consistent and properly
> structured, verified by a CPA. However, significant red flags exist
> regarding earnings quality: (1) Special gains increased 486% to represent
> 41.4% of pre-tax income, driven primarily by a 20,290% surge in
> investment securities sales (from 20M to 4,078M); (2) Net income
> increased 68.5% while operating profit declined 35.6%, indicating
> earnings driven by non-operating activities; (3) Operating cash flow
> declined 28.1% despite higher net income, suggesting a divergence between
> reported earnings and cash generation; (4) Equity surged 21%
> significantly outpacing asset growth of 4.1%. While these patterns
> suggest earnings management through timing of special gains rather than
> outright fraud, they indicate potential manipulation of reported
> profitability and raise concerns about earnings sustainability and
> quality. The reliance on one-time investment gains to boost net income
> is a classic warning sign.

### Run 2

> Multiple red flags indicate potential fraud: (1) Revenue declined 3.34%
> while net income surged 68.54%, which is highly suspicious; (2)
> Operating profit fell 35.7% while net income increased 68.5%, suggesting
> earnings manipulation; (3) Special gains jumped 485% from ¥701M to
> ¥4.1B, appearing to be the primary driver of net income growth; (4)
> Operating cash flow declined 28.1% while net income increased 68.5%,
> indicating divergence between reported earnings and actual cash
> generation; (5) Investment cash flow reversed from negative to positive,
> coinciding with special gains spike; (6) Equity ratio spiked 16.7%
> without corresponding operational improvement; (7) Cash position
> increased 40.3% despite declining operating cash flow. These patterns
> are consistent with earnings manipulation through special gains and
> asset sales rather than genuine operational improvement.

## Trace structure (per-turn outline)

### Run 1 — 23 assistant turns

The per-turn breakdown below is reconstructed from a one-time terminal
capture during Run 1's inspection and may contain small counting errors
in individual rows. The aggregates in the metrics table above are
authoritative (sourced from a `Counter` over all tool calls in the trace
prior to overwrite).

| Turn | Tools called (count by type) |
|---|---|
| 1 | list_attributes × 4 |
| 2 | get_attribute × 8 |
| 3 | get_attribute × 10 |
| 4 | get_attribute × 10 |
| 5 | calculator × 9 |
| 6 | calculator × 3, get_attribute × 3 |
| 7 | calculator × 3, get_attribute × 5 |
| 8 | calculator × 4, get_attribute × 1 |
| 9 | calculator × 3, get_attribute × 2 |
| 10 | calculator × 5 |
| 11 | calculator × 2, get_attribute × 4 |
| 12 | calculator × 2, get_attribute × 3 |
| 13 | calculator × 2, get_attribute × 4 |
| 14 | calculator × 4 |
| 15 | calculator × 2, get_attribute × 3 |
| 16 | calculator × 4 |
| 17 | calculator × 7 |
| 18 | calculator × 6 |
| 19 | calculator × 6 |
| 20 | calculator × 3, get_attribute × 3 |
| 21 | calculator × 4 |
| 22 | calculator × 2, get_attribute × 2 |
| 23 | (final answer) |

### Run 2 — 7 assistant turns

| Turn | Tools called (count by type) |
|---|---|
| 1 | list_attributes × 4 |
| 2 | get_attribute × 8 |
| 3 | get_attribute × 13 |
| 4 | get_attribute × 8 |
| 5 | calculator × 12 |
| 6 | get_attribute × 9 |
| 7 | calculator × 9 |
| 8 | calculator × 7 |
| 9 | (final answer) |

(Run 2's 7 turns refers to assistant turns that issue tool calls; turn 9
is the final answer turn with no tool calls.)
