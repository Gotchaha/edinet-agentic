# EXP-A-0002: Tool-Augmented Verification Agent

## Motivation

EXP-A-0001 (reflection baseline) showed that intrinsic self-correction degrades
performance: the critic produced subjective prose ("overstate", "overreaction"),
the reviser deferred to it, and 21/50 correct predictions flipped to incorrect.
Literature confirms this pattern — Huang et al. (2024) show that LLMs cannot
self-correct reasoning without external feedback, and the Feedback Friction paper
(Jiang et al., NeurIPS 2025) characterizes how models resist incorporating even
correct feedback.

The core idea: replace subjective critique with **tool-grounded factual
verification**. If the verifier uses a calculator and gets "actual change = -3.3%",
that's a fact the reviser can act on — not an opinion it can defer to or dismiss.

## Hypothesis

A verification agent equipped with a calculator tool will reduce M3 (magnitude
insensitivity) errors by enabling accurate numerical verification, and will reduce
M7 (conservatism) degradation compared to EXP-A-0001 by replacing subjective
critique with structured factual feedback.

## Architecture

```
Generate (initial fraud analysis + prediction)
    ↓
Verify (LLM with tools, agentic loop)
    ├─ Identify numerical claims in the analysis
    ├─ Call calculator to recompute ratios/changes
    ├─ ... (loop: more tool calls as needed)
    ├─ Call submit_verification with structured report
    ↓
Revise (receives structured verification, not prose)
    ├─ Incorporate factual corrections
    ├─ Re-assess prediction based on verified evidence
    ↓
Final prediction
```

Key differences from EXP-A-0001 baseline:
- Verifier uses **tools** (calculator) to produce factual checks, not subjective judgments
- Verification output is **structured** (claim-by-claim), not free-form prose
- The reviser receives facts ("your claimed ratio is wrong, actual = X"), not opinions

## Tool design

Tool implementations live in `src/tools/` as a shared library across experiments.
The verifier has both tools available; the LangGraph router checks which tool was
called to decide whether to loop (calculator) or proceed (submit_verification).

### Calculator (`src/tools/calculator.py`)

Evaluates arithmetic expressions using `simpleeval` (AST-based whitelisting,
not `eval()`). The model reads financial values from context (e.g.,
`165215000000` and `159702000000`), constructs an expression, and the tool
returns the precise result. Calling this tool loops back for more verification.

Addresses **M3 (magnitude insensitivity)** — the model cannot reliably compute
ratios, percentage changes, or compare magnitudes of large numbers.

### Submit verification (`src/tools/submit_verification.py`)

Accepts a structured verification report as its argument and exits the verification
loop, transitioning to the reviser. The report schema is enforced through the
tool's parameter definition, which avoids the conflict of using
`with_structured_output()` alongside tool calling (the former forces every model
response to conform to the output schema, preventing intermediate calculator calls).

```json
{
  "checks": [
    {
      "claim": "Revenue declined 15% year-over-year",
      "source": {"sheet": "pl", "field": "売上高"},
      "status": "corrected",
      "detail": "Actual change: (159702000000 - 165215000000) / 165215000000 * 100 = -3.34%"
    }
  ],
  "red_flags_confirmed": 2,
  "red_flags_refuted": 1
}
```

## Design decisions

- **Model:** `claude-haiku-4-5-20251001`, same as EXP-A-0001 for fair comparison.
  Haiku 4.5 fully supports tool use via langchain-anthropic `bind_tools()`.
  Note: Anthropic docs indicate Haiku can be eager to call tools — tool
  descriptions should be precise.
- **Sheets:** `[summary, bs, pl, cf]` (no text), same as EXP-A-0001.
- **Calculator safety:** Use `simpleeval` (AST-based whitelist), not Python
  `eval()`. LangChain's historical approach (`numexpr`) is also unsafe as it
  calls `eval()` internally.
- **No artificial loop cap:** The verifier checks numerical claims from the
  generator's analysis (a finite set), then calls `submit_verification`.
  Cost is tracked per-example; a cap can be added later if needed.

## Implementation

### Files to create

| File | Purpose |
|------|---------|
| `src/tools/__init__.py` | Tools package |
| `src/tools/calculator.py` | Arithmetic evaluation tool |
| `src/tools/submit_verification.py` | Structured verification report tool |
| `src/agents/tool_augmented/__init__.py` | Agent package |
| `src/agents/tool_augmented/graph.py` | LangGraph: generate → verify (tool loop) → revise |
| `src/agents/tool_augmented/prompts.py` | System prompts with structured citation requirements |
| `src/agents/tool_augmented/state.py` | Agent state |
| `scripts/EXP-A-0002/run.py` | Experiment runner |
| `configs/EXP-A-0002.yaml` | Experiment config |

### Steps

1. Implement calculator tool in `src/tools/calculator.py`
2. Build tool-augmented agent in `src/agents/tool_augmented/`
3. Write experiment runner and config
4. Run on dev eval set (N=12) using EVAL-0001 pipeline
5. Evaluate and compare against EXP-A-0001 (reflection baseline)
6. Iterate on prompts/tools based on dev eval results
7. Full benchmark run (N=50) if dev eval looks promising

## Evaluation

Use EVAL-0001 pipeline throughout:

- Dev iteration: `uv run python scripts/EVAL-0001/evaluate.py <results>.jsonl --eval-set dev`
- Comparison against EXP-A-0001:
  `uv run python scripts/EVAL-0001/evaluate.py <results>.jsonl --baseline outputs/EXP-A-0001/claude-haiku-4-5-20251001/results.jsonl`
- Full N=50 benchmark for final numbers
