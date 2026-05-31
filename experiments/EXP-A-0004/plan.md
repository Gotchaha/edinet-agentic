# EXP-A-0004: ReAct Agent with Field-Level Retrieval

## Motivation

EXP-A-0003 (ReAct + calculator) showed that 11/12 predictions matched the
single-call baseline — adding a calculator did not change behavior on the
dev eval set. The dominant failure modes documented in EXP-D-0001 — M2
(evidence drift) and M3 (magnitude insensitivity) — were present in
A-0003's outputs as well.

The literature points to two converging issues that are *not* addressed by
giving the model a calculator:

1. **Lost-in-the-middle for numerical content.** Liu et al. (TACL 2024) and
   the GM-Extract follow-up (Gupte et al., 2025 preprint) document that LLMs
   degrade on positional retrieval in long contexts, and that financial
   documents — dense with structurally similar numbers acting as distractors
   — are particularly susceptible.
2. **Numerical reasoning fragility.** GSM-Symbolic (Mirzadeh et al., ICLR
   2025) shows up to 65% performance drop from a single irrelevant numerical
   clause, indicating that LLMs latch onto distractors rather than performing
   genuine reasoning.

Both point in the same direction: a setup in which the model selects what
numerical content to look at — rather than passively scanning a wall of
numbers in the prompt — is worth testing.

## Hypothesis

A ReAct agent with field-level retrieval tools (replacing inline numerical
data in the prompt) may reduce M3 (magnitude insensitivity) and M2 (evidence
drift) by allowing the model to query numerical context selectively, with
prior-period values inline in each retrieved attribute. Whether this helps
in practice — and on which examples — is what the experiment is intended
to determine.

## Architecture

Standard ReAct loop, identical structure to EXP-A-0003:

```
START → agent ⇄ tools → END
         ↑       ↓
         └───────┘
```

The architectural change vs. A-0003 is in the **tool surface**:

- A-0003: numerical data inline in the user message; calculator is the only tool.
- A-0004: numerical data behind tools; the user message contains only sheet
  names; the agent must explicitly retrieve to see values.

## Tool design

Three tools, all field-level, all using closure capture over the example.
The data in `bs/pl/cf/summary` is JSON-serialized in the dataset; tools parse
once per example and serve queries from the parsed dict.

### `list_attributes(sheet: str) → str`

Returns a JSON-serialized list of all attribute names in the given sheet.
Sheets: `summary`, `bs` (balance sheet), `pl` (profit & loss), `cf` (cash
flow). Used for schema discovery before targeted lookups.

### `get_attribute(sheet: str, name: str) → str`

Returns the raw structure under `data[sheet][name]` as JSON, passing through
without re-shaping. The structure is a dict mapping period names
(`CurrentYear`, `Prior1Year`, `Prior2Year`) to integer values represented as
strings (e.g., `"165215000000"`). Missing values use the upstream sentinel
`"－"`.

If the name is not found, returns a JSON error object — without revealing
available attributes — so the model can fall back to `list_attributes`.

This is the central new tool. Returning the full slice (current + prior
periods inline) lets the model see temporal context in a single call,
intended to make M3-relevant context (period-over-period magnitudes)
cheaper to access than computing it across multiple inline-data scans.

### `calculator(expression: str) → str`

Carried over from A-0003 unchanged. No longer the centerpiece — described to
the model as available for derived metrics, not as the primary reasoning
substrate.

## Prompts

The system prompt and the upstream `base_prompt` (from
`external/EDINET-Bench/prompt/fraud_detection.yaml`) are kept consistent
with EXP-A-0003. The only forced change is the structure of the user
message: A-0003 appended the JSON-serialized sheet contents after
`base_prompt`, while A-0004 cannot — values are no longer inline. In place
of the sheet contents, the user message includes a brief listing of which
sheet names are available (so the model knows what it can pass to
`list_attributes` / `get_attribute`).

This keeps prompt content as close to A-0003 as the design permits.
Whether prompt changes (e.g., reasoning-structure guidance) help is left
to a separate experiment.

## Design decisions

- **Eval set:** `experiments/EVAL-0001/dev_eval_set.json` (N=12, curated).
- **Model:** `claude-haiku-4-5-20251001` only (consistent with A-0003).
- **Sheets exposed:** `[summary, bs, pl, cf]` — same as upstream baseline,
  same as A-0001/A-0002/A-0003. No text field.
- **Termination policy:** LangGraph default `recursion_limit=25`, no
  preemptive bump. Pilot will surface if this is insufficient. Wrap
  `graph.invoke` in try/except for `GraphRecursionError` and record
  `prediction=None` for that example rather than crashing the run.
- **Data passthrough:** `"－"` is preserved as-is; system prompt tells the
  model what it means (matches upstream convention).
- **Schema discovery via tools.** The user prompt contains sheet *names*
  only; the model calls `list_attributes` to find the keys it wants before
  fetching values. The alternative — placing the full schema list in the
  user prompt — is rejected here so we can observe whether explicit
  discovery changes the model's querying behavior.
- **Reuse:** Modify `src/agents/tool_augmented/` in place per project
  convention (per-experiment source-of-truth via git history).

## Implementation

### Files to modify or create

| File | Status | Purpose |
|------|--------|---------|
| `src/tools/retrieval.py` | NEW | `list_attributes`, `get_attribute` (closure factory over example) |
| `src/agents/tool_augmented/graph.py` | MODIFY | Bind retrieval tools + calculator; add `GraphRecursionError` handling |
| `src/agents/tool_augmented/prompts.py` | KEEP | System prompt unchanged from A-0003 |
| `src/agents/tool_augmented/state.py` | MODIFY | Replace `sheets_text` (full sheet contents) with `sheets_listing` (sheet-name list); add `example` (raw dict) for tool closures |
| `scripts/EXP-A-0004/run.py` | NEW | Runner — same shape as A-0003 runner; `build_sheets_text` is replaced by a sheet-name listing |
| `configs/EXP-A-0004.yaml` | NEW | Config: same shape as A-0003, `tools: [list_attributes, get_attribute, calculator]` |

### Steps

1. Implement retrieval tools in `src/tools/retrieval.py` with closure
   factory pattern: `build_retrieval_tools(example) → [list_attrs, get_attr]`.
2. Update graph to bind closure-built tools; add `GraphRecursionError`
   try/except wrapper.
3. Write runner and config.
4. **Pilot:** run on first 1–2 examples (`--limit 2`); inspect traces for:
   - Did the model use tools? Or did it answer without querying?
   - Did it discover the schema (`list_attributes`) before querying?
   - Did it hit the recursion limit?
5. Adjust if needed (tool descriptions), document changes.
6. Main run on dev eval set (N=12).
7. Evaluate, write summary.

## Evaluation

Use EVAL-0001 pipeline:

- **Standalone metrics** (sanity check):
  ```
  uv run python scripts/EVAL-0001/evaluate.py \
    outputs/EXP-A-0004/claude-haiku-4-5-20251001/results.jsonl
  ```
- **vs single-call (EXP-R-0002 Haiku)** — does selective retrieval differ
  from passive long-context reading?
- **vs EXP-A-0003** — does adding retrieval tools change behavior beyond
  what the calculator alone produced? Cleaner isolation since the system
  prompt and `base_prompt` are held constant; the only architectural delta
  is the tool surface (and the corresponding user-message change forced by
  it).
- **Failure-mode breakdown** by classified errors from EXP-D-0001 (only the
  3 M7 + 3 M2 + 2 M3 cases in the dev eval set have classifications).

## Success criteria

A-0004 is informative regardless of headline accuracy. What we want to learn:

1. **Did the model use the tools?** Trace inspection: `>0` tool calls per
   example, with at least one `get_attribute` call.
2. **Did failure modes shift?** Even if accuracy is unchanged, a shift in
   *which kinds of errors* the model makes is a meaningful signal.
3. **Headline performance:** F1 ≥ A-0003 (0.706) is a baseline; F1 ≥
   single-call R-0002 (0.750) on the dev eval set would be a positive
   signal that the architecture helps.

If predictions are largely unchanged from A-0003 (mirroring A-0003's
relationship to single-call), that suggests the tool surface alone is not
moving behavior, and follow-up experiments should explore other variables
(e.g., reasoning structure via prompt design, or different tool
granularities).

## Risks

1. **Model ignores tools.** Frontier models sometimes "answer from memory"
   or refuse to call tools when confidence is high. Without inline sheet
   data, this should be hard, but the model could still hallucinate values.
   Mitigation: pilot trace inspection. If observed, sharpen tool
   descriptions to indicate that data is *only* available via tools.
2. **Schema discovery cost.** The model may over-query `list_attributes`
   or query irrelevant attributes. Mitigation: observe in pilot; if
   excessive, consider including the full schema (attribute list) in the
   user prompt and removing the discovery step.
3. **Japanese attribute names.** The schema list per sheet is 30–50
   Japanese terms. Frontier models handle this, but the model may produce
   English approximations and fail the lookup. Mitigation: error message
   semantics encourage fallback to `list_attributes`.
4. **Recursion limit hit.** With three tools and a setup that requires
   explicit retrieval, the agent may exceed 25 node visits. Mitigation:
   caught by try/except; pilot shows whether to bump.
