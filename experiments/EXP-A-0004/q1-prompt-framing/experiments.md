# Q1 — Experiments

## Experiment 1: System prompt addendum, isolated

### Hypothesis

The calculator-verify addendum in the system prompt — the sentence
"When you need to compute ratios, percentage changes, or other derived
metrics, use the calculator tool to ensure accuracy." — is the cause
of the prediction flip observed between the two earlier pilot runs of
EXP-A-0004 (recorded in `pilot_observations.md`).

### Setting

Two runs on the same example (`doc_id=S100LQX5`, `label=0`, the first
example in the dev eval set, selected via `--limit 1`), differing only
in the system prompt. All other settings — model
(`claude-haiku-4-5-20251001`, `temperature=0.0`, `max_tokens=4096`),
retrieval tools, calculator docstring (current derivation-framing
version), upstream `base_prompt`, sheets listing, recursion limit —
were held constant.

The two prompts:

- **Prompt A**:
  > You are a financial analyst. Reason through the data carefully before concluding.
- **Prompt B** (Prompt A + the calculator-verify addendum sentence
  appended):
  > You are a financial analyst. Reason through the data carefully before concluding. When you need to compute ratios, percentage changes, or other derived metrics, use the calculator tool to ensure accuracy.

Outputs are at `outputs/EXP-A-0004/q1-prompt-framing/exp-1/prompt-A/`
and `.../prompt-B/`, each containing `results.jsonl`, `traces.jsonl`,
and `run_meta.json`.

### Result

Both runs produced the **same prediction (`1`)**. The hypothesis is
falsified: the system prompt addendum alone, with the calculator
docstring held at the current derivation-framing version, did not flip
the prediction back to `0`.

Side-by-side metrics:

| | Prompt A | Prompt B |
|---|---|---|
| prediction | 1 | 1 |
| prob | 0.72 | 0.62 |
| label | 0 | 0 |
| agent turns | 9 | 8 |
| `list_attributes` calls | 4 | 4 |
| `get_attribute` calls | 36 | 27 |
| `calculator` calls | 26 | 20 |
| total tool calls | 66 | 51 |
| input tokens | 50,706 | 39,900 |
| output tokens | 4,794 | 3,724 |
| estimated cost (USD) | 0.0747 | 0.0585 |
| wall-clock (s) | 33.7 | 27.2 |
| parse failures | 0 | 0 |
| recursion errors | 0 | 0 |

Observations from this experiment:

1. Both runs predicted `1` on the same example. Adding the addendum to
   the system prompt did not flip the prediction.
2. The addendum lowered `prob` from 0.72 to 0.62 — a directional
   shift toward less-fraud, but not enough to cross the 0.5 decision
   threshold.
3. The addendum reduced total tool calls (66 → 51), including
   calculator calls (26 → 20), despite explicitly directing the
   model to use the calculator.

## Experiment 2: Calculator docstring, isolated

### Hypothesis

The calculator docstring change is the cause of the prediction flip
observed between the two earlier pilot runs of EXP-A-0004 (recorded in
`pilot_observations.md`).

### Setting

One new run on the same example (`doc_id=S100LQX5`, `label=0`, selected
via `--limit 1`), differing from Experiment 1's Prompt A run only in
the calculator's docstring. All other settings — model
(`claude-haiku-4-5-20251001`, `temperature=0.0`, `max_tokens=4096`),
retrieval tools, system prompt (Variant A from Experiment 1), upstream
`base_prompt`, sheets listing, recursion limit — were held constant.

The two docstrings (description text shown; the two example calls in
each are identical and omitted here):

- **Current (derivation-framing)**:
  > Compute a financial metric or signal from numerical values.
  >
  > Use this for ratios, percentage changes, growth rates, year-over-year differences, margins, sums, or other derived figures based on the values you have retrieved.
- **Previous (verify-framing)**:
  > Evaluate an arithmetic expression and return the result.
  >
  > Use this to verify numerical claims: ratios, percentage changes, differences, etc.

Output for the new run is at
`outputs/EXP-A-0004/q1-prompt-framing/exp-2/docstring-verify/`,
containing `results.jsonl`, `traces.jsonl`, and `run_meta.json`.

### Result

The new run produced **prediction `1`**, the same as the Exp 1 Prompt A
baseline. The hypothesis is falsified: the docstring change alone, with
the system prompt held at Variant A, did not flip the prediction back
to `0`.

Side-by-side metrics:

| | Exp 1 Prompt A (baseline) | Exp 2 docstring-verify |
|---|---|---|
| prediction | 1 | 1 |
| prob | 0.72 | 0.78 |
| label | 0 | 0 |
| agent turns | 9 | 14 |
| `list_attributes` calls | 4 | 4 |
| `get_attribute` calls | 36 | 41 |
| `calculator` calls | 26 | 21 |
| total tool calls | 66 | 66 |
| input tokens | 50,706 | 83,250 |
| output tokens | 4,794 | 4,776 |
| estimated cost (USD) | 0.0747 | 0.1071 |
| wall-clock (s) | 33.7 | 37.7 |
| parse failures | 0 | 0 |
| recursion errors | 0 | 0 |

Observations from this experiment:

1. Switching to the verify-framing docstring did not flip the
   prediction; both runs predicted `1` on the same example.
2. `prob` rose slightly (0.72 → 0.78) — a directional shift toward
   more-fraud, opposite the direction the hypothesis would predict if
   verify-framing pulled the model toward "the numbers are correct, do
   not over-flag."
3. Total tool calls were identical (66 vs 66), but the mix shifted:
   more `get_attribute` (36 → 41), fewer `calculator` (26 → 21),
   despite the verify-framing docstring explicitly directing the model
   to use the calculator to verify numerical claims.
4. Agent turns rose from 9 to 14, driving input tokens from 50,706 to
   83,250 (each turn re-sends the conversation), while output tokens
   were essentially unchanged (4,794 vs 4,776).

## Experiment 3: System prompt addendum and calculator docstring, jointly

### Hypothesis

The combination of the calculator-verify addendum in the system prompt
AND the verify-framing calculator docstring is the cause of the
prediction flip observed between the two earlier pilot runs of
EXP-A-0004 (recorded in `pilot_observations.md`).

### Setting

One new run on the same example (`doc_id=S100LQX5`, `label=0`, selected
via `--limit 1`), differing from Experiment 1's Prompt A run in
**both** the system prompt (Variant B from Experiment 1, with the
calculator-verify addendum appended) and the calculator's docstring
(verify-framing version from Experiment 2). All other settings — model
(`claude-haiku-4-5-20251001`, `temperature=0.0`, `max_tokens=4096`),
retrieval tools, upstream `base_prompt`, sheets listing, recursion
limit — were held constant.

This is the joint condition tested in Pilot 1 of EXP-A-0004, with one
caveat: the CoT phrasing in the system prompt differs from Pilot 1's
original. Pilot 1 used "Think through your reasoning step by step.";
Variant B uses "Reason through the data carefully before concluding."
(introduced as part of the post-pilot prompt cleanup). All other
prompt content matches Pilot 1.

The configuration:

- **System prompt** (Variant B):
  > You are a financial analyst. Reason through the data carefully before concluding. When you need to compute ratios, percentage changes, or other derived metrics, use the calculator tool to ensure accuracy.
- **Calculator docstring** (verify-framing, description text shown;
  the two example calls are identical to the current docstring and
  omitted here):
  > Evaluate an arithmetic expression and return the result.
  >
  > Use this to verify numerical claims: ratios, percentage changes, differences, etc.

Output for the new run is at
`outputs/EXP-A-0004/q1-prompt-framing/exp-3/prompt-B-docstring-verify/`,
containing `results.jsonl`, `traces.jsonl`, and `run_meta.json`.

### Result

The new run produced **prediction `1`**, the same as the Exp 1 Prompt A
baseline. The hypothesis is falsified: the joint condition (both
addendum and verify-framing docstring applied), with the CoT phrasing
held at Variant B's wording, did not flip the prediction back to `0`.

Side-by-side metrics:

| | Exp 1 Prompt A (baseline) | Exp 3 (joint) |
|---|---|---|
| addendum | — | ✓ |
| verify-framing docstring | — | ✓ |
| prediction | 1 | 1 |
| prob | 0.72 | 0.72 |
| label | 0 | 0 |
| agent turns | 9 | 16 |
| `list_attributes` calls | 4 | 4 |
| `get_attribute` calls | 36 | 40 |
| `calculator` calls | 26 | 31 |
| total tool calls | 66 | 75 |
| input tokens | 50,706 | 103,490 |
| output tokens | 4,794 | 5,189 |
| estimated cost (USD) | 0.0747 | 0.1294 |
| wall-clock (s) | 33.7 | 64.6 |
| parse failures | 0 | 0 |
| recursion errors | 0 | 0 |

Observations from this experiment:

1. Both runs predicted `1` on the same example. Applying both changes
   jointly (with Variant B's CoT phrasing) did not flip the prediction.
2. `prob` is identical between the two runs (0.72 in both).
3. Total tool calls rose from 66 to 75; both `get_attribute` (36 → 40)
   and `calculator` (26 → 31) call counts increased.
4. Agent turns rose from 9 to 16, driving input tokens from 50,706 to
   103,490 (each turn re-sends the conversation), while output tokens
   were similar (4,794 vs 5,189).

## Experiment 4: CoT phrasing in the system prompt

### Hypothesis

The original CoT phrasing in the system prompt — "Think through your
reasoning step by step." (used in Pilot 1 but replaced with "Reason
through the data carefully before concluding." in subsequent
experiments) — is the cause of the prediction flip observed in Pilot 1
of EXP-A-0004 (recorded in `pilot_observations.md`).

### Setting

One new run on the same example (`doc_id=S100LQX5`, `label=0`, selected
via `--limit 1`), differing from Experiment 3 only in the CoT phrasing
of the system prompt. With this change, the configuration matches
Pilot 1's exact setting. All other settings — model
(`claude-haiku-4-5-20251001`, `temperature=0.0`, `max_tokens=4096`),
retrieval tools, calculator docstring (verify-framing version, same as
Experiment 3), upstream `base_prompt`, sheets listing, recursion
limit — were held constant.

The configuration:

- **System prompt** (Pilot 1 original):
  > You are a financial analyst. Think through your reasoning step by step. When you need to compute ratios, percentage changes, or other derived metrics, use the calculator tool to ensure accuracy.
- **Calculator docstring**: verify-framing version, identical to
  Experiment 3 (description text omitted here for brevity; see
  Experiment 3's Setting).

Output for the new run is at
`outputs/EXP-A-0004/q1-prompt-framing/exp-4/cot-think/`, containing
`results.jsonl`, `traces.jsonl`, and `run_meta.json`.

### Result

The new run produced **prediction `1`** with `prob=0.72`, the same as
Experiment 3. The hypothesis is falsified: restoring Pilot 1's original
CoT phrasing did not flip the prediction back to `0`.

Side-by-side metrics (Experiment 4 vs Experiment 3 — only the CoT
phrasing differs between them):

| | Exp 3 (Reason-CoT) | Exp 4 cot-think (Think-CoT) |
|---|---|---|
| CoT phrasing | "Reason through the data carefully before concluding." | "Think through your reasoning step by step." |
| addendum | ✓ | ✓ |
| verify-framing docstring | ✓ | ✓ |
| prediction | 1 | 1 |
| prob | 0.72 | 0.72 |
| label | 0 | 0 |
| agent turns | 16 | 14 |
| `list_attributes` calls | 4 | 4 |
| `get_attribute` calls | 40 | 35 |
| `calculator` calls | 31 | 34 |
| total tool calls | 75 | 73 |
| input tokens | 103,490 | 88,037 |
| output tokens | 5,189 | 5,174 |
| estimated cost (USD) | 0.1294 | 0.1139 |
| wall-clock (s) | 64.6 | 47.7 |
| parse failures | 0 | 0 |
| recursion errors | 0 | 0 |

Observations from this experiment:

1. Both runs predicted `1` with `prob=0.72` on the same example.
   Restoring Pilot 1's original CoT phrasing did not flip the
   prediction.
2. Although Experiment 4's configuration matches Pilot 1's exact
   setting (same model, temperature, example, system prompt, and
   calculator docstring), Experiment 4 predicted `1` with
   `prob=0.72` while Pilot 1 predicted `0` with `prob=0.35` (recorded
   in `pilot_observations.md`).
3. Total tool calls were similar (75 vs 73). The mix shifted slightly:
   fewer `get_attribute` (40 → 35), more `calculator` (31 → 34).
4. Agent turns dropped from 16 to 14; input tokens dropped from
   103,490 to 88,037; output tokens were essentially unchanged
   (5,189 vs 5,174).

## Conclusion

Across Experiments 1–4, every prompt-level variable that differed
between Pilots 1 and 2 was isolated and tested individually or jointly:
the calculator-verify addendum (Exp 1), the calculator's docstring
framing (Exp 2), both jointly (Exp 3), and the original CoT phrasing on
top of the joint condition — i.e., Pilot 1's exact configuration
(Exp 4). None of these reproduced the prediction `0` observed in
Pilot 1. Six of seven runs on the same example, including the run that
exactly matched Pilot 1's prompt and tool descriptions, predicted `1`.

The most likely reading of this is that Pilot 1's prediction of `0` was
within-condition variance under nominally `temperature=0` inference,
rather than a systematic effect of the prompt-level differences we
initially attributed it to. Anthropic's own documentation states that
even at `temperature=0`, identical inputs may produce different outputs
across API calls. The technical literature attributes this to a lack
of batch invariance in production LLM inference: server-side batch
composition varies with concurrent load, and several core kernels
produce numerically different outputs depending on batch size, which
can shift logits enough to change the argmax token and cascade through
subsequent generation. For tool-using agents with long trajectories, a
single token-level divergence early in the loop can fan out into
materially different sequences of tool calls and final outputs.

We cannot rule out smaller real effects of the prompt variants —
directional shifts in `prob` and changes in trajectory length and
tool-call mix were observed even where the prediction did not flip —
and we have not estimated the variance directly. A follow-up that
replicates one configuration several times under unchanged conditions
would convert this reading from "consistent with the public technical
record" to "directly observed in our setup."

The practical takeaway for the broader EXP-A-0004 evaluation: on this
example, prompt-level changes within reasonable bounds do not appear
to produce reliable shifts in the agent's prediction. Per-example
outcomes from a single run should be read with this variance in mind.

References:

- Anthropic — Glossary, Temperature section: <https://platform.claude.com/docs/en/about-claude/glossary>
- He et al., "Defeating Nondeterminism in LLM Inference," Thinking Machines Lab, 2025-09-10: <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>
- Yuan et al., "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference," arXiv:2506.09501, 2025 preprint: <https://arxiv.org/abs/2506.09501>
