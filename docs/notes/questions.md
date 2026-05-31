# Research Questions

- template

```
### Q<N>: [Question]
- **Date:** YYYY-MM-DD
- **Origin:** Experiment ID and/or specific file/observation that raised this
- **Status:** open | investigating | resolved
- **Investigation:** Relative path to sub-folder, or "—" if answered inline
- **Context:** Brief background — what was observed, why this question matters
- **Answer:** Inline resolution if short; otherwise one-line summary plus pointer to the sub-folder's findings.
```

### Q1: Why did a seemingly minor change in the system prompt (and the calculator's docstring) flip the agent's prediction on the same example?
- **Date:** 2026-05-08
- **Origin:** `experiments/EXP-A-0004/q1-prompt-framing/pilot_observations.md`
- **Status:** resolved
- **Investigation:** `experiments/EXP-A-0004/q1-prompt-framing/`
- **Context:** Two pilot runs of EXP-A-0004 on the same example differed only in the system prompt's calculator-framing addendum and the calculator's docstring — content tangential to the fraud-detection task itself. The agent identified essentially the same factual observations in both runs but produced opposite predictions (label=0; the first run predicted 0 with prob=0.35, the second predicted 1 with prob=0.78). We do not yet know what mechanism in the prompt-to-judgment chain produces this sensitivity, nor whether the flip is systematic or specific to this example.
- **Answer:** The most likely reading is within-condition variance under nominally `temperature=0` inference (a documented property of production LLM APIs due to lack of batch invariance), rather than a systematic effect of the prompt-level differences. Across four ablation experiments isolating each prompt-level variable (Exp 1–4), six of seven runs on the same example — including one that exactly matched Pilot 1's configuration — predicted `1`; only Pilot 1 predicted `0`. See Conclusion in `experiments/EXP-A-0004/q1-prompt-framing/experiments.md` for full reasoning and references.
