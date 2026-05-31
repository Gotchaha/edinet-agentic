# Personal Notes - EXP-A-004

## Pre-learning

In the pre-discussion of EXP-A-0003, I noticed that Claude Code invoked one skill I created called `/research-advise` in a response, then I let it record that because some of the literature research results are interesting so I decided to read those papers before the next experiment, i.e. this EXP-A-0004. By the way, I also have pasted the skill calling result under the project folder, which is `experiments/EXP-A-0004/llm-numerical-reasoning-literature.md`.

Here is the learning of the papers:

1. In-depth reading (three-pass)
   - [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://openreview.net/forum?id=AjXkRZIvjB)
   - [How well do LLMs reason over tabular data, really?](https://aclanthology.org/2025.trl-1.21/)
   - [Lost in the Middle: How Language Models Use Long Contexts](https://aclanthology.org/2024.tacl-1.9/)

2. Skim (1-1.5 pass, plus possible AI use for reading)
   - [Cutting Through the Noise: Boosting LLM Performance on Math Word Problems](https://openreview.net/forum?id=VnPYbWQjz7)
   - [Can Large language model analyze financial statements well?](https://aclanthology.org/2025.finnlp-1.19/)

*The "three-pass" comes from the "how to read a paper" by S. Keshav.*

## Thinking

There are many proofs confirming that LLMs are bad at reasoning on numbers, or even, we should question that is there any reasoning in the LLM's generating process. This is sort of consistent with what I have been thinking, and it was just the thinking that LLM cannot be used satisfyingly on number-heavy data, or structured tabular data that motivated me to conduct research on representation learning of such data as my research of master course.
Go back here, I also get to know the "lost-in-the-middle" issue, this is also interesting because the attention mechanism should have not gotten such issue. Finally, for above two observations there is still not any satisfied explanation. For the agent research at hand, I think I see (in one aspect) why agent works -- the natural decomposition of the problem mitigates the lost-in-the-middle issue, and I should also do that for the EDINET-Bench, because the input data here is also very long and LLM can get lost.

## About the Result & Next Step

The result is not good, even though there was one run showing a promising result, I could not reproduce it so it was because of the nondeterminism in LLM inference. Honestly, I feel these experiments show no good sign of agentic method, so I reviewed the EXP-D-0001, the diagnosis made at the beginning, I think next I will look into it and we may need to do another diagnosis, a more rigorous one instead.