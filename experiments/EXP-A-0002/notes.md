# Personal Notes - EXP-A-002

## Pre-learning

*(besides the papers listed in the TOREAD part of `experiments/EXP-A-0001/notes.md`)*

- [UCB CS294/194-196 Large Language Model Agents (Fall 2024) - Guest Lecture - LLM Reasoning from Denny Zhou](https://rdi.berkeley.edu/llm-agents/f24)

## Thinking

After reading the papers listed in the `TOREAD` part of `experiments/EXP-A-0001/notes.md`, I have realized a few things:

1. Intrinsic self-correction for reasoning seems to be a bad idea, as illustrated in Huang et al. (2024). This paper critized two previou papers - Reflexion from Shinn et al. (2023) and Self-Refine from Madaan et al. (2023), where our naive baseline design is somewhat similar to. The true important part underscored by the paper and later work is the **tool usage and external feedback**. So I am thinking what tools we can design and provide for the agent (like a calculator to verify the numbers, etc.), also we might adopt the `ReAct` paradigm.
2. Few-shot prompting seems to be the default choice among different researches, however EDINET-Bench provides no such goden-label samples to use as the few-shot exampler I think (e.g., for fraud detection, the label is inferred from the reasoning of amended annual reports by LLMs, and the processing detail is not available).
3. This field is still developing (though it is developing so rapidly), it is tricky to see the papers from top conferences are critized immediately by the following work. I think this indicates that there is a lack of standard or convension in the research conducted. So it is important to be **critic** while reading the papers.
4. By the way, I realized that the papers read are mostly from 2023/2024, which is kind of "from the stone age" in the context of this research field, so I just made some rules for Claude to find and select some up-to-date, valuable papers to read, as listed as todos in the following `Learning` section.


*I also notice that in the research of this field, people often draw inspiration from "human's way" and mimic it to design for agent loop, for example, in the Zhou's talk, he used the book 'How to solve it' greatly to show the intuition; in the paper CRITIC from Gou et al. (2024), they explicitly mentioned that they "draw inspiration from human cognition (Greenfield, 1991; Vaesen, 2012) and critical thinking (Marcus, 1988; Ennis, 1991)."*
*So naturally, for this project, I was thinking: **"What is the workflow of a (professional) human fraud detector?"** (as I know little about finance field, so I just used "fraud detector" to refer to such position or role), then I learned about some background of this (by using LLM) and found that the process of this fraud detection task (or red-flag screening) can be corresponded to multiple roles like external auditors/CPAs, internal auditors, securities regulators and investment analysts, etc. For the possibly most related role of external auditors I found that there actually exists a highly formalized and standardized process, like the ones from `IAASB` and `PCAOB`, and I even found the guidance part of using ATT (automated tools and techniques ) from `ISA 315 (REVISED 2019)`.*
*In all, I am considering a truly marvelous design from these, which this markdown file is too narrow to contain.*

## Learning

- [x] [Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback](https://arxiv.org/abs/2506.11930) (NeurIPS 2025)
  Studies how and why LLMs fail to incorporate external feedback during iterative refinement, characterizing the gap between receiving corrections and actually updating reasoning.
- [ ] [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538) (Survey, Jan 2026)
  Recent survey covering the landscape of agentic reasoning — tool use, self-refinement, planning, and multi-agent collaboration — with taxonomy and comparison of current approaches.
- [ ] [Demystifying Reinforcement Learning in Agentic Reasoning](https://arxiv.org/abs/2510.11701) (Preprint, Oct 2025)
  Investigates the role of RL-based training in shaping agentic behaviors such as tool calling, search, and structured reasoning, analyzing what RL actually teaches models beyond supervised fine-tuning.

## About the Result

It turned out that this experiment is problematic in two ways:

1. **The general logic inconsistency** We explicitely made the verifier to be objective and only do the work of verifying the numerical accuracy (e.g., `Do NOT make subjective judgments about the analysis.`, `Do NOT assess whether the prediction is correct.` indicated by the prompt), however in the tool it would use for submission, there are terms liks `red_flags_confirmed(refuted)` this is not correct, the verification of the number itself does not lead to the confirmation of the red flag. This has led to the issue observed (and as expected) in the some of the traces that the reviser saw the confirmation and was even more confident about the wrong answer.
2. **The implementation error** The run script invokes the `load_prompt_template` to load the original prompt from EDINET-Bench to be the base prompt, part of the user content, for the generator, however we also have written a similar prompt as the system prompt for it, so these contradict.
3. **Other minor issues** The token usage is way too much compared with previous experiment, I assume this comes from the issues mentioned above and the extensive calling of tools. Also, in the sense that the verifier only provides the verification of numbers, the design of the verifier and reviser is actually redundant, it can degrade to a react loop, more clear.

These issues come from my negligence and laziness of being too confident about the code agent (Claude Code), even though I have made well-rounded confirmations and used the Opus-4.6 high, it still made these errors, which are sort of hidden that it has passed the tests before the formal run. This has been recorded in the `docs/notes/lessons_learned.md`.

## Next Step

1. Fix the issue.
2. Consider using ReAct loop.
3. Think about tool design.