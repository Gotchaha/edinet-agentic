# Personal Notes - EXP-A-001

## Thinking

The results from EXP-D-0001 still make me think about the relation of these failures with intelligence limitations, also I am wondering is agentic method able to reduce the effect of llm traits or chosing what llm to use is just a hyper-parameter we need to fintune? And for the next step, an assumption I want to make due to the observation is that the performance will be improved if we decompose the task.

The natural idea of the structure of a baseline agentic model that comes to my mind is a simple reflection one as I noted in the notes in reproduction phase (`reproduction/notes.md`). Also initially it seems fine to start with *prompt chaining*, then I just realized that it is not suitable for our case as the task of fruad detection cannot be clearly and easily decomposed into fixed subtasks (at least for now).

*For the terminology, I noticed that there is a distinction between **workflows** and **agents** but I decide to use the term **agent** uniformly in the project.*

## Learning

- [x] [Building effective agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents) 
- [x] [Thinking in LangGraph - LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [x] [Reflection Agents - LangChain blog](https://blog.langchain.com/reflection-agents/)

## About the Result

Clearly this naive baseline shows an interesting result - we got 43/50 flips of prediction 1->0 after critic, and it ended with all except one cases being predicted as non-fruadlent. From the observation of the trace, I first want to make some assumptions about the issue:
- The critic form and the prompt setting
  The trace shows that words like "overstate/overreaction" occur in most of critiques (46/50) which means the critic is making aggressive judgment, this aligns with our knowledge about haiku - prone to be aggressive. Then our prompt tells the critic to "focus on what is wrong or unsupported" and the reviser to "where critique is valid, correct your analysis", this kind of induces the authority frame that the reviser may treat the critic as higher status then defer to it. So, I *guess* we need to guarantee the critic would make unbiased fact checking only, not strong judgment.
  *(also the current critique is too verbose, another issue. maybe we need structured output instead of long statement, then this choice actually is consistent with the fact-checking form we want)*

- The evidence/feedback
  As far as I know (from experience of using code agent), the llm/agent works best when there is external feedback or evidence in a reflection loop (e.g., the code agent would test the code, get the error message then correct accordingly), but here in our setting, there is no such evidence or tool using, we just use another llm call to verify itself, which may be a problem.

As above is just my immediate assumptions from the observation of the trace, I need to read related papers to see if I am correct and to find out what is (possibly) the true reason. I will find related papers and record them for todo in the next step section.

## Next Step

1. Build the (end-to-end) eval.
2. Read papers in `TOREAD`.
3. Consider adjustment and improvement on current baseline agent.

**TOREAD** 

[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/pdf/2303.17651)
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)

[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/pdf/2310.01798)
[Towards Understanding Sycophancy in Language Models](https://arxiv.org/pdf/2310.13548)

[Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/pdf/2309.11495)
[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://openreview.net/pdf?id=Sx038qxjek)