# Research project conventions

**Check the key points below before drafting an implementation plan**

## flow

1. Always confirm with the user which eval(s) should be used in current experiment.
2. **Always verify actual runtime content of inputs before composing with them.** Before implementing code that composes with existing variables, files, functions, or any external content, read and verify the actual runtime content — never treat inputs as black boxes based on name or signature alone.
3. Usually, pause after the pilot test / sanity check (for implementation correctness), before the eval run / main run, waiting the user to check the implementation.
4. In the experiment wrap-up, do not forget to **register this experiment in `docs/research_log.md`** which is used as a catalog with optional key points recording for each experiment.
5. All the work like git add, commit, push, github create pr, merge, etc. are supposed to be done manually by the user.

## outputs

**Outputs from script can be classified into two kinds (it is normal for one script to have both kinds of outputs) and they go to different folders:**

`outputs/` (untracked): raw run artifacts — large or noisy, often per-example/per-step, easy to regenerate (logs, raw predictions/responses, caches, intermediate files). Stored mainly for debugging/audit, not for comparison or writing.

`experiments/` (tracked): compact, human-readable and stable summaries — small files meant to be cited/compared later (metrics tables/JSON, confidence intervals, short run summaries, curated error notes). Chosen because they capture the conclusions without rerunning.