# Lesson Learned

- template

```
### [Lesson title]
- **Date:** YYYY-MM-DD
- **EXP:** Experiment involved
- **Context:** The background
- **Lesson:** One clear takeaway
- **Why it matters:** Why this matters beyond this one case
- **Rule for future work:** A concrete rule to follow next time
```

### Always verify the plan and implementation of the code agent, even if the tests pass
- **Date:** 2026-03-18
- **EXP:** EXP-A-0002
- **Context:** I have discussed carefully with Claude Code to decide on multiple detailed aspects before the plan mode, and after the implementation with the successful sanity test, I just went for the formal run. However it turned out to have some tricky issues like redundant prompt settings and wrong logic, this had caused terrible results and token waste, where Claude Code even acted as the result was some kind of good, but when I looked into it I just realized something was greatly wrong (even with the model setting of Opus 4.6, high effort).
- **Lesson:** Do not be lazy.
- **Why it matters:** This typical issue represents one tricky, hidden danger from blindly using the code agent.
- **Rule for future work:** Manually always verify the work by the code agent; Also I have required Claude Code to record this case in the feedback memory and explicitly ensure verification of all kinds of variable and inputs used in the implementation in the future.