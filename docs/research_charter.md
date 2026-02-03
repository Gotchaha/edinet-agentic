# Research Charter — EDINET-Bench Agentic Reliability & Evaluation

## 0. One-liner
We study how to make LLM/agent reasoning **more reliable and auditable** on EDINET-Bench’s complex financial tasks (fraud detection & earnings forecasting), focusing on **(i) failure-mode diagnosis, (ii) reliability-layer interventions, (iii) rubric-based evaluation**.

## 1. Context & Motivation (from EDINET-Bench)
EDINET-Bench (Sugiura et al., 2025)[^edinetbench] evaluates frontier LLMs on challenging financial tasks. Empirically, even state-of-the-art LLMs perform only slightly better than logistic regression on fraud detection and earnings forecasting, suggesting substantial room for improvement and motivating “novel agentic pipelines.”

The paper highlights concrete reliability bottlenecks:
- **Parsing inconsistency** can cause LLMs to treat missing parsed fields as if the original report lacks information, leading to **false positives in fraud detection**, and it notes it is necessary to specify this aspect in prompts to avoid misjudgment.
- The benchmark prompts explicitly warn that some data may be missing and represented as “-” due to parsing errors, reinforcing that missingness is a first-class confounder in evaluation.

The Limitations section further argues:
- Fraud detection / earnings forecasting may have an intrinsic upper bound when relying only on a **single annual report**; thus future research could explore using information **beyond the annual report** with **novel agentic pipelines**.
- Benchmark evaluation could be designed using **rubric evaluation** with multiple criteria tailored to each question.

## 2. Research Question
When inputs are long, partially structured, and subject to extraction noise/missingness (as in EDINET-Bench),
**where does an agentic pipeline provide measurable value**?
- Better predictions? (ROC-AUC/MCC)
- Better reliability under noise? (reduced systematic error types)
- Better auditable outputs? (evidence quality under a rubric)

## 3. Hypotheses (H1–H3)

### H1 — Failure Mechanism (Diagnosis)
A major source of LLM errors on fraud/forecasting is **reliability failure** (e.g., treating parsing-missing as business anomalies; evidence drift / inconsistency not detected), rather than purely insufficient “model intelligence.”

**Falsifiable signal**: If we audit errors and do not observe a substantial fraction attributable to missingness/consistency issues (vs. other reasons), H1 is weakened.

### H2 — Reliability Layer (Intervention)
Adding a lightweight **Reliability Layer** (missingness attribution, consistency checks, evidence-binding constraints) will **reduce systematic error types** (especially fraud false positives caused by missing-as-anomaly). Overall ROC-AUC/MCC may or may not improve, but the **error structure** should improve measurably.

**Falsifiable signal**: If reliability-layer variants do not reduce the targeted error categories (even if AUC unchanged), H2 fails.

### H3 — Rubric-Based Evaluation (Measurement)
Traditional metrics alone (ROC-AUC/MCC) may under-measure progress for agentic pipelines; a **rubric** that scores evidence quality (traceability, consistency, uncertainty disclosure) will better capture improvements and distinguish “plausible-sounding” from “auditable” answers.

**Falsifiable signal**: If rubric scores do not correlate with meaningful reliability improvements (or are too noisy to be consistent), H3 is weakened.

## 4. High-level Method Sketch (conceptual)
- Baseline: zero-shot LLM as in EDINET-Bench prompts (fraud/forecasting).
- Reliability Layer (agentic): (a) detect/label missingness & parsing anomalies; (b) enforce evidence-citation; (c) consistency checks; (d) uncertainty reporting.
- Evaluation: (i) ROC-AUC/MCC (paper metrics) + (ii) targeted error analysis + (iii) rubric scoring.

## 5. Scope / Non-goals (for now)
- Focus on **reliability + evaluation methodology** grounded in EDINET-Bench failure modes (fraud & earnings forecasting).
- Aim for **clear causal insights** (diagnosis → intervention → measurement), rather than maximizing leaderboard-style point estimates.
- Not building a large-scale retrieval system in the first phase.
- Not expanding beyond EDINET-Bench inputs in the first phase (any “beyond annual report” information use is deferred until reliability mechanisms are validated).

## 6. Risks & Confounders (from paper)
- Intrinsic difficulty / upper bound when using only a single annual report.
- Label noise / mislabeling risks in fraud dataset construction.
- Contamination risk (reports possibly in pretraining), mitigated by using future reports.
- Parsing inconsistency causing misleading missingness.

## 7. Research Log Convention (pointer)
- Main research log (stage-driven): `docs/research_log.md`
  - Records: stage plans, agent/model versions (A-xxxx), key decisions (ADR-xxxx), and experiment analyses (EXP-xxxx).
  - Each EXP entry should link to: config path, output artifact path, and (optionally) commit hash.

- Personal side notes (kept separate from the main log):
  - Questions: `docs/notes/questions.md`
  - Troubleshooting: `docs/notes/troubleshooting.md`
  - Lessons learned: `docs/notes/lessons_learned.md`

- Reproduction module (official baseline replication, budget-aware):
  - See `reproduction/README.md` for scope and `reproduction/results/` for tracked summaries.
  - Machine-generated artifacts (run logs, raw outputs, audits) live under `reproduction/outputs/` (not tracked).

---

[^edinetbench]: I. Sugiura et al., “EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements,” arXiv:2506.08762, 2025.
