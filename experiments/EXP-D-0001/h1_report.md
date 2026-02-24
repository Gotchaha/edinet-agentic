# H1 Synthesis Report — EXP-D-0001

## Verdict

**H1 is strongly supported.** Errors in EDINET-Bench fraud detection are
overwhelmingly reliability failures — mechanistic, patterned, and avoidable —
not intelligence limitations.

- **100% of errors** (100/100) occur on examples where at least one model/config
  gets the answer right. Zero shared failures across all 4 conditions.
- **98% of errors** are classified into failure modes directly addressable by an
  agentic scaffold (M1/M2/M3/M5/M7). Zero M6 (analytical limitation) as primary.
- **Missingness is not a driver.** No statistical correlation between data
  missingness and errors (all p > 0.6). Only 2/100 errors are M1 (missing-as-anomaly).

---

## Layer 1: Missingness Correlation

Full report: `missingness_analysis.md`

| Condition | Errors | Density (correct) | Density (errors) | MWU p |
|-----------|--------|-------------------|------------------|-------|
| o4-mini / no-text | 26/50 | 0.046 | 0.049 | 0.96 |
| Haiku / no-text | 21/50 | 0.047 | 0.049 | 0.80 |
| o4-mini / with-text | 26/50 | 0.045 | 0.050 | 0.62 |
| Haiku / with-text | 27/50 | 0.032 | 0.062 | 0.95 |

**Conclusion:** Missingness density does not predict errors. The upstream prompt's
mention that "some data may be missing and represented as '-'" does not cause
systematic M1 failures. Only 2 cases (both on the single highest-missingness
example, density=0.40) were classified as M1.

---

## Layer 2: Failure Mode Classification

Classifier: GPT-5, OpenAI Batch API. 100 error cases, 99 successfully classified
(1 truncated to M8). Each request included raw sheet data for claim verification.

### Primary mode distribution

| Mode | Count | Description |
|------|-------|-------------|
| **M7** | **58** | Conservatism bias — dismisses red flags, anchors on "consistent numbers" |
| **M2** | **23** | Evidence drift — reasoning chain disconnected from observations |
| **M3** | **15** | Magnitude insensitivity — flags normal business variation as fraud |
| M1 | 2 | Missing-as-anomaly |
| M4 | 1 | Irrelevant anchoring |
| M8 | 1 | Truncation artifact (token limit) |
| M5 | 0 | Base rate neglect |
| M6 | 0 | Analytical limitation |

### Combined prevalence (primary or secondary)

| Mode | Involved in N cases | As primary | As secondary |
|------|---------------------|-----------|-------------|
| M2 | **80** | 23 | 57 |
| M7 | 66 | 58 | 8 |
| M3 | 27 | 15 | 12 |
| M1 | 10 | 2 | 8 |
| M4 | 4 | 1 | 3 |
| M5 | 1 | 0 | 1 |
| M6 | 1 | 0 | 1 |

**M2 (evidence drift) appears in 80% of all errors** — it is the pervasive
connective tissue of failure. Models regularly cite observations that don't
logically support their conclusion, whether they're being too conservative (M7)
or too aggressive (M3).

### By model × config × error type

| Model | Config | Error type | N | Dominant modes |
|-------|--------|------------|---|----------------|
| o4-mini | no-text | FN | 25 | M7(20), M2(4) |
| o4-mini | with-text | FN | 26 | M7(21), M2(5) |
| o4-mini | no-text | FP | 1 | M2(1) |
| Haiku | no-text | FP | 21 | M3(11), M2(8), M1(1), M4(1) |
| Haiku | with-text | FP | 9 | M2(4), M3(4), M1(1) |
| Haiku | with-text | FN | 18 | M7(17), M2(1) |

**Model failure signatures are distinct and complementary:**

- **o4-mini** is almost purely M7 — conservatism bias. It anchors on surface-level
  consistency and dismisses genuine anomalies. Text inclusion does not change this.
- **Haiku (no-text)** is M3+M2 — it over-reacts to normal fluctuations and makes
  logical leaps, producing false positives.
- **Haiku (with-text)** partially corrects FPs but introduces 18 new FN errors,
  all on examples it got *correct* without text. Text makes Haiku more conservative
  (M7), not more accurate.

---

## Layer 3: Cross-Model Agreement

Full report: `agreement_analysis.md`

| Category | Count | % |
|----------|-------|---|
| All correct (solvable) | 2 | 4% |
| Some correct (reliability failure) | 48 | 96% |
| All wrong same direction | 0 | 0% |
| All wrong mixed | 0 | 0% |

**Every error is avoidable.** For each incorrectly-predicted example, at least one
model/config combination gets it right. This rules out task-intrinsic difficulty
as an explanation and confirms that the errors are reliability failures.

The models fail in complementary ways:
- No-text config: o4-mini correct / Haiku wrong on 40% of examples (Haiku's FP bias);
  Haiku correct / o4-mini wrong on 50% (o4-mini's FN bias)
- With-text config: convergence increases (36% both wrong same direction),
  but still 30% both correct — text helps when it helps

---

## Actionability Assessment

### What fraction of errors can H2 address?

| Category | Count | Addressable? |
|----------|-------|-------------|
| M7 (conservatism) | 58 | Yes — scaffold can flag dismissed anomalies, require explicit engagement |
| M2 (evidence drift) | 23 primary, 57 secondary | Yes — scaffold can verify claim-to-evidence links |
| M3 (magnitude insensitivity) | 15 | Yes — scaffold can provide base-rate context for financial metrics |
| M1 (missing-as-anomaly) | 2 | Yes — scaffold can pre-annotate missingness as benign |
| M6 (analytical limitation) | 0 | N/A — not observed |
| **Total addressable** | **98/100** | |

### H2 design guidance

1. **Evidence verification (targets M2, M7):** The scaffold should require the model
   to cite specific data points from the sheets and verify that cited values match
   the actual data. This directly attacks the 80% M2 prevalence.

2. **Anomaly checklist (targets M7):** For each example, pre-compute a set of
   quantitative anomalies (ratio deviations, sign changes, missing patterns) and
   require the model to explicitly address each one. This prevents the conservatism
   of "everything looks consistent" when it doesn't.

3. **Magnitude calibration (targets M3):** Provide base-rate statistics for key
   financial metrics (revenue growth variance, working capital ratios) so the model
   can distinguish normal variation from genuine outliers.

4. **Multi-perspective deliberation (targets M7 + M3):** Run the model with both
   a "prosecution" and "defense" framing, then reconcile. This exploits the
   complementary failure modes — one perspective catches what the other misses.

5. **Text gating (targets Haiku with-text regression):** Text inclusion is not
   uniformly beneficial. The scaffold should treat text as supplementary evidence
   rather than letting it dominate and flip the model's stance.

---

## Cost Summary

| Phase | Method | Cost |
|-------|--------|------|
| Phase 1 (missingness) | Local computation | $0 |
| Phase 2 (classification) | GPT-5 Batch API, 100 cases | $2.28 |
| Phase 3 (agreement) | Local computation | $0 |
| **Total** | | **$2.28** |
