# Failure Mode Taxonomy — EXP-D-0001

## Codes

| Code | Failure mode | Definition | Typical error type |
|------|-------------|------------|--------------------|
| M1 | Missing-as-anomaly | Treats missing data (`－` or absent fields) as a suspicious signal, when it is just a parsing artifact | FP |
| M2 | Evidence drift | Cites observations that don't logically support the conclusion; reasoning chain is disconnected | FP, FN |
| M3 | Magnitude insensitivity | Flags normal business variation (revenue fluctuations, working capital changes) as fraud signals | FP |
| M4 | Irrelevant anchoring | Fixates on dramatic-sounding but non-diagnostic items (large absolute numbers, foreign-sounding terms) | FP |
| M5 | Base rate neglect | Fails to calibrate probability against actual fraud prevalence; treats any anomaly as high-probability fraud | FP |
| M6 | Analytical limitation | Genuinely lacks domain knowledge needed for this case; analysis is competent but the task is too hard | FP, FN |
| M7 | Conservatism bias | Dismisses genuine red flags, gives unwarranted benefit-of-the-doubt, or anchors on "numbers are consistent" | FN |
| M8 | Other | Does not fit the above categories (describe in notes) | any |

## Usage

- **primary_mode**: required, the dominant failure mode
- **secondary_mode**: optional, a contributing failure mode (null if not applicable)
- **confidence**: high / medium / low — how clearly the reasoning fits the code
- **notes**: 1-2 sentence justification

## Context

The prompt tells models: "some data may be missing and represented as '-' due to parsing errors."
The actual missing indicator in the data is `－` (full-width dash, U+FF0D).
This priming may contribute to M1 failures regardless of actual missingness density.
