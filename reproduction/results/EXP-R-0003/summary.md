# EXP-R-0003: Budgeted Reproduction — fraud_detection

**Date:** 2026-02-16
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf, text
**Sample:** N=50 (stratified, seed=42)

## o4-mini-2025-04-16

- Valid examples: 50/50 (parse failures: 0)
- Cost: $2.1290
- Tokens: 1,629,423 in / 76,515 out

| Metric | Value |
|--------|-------|
| Accuracy  | 0.480 |
| Precision | 1.000 |
| Recall    | 0.037 |
| F1        | 0.071 |
| ROC-AUC   | 0.626 [0.475, 0.771] |
| MCC       | 0.132 [0.000, 0.251] |

## claude-haiku-4-5-20251001

- Valid examples: 50/50 (parse failures: 0)
- Cost: $2.0420
- Tokens: 1,825,697 in / 43,267 out

| Metric | Value |
|--------|-------|
| Accuracy  | 0.460 |
| Precision | 0.500 |
| Recall    | 0.333 |
| F1        | 0.400 |
| ROC-AUC   | 0.521 [0.354, 0.680] |
| MCC       | -0.060 [-0.333, 0.203] |

## Comparison with Paper (Table 6)

Paper config: fraud_detection, summary+bs+cf+pl+text, full test set (224 examples), 3 runs.

| Model | Metric | Ours (N=50, 95% CI) | Paper (N=224, 3-run) |
|-------|--------|---------------------|----------------------|
| o4-mini (same model) | ROC-AUC | 0.626 [0.475, 0.771] | 0.61 +/- 0.01 |
| | MCC | 0.132 [0.000, 0.251] | 0.10 +/- 0.05 |
| Haiku 4.5 (paper: Haiku 3.5) | ROC-AUC | 0.521 [0.354, 0.680] | 0.67 +/- 0.00 |
| | MCC | -0.060 [-0.333, 0.203] | 0.28 +/- 0.02 |

## Notes

- Bootstrap CIs: 1000 resamples, percentile method, seed=42.
- Paper values from Table 6 (3-run mean +/- std).
- Haiku 4.5 (`claude-haiku-4-5-20251001`) is the recommended successor to
  Claude 3.5 Haiku (`claude-3-5-haiku-20241022`) used in the paper.
  Results are not directly comparable due to model version difference.
- Our sample is N=50 (stratified subsample); paper uses full test set N=224.
