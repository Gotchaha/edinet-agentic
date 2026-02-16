# EXP-R-0002: Budgeted Reproduction — fraud_detection

**Date:** 2026-02-13
**Task:** fraud_detection
**Sheets:** summary, bs, pl, cf
**Sample:** N=50 (stratified, seed=42)

## o4-mini-2025-04-16

- Valid examples: 50/50 (parse failures: 0)
- Cost: $0.6340
- Tokens: 191,772 in / 96,156 out

| Metric | Value |
|--------|-------|
| Accuracy  | 0.480 |
| Precision | 0.667 |
| Recall    | 0.074 |
| F1        | 0.133 |
| ROC-AUC   | 0.540 [0.391, 0.684] |
| MCC       | 0.064 [-0.221, 0.284] |

## claude-haiku-4-5-20251001

- Valid examples: 50/50 (parse failures: 0)
- Cost: $0.3583
- Tokens: 202,393 in / 31,181 out

| Metric | Value |
|--------|-------|
| Accuracy  | 0.580 |
| Precision | 0.562 |
| Recall    | 1.000 |
| F1        | 0.720 |
| ROC-AUC   | 0.471 [0.316, 0.623] |
| MCC       | 0.221 [0.000, 0.377] |

## Comparison with Paper (Table 6)

Paper config: fraud_detection, summary+bs+cf+pl, full test set (224 examples), 3 runs.

| Model | Metric | Ours (N=50, 95% CI) | Paper (N=224, 3-run) |
|-------|--------|---------------------|----------------------|
| o4-mini (same model) | ROC-AUC | 0.540 [0.391, 0.684] | 0.52 +/- 0.01 |
| | MCC | 0.064 [-0.221, 0.284] | 0.04 +/- 0.05 |
| Haiku 4.5 (paper: Haiku 3.5) | ROC-AUC | 0.471 [0.316, 0.623] | 0.60 +/- 0.01 |
| | MCC | 0.221 [0.000, 0.377] | 0.18 +/- 0.03 |

## Notes

- Bootstrap CIs: 1000 resamples, percentile method, seed=42.
- Paper values from Table 6 (3-run mean +/- std).
- Haiku 4.5 (`claude-haiku-4-5-20251001`) is the recommended successor to
  Claude 3.5 Haiku (`claude-3-5-haiku-20241022`) used in the paper.
  Results are not directly comparable due to model version difference.
- Our sample is N=50 (stratified subsample); paper uses full test set N=224.
