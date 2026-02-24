# Missingness Analysis — EXP-D-0001

## 1. Missingness Overview (N=50 sample)

- Mean aggregate density: 0.0477
- Std:  0.0720
- Min:  0.0083
- Max:  0.3968
- Median: 0.0295

### Per-sheet missingness

| Sheet | Mean density | Std | Max |
|-------|-------------|-----|-----|
| summary | 0.1031 | 0.1685 | 0.8235 |
| bs | 0.0095 | 0.0153 | 0.0641 |
| pl | 0.0374 | 0.0278 | 0.1000 |
| cf | 0.0389 | 0.0433 | 0.2381 |
| text | 0.0000 | 0.0000 | 0.0000 |

## 2. Missingness vs Prediction Errors

### o4-mini / no-text (R-0002)

- N valid: 50  |  Correct: 24  |  Errors: 26 (FP: 1, FN: 25)
- Mean density — correct: 0.0460 +/- 0.0741
- Mean density — errors:  0.0493 +/- 0.0700
- Mann-Whitney U (correct vs errors): U=315.0, p=0.9613 
- Mean density — FP: 0.0546  |  FN: 0.0491
- Spearman (density vs prob): rho=0.253, p=0.0768 
- Point-biserial (density vs prediction): r=-0.001, p=0.9955 

### Haiku 4.5 / no-text (R-0002)

- N valid: 50  |  Correct: 29  |  Errors: 21 (FP: 21, FN: 0)
- Mean density — correct: 0.0468 +/- 0.0667
- Mean density — errors:  0.0490 +/- 0.0787
- Mann-Whitney U (correct vs errors): U=291.0, p=0.7983 
- Mean density — FP: 0.0490  |  FN: —
- Spearman (density vs prob): rho=-0.096, p=0.5065 
- Point-biserial (density vs prediction): r=0.076, p=0.6003 

### o4-mini / with-text (R-0003)

- N valid: 50  |  Correct: 24  |  Errors: 26 (FP: 0, FN: 26)
- Mean density — correct: 0.0454 +/- 0.0743
- Mean density — errors:  0.0499 +/- 0.0698
- Mann-Whitney U (correct vs errors): U=286.0, p=0.6205 
- Spearman (density vs prob): rho=-0.130, p=0.3695 
- Point-biserial (density vs prediction): r=-0.059, p=0.6835 

### Haiku 4.5 / with-text (R-0003)

- N valid: 50  |  Correct: 23  |  Errors: 27 (FP: 9, FN: 18)
- Mean density — correct: 0.0315 +/- 0.0125
- Mean density — errors:  0.0615 +/- 0.0952
- Mann-Whitney U (correct vs errors): U=306.5, p=0.9457 
- Mean density — FP: 0.0733  |  FN: 0.0555
- Mann-Whitney U (FP vs FN): U=99.0, p=0.3681 
- Spearman (density vs prob): rho=0.156, p=0.2786 
- Point-biserial (density vs prediction): r=0.067, p=0.6430 

## 3. Summary Table

| Condition | Errors | Density (correct) | Density (errors) | MWU p | Spearman rho | Spearman p |
|-----------|--------|-------------------|------------------|-------|-------------|------------|
| o4-mini / no-text (R-0002) | 26/50 | 0.0460 | 0.0493 | 0.9613 | 0.2526 | 0.0768 |
| Haiku 4.5 / no-text (R-0002) | 21/50 | 0.0468 | 0.0490 | 0.7983 | -0.0962 | 0.5065 |
| o4-mini / with-text (R-0003) | 26/50 | 0.0454 | 0.0499 | 0.6205 | -0.1297 | 0.3695 |
| Haiku 4.5 / with-text (R-0003) | 27/50 | 0.0315 | 0.0615 | 0.9457 | 0.1562 | 0.2786 |
