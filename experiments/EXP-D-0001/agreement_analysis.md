# Cross-Model Agreement Analysis — EXP-D-0001

## 1. Four-Way Agreement (2 models × 2 configs)

| Category | Count | % | Interpretation |
|----------|-------|---|----------------|
| all_correct | 2 | 4% | Task solvable by both models |
| some_correct | 48 | 96% | Error is avoidable → **reliability failure** |
| all_wrong_same | 0 | 0% | Task-intrinsic difficulty (shared bias) |
| all_wrong_mixed | 0 | 0% | Both fail but in opposite directions |
| no_valid_predictions | 0 | 0% | Parse failures |

### Models (no-text config)

| Pattern | Count | % |
|---------|-------|---|
| both_correct | 4 | 8% |
| a_correct_b_wrong | 20 | 40% |
| a_wrong_b_correct | 25 | 50% |
| both_wrong_same | 1 | 2% |

### Models (with-text config)

| Pattern | Count | % |
|---------|-------|---|
| both_correct | 15 | 30% |
| a_correct_b_wrong | 9 | 18% |
| a_wrong_b_correct | 8 | 16% |
| both_wrong_same | 18 | 36% |

### Configs (o4-mini)

| Pattern | Count | % |
|---------|-------|---|
| both_correct | 22 | 44% |
| a_correct_b_wrong | 2 | 4% |
| a_wrong_b_correct | 2 | 4% |
| both_wrong_same | 24 | 48% |

### Configs (Haiku 4.5)

| Pattern | Count | % |
|---------|-------|---|
| both_correct | 11 | 22% |
| a_correct_b_wrong | 18 | 36% |
| a_wrong_b_correct | 12 | 24% |
| both_wrong_same | 9 | 18% |

## 2. Reliability vs Intelligence Failure

- Examples with at least one error: **48** / 50
- Of those, avoidable errors (some_correct): **48** (100%)
- Shared failures (all_wrong): **0** (0%)

This means the majority of errors occur on examples where at least one
model/config combination gets it right — suggesting **reliability failures**
rather than fundamental intelligence limitations.
