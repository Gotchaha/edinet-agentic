"""Cross-model agreement analysis — EXP-D-0001, Phase 3.

For each of the 50 examples, compares predictions across all 4 conditions
(2 models × 2 configs) to categorize agreement patterns and distinguish
reliability failures from intelligence failures.

Usage:
    uv run python scripts/agreement.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILES = {
    "o4-mini/no-text": REPO_ROOT / "reproduction/outputs/EXP-R-0002/o4-mini-2025-04-16/results.jsonl",
    "haiku/no-text": REPO_ROOT / "reproduction/outputs/EXP-R-0002/claude-haiku-4-5-20251001/results.jsonl",
    "o4-mini/with-text": REPO_ROOT / "reproduction/outputs/EXP-R-0003/o4-mini-2025-04-16/results.jsonl",
    "haiku/with-text": REPO_ROOT / "reproduction/outputs/EXP-R-0003/claude-haiku-4-5-20251001/results.jsonl",
}

SAMPLE_PATH = REPO_ROOT / "reproduction/sampling/fraud_detection_n50_seed42.json"
PROFILES_PATH = REPO_ROOT / "outputs/EXP-D-0001/missingness_profiles.json"
OUT_PATH = REPO_ROOT / "experiments/EXP-D-0001/agreement_analysis.md"
OUT_DETAIL_PATH = REPO_ROOT / "outputs/EXP-D-0001/agreement_detail.jsonl"


def load_results(path: Path) -> dict[str, dict]:
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["doc_id"]] = r
    return results


def classify_agreement(
    label: int,
    preds: dict[str, int | None],
) -> str:
    """Classify a single example's cross-condition agreement pattern.

    Categories:
    - all_correct:       all conditions predict correctly
    - all_wrong_same:    all conditions wrong, same direction (all FP or all FN)
    - all_wrong_mixed:   all conditions wrong, but different directions
    - some_correct:      mixed — at least one correct and one wrong (reliability issue)
    """
    valid_preds = {k: v for k, v in preds.items() if v is not None}
    if not valid_preds:
        return "no_valid_predictions"

    correct = {k for k, v in valid_preds.items() if v == label}
    wrong = {k for k, v in valid_preds.items() if v != label}

    if len(wrong) == 0:
        return "all_correct"
    if len(correct) == 0:
        # All wrong — check if same direction
        wrong_dirs = {valid_preds[k] for k in wrong}
        if len(wrong_dirs) == 1:
            return "all_wrong_same"
        return "all_wrong_mixed"
    return "some_correct"


def classify_model_pair(
    label: int,
    pred_a: int | None,
    pred_b: int | None,
) -> str:
    """Classify agreement between two conditions."""
    if pred_a is None or pred_b is None:
        return "missing"
    a_ok = pred_a == label
    b_ok = pred_b == label
    if a_ok and b_ok:
        return "both_correct"
    if a_ok and not b_ok:
        return "a_correct_b_wrong"
    if not a_ok and b_ok:
        return "a_wrong_b_correct"
    # Both wrong
    if pred_a == pred_b:
        return "both_wrong_same"
    return "both_wrong_opposite"


def main() -> None:
    sample = json.loads(SAMPLE_PATH.read_text())
    doc_ids = sample["doc_ids"]

    # Load all results
    all_results = {}
    for condition, path in RESULT_FILES.items():
        all_results[condition] = load_results(path)

    # Load missingness profiles
    profiles = json.loads(PROFILES_PATH.read_text())

    conditions = list(RESULT_FILES.keys())
    details = []
    category_counts = Counter()

    # Per-example analysis
    for did in doc_ids:
        preds = {}
        probs = {}
        label = None
        for cond in conditions:
            r = all_results[cond].get(did)
            if r is None:
                preds[cond] = None
                probs[cond] = None
                continue
            label = r["label"]
            preds[cond] = int(r["prediction"]) if r["prediction"] is not None else None
            probs[cond] = float(r["prob"]) if r["prob"] is not None else None

        category = classify_agreement(label, preds)
        category_counts[category] += 1

        # Model-pair comparisons (same config, different model)
        model_pair_notext = classify_model_pair(
            label, preds.get("o4-mini/no-text"), preds.get("haiku/no-text")
        )
        model_pair_text = classify_model_pair(
            label, preds.get("o4-mini/with-text"), preds.get("haiku/with-text")
        )

        # Config-pair comparisons (same model, different config)
        config_pair_o4 = classify_model_pair(
            label, preds.get("o4-mini/no-text"), preds.get("o4-mini/with-text")
        )
        config_pair_haiku = classify_model_pair(
            label, preds.get("haiku/no-text"), preds.get("haiku/with-text")
        )

        miss = profiles.get(did, {}).get("aggregate", {})

        details.append({
            "doc_id": did,
            "label": label,
            "predictions": preds,
            "probs": probs,
            "category": category,
            "model_pair_notext": model_pair_notext,
            "model_pair_text": model_pair_text,
            "config_pair_o4mini": config_pair_o4,
            "config_pair_haiku": config_pair_haiku,
            "missingness_density": miss.get("density"),
        })

    # Save detail JSONL
    OUT_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DETAIL_PATH, "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── Build report ────────────────────────────────────────────────

    lines = []
    lines.append("# Cross-Model Agreement Analysis — EXP-D-0001\n")

    # 4-way agreement
    lines.append("## 1. Four-Way Agreement (2 models × 2 configs)\n")
    lines.append("| Category | Count | % | Interpretation |")
    lines.append("|----------|-------|---|----------------|")

    interpretations = {
        "all_correct": "Task solvable by both models",
        "all_wrong_same": "Task-intrinsic difficulty (shared bias)",
        "all_wrong_mixed": "Both fail but in opposite directions",
        "some_correct": "Error is avoidable → **reliability failure**",
        "no_valid_predictions": "Parse failures",
    }
    n_total = len(doc_ids)
    for cat in ["all_correct", "some_correct", "all_wrong_same", "all_wrong_mixed", "no_valid_predictions"]:
        c = category_counts.get(cat, 0)
        lines.append(f"| {cat} | {c} | {c/n_total*100:.0f}% | {interpretations.get(cat, '')} |")
    lines.append("")

    # Model-pair agreement matrices
    for pair_name, pair_key in [
        ("Models (no-text config)", "model_pair_notext"),
        ("Models (with-text config)", "model_pair_text"),
        ("Configs (o4-mini)", "config_pair_o4mini"),
        ("Configs (Haiku 4.5)", "config_pair_haiku"),
    ]:
        pair_counts = Counter(d[pair_key] for d in details)
        lines.append(f"### {pair_name}\n")
        lines.append("| Pattern | Count | % |")
        lines.append("|---------|-------|---|")
        for pat in ["both_correct", "a_correct_b_wrong", "a_wrong_b_correct", "both_wrong_same", "both_wrong_opposite", "missing"]:
            c = pair_counts.get(pat, 0)
            if c > 0:
                lines.append(f"| {pat} | {c} | {c/n_total*100:.0f}% |")
        lines.append("")

    # Reliability vs intelligence summary
    n_some_correct = category_counts.get("some_correct", 0)
    n_all_wrong = category_counts.get("all_wrong_same", 0) + category_counts.get("all_wrong_mixed", 0)
    n_all_correct = category_counts.get("all_correct", 0)
    n_with_errors = n_total - n_all_correct

    lines.append("## 2. Reliability vs Intelligence Failure\n")
    lines.append(f"- Examples with at least one error: **{n_with_errors}** / {n_total}")
    lines.append(f"- Of those, avoidable errors (some_correct): **{n_some_correct}** ({n_some_correct/max(n_with_errors,1)*100:.0f}%)")
    lines.append(f"- Shared failures (all_wrong): **{n_all_wrong}** ({n_all_wrong/max(n_with_errors,1)*100:.0f}%)")
    lines.append("")

    if n_some_correct > 0:
        lines.append("This means the majority of errors occur on examples where at least one")
        lines.append("model/config combination gets it right — suggesting **reliability failures**")
        lines.append("rather than fundamental intelligence limitations.\n")

    report = "\n".join(lines)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report)
    print(report)
    print(f"\nReport saved to {OUT_PATH}")
    print(f"Detail saved to {OUT_DETAIL_PATH}")


if __name__ == "__main__":
    main()
