"""Prepare error cases for manual audit — EXP-D-0001.

Loads all 4 result sets, identifies prediction errors, enriches with
missingness profiles from Phase 1, and outputs structured JSONL for review.

Usage:
    uv run python scripts/prepare_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILES = {
    "o4-mini/no-text": REPO_ROOT / "reproduction/outputs/EXP-R-0002/o4-mini-2025-04-16/results.jsonl",
    "haiku/no-text": REPO_ROOT / "reproduction/outputs/EXP-R-0002/claude-haiku-4-5-20251001/results.jsonl",
    "o4-mini/with-text": REPO_ROOT / "reproduction/outputs/EXP-R-0003/o4-mini-2025-04-16/results.jsonl",
    "haiku/with-text": REPO_ROOT / "reproduction/outputs/EXP-R-0003/claude-haiku-4-5-20251001/results.jsonl",
}

PROFILES_PATH = REPO_ROOT / "outputs/EXP-D-0001/missingness_profiles.json"
OUT_PATH = REPO_ROOT / "outputs/EXP-D-0001/errors_for_audit.jsonl"


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def main() -> None:
    # Load missingness profiles
    profiles = json.loads(PROFILES_PATH.read_text())

    # Collect all errors
    errors = []
    summary = {}

    for condition, path in RESULT_FILES.items():
        model, config = condition.split("/")
        results = load_results(path)

        n_total = 0
        n_errors = 0
        n_fp = 0
        n_fn = 0

        for r in results:
            if r["prediction"] is None:
                continue
            n_total += 1
            pred = int(r["prediction"])
            label = r["label"]

            if pred != label:
                n_errors += 1
                error_type = "FP" if pred == 1 and label == 0 else "FN"
                if error_type == "FP":
                    n_fp += 1
                else:
                    n_fn += 1

                # Get missingness profile
                miss = profiles.get(r["doc_id"], {}).get("aggregate", {})

                errors.append({
                    "doc_id": r["doc_id"],
                    "model": model,
                    "config": config,
                    "condition": condition,
                    "label": label,
                    "prediction": pred,
                    "error_type": error_type,
                    "prob": r["prob"],
                    "reasoning": r["reasoning"],
                    "missingness_density": miss.get("density"),
                    "missingness_count": miss.get("missing"),
                    "missingness_total": miss.get("total"),
                })

        summary[condition] = {
            "total": n_total,
            "errors": n_errors,
            "fp": n_fp,
            "fn": n_fn,
        }

    # Save errors for audit
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Total error cases: {len(errors)}\n")
    print(f"{'Condition':<25} {'Total':>5} {'Errors':>6} {'FP':>4} {'FN':>4}")
    print("-" * 50)
    for condition, s in summary.items():
        print(f"{condition:<25} {s['total']:>5} {s['errors']:>6} {s['fp']:>4} {s['fn']:>4}")

    print(f"\nSaved to {OUT_PATH}")

    # Distribution by doc_id — how many conditions does each doc appear in errors?
    doc_error_counts: dict[str, int] = {}
    for e in errors:
        doc_error_counts[e["doc_id"]] = doc_error_counts.get(e["doc_id"], 0) + 1

    print(f"\nUnique doc_ids with errors: {len(doc_error_counts)}")
    for count in sorted(set(doc_error_counts.values())):
        n = sum(1 for c in doc_error_counts.values() if c == count)
        print(f"  Errors in {count} condition(s): {n} docs")


if __name__ == "__main__":
    main()
