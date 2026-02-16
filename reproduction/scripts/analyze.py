"""Analyze EDINET-Bench experiment results with bootstrap CIs.

Computes metrics (accuracy, precision, recall, F1, ROC-AUC, MCC) with
bootstrap 95% confidence intervals, and produces a comparison table
against the paper's reported values.

Usage:
    uv run python reproduction/scripts/analyze.py \
        reproduction/outputs/EXP-R-0002/o4-mini-2025-04-16/results.jsonl \
        reproduction/outputs/EXP-R-0002/claude-haiku-4-5-20251001/results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paper-reported values (Table 6, fraud_detection, summary+bs+cf+pl)
# Format: (mean, ± std) — paper reports 3-run std
PAPER_VALUES = {
    "o4-mini-2025-04-16": {"roc_auc": (0.52, 0.01), "mcc": (0.04, 0.05)},
    "claude-3-5-haiku-20241022": {"roc_auc": (0.60, 0.01), "mcc": (0.18, 0.03)},
}


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) == 2 else None,
        "mcc": matthews_corrcoef(labels, preds),
    }


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% CI for ROC-AUC and MCC."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    metrics_boot = {"roc_auc": [], "mcc": []}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bl, bp, bpr = labels[idx], preds[idx], probs[idx]
        # Skip degenerate samples (single class)
        if len(set(bl)) < 2:
            continue
        metrics_boot["roc_auc"].append(roc_auc_score(bl, bpr))
        metrics_boot["mcc"].append(matthews_corrcoef(bl, bp))

    alpha = (1 - ci) / 2
    cis = {}
    for key, vals in metrics_boot.items():
        arr = np.array(vals)
        cis[key] = (float(np.percentile(arr, 100 * alpha)), float(np.percentile(arr, 100 * (1 - alpha))))
    return cis


def analyze_one(path: Path) -> dict:
    results = load_results(path)
    model_id = path.parent.name
    total = len(results)

    # Filter out parse failures
    valid = [r for r in results if r["prediction"] is not None and r["prob"] is not None]
    n_failures = total - len(valid)

    labels = np.array([r["label"] for r in valid])
    preds = np.array([int(r["prediction"]) for r in valid])
    probs = np.array([float(r["prob"]) for r in valid])

    metrics = compute_metrics(labels, preds, probs)
    cis = bootstrap_ci(labels, preds, probs)

    # Load run metadata if available
    meta_path = path.parent / "run_meta.json"
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "model_id": model_id,
        "n_total": total,
        "n_valid": len(valid),
        "n_failures": n_failures,
        "metrics": metrics,
        "bootstrap_ci": cis,
        "cost_usd": run_meta.get("estimated_cost_usd"),
        "total_input_tokens": run_meta.get("total_input_tokens"),
        "total_output_tokens": run_meta.get("total_output_tokens"),
    }


def format_metric_with_ci(value: float, ci: tuple[float, float] | None) -> str:
    if ci is not None:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"


def format_paper_value(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.2f}"


def generate_summary(analyses: list[dict]) -> str:
    lines = []
    lines.append("# EXP-R-0002: Budgeted Reproduction — fraud_detection\n")
    lines.append(f"**Date:** {__import__('datetime').date.today().isoformat()}")
    lines.append("**Task:** fraud_detection")
    lines.append("**Sheets:** summary, bs, pl, cf")
    lines.append(f"**Sample:** N={analyses[0]['n_total']} (stratified, seed=42)\n")

    # Per-model details
    for a in analyses:
        m = a["metrics"]
        ci = a["bootstrap_ci"]
        lines.append(f"## {a['model_id']}\n")
        lines.append(f"- Valid examples: {a['n_valid']}/{a['n_total']} (parse failures: {a['n_failures']})")
        if a["cost_usd"] is not None:
            lines.append(f"- Cost: ${a['cost_usd']:.4f}")
        if a["total_input_tokens"] is not None:
            lines.append(f"- Tokens: {a['total_input_tokens']:,} in / {a['total_output_tokens']:,} out")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Accuracy  | {m['accuracy']:.3f} |")
        lines.append(f"| Precision | {m['precision']:.3f} |")
        lines.append(f"| Recall    | {m['recall']:.3f} |")
        lines.append(f"| F1        | {m['f1']:.3f} |")
        lines.append(f"| ROC-AUC   | {format_metric_with_ci(m['roc_auc'], ci.get('roc_auc'))} |")
        lines.append(f"| MCC       | {format_metric_with_ci(m['mcc'], ci.get('mcc'))} |")
        lines.append("")

    # Comparison table
    lines.append("## Comparison with Paper (Table 6)\n")
    lines.append("Paper config: fraud_detection, summary+bs+cf+pl, full test set (224 examples), 3 runs.\n")
    lines.append("| Model | Metric | Ours (N=50, 95% CI) | Paper (N=224, 3-run) |")
    lines.append("|-------|--------|---------------------|----------------------|")

    for a in analyses:
        m = a["metrics"]
        ci = a["bootstrap_ci"]
        model = a["model_id"]

        # Find matching paper model
        if model == "o4-mini-2025-04-16":
            paper_key = "o4-mini-2025-04-16"
            paper_label = "o4-mini (same model)"
        elif model == "claude-haiku-4-5-20251001":
            paper_key = "claude-3-5-haiku-20241022"
            paper_label = "Haiku 4.5 (paper: Haiku 3.5)"
        else:
            paper_key = None
            paper_label = model

        if paper_key and paper_key in PAPER_VALUES:
            pv = PAPER_VALUES[paper_key]
            lines.append(
                f"| {paper_label} | ROC-AUC | "
                f"{format_metric_with_ci(m['roc_auc'], ci.get('roc_auc'))} | "
                f"{format_paper_value(*pv['roc_auc'])} |"
            )
            lines.append(
                f"| | MCC | "
                f"{format_metric_with_ci(m['mcc'], ci.get('mcc'))} | "
                f"{format_paper_value(*pv['mcc'])} |"
            )
        else:
            lines.append(
                f"| {paper_label} | ROC-AUC | "
                f"{format_metric_with_ci(m['roc_auc'], ci.get('roc_auc'))} | — |"
            )
            lines.append(
                f"| | MCC | "
                f"{format_metric_with_ci(m['mcc'], ci.get('mcc'))} | — |"
            )

    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Bootstrap CIs: 1000 resamples, percentile method, seed=42.")
    lines.append("- Paper values from Table 6 (3-run mean +/- std).")
    lines.append("- Haiku 4.5 (`claude-haiku-4-5-20251001`) is the recommended successor to")
    lines.append("  Claude 3.5 Haiku (`claude-3-5-haiku-20241022`) used in the paper.")
    lines.append("  Results are not directly comparable due to model version difference.")
    lines.append("- Our sample is N=50 (stratified subsample); paper uses full test set N=224.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EDINET-Bench results")
    parser.add_argument("results", type=Path, nargs="+", help="Path(s) to results.jsonl")
    args = parser.parse_args()

    analyses = [analyze_one(p) for p in args.results]

    # Print to console
    for a in analyses:
        m = a["metrics"]
        ci = a["bootstrap_ci"]
        print(f"\n{'='*60}")
        print(f"Model: {a['model_id']}")
        print(f"Valid: {a['n_valid']}/{a['n_total']} (failures: {a['n_failures']})")
        print(f"Accuracy:  {m['accuracy']:.3f}")
        print(f"Precision: {m['precision']:.3f}")
        print(f"Recall:    {m['recall']:.3f}")
        print(f"F1:        {m['f1']:.3f}")
        print(f"ROC-AUC:   {format_metric_with_ci(m['roc_auc'], ci.get('roc_auc'))}")
        print(f"MCC:       {format_metric_with_ci(m['mcc'], ci.get('mcc'))}")
        if a["cost_usd"] is not None:
            print(f"Cost:      ${a['cost_usd']:.4f}")

    # Save summary
    summary = generate_summary(analyses)
    out_dir = REPO_ROOT / "reproduction" / "results" / "EXP-R-0002"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.md"
    out_path.write_text(summary)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
