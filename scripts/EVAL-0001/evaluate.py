"""EVAL-0001: End-to-end evaluation pipeline for EDINET-Bench experiments.

Usage:
    # Single-file mode
    uv run python scripts/EVAL-0001/evaluate.py results.jsonl

    # Comparison mode (agent vs baseline)
    uv run python scripts/EVAL-0001/evaluate.py results.jsonl --baseline baseline.jsonl

    # Filter to dev eval set
    uv run python scripts/EVAL-0001/evaluate.py results.jsonl --eval-set dev

    # Save report to file
    uv run python scripts/EVAL-0001/evaluate.py results.jsonl --output report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.common.metrics import (
    bootstrap_ci,
    compute_metrics,
    format_metric_with_ci,
    load_results,
)

DEV_EVAL_SET_PATH = REPO_ROOT / "experiments" / "EVAL-0001" / "dev_eval_set.json"
ERROR_CLASSIFICATIONS_PATH = (
    REPO_ROOT / "outputs" / "EXP-D-0001" / "error_classifications.jsonl"
)


def filter_results(results: list[dict], doc_ids: set[str]) -> list[dict]:
    """Filter results to only include specified doc_ids."""
    return [r for r in results if r["doc_id"] in doc_ids]


def load_eval_set_doc_ids(eval_set: str) -> set[str] | None:
    """Load doc_ids for the specified eval set. Returns None for no filtering."""
    if eval_set == "benchmark":
        return None
    if eval_set == "dev":
        data = json.loads(DEV_EVAL_SET_PATH.read_text())
        return set(data["doc_ids"])
    raise ValueError(f"Unknown eval set: {eval_set}")


def extract_arrays(results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract labels/preds/probs arrays from results, filtering parse failures.

    Returns (labels, preds, probs, n_parse_failures).
    """
    valid = [r for r in results if r["prediction"] is not None and r["prob"] is not None]
    n_failures = len(results) - len(valid)
    labels = np.array([r["label"] for r in valid])
    preds = np.array([int(r["prediction"]) for r in valid])
    probs = np.array([float(r["prob"]) for r in valid])
    return labels, preds, probs, n_failures


def confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    """Compute TP/FP/TN/FN counts."""
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def cost_summary(results: list[dict]) -> dict | None:
    """Extract cost info from results if token fields are present."""
    total_input = 0
    total_output = 0
    has_tokens = False
    for r in results:
        if "input_tokens" in r:
            total_input += r["input_tokens"]
            total_output += r.get("output_tokens", 0)
            has_tokens = True
    if not has_tokens:
        return None
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "n_examples": len(results),
    }


def analyze(results: list[dict]) -> dict:
    """Run full analysis on a results set."""
    labels, preds, probs, n_failures = extract_arrays(results)
    metrics = compute_metrics(labels, preds, probs)
    cis = bootstrap_ci(labels, preds, probs)
    cm = confusion_counts(labels, preds)
    cost = cost_summary(results)

    return {
        "n_total": len(results),
        "n_valid": len(results) - n_failures,
        "n_parse_failures": n_failures,
        "metrics": metrics,
        "bootstrap_ci": cis,
        "confusion_matrix": cm,
        "cost": cost,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def flip_analysis(
    baseline_results: list[dict],
    agent_results: list[dict],
) -> dict:
    """Analyze prediction transitions from baseline to agent."""
    base_by_id = {r["doc_id"]: r for r in baseline_results}
    agent_by_id = {r["doc_id"]: r for r in agent_results}
    common_ids = sorted(set(base_by_id) & set(agent_by_id))

    transitions: dict[str, list[dict]] = {}
    for doc_id in common_ids:
        br = base_by_id[doc_id]
        ar = agent_by_id[doc_id]

        bp = br.get("prediction")
        ap = ar.get("prediction")
        if bp is None or ap is None:
            continue

        bp, ap = int(bp), int(ap)
        label = br["label"]
        key = f"{bp} -> {ap}"

        correct_before = bp == label
        correct_after = ap == label

        transitions.setdefault(key, []).append({
            "doc_id": doc_id,
            "label": label,
            "baseline_pred": bp,
            "agent_pred": ap,
            "correct_before": correct_before,
            "correct_after": correct_after,
            "improved": not correct_before and correct_after,
            "degraded": correct_before and not correct_after,
        })

    summary = {}
    for key, items in sorted(transitions.items()):
        n_correct = sum(1 for i in items if i["correct_after"])
        n_incorrect = sum(1 for i in items if not i["correct_after"])
        n_improved = sum(1 for i in items if i["improved"])
        n_degraded = sum(1 for i in items if i["degraded"])
        summary[key] = {
            "count": len(items),
            "correct_after": n_correct,
            "incorrect_after": n_incorrect,
            "improved": n_improved,
            "degraded": n_degraded,
            "doc_ids": [i["doc_id"] for i in items],
        }

    return summary


def load_error_classifications() -> dict[str, list[dict]] | None:
    """Load error classifications if available. Keyed by doc_id."""
    if not ERROR_CLASSIFICATIONS_PATH.exists():
        return None
    classifications = {}
    with open(ERROR_CLASSIFICATIONS_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = r["doc_id"]
            classifications.setdefault(key, []).append(r)
    return classifications


def failure_mode_breakdown_from_results(
    baseline_results: list[dict],
    agent_results: list[dict],
    classifications: dict[str, list[dict]],
) -> dict[str, dict[str, int]]:
    """Break down improvements/degradations by failure mode."""
    base_by_id = {r["doc_id"]: r for r in baseline_results}
    agent_by_id = {r["doc_id"]: r for r in agent_results}
    common_ids = set(base_by_id) & set(agent_by_id)

    mode_stats: dict[str, dict[str, int]] = {}

    for doc_id in sorted(common_ids):
        br = base_by_id[doc_id]
        ar = agent_by_id[doc_id]
        bp = br.get("prediction")
        ap = ar.get("prediction")
        if bp is None or ap is None:
            continue

        bp, ap = int(bp), int(ap)
        label = br["label"]
        correct_before = bp == label
        correct_after = ap == label

        if doc_id not in classifications:
            continue

        modes = {c["primary_mode"] for c in classifications[doc_id]}
        for mode in modes:
            if mode not in mode_stats:
                mode_stats[mode] = {"improved": 0, "degraded": 0, "unchanged": 0}

            if not correct_before and correct_after:
                mode_stats[mode]["improved"] += 1
            elif correct_before and not correct_after:
                mode_stats[mode]["degraded"] += 1
            else:
                mode_stats[mode]["unchanged"] += 1

    return mode_stats


# --- Report formatting ---


def format_metrics_table(analysis: dict) -> str:
    """Format metrics as a markdown table."""
    m = analysis["metrics"]
    ci = analysis["bootstrap_ci"]
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy  | {m['accuracy']:.3f} |",
        f"| Precision | {m['precision']:.3f} |",
        f"| Recall    | {m['recall']:.3f} |",
        f"| F1        | {m['f1']:.3f} |",
        f"| ROC-AUC   | {format_metric_with_ci(m['roc_auc'], ci.get('roc_auc'))} |",
        f"| MCC       | {format_metric_with_ci(m['mcc'], ci.get('mcc'))} |",
    ]
    return "\n".join(lines)


def format_confusion_matrix(cm: dict[str, int]) -> str:
    """Format confusion matrix as markdown."""
    lines = [
        "|              | Pred=0 | Pred=1 |",
        "|--------------|--------|--------|",
        f"| **Label=0**  | TN={cm['TN']}  | FP={cm['FP']}   |",
        f"| **Label=1**  | FN={cm['FN']}  | TP={cm['TP']}   |",
    ]
    return "\n".join(lines)


def format_cost(cost: dict | None) -> str:
    """Format cost summary."""
    if cost is None:
        return "No token data available in results."
    lines = [
        f"- Total input tokens: {cost['total_input_tokens']:,}",
        f"- Total output tokens: {cost['total_output_tokens']:,}",
        f"- Examples: {cost['n_examples']}",
    ]
    return "\n".join(lines)


def format_comparison_table(a1: dict, a2: dict, label1: str, label2: str) -> str:
    """Format side-by-side comparison table."""
    m1, m2 = a1["metrics"], a2["metrics"]
    ci1, ci2 = a1["bootstrap_ci"], a2["bootstrap_ci"]
    lines = [
        f"| Metric | {label1} | {label2} |",
        "|--------|" + "-" * (len(label1) + 2) + "|" + "-" * (len(label2) + 2) + "|",
        f"| Accuracy  | {m1['accuracy']:.3f} | {m2['accuracy']:.3f} |",
        f"| Precision | {m1['precision']:.3f} | {m2['precision']:.3f} |",
        f"| Recall    | {m1['recall']:.3f} | {m2['recall']:.3f} |",
        f"| F1        | {m1['f1']:.3f} | {m2['f1']:.3f} |",
        f"| ROC-AUC   | {format_metric_with_ci(m1['roc_auc'], ci1.get('roc_auc'))} | {format_metric_with_ci(m2['roc_auc'], ci2.get('roc_auc'))} |",
        f"| MCC       | {format_metric_with_ci(m1['mcc'], ci1.get('mcc'))} | {format_metric_with_ci(m2['mcc'], ci2.get('mcc'))} |",
    ]
    return "\n".join(lines)


def format_flip_analysis(flips: dict[str, dict]) -> str:
    """Format flip analysis as markdown."""
    lines = ["| Transition | Count | Correct after | Incorrect after | Improved | Degraded |",
             "|-----------|-------|---------------|-----------------|----------|----------|"]
    for transition, info in sorted(flips.items()):
        lines.append(
            f"| {transition} | {info['count']} | {info['correct_after']} | "
            f"{info['incorrect_after']} | {info['improved']} | {info['degraded']} |"
        )
    return "\n".join(lines)


def format_failure_mode_breakdown(mode_stats: dict[str, dict[str, int]]) -> str:
    """Format failure mode breakdown as markdown."""
    lines = [
        "| Mode | Improved | Degraded | Unchanged |",
        "|------|----------|----------|-----------|",
    ]
    for mode in sorted(mode_stats):
        s = mode_stats[mode]
        lines.append(f"| {mode} | {s['improved']} | {s['degraded']} | {s['unchanged']} |")
    return "\n".join(lines)


def generate_report(
    results_path: Path,
    analysis: dict,
    baseline_path: Path | None = None,
    baseline_analysis: dict | None = None,
    baseline_results: list[dict] | None = None,
    agent_results: list[dict] | None = None,
    eval_set: str = "benchmark",
) -> str:
    """Generate the full evaluation report."""
    lines = []
    lines.append(f"# Evaluation Report")
    lines.append("")
    lines.append(f"**Results:** `{results_path}`")
    if baseline_path:
        lines.append(f"**Baseline:** `{baseline_path}`")
    lines.append(f"**Eval set:** {eval_set} (N={analysis['n_total']})")
    lines.append("")

    # 1. Metrics table
    lines.append("## Metrics")
    lines.append("")
    lines.append(format_metrics_table(analysis))
    lines.append("")

    # 2. Confusion matrix
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append(format_confusion_matrix(analysis["confusion_matrix"]))
    lines.append("")

    # 3. Cost summary
    lines.append("## Cost Summary")
    lines.append("")
    lines.append(format_cost(analysis["cost"]))
    lines.append("")

    # 4. Parse failures
    lines.append(f"**Parse failures:** {analysis['n_parse_failures']}/{analysis['n_total']}")
    lines.append("")

    # Comparison mode
    if baseline_analysis and baseline_results and agent_results:
        # 5. Comparison table
        lines.append("## Comparison")
        lines.append("")
        lines.append(format_comparison_table(
            baseline_analysis, analysis, "Baseline", "Agent"
        ))
        lines.append("")

        # 6. Flip analysis
        flips = flip_analysis(baseline_results, agent_results)
        lines.append("## Flip Analysis")
        lines.append("")
        lines.append(format_flip_analysis(flips))
        lines.append("")

        # Per-doc detail for flips
        for transition, info in sorted(flips.items()):
            if info["count"] > 0:
                lines.append(f"### {transition} ({info['count']} examples)")
                lines.append("")
                for doc_id in info["doc_ids"]:
                    lines.append(f"- `{doc_id}`")
                lines.append("")

        # 7. Per-failure-mode breakdown
        classifications = load_error_classifications()
        if classifications:
            mode_stats = failure_mode_breakdown_from_results(
                baseline_results, agent_results, classifications
            )
            if mode_stats:
                lines.append("## Failure Mode Breakdown")
                lines.append("")
                lines.append(format_failure_mode_breakdown(mode_stats))
                lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EVAL-0001: Evaluate EDINET-Bench experiment results"
    )
    parser.add_argument("results", type=Path, help="Path to results.jsonl")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to baseline results.jsonl for comparison mode",
    )
    parser.add_argument(
        "--eval-set", choices=["dev", "benchmark"], default="benchmark",
        help="Eval set to use: 'dev' (N=12) or 'benchmark' (all, default)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save report to markdown file",
    )
    args = parser.parse_args()

    # Load and optionally filter results
    results = load_results(args.results)
    doc_id_filter = load_eval_set_doc_ids(args.eval_set)
    if doc_id_filter is not None:
        results = filter_results(results, doc_id_filter)

    baseline_results = None
    baseline_analysis = None
    if args.baseline:
        baseline_results = load_results(args.baseline)
        if doc_id_filter is not None:
            baseline_results = filter_results(baseline_results, doc_id_filter)
        baseline_analysis = analyze(baseline_results)

    analysis = analyze(results)

    report = generate_report(
        results_path=args.results,
        analysis=analysis,
        baseline_path=args.baseline,
        baseline_analysis=baseline_analysis,
        baseline_results=baseline_results,
        agent_results=results,
        eval_set=args.eval_set,
    )

    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nReport saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
