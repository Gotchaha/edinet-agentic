"""Reusable evaluation metrics for EDINET-Bench experiments.

Decoupled from reproduction/scripts/analyze.py for use across experiments.
"""

from __future__ import annotations

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


def load_results(path: str | Path) -> list[dict]:
    """Load results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Compute classification metrics."""
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
    metrics_boot: dict[str, list[float]] = {"roc_auc": [], "mcc": []}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bl, bp, bpr = labels[idx], preds[idx], probs[idx]
        if len(set(bl)) < 2:
            continue
        metrics_boot["roc_auc"].append(roc_auc_score(bl, bpr))
        metrics_boot["mcc"].append(matthews_corrcoef(bl, bp))

    alpha = (1 - ci) / 2
    cis = {}
    for key, vals in metrics_boot.items():
        arr = np.array(vals)
        cis[key] = (
            float(np.percentile(arr, 100 * alpha)),
            float(np.percentile(arr, 100 * (1 - alpha))),
        )
    return cis


def format_metric_with_ci(value: float, ci: tuple[float, float] | None) -> str:
    """Format a metric value with optional confidence interval."""
    if ci is not None:
        return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{value:.3f}"
