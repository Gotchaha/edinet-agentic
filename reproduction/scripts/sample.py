"""Stratified sampling of EDINET-Bench test split.

Produces a frozen sample list (JSON) for reproducible experiment runs.
Uses doc_id as the stable identifier (not index).

Usage:
    uv run python reproduction/scripts/sample.py --n 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK = "fraud_detection"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified sampling for EDINET-Bench")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reproduction" / "sampling",
    )
    args = parser.parse_args()

    # Load dataset
    ds = load_dataset("SakanaAI/EDINET-Bench", TASK, split="test")
    total = len(ds)
    labels = [ex["label"] for ex in ds]
    doc_ids = [ex["doc_id"] for ex in ds]

    # Label distribution in full set
    n_pos = sum(labels)
    n_neg = total - n_pos
    print(f"Full set: {total} examples ({n_pos} fraud, {n_neg} non-fraud)")

    # Stratified sampling
    sampled_indices, _ = train_test_split(
        list(range(total)),
        train_size=args.n,
        random_state=args.seed,
        stratify=labels,
    )
    sampled_indices.sort()  # deterministic ordering

    sampled_doc_ids = [doc_ids[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    sample_n_pos = sum(sampled_labels)
    sample_n_neg = args.n - sample_n_pos

    print(f"Sample:   {args.n} examples ({sample_n_pos} fraud, {sample_n_neg} non-fraud)")
    print(f"Ratio — full: {n_pos/total:.3f}, sample: {sample_n_pos/args.n:.3f}")

    # Build output
    output = {
        "metadata": {
            "task": TASK,
            "seed": args.seed,
            "n": args.n,
            "total_available": total,
            "label_distribution": {
                "fraud": sample_n_pos,
                "non_fraud": sample_n_neg,
            },
            "full_label_distribution": {
                "fraud": n_pos,
                "non_fraud": n_neg,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "doc_ids": sampled_doc_ids,
    }

    # Write
    out_path = args.output_dir / f"{TASK}_n{args.n}_seed{args.seed}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
