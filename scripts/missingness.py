"""Missingness profiling for EXP-D-0001: failure mechanism diagnosis.

Computes per-example missingness density across financial sheets, then
correlates missingness with prediction errors from EXP-R-0002 and EXP-R-0003.

Usage:
    uv run python scripts/missingness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

# The missing-data indicator in EDINET-Bench is full-width dash (U+FF0D),
# though the prompt tells models it's represented as "-" (ASCII hyphen).
MISSING_INDICATORS = {"－", "-"}

SHEETS_NUMERIC = ["summary", "bs", "pl", "cf"]
SHEETS_ALL = ["summary", "bs", "pl", "cf", "text"]

RESULT_FILES = {
    ("o4-mini", "no-text"): REPO_ROOT / "reproduction/outputs/EXP-R-0002/o4-mini-2025-04-16/results.jsonl",
    ("haiku", "no-text"): REPO_ROOT / "reproduction/outputs/EXP-R-0002/claude-haiku-4-5-20251001/results.jsonl",
    ("o4-mini", "with-text"): REPO_ROOT / "reproduction/outputs/EXP-R-0003/o4-mini-2025-04-16/results.jsonl",
    ("haiku", "with-text"): REPO_ROOT / "reproduction/outputs/EXP-R-0003/claude-haiku-4-5-20251001/results.jsonl",
}

SAMPLE_PATH = REPO_ROOT / "reproduction/sampling/fraud_detection_n50_seed42.json"


# ── Missingness counting ────────────────────────────────────────────


def count_leaf_values(obj: object) -> tuple[int, int]:
    """Recursively count (total_leaves, missing_leaves) in a parsed JSON object."""
    if isinstance(obj, dict):
        total, missing = 0, 0
        for v in obj.values():
            t, m = count_leaf_values(v)
            total += t
            missing += m
        return total, missing
    if isinstance(obj, list):
        total, missing = 0, 0
        for v in obj:
            t, m = count_leaf_values(v)
            total += t
            missing += m
        return total, missing
    # Leaf value
    is_missing = str(obj).strip() in MISSING_INDICATORS
    return 1, int(is_missing)


def compute_missingness_profile(example: dict, sheets: list[str]) -> dict:
    """Compute per-sheet and aggregate missingness for one example."""
    profile = {}
    total_all, missing_all = 0, 0

    for sheet in sheets:
        raw = example.get(sheet)
        if raw is None:
            continue
        parsed = json.loads(raw)
        total, missing = count_leaf_values(parsed)
        density = missing / total if total > 0 else 0.0
        profile[sheet] = {"total": total, "missing": missing, "density": density}
        total_all += total
        missing_all += missing

    profile["aggregate"] = {
        "total": total_all,
        "missing": missing_all,
        "density": missing_all / total_all if total_all > 0 else 0.0,
    }
    return profile


# ── Result loading ──────────────────────────────────────────────────


def load_results(path: Path) -> dict[str, dict]:
    """Load results.jsonl, keyed by doc_id."""
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["doc_id"]] = r
    return results


# ── Statistical analysis ────────────────────────────────────────────


def analyze_condition(
    profiles: dict[str, dict],
    results: dict[str, dict],
    condition_label: str,
) -> dict:
    """Analyze missingness vs prediction correctness for one (model × config)."""
    doc_ids = sorted(set(profiles) & set(results))

    densities = []
    labels = []
    preds = []
    probs = []
    correct_flags = []

    for did in doc_ids:
        r = results[did]
        if r["prediction"] is None or r["prob"] is None:
            continue
        d = profiles[did]["aggregate"]["density"]
        densities.append(d)
        labels.append(r["label"])
        preds.append(int(r["prediction"]))
        probs.append(float(r["prob"]))
        correct_flags.append(int(r["prediction"]) == r["label"])

    densities = np.array(densities)
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    correct_flags = np.array(correct_flags)

    n = len(densities)
    n_correct = correct_flags.sum()
    n_errors = n - n_correct

    # FP and FN
    fp_mask = (preds == 1) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    analysis = {
        "condition": condition_label,
        "n": n,
        "n_correct": int(n_correct),
        "n_errors": int(n_errors),
        "n_fp": int(fp_mask.sum()),
        "n_fn": int(fn_mask.sum()),
    }

    # Missingness: correct vs incorrect
    d_correct = densities[correct_flags]
    d_errors = densities[~correct_flags]
    analysis["density_correct_mean"] = float(d_correct.mean()) if len(d_correct) > 0 else None
    analysis["density_correct_std"] = float(d_correct.std()) if len(d_correct) > 0 else None
    analysis["density_errors_mean"] = float(d_errors.mean()) if len(d_errors) > 0 else None
    analysis["density_errors_std"] = float(d_errors.std()) if len(d_errors) > 0 else None

    # Mann-Whitney U: correct vs errors
    if len(d_correct) >= 2 and len(d_errors) >= 2:
        u_stat, u_p = stats.mannwhitneyu(d_correct, d_errors, alternative="two-sided")
        analysis["mwu_correct_vs_errors"] = {"U": float(u_stat), "p": float(u_p)}
    else:
        analysis["mwu_correct_vs_errors"] = None

    # Missingness: FP vs FN
    d_fp = densities[fp_mask]
    d_fn = densities[fn_mask]
    analysis["density_fp_mean"] = float(d_fp.mean()) if len(d_fp) > 0 else None
    analysis["density_fn_mean"] = float(d_fn.mean()) if len(d_fn) > 0 else None

    if len(d_fp) >= 2 and len(d_fn) >= 2:
        u_stat, u_p = stats.mannwhitneyu(d_fp, d_fn, alternative="two-sided")
        analysis["mwu_fp_vs_fn"] = {"U": float(u_stat), "p": float(u_p)}
    else:
        analysis["mwu_fp_vs_fn"] = None

    # Spearman correlation: missingness vs predicted probability
    if n >= 5:
        rho, p_rho = stats.spearmanr(densities, probs)
        analysis["spearman_density_vs_prob"] = {"rho": float(rho), "p": float(p_rho)}
    else:
        analysis["spearman_density_vs_prob"] = None

    # Point-biserial: missingness vs prediction (binary)
    if n >= 5 and len(set(preds)) == 2:
        rpb, p_rpb = stats.pointbiserialr(preds, densities)
        analysis["pointbiserial_density_vs_pred"] = {"r": float(rpb), "p": float(p_rpb)}
    else:
        analysis["pointbiserial_density_vs_pred"] = None

    return analysis


# ── Report generation ───────────────────────────────────────────────


def format_report(
    analyses: list[dict],
    profiles: dict[str, dict],
    results_all: dict[tuple, dict[str, dict]],
) -> str:
    lines = []
    lines.append("# Missingness Analysis — EXP-D-0001\n")

    # Global missingness overview
    densities = [p["aggregate"]["density"] for p in profiles.values()]
    lines.append("## 1. Missingness Overview (N=50 sample)\n")
    lines.append(f"- Mean aggregate density: {np.mean(densities):.4f}")
    lines.append(f"- Std:  {np.std(densities):.4f}")
    lines.append(f"- Min:  {np.min(densities):.4f}")
    lines.append(f"- Max:  {np.max(densities):.4f}")
    lines.append(f"- Median: {np.median(densities):.4f}")

    # Per-sheet breakdown
    lines.append("\n### Per-sheet missingness\n")
    lines.append("| Sheet | Mean density | Std | Max |")
    lines.append("|-------|-------------|-----|-----|")
    for sheet in SHEETS_ALL:
        sheet_densities = [
            p[sheet]["density"] for p in profiles.values() if sheet in p
        ]
        if sheet_densities:
            lines.append(
                f"| {sheet} | {np.mean(sheet_densities):.4f} "
                f"| {np.std(sheet_densities):.4f} "
                f"| {np.max(sheet_densities):.4f} |"
            )
    lines.append("")

    # Per-condition analysis
    lines.append("## 2. Missingness vs Prediction Errors\n")
    for a in analyses:
        lines.append(f"### {a['condition']}\n")
        lines.append(f"- N valid: {a['n']}  |  Correct: {a['n_correct']}  |  Errors: {a['n_errors']} (FP: {a['n_fp']}, FN: {a['n_fn']})")

        lines.append(f"- Mean density — correct: {_fmt(a['density_correct_mean'])} +/- {_fmt(a['density_correct_std'])}")
        lines.append(f"- Mean density — errors:  {_fmt(a['density_errors_mean'])} +/- {_fmt(a['density_errors_std'])}")

        if a["mwu_correct_vs_errors"]:
            mwu = a["mwu_correct_vs_errors"]
            sig = "**" if mwu["p"] < 0.05 else ""
            lines.append(f"- Mann-Whitney U (correct vs errors): U={mwu['U']:.1f}, p={mwu['p']:.4f} {sig}")
        else:
            lines.append("- Mann-Whitney U (correct vs errors): insufficient data")

        if a["density_fp_mean"] is not None:
            lines.append(f"- Mean density — FP: {_fmt(a['density_fp_mean'])}  |  FN: {_fmt(a['density_fn_mean'])}")
        if a["mwu_fp_vs_fn"]:
            mwu = a["mwu_fp_vs_fn"]
            sig = "**" if mwu["p"] < 0.05 else ""
            lines.append(f"- Mann-Whitney U (FP vs FN): U={mwu['U']:.1f}, p={mwu['p']:.4f} {sig}")

        if a["spearman_density_vs_prob"]:
            sp = a["spearman_density_vs_prob"]
            sig = "**" if sp["p"] < 0.05 else ""
            lines.append(f"- Spearman (density vs prob): rho={sp['rho']:.3f}, p={sp['p']:.4f} {sig}")

        if a.get("pointbiserial_density_vs_pred"):
            pb = a["pointbiserial_density_vs_pred"]
            sig = "**" if pb["p"] < 0.05 else ""
            lines.append(f"- Point-biserial (density vs prediction): r={pb['r']:.3f}, p={pb['p']:.4f} {sig}")

        lines.append("")

    # Summary table
    lines.append("## 3. Summary Table\n")
    lines.append("| Condition | Errors | Density (correct) | Density (errors) | MWU p | Spearman rho | Spearman p |")
    lines.append("|-----------|--------|-------------------|------------------|-------|-------------|------------|")
    for a in analyses:
        sp = a.get("spearman_density_vs_prob") or {}
        mwu = a.get("mwu_correct_vs_errors") or {}
        lines.append(
            f"| {a['condition']} | {a['n_errors']}/{a['n']} "
            f"| {_fmt(a['density_correct_mean'])} "
            f"| {_fmt(a['density_errors_mean'])} "
            f"| {_fmt(mwu.get('p'))} "
            f"| {_fmt(sp.get('rho'))} "
            f"| {_fmt(sp.get('p'))} |"
        )
    lines.append("")

    return "\n".join(lines)


def _fmt(val, decimals=4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    # Load frozen sample
    sample = json.loads(SAMPLE_PATH.read_text())
    doc_ids = set(sample["doc_ids"])

    # Load dataset and filter to sample
    print("Loading dataset...")
    ds = load_dataset("SakanaAI/EDINET-Bench", "fraud_detection", split="test")
    ds = ds.filter(lambda ex: ex["doc_id"] in doc_ids)
    assert len(ds) == len(doc_ids), f"Expected {len(doc_ids)}, got {len(ds)}"

    # Compute missingness profiles
    print("Computing missingness profiles...")
    profiles: dict[str, dict] = {}
    for ex in ds:
        did = ex["doc_id"]
        profiles[did] = compute_missingness_profile(ex, SHEETS_ALL)

    # Save raw profiles for downstream use
    profiles_path = REPO_ROOT / "outputs/EXP-D-0001/missingness_profiles.json"
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved profiles to {profiles_path}")

    # Load all result sets
    print("Loading result sets...")
    results_all = {}
    for key, path in RESULT_FILES.items():
        results_all[key] = load_results(path)

    # Run analysis per condition
    condition_labels = {
        ("o4-mini", "no-text"): "o4-mini / no-text (R-0002)",
        ("haiku", "no-text"): "Haiku 4.5 / no-text (R-0002)",
        ("o4-mini", "with-text"): "o4-mini / with-text (R-0003)",
        ("haiku", "with-text"): "Haiku 4.5 / with-text (R-0003)",
    }

    analyses = []
    for key in RESULT_FILES:
        a = analyze_condition(profiles, results_all[key], condition_labels[key])
        analyses.append(a)

    # Generate report
    report = format_report(analyses, profiles, results_all)

    out_path = REPO_ROOT / "experiments/EXP-D-0001/missingness_analysis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport saved to {out_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
