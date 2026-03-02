"""Run EXP-A-0001 baseline reflection agent.

Usage:
    uv run python scripts/EXP-A-0001/run.py --config configs/EXP-A-0001.yaml
    uv run python scripts/EXP-A-0001/run.py --config configs/EXP-A-0001.yaml --limit 1
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from agents.baseline.graph import build_graph  # noqa: E402
from common.data import (  # noqa: E402
    build_sheets_text,
    load_dataset_for_sample,
    load_prompt_template,
    load_sample,
)
from common.parsing import extract_json_between_markers  # noqa: E402

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


def run_one(graph, example: dict, base_prompt: str, sheets: list[str]) -> dict:
    """Run the agent on a single example and return result + trace."""
    sheets_text = build_sheets_text(example, sheets)

    initial_state = {
        "messages": [],
        "doc_id": example["doc_id"],
        "sheets_text": sheets_text,
        "base_prompt": base_prompt,
        "final_prediction": None,
        "final_prob": None,
        "final_reasoning": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    t0 = time.monotonic()
    final_state = graph.invoke(initial_state)
    elapsed = time.monotonic() - t0

    # If reviser didn't parse, try parsing generator output as fallback
    prediction = final_state.get("final_prediction")
    prob = final_state.get("final_prob")
    reasoning = final_state.get("final_reasoning")

    if prediction is None and final_state["messages"]:
        # Try the last message (reviser), then first message (generator)
        for msg in [final_state["messages"][-1], final_state["messages"][0]]:
            parsed = extract_json_between_markers(msg.content)
            if parsed:
                prediction = parsed.get("prediction")
                prob = parsed.get("prob")
                reasoning = parsed.get("reasoning")
                break

    result = {
        "doc_id": example["doc_id"],
        "edinet_code": example["edinet_code"],
        "label": example["label"],
        "prediction": prediction,
        "prob": prob,
        "reasoning": reasoning,
        "input_tokens": final_state.get("input_tokens", 0),
        "output_tokens": final_state.get("output_tokens", 0),
        "elapsed_sec": round(elapsed, 2),
    }

    trace = {
        "doc_id": example["doc_id"],
        "messages": [
            {"role": msg.type, "content": msg.content}
            for msg in final_state["messages"]
        ],
    }

    return result, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDINET-Bench agent experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N examples")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model_id = cfg["model"]["id"]
    task = cfg["task"]
    sheets = cfg["sheets"]
    experiment_id = cfg["experiment_id"]

    doc_ids = load_sample(cfg["sample"])
    ds = load_dataset_for_sample(task, doc_ids)
    base_prompt = load_prompt_template(task)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Pilot mode: running {len(ds)} examples")

    graph = build_graph(model_id)

    results = []
    traces = []
    total_in, total_out = 0, 0
    parse_failures = 0
    t_start = time.monotonic()

    for example in tqdm(ds, desc=f"{experiment_id}/{model_id}"):
        result, trace = run_one(graph, example, base_prompt, sheets)
        results.append(result)
        traces.append(trace)
        total_in += result["input_tokens"]
        total_out += result["output_tokens"]
        if result["prediction"] is None:
            parse_failures += 1

    total_elapsed = time.monotonic() - t_start

    # Cost estimate
    prices = PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    cost = (total_in * prices["input"] + total_out * prices["output"]) / 1_000_000

    # Save outputs
    out_dir = REPO_ROOT / "outputs" / experiment_id / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    traces_path = out_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    run_meta = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "task": task,
        "agent_type": cfg["agent"]["type"],
        "agent_rounds": cfg["agent"]["rounds"],
        "n_examples": len(results),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": round(cost, 4),
        "total_elapsed_sec": round(total_elapsed, 1),
        "parse_failures": parse_failures,
        "config": cfg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Experiment:      {experiment_id}")
    print(f"Model:           {model_id}")
    print(f"Agent:           {cfg['agent']['type']} (rounds={cfg['agent']['rounds']})")
    print(f"Examples:        {len(results)}")
    print(f"Parse failures:  {parse_failures}")
    print(f"Input tokens:    {total_in:,}")
    print(f"Output tokens:   {total_out:,}")
    print(f"Estimated cost:  ${cost:.4f}")
    print(f"Wall-clock time: {total_elapsed:.1f}s")
    print(f"Results:         {results_path}")
    print(f"Traces:          {traces_path}")
    print(f"Metadata:        {meta_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
