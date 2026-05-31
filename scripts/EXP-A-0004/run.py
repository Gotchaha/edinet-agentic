"""Run EXP-A-0004 ReAct agent with field-level retrieval tools.

Usage:
    uv run python scripts/EXP-A-0004/run.py --config configs/EXP-A-0004.yaml
    uv run python scripts/EXP-A-0004/run.py --config configs/EXP-A-0004.yaml --limit 1
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

from agents.tool_augmented.graph import build_graph  # noqa: E402
from agents.tool_augmented.prompts import SYSTEM_PROMPT  # noqa: E402
from common.data import (  # noqa: E402
    load_dataset_for_sample,
    load_prompt_template,
    load_sample,
)
from common.parsing import extract_json_between_markers  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


def build_sheets_listing(sheets: list[str]) -> str:
    """Build the data section appended after base_prompt.

    A-0003 appended JSON-serialized sheet contents here. A-0004 appends
    only a brief listing of available sheets — values are accessed via
    tools.
    """
    return (
        "\nThe report data is accessible via the available tools "
        "(list_attributes, get_attribute, calculator). "
        f"Sheets exposed: {', '.join(sheets)}.\n"
    )


def _serialize_message(msg) -> dict:
    """Serialize a LangChain message for trace output, including tool calls."""
    content = msg.content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    entry = {"role": msg.type, "content": content}
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        entry["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"]}
            for tc in msg.tool_calls
        ]
    return entry


def run_one(graph, example: dict, base_prompt: str, sheets: list[str]) -> tuple[dict, dict, bool]:
    """Run the agent on a single example.

    Returns (result, trace, recursion_error).
    """
    sheets_listing = build_sheets_listing(sheets)

    initial_state = {
        "messages": [],
        "doc_id": example["doc_id"],
        "sheets_listing": sheets_listing,
        "base_prompt": base_prompt,
        "final_prediction": None,
        "final_prob": None,
        "final_reasoning": None,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    recursion_error = False
    t0 = time.monotonic()
    try:
        final_state = graph.invoke(initial_state)
    except GraphRecursionError:
        recursion_error = True
        final_state = {
            "messages": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "final_prediction": None,
            "final_prob": None,
            "final_reasoning": "GraphRecursionError",
        }
    elapsed = time.monotonic() - t0

    prediction = final_state.get("final_prediction")
    prob = final_state.get("final_prob")
    reasoning = final_state.get("final_reasoning")

    # Fallback: parse from messages if agent node didn't set finals
    if prediction is None and final_state.get("messages"):
        for msg in reversed(final_state["messages"]):
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if content:
                parsed = extract_json_between_markers(content)
                if parsed and "prediction" in parsed:
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
            _serialize_message(msg) for msg in final_state.get("messages", [])
        ],
    }
    if recursion_error:
        trace["recursion_error"] = True

    return result, trace, recursion_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDINET-Bench agent experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N examples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override default output directory (outputs/<experiment_id>/<model_id>/)",
    )
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

    results = []
    traces = []
    total_in, total_out = 0, 0
    parse_failures = 0
    recursion_errors = 0
    t_start = time.monotonic()

    for example in tqdm(ds, desc=f"{experiment_id}/{model_id}"):
        # Rebuild graph per example: retrieval tools capture this example via closure
        graph = build_graph(model_id, example)
        result, trace, recursion_error = run_one(graph, example, base_prompt, sheets)
        results.append(result)
        traces.append(trace)
        total_in += result["input_tokens"]
        total_out += result["output_tokens"]
        if result["prediction"] is None:
            parse_failures += 1
        if recursion_error:
            recursion_errors += 1

    total_elapsed = time.monotonic() - t_start

    # Cost estimate
    prices = PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    cost = (total_in * prices["input"] + total_out * prices["output"]) / 1_000_000

    # Save outputs
    if args.output_dir is not None:
        out_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
        out_dir = out_dir.resolve()
    else:
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
        "agent_tools": cfg["agent"].get("tools", []),
        "system_prompt": SYSTEM_PROMPT,
        "n_examples": len(results),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_cost_usd": round(cost, 4),
        "total_elapsed_sec": round(total_elapsed, 1),
        "parse_failures": parse_failures,
        "recursion_errors": recursion_errors,
        "config": cfg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Experiment:        {experiment_id}")
    print(f"Model:             {model_id}")
    print(f"Agent:             {cfg['agent']['type']} (tools={cfg['agent'].get('tools', [])})")
    print(f"Examples:          {len(results)}")
    print(f"Parse failures:    {parse_failures}")
    print(f"Recursion errors:  {recursion_errors}")
    print(f"Input tokens:      {total_in:,}")
    print(f"Output tokens:     {total_out:,}")
    print(f"Estimated cost:    ${cost:.4f}")
    print(f"Wall-clock time:   {total_elapsed:.1f}s")
    print(f"Results:           {results_path}")
    print(f"Traces:            {traces_path}")
    print(f"Metadata:          {meta_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
