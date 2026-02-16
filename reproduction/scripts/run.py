"""Run EDINET-Bench experiment with token usage tracking.

Bypasses upstream model classes to capture token counts from API responses.
Reuses upstream extract_json_between_markers for response parsing.
Replicates prompt construction logic from upstream process_example.

Usage:
    uv run python reproduction/scripts/run.py configs/EXP-R-0002_o4-mini.yaml
    uv run python reproduction/scripts/run.py configs/EXP-R-0002_o4-mini.yaml --limit 5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import backoff
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env before any SDK imports (upstream utils.py calls load_dotenv at import time)
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

import anthropic  # noqa: E402
import openai  # noqa: E402
from edinet_bench.utils import extract_json_between_markers  # noqa: E402

# ---------- Pricing (USD per 1M tokens, as of 2025-04) ----------
PRICING = {
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


@dataclass
class RunResult:
    doc_id: str
    edinet_code: str
    label: int
    prediction: int | None
    prob: float | None
    reasoning: str | None
    input_tokens: int
    output_tokens: int
    elapsed_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- API callers with usage capture ----------

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIError, openai.APITimeoutError),
    max_tries=5,
)
def call_openai(client: openai.OpenAI, model_id: str, system_prompt: str, user_prompt: str):
    """Call OpenAI API, return (text, input_tokens, output_tokens).

    Replicates upstream OpenAIModel.get_completion message format exactly.
    For o4-mini: no temperature, max_tokens, or seed (matches upstream).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if model_id == "o4-mini-2025-04-16":
        response = client.chat.completions.create(model=model_id, messages=messages)
    else:
        response = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=4096, temperature=0.0, seed=0,
        )
    text = response.choices[0].message.content
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens


@backoff.on_exception(
    backoff.expo,
    (anthropic.RateLimitError, anthropic.APIError, anthropic.InternalServerError),
    max_tries=5,
)
def call_anthropic(client: anthropic.Anthropic, model_id: str, system_prompt: str, user_prompt: str):
    """Call Anthropic API, return (text, input_tokens, output_tokens).

    Replicates upstream AnthropicModel.get_completion message format exactly.
    """
    response = client.messages.create(
        model=model_id,
        system=system_prompt,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        max_tokens=4096,
        temperature=0.0,
    )
    text = response.content[0].text
    return text, response.usage.input_tokens, response.usage.output_tokens


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
}

CLIENT_FACTORIES = {
    "openai": openai.OpenAI,
    "anthropic": anthropic.Anthropic,
}


def build_prompt(base_prompt: str, example: dict, sheets: list[str]) -> str:
    """Replicate upstream process_example prompt construction."""
    return base_prompt + "\n".join(
        f"{sheet}: {example[sheet]}" for sheet in sheets if sheet in example
    )


def run_one(
    example: dict,
    client,
    caller_fn,
    model_id: str,
    system_prompt: str,
    base_prompt: str,
    sheets: list[str],
) -> RunResult:
    prompt = build_prompt(base_prompt, example, sheets)
    t0 = time.monotonic()
    text, in_tok, out_tok = caller_fn(client, model_id, system_prompt, prompt)
    elapsed = time.monotonic() - t0

    json_data = extract_json_between_markers(text)
    if json_data is None:
        prob, prediction, reasoning = None, None, None
    else:
        prob = json_data.get("prob")
        prediction = json_data.get("prediction")
        reasoning = json_data.get("reasoning")

    return RunResult(
        doc_id=example["doc_id"],
        edinet_code=example["edinet_code"],
        label=example["label"],
        prediction=prediction,
        prob=prob,
        reasoning=reasoning,
        input_tokens=in_tok,
        output_tokens=out_tok,
        elapsed_sec=round(elapsed, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDINET-Bench experiment")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N examples (for pilot)")
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(args.config.read_text())
    model_id = cfg["model"]["id"]
    provider = cfg["model"]["provider"]
    task = cfg["task"]
    sheets = cfg["sheets"]
    system_prompt = cfg["system_prompt"]
    experiment_id = cfg["experiment_id"]

    # Load frozen sample
    sample_path = REPO_ROOT / cfg["sample"]
    sample_data = json.loads(sample_path.read_text())
    doc_ids = set(sample_data["doc_ids"])

    # Load dataset & filter to sample
    ds = load_dataset("SakanaAI/EDINET-Bench", task, split="test")
    ds = ds.filter(lambda ex: ex["doc_id"] in doc_ids)
    assert len(ds) == len(doc_ids), f"Expected {len(doc_ids)} examples, got {len(ds)}"

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Pilot mode: running {len(ds)} examples")

    # Load prompt
    prompt_path = REPO_ROOT / "external" / "EDINET-Bench" / "prompt" / f"{task}.yaml"
    base_prompt = yaml.safe_load(prompt_path.read_text())["prompt"]

    # Set up API client
    client = CLIENT_FACTORIES[provider]()
    caller_fn = CALLERS[provider]

    # Run
    results: list[RunResult] = []
    total_in, total_out = 0, 0
    parse_failures = 0
    t_start = time.monotonic()

    for example in tqdm(ds, desc=f"{model_id}"):
        result = run_one(example, client, caller_fn, model_id, system_prompt, base_prompt, sheets)
        results.append(result)
        total_in += result.input_tokens
        total_out += result.output_tokens
        if result.prediction is None:
            parse_failures += 1

    total_elapsed = time.monotonic() - t_start

    # Cost estimate
    prices = PRICING.get(model_id, {"input": 0.0, "output": 0.0})
    cost = (total_in * prices["input"] + total_out * prices["output"]) / 1_000_000

    # Save results
    out_dir = REPO_ROOT / "reproduction" / "outputs" / experiment_id / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    run_meta = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "provider": provider,
        "task": task,
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
    print(f"Model:           {model_id}")
    print(f"Examples:        {len(results)}")
    print(f"Parse failures:  {parse_failures}")
    print(f"Input tokens:    {total_in:,}")
    print(f"Output tokens:   {total_out:,}")
    print(f"Estimated cost:  ${cost:.4f}")
    print(f"Wall-clock time: {total_elapsed:.1f}s")
    print(f"Results:         {results_path}")
    print(f"Metadata:        {meta_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
