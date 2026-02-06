"""EXP-R-0001: Smoke test — verify upstream pipeline end-to-end on 3 examples."""

import json
import yaml
import datasets
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from edinet_bench.model import MODEL_TABLE  # noqa: E402
from edinet_bench.predict import process_example, Result  # noqa: E402

TASK = "fraud_detection"
MODEL_ID = "o4-mini-2025-04-16"
NUM_EXAMPLES = 3
SHEETS = ["summary", "bs", "pl", "cf"]
OUTPUT_DIR = REPO_ROOT / "reproduction" / "outputs" / "EXP-R-0001"


def main():
    # Load prompt from upstream repo (absolute path)
    prompt_path = REPO_ROOT / "external" / "EDINET-Bench" / "prompt" / f"{TASK}.yaml"
    with open(prompt_path) as f:
        prompt = yaml.safe_load(f)["prompt"]

    # Load dataset (first N examples)
    ds = datasets.load_dataset("SakanaAI/EDINET-Bench", TASK, split="test")
    ds = ds.select(range(NUM_EXAMPLES))

    # Model from upstream MODEL_TABLE
    model = MODEL_TABLE[MODEL_ID](MODEL_ID, "You are a financial analyst.")

    # Run predictions sequentially
    results: list[Result] = []
    for i, example in enumerate(ds):
        print(f"[{i+1}/{NUM_EXAMPLES}] edinet_code={example['edinet_code']}")
        result = process_example(example, model, prompt, SHEETS)
        results.append(result)
        print(f"  label={result.label} pred={result.prediction} prob={result.prob}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{TASK}_smoke.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # Summary
    n_ok = sum(1 for r in results if r.prediction is not None)
    print(f"\nDone. {n_ok}/{NUM_EXAMPLES} parsed successfully. Output: {out_path}")


if __name__ == "__main__":
    main()
