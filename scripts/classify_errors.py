"""LLM-assisted failure mode classification — EXP-D-0001, Phase 2.3.

Classifies each error case's reasoning into failure mode categories using
GPT-5 via OpenAI Batch API. Each request includes raw sheet data so the
classifier can verify the model's claims against actual financial data.

Subcommands:
    uv run python scripts/classify_errors.py submit    # upload batch, print batch_id
    uv run python scripts/classify_errors.py status    # check batch status
    uv run python scripts/classify_errors.py download  # download results when complete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

ERRORS_PATH = REPO_ROOT / "outputs/EXP-D-0001/errors_for_audit.jsonl"
BATCH_INPUT_PATH = REPO_ROOT / "outputs/EXP-D-0001/batch_input.jsonl"
BATCH_META_PATH = REPO_ROOT / "outputs/EXP-D-0001/batch_meta.json"
OUT_PATH = REPO_ROOT / "outputs/EXP-D-0001/error_classifications.jsonl"

SAMPLE_PATH = REPO_ROOT / "reproduction/sampling/fraud_detection_n50_seed42.json"

CLASSIFIER_MODEL = "gpt-5"

SHEETS_NO_TEXT = ["summary", "bs", "pl", "cf"]
SHEETS_WITH_TEXT = ["summary", "bs", "pl", "cf", "text"]

TAXONOMY = """\
## Failure Mode Taxonomy

Classify the model's reasoning into one or more of these failure modes. Use the
code that best describes the PRIMARY cause of the incorrect prediction.

| Code | Failure mode | Definition |
|------|-------------|------------|
| M1 | Missing-as-anomaly | Treats missing data ("-" or absent fields) as a suspicious signal, when it is just a parsing artifact |
| M2 | Evidence drift | Cites observations that don't logically support the conclusion; reasoning chain is disconnected |
| M3 | Magnitude insensitivity | Flags normal business variation (revenue fluctuations, working capital changes) as fraud signals |
| M4 | Irrelevant anchoring | Fixates on dramatic-sounding but non-diagnostic items (large absolute numbers, foreign-sounding terms) |
| M5 | Base rate neglect | Fails to calibrate probability against actual fraud prevalence; treats any anomaly as high-probability fraud |
| M6 | Analytical limitation | Genuinely lacks domain knowledge needed for this case; analysis is competent but the task is too hard |
| M7 | Conservatism bias | For FN errors: dismisses genuine red flags, gives unwarranted benefit-of-the-doubt, or anchors on "numbers are consistent" |
| M8 | Other | Does not fit the above categories (describe in notes) |

For FP errors (predicted fraud, actually non-fraud): M1-M6 are most relevant.
For FN errors (predicted non-fraud, actually fraud): M7, M6, M2 are most relevant.
"""

SYSTEM_PROMPT = f"""\
You are an expert financial analyst and error auditor. Your task is to classify
why a fraud-detection model made an incorrect prediction by analyzing its
reasoning against the actual financial data provided.

{TAXONOMY}

You are given:
1. The model's reasoning for its (incorrect) prediction
2. The actual raw financial data sheets the model was given
3. Metadata (error type, predicted probability, missingness info)

Use the raw data to VERIFY claims in the model's reasoning. For example:
- If the model says "revenue declined sharply", check the actual PL data
- If the model flags missing data, check actual missingness density
- If the model cites specific numbers, verify them against the sheets

Respond in JSON format:
```json
{{
  "primary_mode": "M1",
  "secondary_mode": "M3",
  "confidence": "high",
  "notes": "Brief explanation of why this classification fits"
}}
```

Rules:
- primary_mode is REQUIRED (one of M1-M8)
- secondary_mode is OPTIONAL (null if not applicable)
- confidence is one of: "high", "medium", "low"
- notes should be 1-2 sentences max
- Focus on the REASONING quality, not whether the prediction was right/wrong
"""


def load_errors() -> list[dict]:
    errors = []
    with open(ERRORS_PATH) as f:
        for line in f:
            errors.append(json.loads(line))
    return errors


def load_sheet_data() -> dict[str, dict[str, str]]:
    """Load raw sheet data from dataset, keyed by doc_id."""
    sample = json.loads(SAMPLE_PATH.read_text())
    doc_ids = set(sample["doc_ids"])

    print("Loading dataset for sheet data...")
    ds = load_dataset("SakanaAI/EDINET-Bench", "fraud_detection", split="test")
    ds = ds.filter(lambda ex: ex["doc_id"] in doc_ids)

    sheet_data: dict[str, dict[str, str]] = {}
    for ex in ds:
        did = ex["doc_id"]
        sheet_data[did] = {}
        for sheet in SHEETS_WITH_TEXT:
            if ex.get(sheet):
                sheet_data[did][sheet] = ex[sheet]

    print(f"Loaded sheet data for {len(sheet_data)} documents")
    return sheet_data


def build_user_message(error: dict, sheets: dict[str, str]) -> str:
    """Build the user message for one error case."""
    # Determine which sheets apply based on config
    if error["config"] == "no-text":
        sheet_names = SHEETS_NO_TEXT
    else:
        sheet_names = SHEETS_WITH_TEXT

    parts = []
    parts.append("## Case to Classify\n")
    parts.append(f"- Model: {error['model']}")
    parts.append(f"- Config: {error['config']}")
    parts.append(f"- Error type: {error['error_type']} (label={error['label']}, prediction={error['prediction']})")
    parts.append(f"- Predicted probability: {error['prob']}")
    parts.append(f"- Missingness density: {error.get('missingness_density', 'unknown')}")

    parts.append("\n## Model's Reasoning\n")
    parts.append(error["reasoning"])

    parts.append("\n## Raw Financial Data (sheets the model was given)\n")
    for sheet in sheet_names:
        raw = sheets.get(sheet, "")
        if raw:
            # Truncate very long text sheets to stay within token limits
            if sheet == "text" and len(raw) > 30000:
                raw = raw[:30000] + "\n... [truncated]"
            parts.append(f"### {sheet.upper()}\n")
            parts.append(raw)
            parts.append("")

    return "\n".join(parts)


def build_batch_request(custom_id: str, user_message: str) -> dict:
    """Build a single batch request in OpenAI Batch API format."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": CLASSIFIER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
    }


# ── Subcommands ────────────────────────────────────────────────────


def cmd_submit(args: argparse.Namespace) -> None:
    """Build batch input, upload, and create batch."""
    errors = load_errors()
    sheet_data = load_sheet_data()

    print(f"Building batch input for {len(errors)} error cases...")

    BATCH_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_est_tokens = 0

    with open(BATCH_INPUT_PATH, "w") as f:
        for error in errors:
            did = error["doc_id"]
            sheets = sheet_data.get(did, {})
            user_msg = build_user_message(error, sheets)

            custom_id = f"{did}__{error['model']}__{error['config']}"
            request = build_batch_request(custom_id, user_msg)
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

            # Rough token estimate (1 token ~ 4 chars)
            est = (len(SYSTEM_PROMPT) + len(user_msg)) // 4
            total_est_tokens += est

    print(f"Wrote {len(errors)} requests to {BATCH_INPUT_PATH}")
    print(f"Estimated input tokens: ~{total_est_tokens:,}")
    est_cost = total_est_tokens * 0.625 / 1_000_000  # GPT-5 batch input: $0.625/M
    est_cost += len(errors) * 800 * 5.0 / 1_000_000  # output incl reasoning: $5.00/M
    print(f"Estimated batch cost: ~${est_cost:.2f}")

    if args.dry_run:
        print("Dry run — not uploading to OpenAI.")
        return

    client = OpenAI()

    # Upload input file
    print("Uploading batch input file...")
    with open(BATCH_INPUT_PATH, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {upload.id}")

    # Create batch
    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "EXP-D-0001", "phase": "2.3"},
    )
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.status}")

    # Save metadata
    meta = {
        "batch_id": batch.id,
        "input_file_id": upload.id,
        "n_requests": len(errors),
        "model": CLASSIFIER_MODEL,
        "status": batch.status,
    }
    BATCH_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved batch metadata to {BATCH_META_PATH}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check batch status."""
    if not BATCH_META_PATH.exists():
        print(f"No batch metadata found at {BATCH_META_PATH}")
        print("Run 'submit' first.")
        sys.exit(1)

    meta = json.loads(BATCH_META_PATH.read_text())
    client = OpenAI()

    batch = client.batches.retrieve(meta["batch_id"])
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Requests — completed: {batch.request_counts.completed}, "
              f"failed: {batch.request_counts.failed}, "
              f"total: {batch.request_counts.total}")
    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file: {batch.error_file_id}")

    # Update stored status
    meta["status"] = batch.status
    if batch.output_file_id:
        meta["output_file_id"] = batch.output_file_id
    if batch.error_file_id:
        meta["error_file_id"] = batch.error_file_id
    BATCH_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def cmd_download(args: argparse.Namespace) -> None:
    """Download completed batch results and merge with error metadata."""
    if not BATCH_META_PATH.exists():
        print(f"No batch metadata found at {BATCH_META_PATH}")
        sys.exit(1)

    meta = json.loads(BATCH_META_PATH.read_text())
    client = OpenAI()

    # Check status
    batch = client.batches.retrieve(meta["batch_id"])
    if batch.status != "completed":
        print(f"Batch status is '{batch.status}', not 'completed'. Wait and try again.")
        sys.exit(1)

    if not batch.output_file_id:
        print("No output file available.")
        sys.exit(1)

    # Download output
    print(f"Downloading output file {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id)

    # Parse responses
    responses: dict[str, dict] = {}
    for line in content.text.strip().split("\n"):
        resp = json.loads(line)
        custom_id = resp["custom_id"]
        if resp.get("error"):
            print(f"  Error for {custom_id}: {resp['error']}")
            responses[custom_id] = {
                "primary_mode": "M8",
                "secondary_mode": None,
                "confidence": "low",
                "notes": f"API error: {resp['error']}",
                "input_tokens": 0,
                "output_tokens": 0,
            }
            continue

        body = resp["response"]["body"]
        usage = body.get("usage", {})
        choice = body["choices"][0]
        text = choice["message"]["content"]

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {
                "primary_mode": "M8",
                "secondary_mode": None,
                "confidence": "low",
                "notes": f"Parse failure. Raw: {text[:200]}",
            }

        # Validate primary_mode
        VALID_MODES = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"}
        pm = result.get("primary_mode", "M8")
        if pm not in VALID_MODES:
            print(f"  Warning: invalid primary_mode '{pm}' for {custom_id}, defaulting to M8")
            pm = "M8"
        sm = result.get("secondary_mode")
        if sm is not None and sm not in VALID_MODES:
            print(f"  Warning: invalid secondary_mode '{sm}' for {custom_id}, setting to null")
            sm = None

        responses[custom_id] = {
            "primary_mode": pm,
            "secondary_mode": sm,
            "confidence": result.get("confidence", "low"),
            "notes": result.get("notes", ""),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    # Merge with error metadata
    errors = load_errors()
    total_in, total_out = 0, 0

    with open(OUT_PATH, "w") as f:
        for error in errors:
            custom_id = f"{error['doc_id']}__{error['model']}__{error['config']}"
            classification = responses.get(custom_id)
            if classification is None:
                print(f"  WARNING: Missing response for {custom_id}", file=sys.stderr)
                classification = {
                    "primary_mode": "M8",
                    "secondary_mode": None,
                    "confidence": "low",
                    "notes": "No response received",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

            total_in += classification["input_tokens"]
            total_out += classification["output_tokens"]

            record = {
                "doc_id": error["doc_id"],
                "model": error["model"],
                "config": error["config"],
                "error_type": error["error_type"],
                "label": error["label"],
                "prediction": error["prediction"],
                "prob": error["prob"],
                "missingness_density": error.get("missingness_density"),
                "primary_mode": classification["primary_mode"],
                "secondary_mode": classification["secondary_mode"],
                "confidence": classification["confidence"],
                "notes": classification["notes"],
                "input_tokens": classification["input_tokens"],
                "output_tokens": classification["output_tokens"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # GPT-5 batch pricing: $0.625/M input, $5.00/M output
    cost = (total_in * 0.625 + total_out * 5.0) / 1_000_000
    n_classified = len(responses)
    n_errors_api = sum(1 for r in responses.values() if "API error" in r.get("notes", ""))

    print(f"\nClassified {n_classified} cases ({n_errors_api} API errors)")
    print(f"Tokens: {total_in:,} in / {total_out:,} out")
    print(f"Estimated cost: ${cost:.4f}")
    print(f"Saved to {OUT_PATH}")

    # Download errors if any
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id)
        err_path = REPO_ROOT / "outputs/EXP-D-0001/batch_errors.jsonl"
        err_path.write_text(err_content.text)
        print(f"Batch errors saved to {err_path}")

    # Print distribution
    from collections import Counter
    with open(OUT_PATH) as f:
        classifications = [json.loads(line) for line in f]
    mode_counts = Counter(r["primary_mode"] for r in classifications)
    print("\nPrimary failure mode distribution:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")

    # By model × error_type
    print("\nBy model × error_type:")
    by_model = {}
    for r in classifications:
        key = (r["model"], r["error_type"])
        by_model.setdefault(key, Counter())[r["primary_mode"]] += 1
    for key in sorted(by_model):
        print(f"  {key[0]}/{key[1]}: {dict(sorted(by_model[key].items()))}")


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify error failure modes using GPT-5 Batch API"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Build and upload batch")
    p_submit.add_argument("--dry-run", action="store_true",
                          help="Build batch input but don't upload")

    sub.add_parser("status", help="Check batch status")
    sub.add_parser("download", help="Download and process results")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
