"""Dataset loading and prompt construction for EDINET-Bench tasks."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sample(sample_path: str | Path) -> list[str]:
    """Load frozen doc_ids from a sample JSON file."""
    path = REPO_ROOT / sample_path
    data = json.loads(path.read_text())
    return data["doc_ids"]


def load_dataset_for_sample(task: str, doc_ids: list[str]) -> Dataset:
    """Load HF dataset and filter to the given doc_ids."""
    ds = load_dataset("SakanaAI/EDINET-Bench", task, split="test")
    id_set = set(doc_ids)
    ds = ds.filter(lambda ex: ex["doc_id"] in id_set)
    assert len(ds) == len(id_set), (
        f"Expected {len(id_set)} examples, got {len(ds)}"
    )
    return ds


def load_prompt_template(task: str) -> str:
    """Load base prompt from upstream prompt YAML."""
    prompt_path = REPO_ROOT / "external" / "EDINET-Bench" / "prompt" / f"{task}.yaml"
    return yaml.safe_load(prompt_path.read_text())["prompt"]


def build_sheets_text(example: dict, sheets: list[str]) -> str:
    """Format sheet data as text, matching upstream process_example logic."""
    return "\n".join(
        f"{sheet}: {example[sheet]}" for sheet in sheets if sheet in example
    )
