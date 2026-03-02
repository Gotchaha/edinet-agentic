"""JSON extraction from LLM output.

Decoupled copy of extract_json_between_markers from upstream
external/EDINET-Bench/src/edinet_bench/utils.py.
"""

from __future__ import annotations

import json
import re


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON from ```json ... ``` markers, with fallback to bare JSON."""
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue

    return None
