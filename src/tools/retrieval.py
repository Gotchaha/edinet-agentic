"""Field-level retrieval tools over an EDINET-Bench example.

Closure factory: pre-parses the example's sheet JSON strings once and
returns [list_attributes, get_attribute] with the parsed data captured.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

SHEET_KEYS = ["summary", "bs", "pl", "cf"]


def build_retrieval_tools(example: dict):
    """Return retrieval tools bound to the given example via closure."""
    parsed = {
        sheet: json.loads(example[sheet])
        for sheet in SHEET_KEYS
        if sheet in example and example[sheet]
    }

    @tool
    def list_attributes(sheet: str) -> str:
        """List all available attribute names in a financial sheet.

        Sheets: 'summary', 'bs' (balance sheet), 'pl' (profit & loss),
        'cf' (cash flow).
        Returns a JSON array of attribute names (Japanese strings).
        """
        if sheet not in parsed:
            return json.dumps(
                {"error": f"unknown sheet '{sheet}'. Valid sheets: {list(parsed)}"},
                ensure_ascii=False,
            )
        return json.dumps(list(parsed[sheet].keys()), ensure_ascii=False)

    @tool
    def get_attribute(sheet: str, name: str) -> str:
        """Retrieve values for an attribute across all available periods.

        Returns a JSON object mapping period name (e.g. 'CurrentYear',
        'Prior1Year', 'Prior2Year') to value (integer string in JPY).
        Missing values are represented as the string '－'.
        Use list_attributes first if you do not know the attribute name.
        """
        if sheet not in parsed:
            return json.dumps(
                {"error": f"unknown sheet '{sheet}'"},
                ensure_ascii=False,
            )
        if name not in parsed[sheet]:
            return json.dumps(
                {"error": f"attribute '{name}' not found in sheet '{sheet}'"},
                ensure_ascii=False,
            )
        return json.dumps(parsed[sheet][name], ensure_ascii=False)

    return [list_attributes, get_attribute]
