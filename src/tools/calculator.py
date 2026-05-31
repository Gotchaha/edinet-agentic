"""Calculator tool for arithmetic verification."""

from __future__ import annotations

from langchain_core.tools import tool
from simpleeval import simple_eval


@tool
def calculator(expression: str) -> str:
    """Compute a financial metric or signal from numerical values.

    Use this for ratios, percentage changes, growth rates, year-over-year
    differences, margins, sums, or other derived figures based on the values
    you have retrieved. Examples:
    - calculator("(159702000000 - 165215000000) / 165215000000 * 100")
    - calculator("7468000000 / 94364000000")
    """
    try:
        result = simple_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
