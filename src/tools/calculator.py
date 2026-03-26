"""Calculator tool for arithmetic verification."""

from __future__ import annotations

from langchain_core.tools import tool
from simpleeval import simple_eval


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Use this to verify numerical claims: ratios, percentage changes,
    differences, etc. Examples:
    - calculator("(159702000000 - 165215000000) / 165215000000 * 100")
    - calculator("7468000000 / 94364000000")
    """
    try:
        result = simple_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
