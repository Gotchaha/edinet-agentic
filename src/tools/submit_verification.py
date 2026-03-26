"""Submit verification tool for structured verification reports."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def submit_verification(
    checks: list[dict],
    red_flags_confirmed: int,
    red_flags_refuted: int,
) -> str:
    """Submit the structured verification report.

    Call this after you have finished checking all numerical claims.

    Args:
        checks: List of checked claims, each with keys:
            claim, source (sheet + field), status ("confirmed"/"corrected"),
            detail (computation or explanation)
        red_flags_confirmed: Number of red flags that were confirmed
        red_flags_refuted: Number of red flags that were refuted
    """
    return (
        f"Verification submitted: {len(checks)} checks, "
        f"{red_flags_confirmed} confirmed, {red_flags_refuted} refuted."
    )
