"""Agent state definition for the baseline reflection agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the generate-critique-revise reflection loop.

    The `messages` list (from MessagesState) holds the full conversation:
    generator output, critic feedback, and revised output — serving as the trace.
    """

    doc_id: str
    sheets_text: str
    base_prompt: str
    # Parsed final output
    final_prediction: int | None
    final_prob: float | None
    final_reasoning: str | None
    # Token tracking
    input_tokens: int
    output_tokens: int
