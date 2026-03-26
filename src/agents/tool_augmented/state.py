"""Agent state definition for the tool-augmented verification agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State for the generate-verify-revise loop.

    The `messages` list (from MessagesState) holds the verifier's tool-calling
    conversation. `generator_output` is stored separately so the reviser can
    access it after the verify loop fills `messages` with tool calls.
    """

    doc_id: str
    sheets_text: str
    base_prompt: str
    generator_output: str
    # Parsed final output
    final_prediction: int | None
    final_prob: float | None
    final_reasoning: str | None
    # Token tracking
    input_tokens: int
    output_tokens: int
