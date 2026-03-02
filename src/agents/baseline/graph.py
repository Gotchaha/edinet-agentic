"""LangGraph definition for the baseline reflection agent.

Graph: START → generate → critique → revise → END
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.baseline.prompts import (
    CRITIC_SYSTEM,
    GENERATOR_SYSTEM,
    REVISER_SYSTEM,
)
from agents.baseline.state import AgentState
from common.parsing import extract_json_between_markers


def _track_tokens(state: AgentState, response) -> dict:
    """Extract token usage from response metadata and return update dict."""
    usage = response.usage_metadata or {}
    return {
        "input_tokens": state.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
    }


def build_graph(model_id: str):
    """Build and compile the reflection agent graph."""
    llm = ChatAnthropic(
        model=model_id,
        max_tokens=4096,
        temperature=0.0,
    )

    def generate(state: AgentState) -> dict:
        user_content = state["base_prompt"] + state["sheets_text"]
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            **_track_tokens(state, response),
        }

    def critique(state: AgentState) -> dict:
        generator_output = state["messages"][-1].content
        user_content = (
            "## Original Financial Data\n\n"
            f"{state['sheets_text']}\n\n"
            "## Analyst's Analysis\n\n"
            f"{generator_output}"
        )
        messages = [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            **_track_tokens(state, response),
        }

    def revise(state: AgentState) -> dict:
        generator_output = state["messages"][-2].content
        critic_feedback = state["messages"][-1].content
        user_content = (
            "## Original Financial Data\n\n"
            f"{state['sheets_text']}\n\n"
            "## Your Previous Analysis\n\n"
            f"{generator_output}\n\n"
            "## Auditor Feedback\n\n"
            f"{critic_feedback}"
        )
        messages = [
            SystemMessage(content=REVISER_SYSTEM),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        tokens = _track_tokens(state, response)

        # Parse final JSON from revised output
        parsed = extract_json_between_markers(response.content)
        final_prediction = None
        final_prob = None
        final_reasoning = None
        if parsed:
            final_prediction = parsed.get("prediction")
            final_prob = parsed.get("prob")
            final_reasoning = parsed.get("reasoning")

        return {
            "messages": [response],
            "final_prediction": final_prediction,
            "final_prob": final_prob,
            "final_reasoning": final_reasoning,
            **tokens,
        }

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("revise", revise)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "revise")
    graph.add_edge("revise", END)

    return graph.compile()
