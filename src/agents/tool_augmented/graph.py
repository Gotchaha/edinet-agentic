"""LangGraph definition for the tool-augmented verification agent.

Graph: START → generate → verify ←→ tools (calculator loop)
                              └→ revise → END
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.tool_augmented.prompts import (
    GENERATOR_SYSTEM,
    REVISER_SYSTEM,
    VERIFIER_SYSTEM,
)
from agents.tool_augmented.state import AgentState
from common.parsing import extract_json_between_markers
from tools.calculator import calculator
from tools.submit_verification import submit_verification


def _track_tokens(state: AgentState, response) -> dict:
    """Extract token usage from response metadata and return update dict."""
    usage = response.usage_metadata or {}
    return {
        "input_tokens": state.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
    }


def build_graph(model_id: str):
    """Build and compile the tool-augmented verification agent graph."""
    llm = ChatAnthropic(
        model=model_id,
        max_tokens=4096,
        temperature=0.0,
    )
    tools = [calculator, submit_verification]
    llm_with_tools = llm.bind_tools(tools)

    def generate(state: AgentState) -> dict:
        user_content = state["base_prompt"] + state["sheets_text"]
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        generator_output = response.content

        # Set up verifier context as the first message in the messages list
        verifier_context = (
            "## Original Financial Data\n\n"
            f"{state['sheets_text']}\n\n"
            "## Analyst's Analysis to Verify\n\n"
            f"{generator_output}\n\n"
            "Identify all numerical claims in the analysis above. "
            "For each claim, use the `calculator` tool to verify the arithmetic. "
            "When all claims are checked, call `submit_verification` with the structured report."
        )

        return {
            "generator_output": generator_output,
            "messages": [HumanMessage(content=verifier_context)],
            **_track_tokens(state, response),
        }

    def verify(state: AgentState) -> dict:
        messages = [SystemMessage(content=VERIFIER_SYSTEM)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            **_track_tokens(state, response),
        }

    def route_after_verify(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return "revise"  # fallback: no tool calls
        for tc in last_msg.tool_calls:
            if tc["name"] == "submit_verification":
                return "revise"
        return "tools"  # calculator calls — execute and loop

    def revise(state: AgentState) -> dict:
        # Extract verification report from submit_verification tool call
        verification_report = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "submit_verification":
                        verification_report = tc["args"]
                        break
            if verification_report:
                break

        # Format verification report for the reviser
        if verification_report:
            report_lines = ["## Verification Report\n"]
            for i, check in enumerate(verification_report.get("checks", []), 1):
                report_lines.append(f"### Check {i}")
                report_lines.append(f"- **Claim:** {check.get('claim', 'N/A')}")
                report_lines.append(f"- **Source:** {check.get('source', 'N/A')}")
                report_lines.append(f"- **Status:** {check.get('status', 'N/A')}")
                report_lines.append(f"- **Detail:** {check.get('detail', 'N/A')}")
                report_lines.append("")
            confirmed = verification_report.get("red_flags_confirmed", 0)
            refuted = verification_report.get("red_flags_refuted", 0)
            report_lines.append(
                f"**Summary:** {confirmed} red flags confirmed, {refuted} refuted."
            )
            report_text = "\n".join(report_lines)
        else:
            report_text = "## Verification Report\n\nNo structured report was submitted."

        user_content = (
            "## Original Financial Data\n\n"
            f"{state['sheets_text']}\n\n"
            "## Your Previous Analysis\n\n"
            f"{state['generator_output']}\n\n"
            f"{report_text}"
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
    graph.add_node("verify", verify)
    graph.add_node("tools", ToolNode([calculator, submit_verification]))
    graph.add_node("revise", revise)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("tools", "verify")
    graph.add_edge("revise", END)

    return graph.compile()
