"""LangGraph definition for the tool-augmented ReAct agent.

Graph: START → agent ←→ tools
                  └→ END
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.tool_augmented.prompts import SYSTEM_PROMPT
from agents.tool_augmented.state import AgentState
from common.parsing import extract_json_between_markers
from tools.calculator import calculator
from tools.retrieval import build_retrieval_tools


def _track_tokens(state: AgentState, response) -> dict:
    """Extract token usage from response metadata and return update dict."""
    usage = response.usage_metadata or {}
    return {
        "input_tokens": state.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens", 0),
    }


def build_graph(model_id: str, example: dict):
    """Build and compile the ReAct agent graph.

    Tools are built per-example: retrieval tools (list_attributes,
    get_attribute) capture the example via closure, plus calculator.
    """
    llm = ChatAnthropic(
        model=model_id,
        max_tokens=4096,
        temperature=0.0,
    )
    tools = [*build_retrieval_tools(example), calculator]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        # On first call, add the user message to state so it persists across turns
        if not state["messages"]:
            user_content = state["base_prompt"] + state["sheets_listing"]
            user_msg = HumanMessage(content=user_content)
            state_messages = [user_msg]
        else:
            state_messages = []

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"] + state_messages
        response = llm_with_tools.invoke(messages)
        tokens = _track_tokens(state, response)

        # If no tool calls, parse final output
        result = {"messages": state_messages + [response], **tokens}
        if not isinstance(response, AIMessage) or not response.tool_calls:
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            parsed = extract_json_between_markers(content)
            if parsed:
                result["final_prediction"] = parsed.get("prediction")
                result["final_prob"] = parsed.get("prob")
                result["final_reasoning"] = parsed.get("reasoning")

        return result

    def route_after_agent(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")

    return graph.compile()
