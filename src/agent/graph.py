from agent.tools import GooglePlacesTool, LeadExtractorTool
from langchain_openai import ChatOpenAI
import functools
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from agent.helpers import agent_node, create_agent


class State(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str


llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return END
    return "continue"


# Build graph
graph = StateGraph(State)

# Tool definitions
lead_finder_tools = [GooglePlacesTool()]
lead_enricher_tools = [LeadExtractorTool()]

tools = lead_finder_tools + lead_enricher_tools
tool_node = ToolNode(tools)
graph.add_node("call_tool", tool_node)

# Lead Finder
LeadFinderAgent = create_agent(
    llm,
    lead_finder_tools,
    "You are a professional lead finder. Your job is to find leads through Google Places, accessible via the GooglePlacesTool. Return only the list of found lead url's, along with their addresses.",
)
LeadFinderNode = functools.partial(
    agent_node, agent=LeadFinderAgent, name="lead_finder"
)
graph.add_node("lead_finder", LeadFinderNode)

# Lead Enricher
LeadEnricherAgent = create_agent(
    llm,
    lead_enricher_tools,
    "You are a professional lead enricher. Your job is to gather as much relevant lead information from a given website as possible. You will receive a list of url's from leads and their addresses. You will use the LeadExtractorTool to gather additional lead data from the provided url's.",
)
LeadEnricherNode = functools.partial(
    agent_node, agent=LeadEnricherAgent, name="lead_enricher"
)
graph.add_node("lead_enricher", LeadEnricherNode)

# Edges
graph.add_conditional_edges(
    "lead_finder",
    router,
    {"continue": "lead_enricher", "call_tool": "call_tool", END: "lead_enricher"},
)

graph.add_conditional_edges(
    "lead_enricher", router, {"continue": END, "call_tool": "call_tool", END: END}
)

graph.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"lead_finder": "lead_finder", "lead_enricher": "lead_enricher"},
)


graph.add_edge(START, "lead_finder")

graph = graph.compile()
graph.name = "Leadgen Graph"
