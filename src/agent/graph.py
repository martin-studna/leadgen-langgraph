import functools
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode
from agent.helpers import agent_node, create_agent
from agent.tools import LeadFinderTool, LeadExtractorTool


class State(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str
    iteration: int


llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)



def finder_router(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    
    
    if last_message.tool_calls:
        state["iteration"] += 1
        return "call_tool"
    if state["iteration"] < 6:
        return "call_tool"
    
    return "continue"

def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    return "continue"


# Tool definitions
lead_finder_tools = [LeadFinderTool()]
lead_enricher_tools = [LeadExtractorTool()]

graph = StateGraph(State)

# Lead Finder
lead_finder_agent = create_agent(
    llm,
    lead_finder_tools,
    """
    You are a professional lead finder. 
    Your job is to find leads through Google Places, accessible via the LeadFinderTool. 
    Return only the list of found lead url's, along with their address (do label them as 'url' and 'address'). Do not include any other information or markdown styling in your response.
    If an address or url from a lead is not available, pick a different lead, or do not include the lead in the list.
    """,
)
lead_finder_node = functools.partial(
    agent_node, agent=lead_finder_agent, name="lead_finder"
)

# Lead Enricher
lead_enricher_agent = create_agent(
    llm,
    lead_enricher_tools,
    """
    You are a professional lead enricher. Your job is to gather as much relevant lead information from a given website as possible. 
    You will receive a list of url's from leads and their addresses, 
    and use the LeadExtractorTool to gather additional lead data from the provided url's.
    The enriched lead data must contain the following fields: company name, website, email, address, phone, CEO, company_mission. If any of these fields are not available, mark them "not found".
    Do not include any markdown styling in your response.
    """,
)
lead_enricher_node = functools.partial(
    agent_node, agent=lead_enricher_agent, name="lead_enricher"
)

# Nodes
graph.add_node("lead_finder", lead_finder_node)
graph.add_node("lead_enricher", lead_enricher_node)
graph.add_node("call_tool", ToolNode(lead_finder_tools + lead_enricher_tools))

# Edges
graph.add_conditional_edges(
    "lead_finder",
    router,
    {"continue": "lead_enricher", "call_tool": "call_tool"},
)
graph.add_conditional_edges(
    "lead_enricher", router, {"continue": END, "call_tool": "call_tool"}
)
graph.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"lead_finder": "lead_finder", "lead_enricher": "lead_enricher"},
)


graph.add_edge(START, "lead_finder")

graph = graph.compile()
graph.name = "Leadgen Graph"

image = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(image)

# Uncomment when you want to run the graph locally. If doing so, check the Firecrawl url in tools.py.
import asyncio


async def main():
    inputs = [
        {
            "role": "user",
            "content": "Find factories in czechia which have assembly lines where we could use AI cameras to automate. No cars and automative, Only big companies",
        }
    ]
    async for chunk in graph.astream({"messages": inputs}, stream_mode="values",config={"recursion_limit":20}):
        chunk["messages"][-1].pretty_print()


asyncio.run(main())
