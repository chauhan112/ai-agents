from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

MODEL = "gemma3:4b"

class AgentState(TypedDict):
    destination: str 
    research: str     
    plan: str 
    human_feedback: Literal["approve", "revise"] 

def research_node(state: AgentState) -> AgentState:
    search = DuckDuckGoSearchResults()
    results = search.invoke(f"Top attractions in {state['destination']}")
    return {"research": results}

def planning_node(state: AgentState) -> AgentState:
    llm = ChatOllama(model=MODEL, temperature=0.7)
    prompt = f"""
    Create a 3-day trip plan for {state['destination']} using this research:
    {state['research']}
    
    Format strictly as:
    Day 1: [Activities]
    Day 2: [Activities]
    Day 3: [Activities]
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"plan": response.content}

def review_node(state: AgentState) -> AgentState:
    """Asks for human approval"""
    print("\n📝 Proposed Trip Plan:")
    print(state["plan"])
    feedback = input("\nApprove? (yes/no): ").strip().lower()
    return {"human_feedback": "approve" if feedback == "yes" else "revise"}

def should_revise(state: AgentState):
    return "revise" if state["human_feedback"] == "revise" else END

def get_pipeline_graph():
    builder = StateGraph(AgentState)
    builder.add_node("research", research_node)
    builder.add_node("plan", planning_node)
    builder.add_node("review", review_node)
    builder.set_entry_point("research")

    builder.add_edge("research", "plan")
    builder.add_edge("plan", "review")
    builder.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "plan",  # Loop back to planning if rejected
            END: END
        }
    )

    graph = builder.compile()
    return graph

def trip_planner(destination, printPlan=True):
    graph = get_pipeline_graph()
    
    result = graph.invoke({"destination": destination, "human_feedback": ""})
    if printPlan:
        print("\n✅ Final Approved Plan:")
        print(result["plan"])
    return result