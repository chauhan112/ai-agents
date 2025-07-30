#%%
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

MODEL = "gemma3:4b"

# ===== 1. DEFINE STATE (Critical for LangGraph) =====
class AgentState(TypedDict):
    destination: str  # User's destination
    research: str     # Research results
    plan: str         # Generated trip plan
    human_feedback: str  # "approve" or "revise"

# ===== 2. BUILD NODES (Functions that update state) =====
def research_node(state: AgentState) -> AgentState:
    """Searches the web for destination info"""
    search = DuckDuckGoSearchResults()
    results = search.invoke(f"Top attractions in {state['destination']}")
    return {"research": results}

def planning_node(state: AgentState) -> AgentState:
    """Generates trip plan using local LLM"""
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

# ===== 3. DEFINE CONDITIONAL EDGES =====
def should_revise(state: AgentState):
    """Route to planning_node if revision needed"""
    return "revise" if state["human_feedback"] == "revise" else END

# ===== 4. CONSTRUCT GRAPH =====
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("research", research_node)
builder.add_node("plan", planning_node)
builder.add_node("review", review_node)

# Set entry point
builder.set_entry_point("research")

# Define edges
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

# Compile graph
graph = builder.compile()

# ===== 5. RUN IT! =====
if __name__ == "__main__":
    inputs = {
        "destination": "Munich",  # Change this to any destination
        "human_feedback": ""     # Initialize empty
    }
    
    print("✈️ Starting trip planning for", inputs["destination"])
    result = graph.invoke(inputs)
    
    print("\n✅ Final Approved Plan:")
    print(result["plan"])
# %%
