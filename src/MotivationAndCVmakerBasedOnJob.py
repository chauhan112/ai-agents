from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

MODEL = "gemma3:4b"
REASONING_MODEL = "deepseek-r1:1.5b"

class AgentState(TypedDict):
    job_description: str 
    job_summarization: str
    candidate_profile_info: str
    relevant_content: str
    motivation: str
    cvContent: str
    # revise_motivation: Literal["approve", "revise"] 
    revise_cv: Literal["approve", "revise"]

def relevant_content_matcher(state: AgentState) -> AgentState:
    llm = ChatOllama(model=MODEL, temperature=0.7)
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Given below is the candidate profile
    {state['candidate_profile_info']}

    extract the relevant content from the candidate profile that is relevant to the job description

    list the relevant content as bullet points
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"relevant_content": response.content}

def job_description_summarizer(state: AgentState) -> AgentState:
    llm = ChatOllama(model=MODEL, temperature=0.7)
    prompt = f"""
    Given below is the job description
    {state['job_description']}

    Give a summary of the job description (skills, experience, etc. things that are relevant to job application)
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"job_summarization": response.content}

def motivation_writer(state: AgentState) -> AgentState:
    llm = ChatOllama(model=MODEL, temperature=0.7)
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content that is relevant
    {state['relevant_content']}

    Write a motivation letter for the candidate
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"motivation": response.content}

def cv_writer(state: AgentState) -> AgentState:
    llm = ChatOllama(model=REASONING_MODEL, temperature=0.7)
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content that is relevant
    {state['relevant_content']}

    make a CV for the candidate
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"cvContent": response.content}

def review_node(state: AgentState) -> AgentState:
    """Asks for human approval"""
    print("\n📝 Motivation:")
    print(state["motivation"])

    print("\n📝 cv content:")
    print(state["cvContent"])
    feedback = input("\nApprove? (yes/no): ").strip().lower()
    return {"revise_cv": "approve" if feedback == "yes" else "revise"}

def should_revise(state: AgentState):
    return "revise" if state["revise_cv"] == "revise" else END

def get_pipeline_graph():
    builder = StateGraph(AgentState)
    builder.add_node("job_summary", job_description_summarizer)
    builder.add_node("candidate_info_extraction", relevant_content_matcher)
    builder.add_node("motivation", motivation_writer)
    builder.add_node("cv", cv_writer)
    builder.add_node("review", review_node)
    builder.set_entry_point("job_summary")

    builder.add_edge("job_summary", "candidate_info_extraction")
    builder.add_edge("candidate_info_extraction", "motivation")
    builder.add_edge("motivation", "cv")
    builder.add_edge("cv", "review")
    builder.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "cv",  
            END: END
        }
    )
    graph = builder.compile()
    return graph

def job_apply_helper(candidate_profile, job_description):
    graph = get_pipeline_graph()
    result = graph.invoke({"job_description": job_description, "candidate_profile_info": candidate_profile, "revise_cv": ""})
    return result