from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from .db.dbInfo import db, LLMAgentData
import inspect
import json
import hashlib


MODEL = "gemma3:4b"
REASONING_MODEL = "deepseek-r1:1.5b"

class AgentState(TypedDict):
    task_id: str
    job_description: str 
    job_summarization: str
    candidate_profile_info: str
    motivation: str
    cvContent: str
    revise_cv: Literal["approve", "revise"]

class Log:
    @staticmethod
    def input(agent_state, data, func_name= "-no-name-", var_name= "-no-name-"):
        Log.store(agent_state, data, func_name, var_name, "input")
    @staticmethod
    def store(agent_state, data, func_name= "-no-name-", var_name= "-no-name-", typ="output"):
        db.connect()
        LLMAgentData.create(task_id=agent_state['task_id'], 
            function_name=func_name, 
            type=typ, 
            variable_name=var_name, 
            data=data)
        db.close()
    @staticmethod
    def output(agent_state, data, func_name= "-no-name-", var_name= "-no-name-"):
        Log.store(agent_state, data, func_name, var_name, "output")
    @staticmethod
    def process(agent_state, data, func_name= "-no-name-", var_name= "-no-name-"):
        Log.store(agent_state, data, func_name, var_name, "process")

class ReadLog:
    @staticmethod
    def read(agent_state, func_name= "-no-name-", var_name= "-no-name-", typ="output"):
        db.connect()
        entry = LLMAgentData.get(task_id=agent_state['task_id'], 
            function_name=func_name, 
            type=typ, 
            variable_name=var_name)
        db.close()
        return entry
    @staticmethod
    def output(agent_state, func_name= "-no-name-", var_name= "-no-name-"):
        return ReadLog.read(agent_state, func_name, var_name, "output")
    @staticmethod
    def exists(agent_state, func_name= "-no-name-", var_name= "-no-name-", typ="output"):
        db.connect()
        entry = LLMAgentData.get_or_none(task_id=agent_state['task_id'], 
            function_name=func_name, 
            type=typ, 
            variable_name=var_name)
        db.close()
        return entry is not None
    @staticmethod
    def readOrCreate(agent_state,func, func_name= "-no-name-", var_name= "-no-name-", typ="output"):
        db.connect()
        entry = LLMAgentData.get_or_none(task_id=agent_state['task_id'], 
            function_name=func_name, 
            type=typ, 
            variable_name=var_name)
        db.close()
        if entry is None:
            entry = func()
            Log.store(agent_state, entry, func_name, var_name, "output")
        return entry

def runModel(model, prompt, state, func_name= "-no-name-"):
    newPrompt = ReadLog.readOrCreate(state, lambda: prompt, func_name, "prompt", "process")
    llm = ChatOllama(model=model, temperature=0.7)
    response = ReadLog.readOrCreate(state, lambda: llm.invoke([HumanMessage(content=newPrompt)]), func_name, "response")
    return response

def relevant_content_matcher(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Given below is the candidate profile
    {state['candidate_profile_info']}

    extract the relevant content from the candidate profile that is relevant to the job description

    list the relevant content as bullet points
    """
    response = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"relevant_content": response.content}

def job_description_summarizer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the job description
    {state['job_description']}

    Give a summary of the job description (skills, experience, etc. things that are relevant to job application)
    """
    response = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"job_summarization": response.content}

def motivation_writer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content
    {state['candidate_profile_info']}

    Write a motivation letter for the candidate
    """
    response = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"motivation": response.content}

def cv_writer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content 
    {state['candidate_profile_info']}

    make a CV for the candidate
    """
    response = runModel(REASONING_MODEL, prompt, state, inspect.currentframe().f_code.co_name)
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
    builder.add_node("motivation", motivation_writer)
    builder.add_node("cv", cv_writer)
    builder.add_node("review", review_node)
    builder.set_entry_point("job_summary")

    builder.add_edge("job_summary", "motivation")
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

def generate_key(params) -> str:
    serialized_params = json.dumps(params, sort_keys=True, separators=(',', ':'))
    hash_object = hashlib.sha256(serialized_params.encode('utf-8'))
    return hash_object.hexdigest()
def job_apply_helper(candidate_profile, job_description):
    key = generate_key((candidate_profile.strip(), job_description.strip()))
    graph = get_pipeline_graph()
    result = graph.invoke({"task_id": key,"job_description": job_description, "candidate_profile_info": candidate_profile, "revise_cv": ""})
    return result
