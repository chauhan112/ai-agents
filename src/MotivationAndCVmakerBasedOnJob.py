from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from .db.dbInfo import db, LLMAgentData
import inspect
import json
import hashlib


MODEL = "gemma3n:e4b" # gemma3:4b deepseek-r1:1.5b deepseek-r1:7b
TEMPERATURE = 0.2

class AgentState(TypedDict):
    task_id: str
    job_description: str 
    job_summarization: str
    candidate_profile_info: str
    motivation: str
    cvContent: str
    revise_cv: Literal["approve", "revise"]

class DbWrapper:
    inst = None
    def read_or_create(self, task_id, func, func_name= "-no-name-", var_name= "-no-name-", more_info={}):
        val = self.read(task_id, func_name, var_name)
        if val is None:
            val = func()
            self.create(task_id, val, func_name, var_name, more_info )
            return val
        return val.data
    def _run_command(self, func, *params, **kwargs):
        db.connect()
        try:
            res = func(*params, **kwargs)
        except Exception as e:
            print(f"Error : {e}")
            res = None
        finally:
            db.close()
        return res
    def create(self, task_id, data, func_name= "-no-name-", var_name= "-no-name-", more_info ={}):
        return self._run_command(LLMAgentData.create, task_id=task_id, 
            function_name=func_name, 
            variable_name=var_name, 
            data=data, more_info=more_info)
    def read(self, task_id, func_name= "-no-name-", var_name= "-no-name-"):
        return self._run_command(LLMAgentData.get_or_none, task_id=task_id, 
            function_name=func_name, 
            variable_name=var_name)
    def delete(self, task_id):
        return self._run_command(lambda : LLMAgentData.delete().where(LLMAgentData.task_id == task_id).execute())
class ReadLog:
    inst = None
    @staticmethod
    def get_instance():
        if ReadLog.inst is None:
            ReadLog.inst = DbWrapper()
        return ReadLog.inst
    @staticmethod
    def readOrCreate(task_id, func, func_name= "-no-name-", var_name= "-no-name-", more_info={}):
        inst = ReadLog.get_instance()
        return inst.read_or_create(task_id, func, func_name, var_name, more_info)
    @staticmethod
    def delete(task_id):
        inst = ReadLog.get_instance()
        return inst.delete(task_id)


def runModel(model, prompt, state, func_name= "-no-name-"):
    newPrompt = ReadLog.readOrCreate(state["task_id"], lambda: prompt, func_name, "prompt")
    llm = ChatOllama(model=model, temperature=TEMPERATURE)
    key = generate_key((state["task_id"], model, TEMPERATURE))
    response = ReadLog.readOrCreate(key, lambda: llm.invoke([HumanMessage(content=newPrompt)]).content, 
        func_name, "response", {"model":model, "temperature":TEMPERATURE})
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
    response_content = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"relevant_content": response_content}

def job_description_summarizer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the job description
    {state['job_description']}

    Give a summary of the job description (skills, experience, etc. things that are relevant to job application)
    """
    response_content = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"job_summarization": response_content}

def motivation_writer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content
    {state['candidate_profile_info']}

    Write a motivation letter for the candidate (in the perspective of the candidate)
    """
    response_content = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"motivation": response_content}

def cv_writer(state: AgentState) -> AgentState:
    prompt = f"""
    Given below is the summary of job description
    {state['job_summarization']}

    Here is also candidate profile content 
    {state['candidate_profile_info']}

    make a CV for the candidate. Just make it a list of bullet points instead of markdown
    """
    response_content = runModel(MODEL, prompt, state, inspect.currentframe().f_code.co_name)
    return {"cvContent": response_content}

def review_node(state: AgentState) -> AgentState:
    """Asks for human approval"""
    print("\n📝 Motivation:")
    print(state["motivation"])

    print("\n📝 cv content:")
    print(state["cvContent"])
    feedback = input("\nApprove? (yes/no): ").strip().lower()
    res = {"revise_cv": "approve" if feedback == "yes" else "revise"}
    return res

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

def removeThinkTag(content):
    import re
    return re.sub("<think>.*</think>","", content, flags=re.DOTALL)

def job_apply_helper(candidate_profile, job_description, outCV ="cv.txt", outMotivation ="motivation.txt", redo=False):
    key = generate_key((candidate_profile.strip(), job_description.strip()))
    graph = get_pipeline_graph()
    if redo:
        newkey = generate_key((key, MODEL, TEMPERATURE))
        ReadLog.delete(newkey)
        ReadLog.delete(key)
    result = graph.invoke({"task_id": key,"job_description": job_description, "candidate_profile_info": candidate_profile, "revise_cv": ""})
    
    result["motivation"] = removeThinkTag(result["motivation"])
    result["cvContent"] = removeThinkTag(result["cvContent"])

    with open(outCV, "w") as f:
        f.write(result["cvContent"])
    with open(outMotivation, "w") as f:
        f.write(result["motivation"])
    
    return result