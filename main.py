#%%
from src.trip_planner import trip_planner

trip_planner("Tokyo")
# %%
import sys
sys.path.insert(0, 'src/rlib')
from src.MotivationAndCVmakerBasedOnJob import job_apply_helper, generate_key
from src.rlib.useful.FileDatabase import File

job_content = File.getFileContent("./tests/job_apply_helper/job_description.txt").strip()
candidate_profile = File.getFileContent("./tests/job_apply_helper/candidate_profile.txt").strip()
#%%
job_apply_helper(candidate_profile, job_content, 
    outCV="output/cv.txt", 
    outMotivation="output/motivation.txt", redo=True)

#%%
from src.db.dbInfo import db, LLMAgentData

vals = LLMAgentData.select()
for val in vals:
    print(val.task_id, val.function_name, val.variable_name, val.more_info)

# %%







key = generate_key((candidate_profile, job_content))

state = {"task_id": key,"job_description": job_content, "candidate_profile_info": candidate_profile, "revise_cv": ""}
# %%
from src.MotivationAndCVmakerBasedOnJob import job_description_summarizer, ReadLog
# %%
job_description_summarizer(state)
# %%
from src.db.dbInfo import db, LLMAgentData

vals = LLMAgentData.select()
for val in vals:
    print(val.task_id, val.function_name, val.variable_name)
# %%
# ReaLog.readOrCreate({"task_id": val.task_id,"job_ddescription": job_content, "candidate_profile_info": candidate_profile, "revise_cv": ""}, lambda: job_content, "func_n", "job_description", "input")

db.connect()


res =LLMAgentData.get_or_none(task_id="1384cbbd9e0359f38a29d521c5fb4632d61e07806fc8165e389971a96506cc5e",
    function_name="job_description_summarizer",
    variable_name="response"
)



db.close()


# ReadLog.readOrCreate(
#     agent_state={"task_id": "1384cbbd9e0359f38a29d521c5fb4632d61e07806fc8165e389971a96506cc5e"},
#     func = lambda: "test",
#     func_name="job_description_summarizer",
#     var_name="response",
#     typ="output"
# )

# %%
prompt = f"""
Given below is the job description
{state['job_description']}

Give a summary of the job description (skills, experience, etc. things that are relevant to job application)
"""
ReadLog.readOrCreate(state, lambda: prompt, "func_n", "prompt", "process")
# %%
db.close()
# %%
