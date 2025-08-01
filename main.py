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
job_apply_helper(candidate_profile, job_content)
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
    print(val.task_id, val.function_name, val.type, val.variable_name, val.data)
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
