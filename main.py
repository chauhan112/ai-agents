#%%
from src.trip_planner import trip_planner

trip_planner("Tokyo")
# %%
import sys
sys.path.insert(0, 'src/rlib')
from src.MotivationAndCVmakerBasedOnJob import job_apply_helper
from src.rlib.useful.FileDatabase import File

job_content = File.getFileContent("./tests/job_apply_helper/job_description.txt")
candidate_profile = File.getFileContent("./tests/job_apply_helper/candidate_profile.txt")
#%%
job_apply_helper(candidate_profile, job_content)
# %%
