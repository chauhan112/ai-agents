# #%%
# from crewai import Agent, Task, Crew, Process
# from langchain_community.llms import Ollama
# #%%
# ollama_llm = Ollama(model="gemma3:4b")

# researcher = Agent(
#   role='Senior Research Analyst',
#   goal='Uncover cutting-edge developments in AI and data science',
#   backstory="""You work at a leading tech think tank.
#   Your expertise lies in identifying emerging trends.
#   You have a knack for dissecting complex data and presenting
#   actionable insights.""",
#   verbose=True,
#   allow_delegation=False,
#   llm=ollama_llm 
# )


# writer = Agent(
#   role='Tech Content Strategist',
#   goal='Craft compelling content on tech advancements',
#   backstory="""You are a renowned Content Strategist, known for
#   your insightful and engaging articles.
#   You transform complex concepts into compelling narratives.""",
#   verbose=True,
#   allow_delegation=True,
#   llm=ollama_llm # Pass the Ollama LLM model to the agent
# )


# # --- 2. Create Your Tasks ---
# # The tasks remain the same.

# research_task = Task(
#   description="""Conduct a comprehensive analysis of the latest advancements in AI in 2025.
#   Identify key trends, breakthrough technologies, and potential industry impacts.
#   Your final answer MUST be a full analysis report.""",
#   expected_output='A comprehensive 3-paragraph report on the latest AI advancements.',
#   agent=researcher
# )

# write_task = Task(
#   description="""Using the insights provided, develop an engaging blog post
#   that highlights the most significant AI advancements.
#   Your post should be informative yet accessible, catering to a tech-savvy audience.
#   Make it sound cool, avoid complex words so it doesn't sound like AI.
#   Your final answer MUST be the full blog post of at least 4 paragraphs.""",
#   expected_output='A 4-paragraph blog post on AI advancements, formatted in markdown.',
#   agent=writer,
#   context=[research_task]
# )


# # --- 3. Assemble Your Crew ---
# # The crew definition remains the same.
# my_crew = Crew(
#   agents=[researcher, writer],
#   tasks=[research_task, write_task],
#   process=Process.sequential,
#   verbose=2
# )

# # --- 4. Kick Off the Work! ---
# print("## Welcome to the Tech News Crew (Ollama Edition)! ##")
# print("--------------------------------------------------")
# result = my_crew.kickoff()

# print("\n\n########################")
# print("## Here is the final result:")
# print("########################\n")
# print(result)

#%%
from crewai import Agent, Task, Crew
#%%
# Configure LOCAL LLM (no API keys needed!)
from langchain_community.llms import Ollama
llm = Ollama(model="gemma3:4b", temperature=0.3)

# Define AGENTS
researcher = Agent(
    role='Greeting Specialist',
    goal='Craft the perfect hello message',
    backstory='You are a world-class greeting expert',
    llm=llm,
    verbose=True
)

translator = Agent(
    role='Spanish Translator',
    goal='Translate greetings accurately',
    backstory='You are a fluent Spanish translator',
    llm=llm,
    verbose=True
)

# Define TASKS
task1 = Task(
    description='Say "Hello World" in a creative way',
    agent=researcher,
    expected_output='A creative English greeting'
)

task2 = Task(
    description='Translate the greeting to Spanish',
    agent=translator,
    expected_output='The greeting in Spanish'
)

# Build the CREW
crew = Crew(
    agents=[researcher, translator],
    tasks=[task1, task2],
    verbose=2  # Show agent thinking process
)

# Run the crew
result = crew.kickoff()
print("\n\n########################")
print("## Here is your result:")
print("########################\n")
print(result)