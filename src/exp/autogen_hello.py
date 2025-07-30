#%%
import autogen

# Configuration for the LLM
config_list = [
    {
        "model": "gemma3:4b",
        "base_url": "http://localhost:11434",
        "api_key": "ollama",  # placeholder
    }
]

# Create an Assistant Agent
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful assistant. Answer as concisely as possible."
)

# Create a User Proxy Agent (acts on behalf of the user)
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  # Auto-run; set to "ALWAYS" for manual input
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

# Initiate a chat from the user proxy to the assistant
user_proxy.initiate_chat(
    assistant,
    message="Hello, Assistant! Can you tell me what AutoGen is in two sentences? Then say TERMINATE."
)
# %%
