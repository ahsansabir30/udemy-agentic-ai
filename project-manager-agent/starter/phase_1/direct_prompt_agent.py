# Test script for DirectPromptAgent class

# TODO 1: Import the DirectPromptAgent class from BaseAgents
from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO 2: Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

prompt = "What is the Capital of France?"

# TODO 3: Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)

# TODO 4: Use direct_agent to send the prompt and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# TODO 5: Print an explanatory message describing the knowledge source
print(
    "\nExplanation: The DirectPromptAgent used the model's built-in general world "
    "knowledge (geography) learned during training to answer the question. "
    "No external knowledge base or tools were used—the response is generated "
    "directly from the language model provided by "
    "."
)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_prompt_agent_output.txt"), "w") as f:
    f.write(direct_agent_response + "\n")   
