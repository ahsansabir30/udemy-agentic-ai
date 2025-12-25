# TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor. ALWAYS start your answer with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
agent = AugmentedPromptAgent(
    openai_api_key = openai_api_key,
    persona = persona
)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
"The agent has used general world knowledge learned during training - mainly pertaining to basic geography knowledge that Paris is the capital of France"
# - How the system prompt specifying the persona affected the agent's response.
"By specifying a persona in a system prompt, it forces the agent to adapt it tone and style - from abovethe persona would be a college professor and will always begin with the phrase 'Dear students,'"

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augmented_prompt_agent.txt"), "w") as f:
    f.write(augmented_agent_response + "\n")

