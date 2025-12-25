import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print("Agent Response Using Provided Knowledge: %s" % knowledge)
response = knowledge_agent.respond(prompt)
print("Response from the agent: %s" % response)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_augmented_prompt_agent_output.txt"), "w") as f:
    f.write("Agent Response Using Provided Knowledge:\n")
    f.write(knowledge + "\n")
    f.write("Response from the agent:\n")
    f.write(response + "\n")
