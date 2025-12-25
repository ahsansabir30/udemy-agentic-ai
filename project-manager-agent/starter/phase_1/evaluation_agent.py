import os
from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent.respond,
    max_interactions=10
)

result = evaluation_agent.evaluate(prompt)
print("\nFinal Evaluation Result:")
print(result['final_response'])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_agent_output.txt"), "w") as f:
    f.write("Final Evaluation Result:\n")
    f.write(result['final_response'] + "\n")  
    f.write("Evaluation Result:\n")
    f.write(result['final_evaluation'] + "\n")