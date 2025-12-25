# agentic_workflow.py

import os
from dotenv import load_dotenv
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt"), "r") as f:
    product_spec = f.read()

# Instantiate all the agents
# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)


# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

persona_product_manager_eval = "You are an evaluation agent that checks user stories."
evaluation_criteria_project_manager = "Each story must follow this exact structure:\n As a [type of user], I want [an action or feature] so that [benefit/value]."

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_project_manager,
    worker_agent=product_manager_knowledge_agent.respond,
    max_interactions=5
)


# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = f"Features of a product are defined by organizing similar user stories into cohesive groups.\n{product_spec}"
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)


# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: " 
    "Feature Name: A clear, concise title\n"
    "Description: A brief explanation\n"
    "Key Functionality: The main capabilities\n"
    "User Benefit: The value to the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent.respond,
    max_interactions=5
)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = f"Development tasks are defined by identifying what needs to be built to implement each user story.\n{product_spec}"
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev = (
    "The answer should be tasks following this exact structure: " 
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n" \
    "Related User Story: Reference to the parent user story\n" \
    "Description: Detailed explanation of the technical work required\n" \
    "Acceptance Criteria: Specific requirements that must be met for completion\n" \
    "Estimated Effort: Time or complexity estimation\n" \
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev,
    worker_agent=development_engineer_knowledge_agent.respond,
    max_interactions=5
)

# Routing Agent
routing_agent = RoutingAgent(openai_api_key, [])

def product_manager_support(input):
    return product_manager_evaluation_agent.evaluate(input)

def program_manager_support(input):
    return program_manager_evaluation_agent.evaluate(input)

def development_engineer_support(input):
    return development_engineer_evaluation_agent.evaluate(input)

routing_agent.agents = [
    {
        "name": "product manager",
        "description": "Define user stories from a product specification",
        "func": product_manager_support,
    },
    {
        "name": "program manager",
        "description": "Define product features from user stories",
        "func": program_manager_support,
    },
    {
        "name": "development engineer",
        "description": "Define development tasks from user stories and features",
        "func": development_engineer_support,
    },
]

# Run the workflow
print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

completed_steps = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic_workflow_output.txt"), "a") as f:
    for i, step in enumerate(steps, start=1):
        print(f"\n--- Executing step {i}: {step} ---")
        f.write(f"{step.upper()}\n\n")
        result = routing_agent.route(step)
        completed_steps.append(result)
        print("Result:")
        f.write(f"{result.get('final_response')}\n")
        print(result.get('final_response'))
        print("Evaluation:")
        print(result.get('final_evaluation'))
        print(f"--- Step {i} completed ---\n")
        f.write("--------------------------\n\n")

print("\n*** Final Workflow Output ***\n")
print(completed_steps[-1]['final_response'])

print("\n*** Final Evaluation Output ***\n")
print(completed_steps[-1]['final_evaluation'])
