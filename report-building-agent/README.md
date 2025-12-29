# Document Assistant Project

The Document Assistant project is utilising an AI-powered document processing system using LangChain and LangGraph.

It uses a multi-agent architecture to handle different tasks, including answering questions about documents, generating summaries and key points, and performing calculations on financial and healthcare data.

## Getting Started

### Dependencies

Make sure you have the following installed:

- Python 3.9+
- Git
- pip

### Installation

```bash
git clone https://github.com/username/repository-name.git
cd report-building-agent

python -m venv venv

source venv/bin/activate

# venv\Scripts\Activate.ps1 (Windows)

pip install -r requirements.txt

cp .env.example .env # update parameters in this file 

python main.py
```

1. Clone the repository:
```bash

git clone https://github.com/username/repository-name.git
cd report-building-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows -> venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
# Edit file with env variables
cp .env.example .env
```

### Implementation Decision
#### Overview
The current implementation of the agentic system manages user interactions through session-based state tracking. Each session represents an isolated conversational context in which agents can reason, act, and persist intermediate results.

#### Session and State Management
Users are required to create a unique user ID at the start of a conversation. This user ID is used to initialize a unique session, which is persisted as a file within the sessions directory.

For each user query:
1. The system generates a response via the appropriate agent.
2. The agent state (AgentState) is updated to reflect the latest interaction (e.g., updating the current_response field).
3. The full interaction is logged to the corresponding session file.

In addition, a separate evaluation and summarization process is run on the conversation. The resulting condensed representation is stored as conversation_summary, enabling efficient context retrieval and long-term session tracking.

#### System Purpose
The system is designed to support:
- Question answering over document collections
- Summarization of key document insights
- Calculations over financial and healthcare-related data

#### Agent Architecture
The workflow currently consists of three specialized agents:
1. QA Agent
    - Handles question-and-answer tasks.
    - Records query intent and response metadata.
2. Summarization Agent
    - Produces summaries of conversations or documents.
    - Logs summarization actions to the session state.
3. Calculation Agent
    - Performs numerical and analytical computations.
    - Records calculation inputs and outputs.

####  Tooling and Agent Autonomy
All agents have access to a shared set of four tools. Tool invocation is autonomously determined by each agent based on task requirements:
1. Calculator Tool 
    - Safely evaluates mathematical expressions and returns computed results.
2. Document Search Tool 
    - Searches for relevant documents using structured filters and natural language queries.
3. Document Reader Tool
    - Retrieves the full content of a document by its unique identifier.
4. Document Statistics Tool
    - Provides aggregate statistics across the document corpus.

Agents decide when and how to invoke these tools as part of their reasoning and action-planning process.

To ensure that each agent produces outputs aligned with its intended purpose, the system relies on strictly specialized prompts rather than runtime deferral or agent handoff mechanisms. Each agent is initialized with a role-specific prompt that explicitly constrains the agent’s scope of responsibility. These prompts act as behavioral contracts, guiding the language model to generate responses that are correct for the agent’s role.

#### Checkpointing and Persistence
The system employs a checkpointer responsible for saving and restoring execution state while workflows, agents, or graphs are running. In the current implementation, checkpoint data is stored exclusively in memory (RAM) for testing and development purposes. This design allows for future extension to persistent storage (e.g., databases or object stores) in production environments.

### Running the Assistant

```bash
python main.py
```

