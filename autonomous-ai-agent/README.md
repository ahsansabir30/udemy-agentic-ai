# Uda-hub Autonomous AI Agent

An agentic AI system for automated customer support, integrating with external services like CultPass for cultural experiences.

## Project Overview

This project implements a multi-agent system for Uda-hub's customer support, featuring:

- **Supervisor Agent**: Routes queries to appropriate specialized agents
- **Knowledge Agent**: Handles FAQ and information retrieval
- **Action Agent**: Performs database operations and user actions
- **Escalation Agent**: Manages complex issues requiring human intervention

## Architecture

The system uses:
- LangGraph for workflow orchestration
- MCP (Model Context Protocol) servers for database abstraction
- RAG (Retrieval-Augmented Generation) for knowledge base search
- SQLite databases for data persistence

## Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd autonomous-ai-agent/starter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Initialize databases:**
   Run the setup notebooks in order:
   ```bash
   # Run in Jupyter or Python
   # 01_external_db_setup.ipynb - Sets up CultPass data
   # 02_core_db_setup.ipynb - Sets up Uda-hub data
   ```

### Running the Application

**Option 1: Using the Python script**
```bash
python 03_agentic_app.py
```

**Option 2: Using the notebook**
Open `03_agentic_app.ipynb` and run the cells.

## Testing

Run the test suite:
```bash
python test_agent.py
```

## Project Structure

```
starter/
├── agentic/
│   ├── agents/          # Agent implementations
│   ├── design/          # Architecture documentation
│   ├── tools/           # MCP servers and utilities
│   └── workflow.py      # Main orchestration
├── data/
│   ├── core/            # Uda-hub database
│   ├── external/        # CultPass database
│   └── models/          # SQLAlchemy models
├── .env                 # Environment variables
├── 03_agentic_app.py    # Main application
├── test_agent.py        # Test suite
└── utils.py             # Utilities
```

## Key Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Database Abstraction**: MCP servers for secure database access
- **Knowledge Base RAG**: Intelligent information retrieval
- **Memory Management**: Thread-based conversation context
- **Scalable Design**: Modular components for easy extension

## API Documentation

### MCP Servers

#### CultPass Database Server
- `search_experiences(query, limit)`: Search experiences
- `get_user_info(user_id)`: Get user details
- `create_reservation(user_id, experience_id)`: Book experience

#### Uda-hub Database Server
- `search_knowledge_base(query, account_id)`: Search knowledge base
- `get_ticket_info(ticket_id)`: Get ticket details
- `create_ticket_message(ticket_id, role, content)`: Add message to ticket

## Built With

- **LangChain/LangGraph**: Agent orchestration
- **FastMCP**: Model Context Protocol implementation
- **SQLAlchemy**: Database ORM
- **OpenAI**: LLM provider
- **Python**: Core language

## Contributing

1. Follow the established agent and tool patterns
2. Add tests for new functionality
3. Update documentation
4. Use relative paths for database access

## License

This project is part of the Udemy Agentic AI course.

## License
[License](../LICENSE.md)
