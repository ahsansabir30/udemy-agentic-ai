# Uda-hub Autonomous AI Agent - Solution

This is a complete, working implementation of an autonomous AI agent for Uda-hub customer support. The system uses a multi-agent architecture with specialized agents for different types of customer queries.

## Features

- **Multi-Agent Architecture**: Supervisor agent routes queries to specialized agents
- **Knowledge Base Search**: RAG-powered knowledge retrieval for informational queries
- **Database Operations**: Secure MCP server abstraction for database interactions
- **Action Handling**: Automated ticket management and reservation operations
- **Escalation Management**: Human-in-the-loop for complex issues
- **Stateful Conversations**: Memory-enabled chat with conversation history

## Architecture

### Agents
- **Supervisor Agent**: Routes queries to appropriate specialists based on content analysis
- **Knowledge Agent**: Handles informational queries using RAG search
- **Action Agent**: Performs database operations (reservations, ticket updates)
- **Escalation Agent**: Manages complex issues requiring human intervention

### Tools & Infrastructure
- **MCP Servers**: Secure database abstraction using Model Context Protocol
- **RAG System**: Knowledge base retrieval with ChromaDB and OpenAI embeddings
- **LangGraph Workflow**: Orchestrates agent interactions with state management
- **SQLite Databases**: Separate databases for external (CultPass) and internal (Uda-hub) data

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   Edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

### Running the Application

```bash
python3 03_agentic_app.py
```

The application will start an interactive chat interface. Type your customer support queries and the AI agent will respond appropriately.

### Example Queries to Try

- **Knowledge queries**: "What is included in my subscription?", "How do I reset my password?"
- **Action queries**: "Book a reservation for tomorrow", "Check my ticket status"
- **Escalation queries**: "I need to speak to a human representative"

## Project Structure

```
solution/
├── 03_agentic_app.py          # Main application entry point
├── agentic/                   # Agent implementations
│   ├── workflow.py           # Main orchestration logic
│   ├── agents/               # Individual agent classes
│   └── tools/                # MCP servers and RAG tools
├── data/                     # Database models and data files
│   ├── models/               # SQLAlchemy model definitions
│   ├── core/                 # Uda-hub internal database
│   └── external/             # CultPass external database
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
└── README.md                 # This file
```

## Database Setup

The application uses two SQLite databases that are pre-populated:

- **CultPass Database** (`data/external/cultpass.db`): External partner data
- **Uda-hub Database** (`data/core/udahub.db`): Internal ticket and user data

The databases are automatically set up when you run the notebooks in the starter folder, but the populated databases are included here for immediate use.

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY environment variable not set"**
   - Make sure your `.env` file contains a valid OpenAI API key

2. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Make sure you're running from the solution directory

3. **Database connection errors**
   - The database files should be present in the `data/` folders
   - If missing, run the setup notebooks from the starter folder

### Getting Help

If you encounter issues:
1. Check that all files are present in the solution folder
2. Verify your Python version (3.8+ required)
3. Ensure your OpenAI API key is valid and has sufficient credits
4. Check the terminal output for specific error messages

## Development

This solution was developed using:
- **LangChain/LangGraph**: Agent orchestration
- **FastMCP**: Model Context Protocol for secure database access
- **SQLAlchemy**: Database ORM
- **OpenAI GPT-4o-mini**: Language model
- **ChromaDB**: Vector database for knowledge retrieval
