# Agentic System Design for Uda-hub Customer Support

## Overview
This agentic system is designed to provide automated customer support for Uda-hub, a platform that integrates with external services like CultPass (a cultural experiences subscription service). The system handles customer inquiries, retrieves information from knowledge bases, and performs actions on behalf of users.

## Architecture

### Multi-Agent Orchestration
The system uses a hierarchical agent structure with a supervisor agent coordinating specialized agents:

1. **Supervisor Agent**: Routes queries and coordinates between specialized agents
2. **Knowledge Retrieval Agent**: Handles FAQ and knowledge base queries
3. **Database Action Agent**: Performs read/write operations on databases
4. **Support Escalation Agent**: Handles complex issues requiring human intervention

### Workflow
The system follows a state machine pattern using LangGraph:
- **Initial Classification**: Determine query type and intent
- **Information Retrieval**: Search knowledge base and external data
- **Action Execution**: Perform database operations if needed
- **Response Generation**: Provide helpful, contextual responses
- **Escalation**: Route to human support when appropriate

### Memory Management
- **Short-term Memory**: Thread-based conversation context using LangGraph checkpointer
- **Long-term Memory**: Semantic search over conversation history and knowledge base

### Tools and Integrations

#### MCP Servers
- **CultPass Database Server**: Provides access to external CultPass data (experiences, users, subscriptions, reservations)
- **Uda-hub Database Server**: Manages internal Uda-hub data (accounts, users, tickets, knowledge base)

#### RAG System
- **Vector Database**: ChromaDB for storing and retrieving knowledge base articles
- **Embedding Model**: Uses sentence transformers for semantic search
- **Retrieval Strategy**: Hybrid search combining keyword and semantic matching

## Data Flow

1. User submits query via chat interface
2. Supervisor agent analyzes query and routes to appropriate agent
3. Specialized agents use tools to gather information
4. Actions are performed on databases through MCP servers
5. Response is generated and sent back to user
6. Conversation state is maintained for context

## Security Considerations
- Database access is abstracted through MCP servers
- Relative paths are used to avoid exposing absolute paths
- User data is accessed only through authorized queries
- Sensitive operations require explicit confirmation

## Scalability
- Modular agent design allows for easy addition of new capabilities
- Tool abstraction enables integration with new data sources
- State management supports concurrent conversations

## Evaluation Metrics
- Response accuracy
- Query resolution time
- User satisfaction scores
- Escalation rates

## Future Enhancements
- Multi-language support
- Voice interaction capabilities
- Integration with additional external services
- Advanced analytics and reporting