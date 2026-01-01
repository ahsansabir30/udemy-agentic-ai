"""
Main workflow orchestration for the Uda-hub agentic system.
"""

import logging
import uuid
from typing import Dict, Any, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Import agents
from agentic.agents.supervisor_agent import SupervisorAgent
from agentic.agents.knowledge_agent import KnowledgeAgent
from agentic.agents.action_agent import ActionAgent
from agentic.agents.escalation_agent import EscalationAgent

# Import tools
from agentic.tools.rag_tool import search_knowledge_base
from agentic.tools.cultpass_db_server import (
    search_experiences, get_user_info, check_reservation_availability, create_reservation
)
from agentic.tools.udahub_db_server import (
    search_knowledge_base as search_kb_udahub,
    get_ticket_info, create_ticket_message, update_ticket_status, get_user_tickets
)

# Import memory system
from agentic.memory import (
    save_conversation_message, get_conversation_history,
    save_user_preference, get_user_preference, get_all_user_preferences
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    user_context: Dict[str, Any]
    ticket_id: str
    account_id: str

# Global variables for agents and executors
supervisor = None
knowledge_executor = None
action_executor = None
escalation_executor = None

def initialize_agents():
    """Initialize agents and executors with API keys."""
    global supervisor, knowledge_executor, action_executor, escalation_executor

    if supervisor is None:
        supervisor = SupervisorAgent()

        # Create tools
        knowledge_tools = [
            Tool(
                name="search_knowledge_base",
                func=search_knowledge_base,
                description="Search the knowledge base for information about CultPass and Uda-hub services."
            ),
            Tool(
                name="search_kb_udahub",
                func=search_kb_udahub.fn,
                description="Search Uda-hub knowledge base for account-specific information."
            )
        ]

        action_tools = [
            Tool(
                name="search_experiences",
                func=search_experiences.fn,
                description="Search for available experiences in CultPass."
            ),
            Tool(
                name="get_user_info",
                func=get_user_info.fn,
                description="Get user information from CultPass database."
            ),
            Tool(
                name="check_reservation_availability",
                func=check_reservation_availability.fn,
                description="Check if an experience has available slots."
            ),
            Tool(
                name="create_reservation",
                func=create_reservation.fn,
                description="Create a new reservation for a user."
            ),
            Tool(
                name="get_ticket_info",
                func=get_ticket_info.fn,
                description="Get ticket information from Uda-hub."
            ),
            Tool(
                name="create_ticket_message",
                func=create_ticket_message.fn,
                description="Add a message to an existing ticket."
            ),
            Tool(
                name="update_ticket_status",
                func=update_ticket_status.fn,
                description="Update ticket status and metadata."
            ),
            Tool(
                name="get_user_tickets",
                func=get_user_tickets.fn,
                description="Get all tickets for a specific user."
            )
        ]

        knowledge_agent = KnowledgeAgent(tools=knowledge_tools)
        action_agent = ActionAgent(tools=action_tools)
        escalation_agent = EscalationAgent(tools=action_tools)

        # Create agent executors
        knowledge_executor = knowledge_agent.create_agent(knowledge_agent.get_system_prompt())
        action_executor = action_agent.create_agent(action_agent.get_system_prompt())
        escalation_executor = escalation_agent.create_agent(escalation_agent.get_system_prompt())

# Workflow functions
def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """Supervisor agent routing logic."""
    logger.info("Supervisor node processing request")

    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        query = last_message.content
        logger.info(f"Routing query: {query[:100]}...")

        routing = supervisor.route_query(query)
        logger.info(f"Routed to agent: {routing['agent']} - Reason: {routing['reason']}")

        # Save user message to persistent memory
        ticket_id = state.get("ticket_id", "unknown")
        user_id = state.get("user_context", {}).get("user_id", "unknown")
        if user_id != "unknown":
            try:
                save_conversation_message(user_id, ticket_id, "human", query)
                logger.info(f"Saved user message to persistent memory for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to save user message to memory: {e}")

        return {
            "current_agent": routing["agent"],
            "messages": messages,
            "user_context": state.get("user_context", {}),
            "ticket_id": ticket_id,
            "account_id": state.get("account_id", "cultpass")
        }

    return state

def knowledge_node(state: AgentState) -> Dict[str, Any]:
    """Knowledge agent processing with confidence scoring and escalation."""
    logger.info("Knowledge node processing request")

    initialize_agents()  # Ensure agents are initialized

    # Get the last human message for analysis
    messages = state["messages"]
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    if last_human_message:
        query = last_human_message.content
        logger.info(f"Processing knowledge query: {query[:100]}...")

        # Retrieve memory context for personalized responses
        user_id = state.get("user_context", {}).get("user_id", "unknown")
        ticket_id = state.get("ticket_id", "unknown")
        
        memory_context = ""
        conversation_history = []
        user_prefs = {}
        
        if user_id != "unknown":
            try:
                # Get conversation history
                conversation_history = get_conversation_history(user_id, ticket_id, limit=10)
                if conversation_history:
                    memory_context += f"\n\nCONVERSATION HISTORY:\n"
                    for msg in conversation_history[-5:]:  # Last 5 messages for context
                        role = "User" if msg["message_type"] == "human" else "Assistant"
                        memory_context += f"{role}: {msg['content'][:200]}...\n"
                
                # Get user preferences
                user_prefs = get_all_user_preferences(user_id)
                if user_prefs:
                    memory_context += f"\nUSER PREFERENCES:\n"
                    for key, value in user_prefs.items():
                        memory_context += f"- {key}: {value}\n"
                        
                logger.info(f"Retrieved memory context for user {user_id}: {len(conversation_history)} messages, {len(user_prefs)} preferences")
            except Exception as e:
                logger.error(f"Failed to retrieve memory context: {e}")
                memory_context = ""

        # Check if we should escalate based on query complexity or previous failures
        should_escalate = False

        # Simple heuristic: escalate if query contains certain keywords
        escalation_keywords = ["complaint", "problem", "issue", "wrong", "error", "bug", "refund", "cancel"]
        if any(keyword in query.lower() for keyword in escalation_keywords):
            should_escalate = True
            logger.info("Query contains escalation keywords, routing to escalation")

    # Create enhanced messages with memory context
    enhanced_messages = messages.copy()
    if memory_context:
        # Add memory context as a system message at the beginning
        from langchain_core.messages import SystemMessage
        memory_message = SystemMessage(content=f"USER CONTEXT INFORMATION:{memory_context}")
        enhanced_messages.insert(0, memory_message)

    result = knowledge_executor.invoke({"messages": enhanced_messages})

    # Extract confidence information from the result if available
    confidence = 0.5  # Default confidence
    ai_response = ""

    # Try to extract confidence from the AI response
    if result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            ai_response = last_message.content

            # Look for confidence indicators in the response
            if "No relevant information found" in ai_response:
                confidence = 0.0
                should_escalate = True
                logger.warning("No relevant information found, escalating")
            elif "don't know" in ai_response.lower() or "uncertain" in ai_response.lower():
                confidence = 0.2
                if confidence < 0.3:
                    should_escalate = True
                    logger.info("Low confidence response, escalating")

    # Save AI response to persistent memory
    ticket_id = state.get("ticket_id", "unknown")
    user_id = state.get("user_context", {}).get("user_id", "unknown")
    if user_id != "unknown" and ai_response:
        try:
            save_conversation_message(user_id, ticket_id, "ai", ai_response)
            logger.info(f"Saved AI response to persistent memory for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save AI response to memory: {e}")

    # Determine next agent based on confidence and escalation logic
    next_agent = "knowledge_agent"  # Default to end
    if should_escalate:
        next_agent = "escalation_agent"
        logger.info("Knowledge agent escalating to human support")

    return {
        "messages": result["messages"],
        "current_agent": next_agent,
        "user_context": state.get("user_context", {}),
        "ticket_id": ticket_id,
        "account_id": state.get("account_id", "cultpass")
    }

def action_node(state: AgentState) -> Dict[str, Any]:
    """Action agent processing."""
    logger.info("Action node processing request")

    initialize_agents()  # Ensure agents are initialized
    
    # Retrieve memory context for personalized responses
    user_id = state.get("user_context", {}).get("user_id", "unknown")
    ticket_id = state.get("ticket_id", "unknown")
    
    memory_context = ""
    conversation_history = []
    user_prefs = {}
    
    if user_id != "unknown":
        try:
            # Get conversation history
            conversation_history = get_conversation_history(user_id, ticket_id, limit=10)
            if conversation_history:
                memory_context += f"\n\nCONVERSATION HISTORY:\n"
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    role = "User" if msg["message_type"] == "human" else "Assistant"
                    memory_context += f"{role}: {msg['content'][:200]}...\n"
            
            # Get user preferences
            user_prefs = get_all_user_preferences(user_id)
            if user_prefs:
                memory_context += f"\nUSER PREFERENCES:\n"
                for key, value in user_prefs.items():
                    memory_context += f"- {key}: {value}\n"
                    
            logger.info(f"Retrieved memory context for action agent - user {user_id}: {len(conversation_history)} messages, {len(user_prefs)} preferences")
        except Exception as e:
            logger.error(f"Failed to retrieve memory context for action agent: {e}")
            memory_context = ""

    # Create enhanced messages with memory context
    messages = state["messages"]
    enhanced_messages = messages.copy()
    if memory_context:
        # Add memory context as a system message at the beginning
        from langchain_core.messages import SystemMessage
        memory_message = SystemMessage(content=f"USER CONTEXT INFORMATION:{memory_context}")
        enhanced_messages.insert(0, memory_message)

    result = action_executor.invoke({"messages": enhanced_messages})

    # Save AI response to persistent memory
    ticket_id = state.get("ticket_id", "unknown")
    user_id = state.get("user_context", {}).get("user_id", "unknown")
    if result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content') and user_id != "unknown":
            try:
                save_conversation_message(user_id, ticket_id, "ai", last_message.content)
                logger.info(f"Saved action agent response to persistent memory for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to save action response to memory: {e}")

    logger.info("Action agent completed processing")
    return {
        "messages": result["messages"],
        "current_agent": "action_agent",
        "user_context": state.get("user_context", {}),
        "ticket_id": ticket_id,
        "account_id": state.get("account_id", "cultpass")
    }

def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Escalation agent processing."""
    logger.info("Escalation node processing - routing to human support")

    initialize_agents()  # Ensure agents are initialized
    
    # Retrieve memory context for personalized responses
    user_id = state.get("user_context", {}).get("user_id", "unknown")
    ticket_id = state.get("ticket_id", "unknown")
    
    memory_context = ""
    conversation_history = []
    user_prefs = {}
    
    if user_id != "unknown":
        try:
            # Get conversation history
            conversation_history = get_conversation_history(user_id, ticket_id, limit=10)
            if conversation_history:
                memory_context += f"\n\nCONVERSATION HISTORY:\n"
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    role = "User" if msg["message_type"] == "human" else "Assistant"
                    memory_context += f"{role}: {msg['content'][:200]}...\n"
            
            # Get user preferences
            user_prefs = get_all_user_preferences(user_id)
            if user_prefs:
                memory_context += f"\nUSER PREFERENCES:\n"
                for key, value in user_prefs.items():
                    memory_context += f"- {key}: {value}\n"
                    
            logger.info(f"Retrieved memory context for escalation agent - user {user_id}: {len(conversation_history)} messages, {len(user_prefs)} preferences")
        except Exception as e:
            logger.error(f"Failed to retrieve memory context for escalation agent: {e}")
            memory_context = ""

    # Create enhanced messages with memory context
    messages = state["messages"]
    enhanced_messages = messages.copy()
    if memory_context:
        # Add memory context as a system message at the beginning
        from langchain_core.messages import SystemMessage
        memory_message = SystemMessage(content=f"USER CONTEXT INFORMATION:{memory_context}")
        enhanced_messages.insert(0, memory_message)

    result = escalation_executor.invoke({"messages": enhanced_messages})

    # Save AI response to persistent memory
    ticket_id = state.get("ticket_id", "unknown")
    user_id = state.get("user_context", {}).get("user_id", "unknown")
    if result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content') and user_id != "unknown":
            try:
                save_conversation_message(user_id, ticket_id, "ai", last_message.content)
                logger.info(f"Saved escalation response to persistent memory for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to save escalation response to memory: {e}")

    logger.info("Escalation agent completed - human support notified")
    return {
        "messages": result["messages"],
        "current_agent": "escalation_agent",
        "user_context": state.get("user_context", {}),
        "ticket_id": ticket_id,
        "account_id": state.get("account_id", "cultpass")
    }

# Create workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("knowledge_agent", knowledge_node)
workflow.add_node("action_agent", action_node)
workflow.add_node("escalation_agent", escalation_node)

# Add edges
workflow.set_entry_point("supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["current_agent"],
    {
        "knowledge_agent": "knowledge_agent",
        "action_agent": "action_agent",
        "escalation_agent": "escalation_agent"
    }
)

# Conditional routing from knowledge agent (can escalate)
workflow.add_conditional_edges(
    "knowledge_agent",
    lambda state: state["current_agent"],
    {
        "knowledge_agent": END,  # End conversation
        "escalation_agent": "escalation_agent"  # Escalate to human
    }
)

# All other agents lead to end
workflow.add_edge("action_agent", END)
workflow.add_edge("escalation_agent", END)

# Add memory
checkpointer = MemorySaver()

# Create orchestrator (will initialize agents when first called)
def get_orchestrator():
    """Get the orchestrator, initializing agents if needed."""
    initialize_agents()
    return workflow.compile(checkpointer=checkpointer)

orchestrator = None