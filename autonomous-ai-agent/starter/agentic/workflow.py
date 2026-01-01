"""
Main workflow orchestration for the Uda-hub agentic system.
"""

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
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        query = last_message.content
        routing = supervisor.route_query(query)

        return {
            "current_agent": routing["agent"],
            "messages": messages,
            "user_context": state.get("user_context", {}),
            "ticket_id": state.get("ticket_id", ""),
            "account_id": state.get("account_id", "cultpass")
        }

    return state

def knowledge_node(state: AgentState) -> Dict[str, Any]:
    """Knowledge agent processing."""
    initialize_agents()  # Ensure agents are initialized
    result = knowledge_executor.invoke({"messages": state["messages"]})

    return {
        "messages": result["messages"],
        "current_agent": "knowledge_agent",
        "user_context": state.get("user_context", {}),
        "ticket_id": state.get("ticket_id", ""),
        "account_id": state.get("account_id", "cultpass")
    }

def action_node(state: AgentState) -> Dict[str, Any]:
    """Action agent processing."""
    initialize_agents()  # Ensure agents are initialized
    result = action_executor.invoke({"messages": state["messages"]})

    return {
        "messages": result["messages"],
        "current_agent": "action_agent",
        "user_context": state.get("user_context", {}),
        "ticket_id": state.get("ticket_id", ""),
        "account_id": state.get("account_id", "cultpass")
    }

def escalation_node(state: AgentState) -> Dict[str, Any]:
    """Escalation agent processing."""
    initialize_agents()  # Ensure agents are initialized
    result = escalation_executor.invoke({"messages": state["messages"]})

    return {
        "messages": result["messages"],
        "current_agent": "escalation_agent",
        "user_context": state.get("user_context", {}),
        "ticket_id": state.get("ticket_id", ""),
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

# All agents lead to end for now
workflow.add_edge("knowledge_agent", END)
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