"""
Supervisor agent that coordinates between specialized agents.
"""

from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from .base_agent import BaseAgent


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes queries to appropriate specialized agents."""

    def __init__(self, tools: List[BaseTool] = None):
        # Initialize without model for routing logic
        self.name = "Supervisor"
        self.tools = tools or []
        self.model = None  # Not needed for routing

    def get_system_prompt(self) -> str:
        return """You are the Supervisor Agent for Uda-hub customer support.

Your role is to analyze user queries and route them to the appropriate specialized agent.

Available agents:
- knowledge_agent: For FAQ, general information, and knowledge base queries
- action_agent: For performing actions like updating data, making reservations, managing accounts
- escalation_agent: For complex issues requiring human intervention

Analyze the user's query and determine which agent should handle it.

For knowledge queries (FAQs, how-tos, general information):
- Route to knowledge_agent

For action requests (bookings, updates, cancellations):
- Route to action_agent

For complaints, technical issues, or unclear requests:
- Route to escalation_agent

Always provide a brief explanation of your routing decision.

Response format:
AGENT: <agent_name>
REASON: <brief explanation>
"""

    def route_query(self, query: str) -> Dict[str, str]:
        """Route the query to appropriate agent using simple rules."""
        query_lower = query.lower()

        # Knowledge queries - informational, how-to, FAQs
        knowledge_keywords = [
            'how', 'what', 'why', 'when', 'where', 'who', 'tell me', 'explain',
            'faq', 'help', 'guide', 'login', 'password', 'subscription', 'premium',
            'included', 'about', 'information', 'details', 'features'
        ]

        # Action queries - things that modify state or perform operations
        action_keywords = [
            'book', 'reserve', 'cancel', 'update', 'change', 'create', 'make',
            'buy', 'purchase', 'add', 'remove', 'delete', 'modify', 'set'
        ]

        # Check for knowledge keywords
        if any(word in query_lower for word in knowledge_keywords):
            return {
                "agent": "knowledge_agent",
                "reason": "Query appears to be seeking information or guidance"
            }
        # Check for action keywords
        elif any(word in query_lower for word in action_keywords):
            return {
                "agent": "action_agent",
                "reason": "Query involves performing an action or modification"
            }
        # Default to escalation for complex or unclear queries
        else:
            return {
                "agent": "escalation_agent",
                "reason": "Query is complex or unclear, requires human assistance"
            }