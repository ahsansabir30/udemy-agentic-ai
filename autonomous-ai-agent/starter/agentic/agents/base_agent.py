"""
Base agent utilities and common functionality for Uda-hub agentic system.
"""
import os
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


class BaseAgent:
    """Base class for all agents in the system."""

    def __init__(self, name: str, model_name: str = "gpt-4o-mini", tools: List[BaseTool] = None):
        self.name = name
        self.model = ChatOpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY"),
            model=model_name)
        self.tools = tools or []

    def create_agent(self, system_prompt: str):
        """Create a LangGraph agent with the given system prompt."""
        return create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=system_prompt
        )

    def get_system_prompt(self) -> str:
        """Override this method to provide agent-specific system prompt."""
        return f"You are {self.name}, a helpful AI assistant."


class AgentState:
    """State management for the agentic workflow."""

    messages: List[BaseMessage]
    current_agent: str
    user_context: Dict[str, Any]
    ticket_id: str
    account_id: str

    def __init__(self):
        self.messages = []
        self.current_agent = "supervisor"
        self.user_context = {}
        self.ticket_id = ""
        self.account_id = "cultpass"