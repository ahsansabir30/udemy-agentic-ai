from logging import config
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    # Action tracking
    actions_taken: Annotated[List[str], operator.add]

def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> tuple[Dict[str, Any], List[str]]:
    agent = create_react_agent(llm, tools)
    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]
    return result, tools_used


# TODO: Implement the classify_intent function.
# This function should classify the user's intent and set the next step in the workflow.
# Refer to README.md Task 2.2
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    # Configure the llm chat model for structured output
    llm_structured = llm.with_structured_output(UserIntent)

    # Create a formatted prompt with conversation history and user input
    prompt_template = get_intent_classification_prompt()
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    formatted_prompt = prompt_template.format(user_input=state["user_input"], conversation_history=conversation_history)

    # Invoke the LLM with the prompt
    intent = llm_structured.invoke(formatted_prompt)

    # Add conditional logic to set next_step based on intent
    if intent.intent_type == "qa":
        next_step = "qa_agent"
    elif intent.intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent.intent_type == "calculation":
        next_step = "calculation_agent"
    else:
        next_step = "qa_agent"

    return {
        "intent": intent,
        "next_step": next_step,
        "actions_taken": ["classify_intent"],
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the summarization_agent function. Refer to README.md Task 2.3
def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }   


# TODO: Finish implementing the update_memory function. Refer to README.md Task 2.4
def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(
        UpdateMemoryResponse
    )

    response = structured_llm.invoke(prompt_with_history)
    return {
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "end"
    }

def should_continue(state: AgentState) -> str:
    """Router function"""
    intent = state.get("intent")
    if intent:
        return intent.intent_type
    return "unknown"

def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent, configurable={"llm": llm, "tools": tools})
    workflow.add_node("summarization_agent", summarization_agent, configurable={"llm": llm, "tools": tools})
    workflow.add_node("calculation_agent", calculation_agent, configurable={"llm": llm, "tools": tools})
    workflow.add_node("update_memory", update_memory, configurable={"llm": llm})    
    
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa": "qa_agent",
            "summarization": "summarization_agent",
            "calculation": "calculation_agent",
            "unknown": "qa_agent"
        }
    )

    # TODO: For each node add an edge that connects it to the update_memory node
    # qa_agent -> update_memory
    workflow.add_edge("qa_agent", "update_memory")
    # summarization_agent -> update_memory
    workflow.add_edge("summarization_agent", "update_memory")
    # calculation_agent -> update_memory
    workflow.add_edge("calculation_agent", "update_memory")
    workflow.add_edge("update_memory", END)
    return workflow.compile(checkpointer=InMemorySaver())
