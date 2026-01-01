# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
)
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

def chat_interface(agent:CompiledStateGraph, ticket_id:str):
    # Import memory functions
    from agentic.memory import save_user_preference
    
    is_first_iteration = True
    user_name = None
    user_id = None
    messages = [SystemMessage(content = f"ThreadId: {ticket_id}")]
    
    while True:
        if is_first_iteration:
            user_input = input("What is your name? ")
            user_name = user_input.strip()
            print(f"Name: {user_name}")
            
            # Create or get user ID for memory system
            # For demo purposes, we'll use the name as a simple user identifier
            # In a real system, this would be looked up from a user database
            user_id = f"user_{user_name.lower().replace(' ', '_')}"
            
            prompt_text = "Question: "
        else:
            user_input = input(prompt_text)
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        
        # Add user name context to the message if available
        message_content = user_input
        if user_name and not is_first_iteration:
            message_content = f"User: {user_name}\nQuestion: {user_input}"
        
        messages = [HumanMessage(content=message_content)]
        
        # Prepare the state with user context
        trigger = {
            "messages": messages,
            "user_context": {"user_id": user_id, "user_name": user_name} if user_id else {},
            "ticket_id": ticket_id,
            "account_id": "cultpass"
        }
        config = {
            "configurable": {
                "thread_id": ticket_id,
            }
        }
        
        result = agent.invoke(input=trigger, config=config)
        print("Assistant:", result["messages"][-1].content)
        is_first_iteration = False