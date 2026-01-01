"""
MCP Server for Uda-hub internal database access.
Provides tools for querying accounts, users, tickets, and knowledge base.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastmcp import FastMCP
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.models import udahub

# Initialize FastMCP server
app = FastMCP("Uda-hub Database Server")

# Database setup
def get_db_path() -> str:
    """Get the relative path to the Uda-hub database."""
    return "data/core/udahub.db"

def get_engine():
    """Create database engine."""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    return create_engine(f"sqlite:///{db_path}", echo=False)

def get_session():
    """Get database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

@app.tool()
def search_knowledge_base(query: str, account_id: str = "cultpass", limit: int = 5) -> List[Dict[str, Any]]:
    """Search the knowledge base for relevant articles.

    Args:
        query: Search term for title, content, or tags
        account_id: Account to search in (default: cultpass)
        limit: Maximum number of results to return

    Returns:
        List of knowledge article dictionaries
    """
    with get_session() as session:
        if query:
            articles = session.query(udahub.Knowledge).filter(
                udahub.Knowledge.account_id == account_id,
                (udahub.Knowledge.title.contains(query) |
                 udahub.Knowledge.content.contains(query) |
                 udahub.Knowledge.tags.contains(query))
            ).limit(limit).all()
        else:
            articles = session.query(udahub.Knowledge).filter(
                udahub.Knowledge.account_id == account_id
            ).limit(limit).all()

        return [{
            "article_id": article.article_id,
            "title": article.title,
            "content": article.content,
            "tags": article.tags
        } for article in articles]

@app.tool()
def get_ticket_info(ticket_id: str) -> Optional[Dict[str, Any]]:
    """Get ticket information by ticket ID.

    Args:
        ticket_id: The ticket ID to look up

    Returns:
        Ticket information dictionary or None if not found
    """
    with get_session() as session:
        ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()
        if not ticket:
            return None

        messages = [{
            "message_id": msg.message_id,
            "role": msg.role.value,
            "content": msg.content,
            "created_at": str(msg.created_at)
        } for msg in ticket.messages]

        metadata = ticket.ticket_metadata
        metadata_info = {
            "status": metadata.status,
            "main_issue_type": metadata.main_issue_type,
            "tags": metadata.tags
        } if metadata else None

        return {
            "ticket_id": ticket.ticket_id,
            "account_id": ticket.account_id,
            "user_id": ticket.user_id,
            "channel": ticket.channel,
            "created_at": str(ticket.created_at),
            "metadata": metadata_info,
            "messages": messages
        }

@app.tool()
def create_ticket_message(ticket_id: str, role: str, content: str) -> Dict[str, Any]:
    """Add a new message to an existing ticket.

    Args:
        ticket_id: The ticket ID to add message to
        role: Message role (user, agent, ai, system)
        content: Message content

    Returns:
        Message creation result
    """
    import uuid
    from datetime import datetime

    with get_session() as session:
        # Check if ticket exists
        ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()
        if not ticket:
            return {"success": False, "reason": "Ticket not found"}

        # Validate role
        valid_roles = ["user", "agent", "ai", "system"]
        if role not in valid_roles:
            return {"success": False, "reason": f"Invalid role. Must be one of: {valid_roles}"}

        # Create message
        message = udahub.TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=udahub.RoleEnum[role],
            content=content
        )

        session.add(message)
        session.commit()

        return {
            "success": True,
            "message_id": message.message_id,
            "ticket_id": ticket_id,
            "role": role
        }

@app.tool()
def update_ticket_status(ticket_id: str, status: str, tags: str = None) -> Dict[str, Any]:
    """Update ticket status and metadata.

    Args:
        ticket_id: The ticket ID to update
        status: New status (open, closed, pending, etc.)
        tags: Optional tags to update

    Returns:
        Update result
    """
    with get_session() as session:
        # Check if ticket exists
        ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()
        if not ticket:
            return {"success": False, "reason": "Ticket not found"}

        # Update metadata
        metadata = ticket.ticket_metadata
        if metadata:
            metadata.status = status
            if tags:
                metadata.tags = tags
            session.commit()
        else:
            return {"success": False, "reason": "Ticket metadata not found"}

        return {
            "success": True,
            "ticket_id": ticket_id,
            "new_status": status,
            "updated_tags": tags
        }

@app.tool()
def get_user_tickets(account_id: str, external_user_id: str) -> List[Dict[str, Any]]:
    """Get all tickets for a specific user.

    Args:
        account_id: The account ID
        external_user_id: The external user ID

    Returns:
        List of user's tickets
    """
    with get_session() as session:
        # Find user
        user = session.query(udahub.User).filter_by(
            account_id=account_id,
            external_user_id=external_user_id
        ).first()

        if not user:
            return []

        tickets = []
        for ticket in user.tickets:
            metadata = ticket.ticket_metadata
            tickets.append({
                "ticket_id": ticket.ticket_id,
                "channel": ticket.channel,
                "created_at": str(ticket.created_at),
                "status": metadata.status if metadata else "unknown",
                "tags": metadata.tags if metadata else "",
                "message_count": len(ticket.messages)
            })

        return tickets

if __name__ == "__main__":
    app.run()