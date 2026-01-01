"""
Persistent memory system for the Uda-hub autonomous AI agent.
Provides long-term storage and retrieval of conversation history and user preferences.
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.models import udahub
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class PersistentMemory:
    """Persistent memory system for storing conversation history and user preferences."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use absolute path based on script location
            script_dir = Path(__file__).parent.parent  # Go up to solution directory
            db_path = str(script_dir / "data" / "core" / "udahub.db")
        
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def _get_session(self):
        """Get a database session."""
        return self.Session()

    def save_conversation_message(self, user_id: str, ticket_id: str,
                                message_type: str, content: str) -> str:
        """Save a conversation message to persistent storage.

        Args:
            user_id: The user identifier
            ticket_id: The ticket/conversation identifier
            message_type: 'human' or 'ai'
            content: The message content

        Returns:
            The message ID
        """
        message_id = str(uuid.uuid4())

        with self._get_session() as session:
            conversation = udahub.ConversationHistory(
                id=message_id,
                user_id=user_id,
                ticket_id=ticket_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now()
            )
            session.add(conversation)
            session.commit()

        return message_id

    def get_conversation_history(self, user_id: str, ticket_id: str,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a user and ticket.

        Args:
            user_id: The user identifier
            ticket_id: The ticket/conversation identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of conversation messages with metadata
        """
        with self._get_session() as session:
            messages = session.query(udahub.ConversationHistory)\
                .filter_by(user_id=user_id, ticket_id=ticket_id)\
                .order_by(udahub.ConversationHistory.timestamp)\
                .limit(limit)\
                .all()

            return [{
                'id': msg.id,
                'message_type': msg.message_type,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            } for msg in messages]

    def save_user_preference(self, user_id: str, key: str, value: str) -> bool:
        """Save or update a user preference.

        Args:
            user_id: The user identifier
            key: Preference key
            value: Preference value

        Returns:
            True if successful, False otherwise
        """
        with self._get_session() as session:
            # Check if preference already exists
            existing = session.query(udahub.UserPreferences)\
                .filter_by(user_id=user_id, preference_key=key)\
                .first()

            if existing:
                existing.preference_value = value
                existing.updated_at = datetime.now()
            else:
                preference = udahub.UserPreferences(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    preference_key=key,
                    preference_value=value
                )
                session.add(preference)

            session.commit()
            return True

    def get_user_preference(self, user_id: str, key: str) -> Optional[str]:
        """Retrieve a user preference.

        Args:
            user_id: The user identifier
            key: Preference key

        Returns:
            Preference value or None if not found
        """
        with self._get_session() as session:
            preference = session.query(udahub.UserPreferences)\
                .filter_by(user_id=user_id, preference_key=key)\
                .first()

            return preference.preference_value if preference else None

    def get_all_user_preferences(self, user_id: str) -> Dict[str, str]:
        """Retrieve all preferences for a user.

        Args:
            user_id: The user identifier

        Returns:
            Dictionary of preference key-value pairs
        """
        with self._get_session() as session:
            preferences = session.query(udahub.UserPreferences)\
                .filter_by(user_id=user_id)\
                .all()

            return {pref.preference_key: pref.preference_value for pref in preferences}

    def get_recent_conversations(self, user_id: str, days: int = 7,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation summaries for a user.

        Args:
            user_id: The user identifier
            days: Number of days to look back
            limit: Maximum number of conversations to return

        Returns:
            List of recent conversation summaries
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)

        with self._get_session() as session:
            # Get distinct ticket_ids for recent conversations
            recent_tickets = session.query(
                udahub.ConversationHistory.ticket_id,
                udahub.Ticket.title,
                udahub.ConversationHistory.timestamp
            )\
            .join(udahub.Ticket, udahub.ConversationHistory.ticket_id == udahub.Ticket.ticket_id)\
            .filter(
                udahub.ConversationHistory.user_id == user_id,
                udahub.ConversationHistory.timestamp >= cutoff_date
            )\
            .order_by(udahub.ConversationHistory.timestamp.desc())\
            .distinct()\
            .limit(limit)\
            .all()

            return [{
                'ticket_id': ticket_id,
                'title': title,
                'last_message': timestamp.isoformat()
            } for ticket_id, title, timestamp in recent_tickets]


# Global memory instance
_memory_instance = None

def get_memory() -> PersistentMemory:
    """Get the global memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = PersistentMemory()
    return _memory_instance


# Convenience functions for easy access
def save_conversation_message(user_id: str, ticket_id: str,
                            message_type: str, content: str) -> str:
    """Save a conversation message."""
    return get_memory().save_conversation_message(user_id, ticket_id, message_type, content)

def get_conversation_history(user_id: str, ticket_id: str,
                           limit: int = 50) -> List[Dict[str, Any]]:
    """Get conversation history."""
    return get_memory().get_conversation_history(user_id, ticket_id, limit)

def save_user_preference(user_id: str, key: str, value: str) -> bool:
    """Save a user preference."""
    return get_memory().save_user_preference(user_id, key, value)

def get_user_preference(user_id: str, key: str) -> Optional[str]:
    """Get a user preference."""
    return get_memory().get_user_preference(user_id, key)

def get_all_user_preferences(user_id: str) -> Dict[str, str]:
    """Get all user preferences."""
    return get_memory().get_all_user_preferences(user_id)

def get_recent_conversations(user_id: str, days: int = 7,
                           limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversations."""
    return get_memory().get_recent_conversations(user_id, days, limit)