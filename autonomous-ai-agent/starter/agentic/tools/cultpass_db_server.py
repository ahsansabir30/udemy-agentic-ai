"""
MCP Server for CultPass external database access.
Provides tools for querying CultPass experiences, users, subscriptions, and reservations.
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

from data.models import cultpass

# Initialize FastMCP server
app = FastMCP("CultPass Database Server")

# Database setup
def get_db_path() -> str:
    """Get the absolute path to the CultPass database."""
    script_dir = Path(__file__).parent.parent.parent  # Go up to solution directory
    return str(script_dir / "data" / "external" / "cultpass.db")

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
def search_experiences(query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
    """Search for experiences in CultPass database.

    Args:
        query: Search term for title or description
        limit: Maximum number of results to return

    Returns:
        List of experience dictionaries
    """
    with get_session() as session:
        if query:
            experiences = session.query(cultpass.Experience).filter(
                cultpass.Experience.title.contains(query) |
                cultpass.Experience.description.contains(query)
            ).limit(limit).all()
        else:
            experiences = session.query(cultpass.Experience).limit(limit).all()

        return [{
            "experience_id": exp.experience_id,
            "title": exp.title,
            "description": exp.description,
            "location": exp.location,
            "when": str(exp.when),
            "slots_available": exp.slots_available,
            "is_premium": exp.is_premium
        } for exp in experiences]

@app.tool()
def get_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user information by user ID.

    Args:
        user_id: The user ID to look up

    Returns:
        User information dictionary or None if not found
    """
    with get_session() as session:
        user = session.query(cultpass.User).filter_by(user_id=user_id).first()
        if not user:
            return None

        subscription = user.subscription
        reservations = [{
            "reservation_id": r.reservation_id,
            "experience_id": r.experience_id,
            "status": r.status
        } for r in user.reservations]

        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "subscription": {
                "subscription_id": subscription.subscription_id if subscription else None,
                "status": subscription.status if subscription else None,
                "tier": subscription.tier if subscription else None,
                "monthly_quota": subscription.monthly_quota if subscription else None
            } if subscription else None,
            "reservations": reservations
        }

@app.tool()
def check_reservation_availability(experience_id: str) -> Dict[str, Any]:
    """Check availability for a specific experience.

    Args:
        experience_id: The experience ID to check

    Returns:
        Availability information
    """
    with get_session() as session:
        experience = session.query(cultpass.Experience).filter_by(experience_id=experience_id).first()
        if not experience:
            return {"available": False, "reason": "Experience not found"}

        # Count current reservations
        reservation_count = session.query(cultpass.Reservation).filter_by(
            experience_id=experience_id,
            status="reserved"
        ).count()

        available = reservation_count < experience.slots_available
        return {
            "available": available,
            "slots_available": experience.slots_available,
            "slots_taken": reservation_count,
            "experience_title": experience.title
        }

@app.tool()
def create_reservation(user_id: str, experience_id: str) -> Dict[str, Any]:
    """Create a new reservation for a user.

    Args:
        user_id: The user ID making the reservation
        experience_id: The experience ID to reserve

    Returns:
        Reservation result
    """
    import uuid
    from datetime import datetime

    with get_session() as session:
        # Check if user exists
        user = session.query(cultpass.User).filter_by(user_id=user_id).first()
        if not user:
            return {"success": False, "reason": "User not found"}

        # Check if experience exists and is available
        experience = session.query(cultpass.Experience).filter_by(experience_id=experience_id).first()
        if not experience:
            return {"success": False, "reason": "Experience not found"}

        # Check availability
        availability = check_reservation_availability(experience_id)
        if not availability["available"]:
            return {"success": False, "reason": "Experience is fully booked"}

        # Create reservation
        reservation = cultpass.Reservation(
            reservation_id=str(uuid.uuid4())[:6],
            user_id=user_id,
            experience_id=experience_id,
            status="reserved"
        )

        session.add(reservation)
        session.commit()

        return {
            "success": True,
            "reservation_id": reservation.reservation_id,
            "experience_title": experience.title,
            "user_name": user.full_name
        }

if __name__ == "__main__":
    app.run()