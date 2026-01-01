"""
Database setup script for the enhanced autonomous AI agent.
Creates the necessary tables for persistent memory functionality.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from data.models import udahub
from sqlalchemy import create_engine

def setup_database():
    """Create all database tables."""
    db_path = "data/core/udahub.db"

    # Create engine
    engine = create_engine(f"sqlite:///{db_path}", echo=True)

    # Create all tables
    print("Creating database tables...")
    udahub.Base.metadata.create_all(engine)
    print("✅ Database setup complete!")

if __name__ == "__main__":
    setup_database()