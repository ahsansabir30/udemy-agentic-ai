"""
Uda-hub Autonomous AI Agent Application
Main application file for running the agentic customer support system.
"""

import os
from dotenv import load_dotenv
from utils import chat_interface
from agentic.workflow import get_orchestrator

def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key.")
        return

    print("🤖 Uda-hub Autonomous AI Agent")
    print("==============================")
    print("Welcome to the Uda-hub customer support system.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print()

    # Get the orchestrator (initializes agents)
    try:
        orchestrator = get_orchestrator()
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return

    # Start the chat interface
    try:
        chat_interface(orchestrator, "test_ticket_001")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()