"""
Test cases for the Uda-hub Autonomous AI Agent system.
"""

import os
import sys
from pathlib import Path
import pytest

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agentic.tools.rag_tool import search_knowledge_base

class TestKnowledgeBaseSearch:
    """Test knowledge base search functionality."""

    def test_search_login_issues(self):
        """Test searching for login-related information."""
        results = search_knowledge_base("login")
        assert "How to Handle Login Issues?" in results
        assert "password" in results.lower()

    def test_search_reservation(self):
        """Test searching for reservation information."""
        results = search_knowledge_base("reserve")
        assert "How to Reserve a Spot for an Event" in results

    def test_search_nonexistent_topic(self):
        """Test searching for topic that doesn't exist."""
        results = search_knowledge_base("nonexistent_topic_xyz")
        assert "No relevant information found" in results

    def test_search_subscription(self):
        """Test searching for subscription information."""
        results = search_knowledge_base("subscription")
        assert len(results) > 0  # Should find some results

class TestSupervisorAgent:
    """Test the supervisor agent routing logic."""

    def test_supervisor_routing_knowledge(self):
        """Test supervisor routes knowledge queries correctly."""
        from agentic.agents.supervisor_agent import SupervisorAgent

        supervisor = SupervisorAgent()
        routing = supervisor.route_query("How do I reset my password?")

        assert routing["agent"] == "knowledge_agent"
        assert "information" in routing["reason"].lower()

    def test_supervisor_routing_action(self):
        """Test supervisor routes action queries correctly."""
        from agentic.agents.supervisor_agent import SupervisorAgent

        supervisor = SupervisorAgent()
        routing = supervisor.route_query("I want to book a reservation")

        assert routing["agent"] == "action_agent"
        assert "action" in routing["reason"].lower()

    def test_supervisor_routing_escalation(self):
        """Test supervisor routes complex queries to escalation."""
        from agentic.agents.supervisor_agent import SupervisorAgent

        supervisor = SupervisorAgent()
        routing = supervisor.route_query("My account is completely broken and nothing works")

        assert routing["agent"] == "escalation_agent"
        assert "complex" in routing["reason"].lower() or "unclear" in routing["reason"].lower()

    def test_supervisor_routing_knowledge_variations(self):
        """Test various knowledge query patterns."""
        from agentic.agents.supervisor_agent import SupervisorAgent

        supervisor = SupervisorAgent()
        queries = [
            "What is included in my subscription?",
            "How do I cancel?",
            "Tell me about premium experiences",
            "FAQ about payments"
        ]

        for query in queries:
            routing = supervisor.route_query(query)
            assert routing["agent"] == "knowledge_agent", f"Query '{query}' was not routed to knowledge_agent"

class TestDatabaseModels:
    """Test database model imports and basic functionality."""

    def test_model_imports(self):
        """Test that database models can be imported."""
        try:
            from data.models import cultpass, udahub
            assert cultpass.Base is not None
            assert udahub.Base is not None
        except ImportError as e:
            pytest.fail(f"Failed to import models: {e}")

class TestTools:
    """Test tool functionality."""

    def test_rag_tool_import(self):
        """Test RAG tool can be imported."""
        from agentic.tools.rag_tool import search_knowledge_base
        assert callable(search_knowledge_base)

    def test_mcp_servers_import(self):
        """Test MCP servers can be imported."""
        from agentic.tools.cultpass_db_server import app as cultpass_app
        from agentic.tools.udahub_db_server import app as udahub_app
        assert cultpass_app is not None
        assert udahub_app is not None

def run_tests():
    """Run all tests manually."""
    print("Running Uda-hub Agent Tests...")
    print("=" * 40)

    test_classes = [TestKnowledgeBaseSearch, TestSupervisorAgent, TestDatabaseModels, TestTools]
    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"Passed:  {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"Failed:  {method_name}: {e}")
                    failed += 1

    print(f"\n{'='*40}")
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)