"""Tests for AgentIdentityRegistry and AgentIdentityVerifier"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pytest
import tempfile
from pathlib import Path
from server import AgentIdentityRegistry


class TestAgentIdentityRegistry:
    """Tests for the JSON-backed agent identity registry"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        self.registry = AgentIdentityRegistry(self.registry_path)

    @pytest.mark.unit
    def test_register_new_agent(self):
        """Register a new agent and retrieve it"""
        self.registry.register("stripe-acp:agent-1", agent_type="stripe_acp")
        agent = self.registry.lookup("stripe-acp:agent-1")
        assert agent is not None
        assert agent["agent_type"] == "stripe_acp"
        assert agent["transaction_count"] == 0
        assert agent["trust_score"] == 0.5  # default neutral trust

    @pytest.mark.unit
    def test_lookup_missing_agent(self):
        """Lookup non-existent agent returns None"""
        assert self.registry.lookup("nonexistent") is None

    @pytest.mark.unit
    def test_update_agent_transaction_count(self):
        """Recording a transaction increments count and updates last_seen"""
        self.registry.register("agent-1", agent_type="visa_tap")
        self.registry.record_transaction("agent-1")
        agent = self.registry.lookup("agent-1")
        assert agent["transaction_count"] == 1

    @pytest.mark.unit
    def test_update_trust_score(self):
        """Trust score can be updated"""
        self.registry.register("agent-1")
        self.registry.update_trust("agent-1", 0.9)
        agent = self.registry.lookup("agent-1")
        assert agent["trust_score"] == 0.9

    @pytest.mark.unit
    def test_trust_score_clamped(self):
        """Trust score clamped to [0, 1]"""
        self.registry.register("agent-1")
        self.registry.update_trust("agent-1", 1.5)
        assert self.registry.lookup("agent-1")["trust_score"] == 1.0
        self.registry.update_trust("agent-1", -0.5)
        assert self.registry.lookup("agent-1")["trust_score"] == 0.0

    @pytest.mark.unit
    def test_persistence(self):
        """Registry persists to disk and loads back"""
        self.registry.register("agent-1", agent_type="openai")
        self.registry.record_transaction("agent-1")
        # Create new registry from same path
        registry2 = AgentIdentityRegistry(self.registry_path)
        agent = registry2.lookup("agent-1")
        assert agent is not None
        assert agent["agent_type"] == "openai"
        assert agent["transaction_count"] == 1

    @pytest.mark.unit
    def test_registry_creates_file(self):
        """Registry creates file on first write"""
        assert not self.registry_path.exists()
        self.registry.register("agent-1")
        assert self.registry_path.exists()

    @pytest.mark.unit
    def test_register_duplicate_no_overwrite(self):
        """Registering same agent twice doesn't reset data"""
        self.registry.register("agent-1", agent_type="stripe_acp")
        self.registry.record_transaction("agent-1")
        self.registry.register("agent-1", agent_type="stripe_acp")
        agent = self.registry.lookup("agent-1")
        assert agent["transaction_count"] == 1  # not reset

    @pytest.mark.unit
    def test_list_agents(self):
        """List all registered agents"""
        self.registry.register("agent-1")
        self.registry.register("agent-2")
        agents = self.registry.list_agents()
        assert len(agents) == 2
        assert "agent-1" in agents
        assert "agent-2" in agents
