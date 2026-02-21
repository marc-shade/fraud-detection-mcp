"""Tests for AgentIdentityRegistry and AgentIdentityVerifier"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pytest
import tempfile
from pathlib import Path
from server import AgentIdentityRegistry, AgentIdentityVerifier


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


class TestAgentIdentityVerifier:
    """Tests for agent credential validation"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        self.registry = AgentIdentityRegistry(self.registry_path)
        self.verifier = AgentIdentityVerifier(self.registry)

    @pytest.mark.unit
    def test_verify_with_valid_agent_id(self):
        """Known agent in registry is verified"""
        self.registry.register("stripe-acp:agent-1", agent_type="stripe_acp")
        self.registry.update_trust("stripe-acp:agent-1", 0.8)
        result = self.verifier.verify(agent_identifier="stripe-acp:agent-1")
        assert result["verified"] is True
        assert result["trust_score"] >= 0.5
        assert result["identity"]["agent_id"] == "stripe-acp:agent-1"

    @pytest.mark.unit
    def test_verify_unknown_agent(self):
        """Unknown agent not in registry gets low trust"""
        result = self.verifier.verify(agent_identifier="unknown-agent")
        assert result["verified"] is False
        assert result["trust_score"] < 0.5
        assert "not_in_registry" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_api_key_valid_format(self):
        """Valid API key format passes basic validation"""
        result = self.verifier.verify(
            agent_identifier="test-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        )
        assert "invalid_key_format" not in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_api_key_too_short(self):
        """API key that's too short gets a warning"""
        result = self.verifier.verify(
            agent_identifier="test-agent",
            api_key="short"
        )
        assert "invalid_key_format" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_jwt_expired(self):
        """Expired JWT token gets a warning"""
        import base64
        # Create a fake expired JWT (header.payload.signature)
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload = base64.urlsafe_b64encode(b'{"exp":1000000000,"sub":"agent-1"}').rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(agent_identifier="agent-1", token=token)
        assert "token_expired" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_jwt_valid_expiry(self):
        """Non-expired JWT doesn't get expiry warning"""
        import base64
        import time
        future_exp = int(time.time()) + 3600  # 1 hour from now
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "agent-1"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(agent_identifier="agent-1", token=token)
        assert "token_expired" not in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_malformed_token(self):
        """Malformed token gets a warning"""
        result = self.verifier.verify(agent_identifier="agent-1", token="not-a-jwt")
        assert "token_parse_error" in result["warnings"]

    @pytest.mark.unit
    def test_verify_result_structure(self):
        """Result contains all required fields"""
        result = self.verifier.verify(agent_identifier="test-agent")
        assert "verified" in result
        assert "identity" in result
        assert "trust_score" in result
        assert "warnings" in result
        assert isinstance(result["verified"], bool)
        assert isinstance(result["trust_score"], float)
        assert isinstance(result["warnings"], list)

    @pytest.mark.unit
    def test_verify_no_identifier(self):
        """No identifier at all returns unverified"""
        result = self.verifier.verify()
        assert result["verified"] is False
        assert "no_identifier" in result["warnings"]

    @pytest.mark.unit
    def test_verify_auto_registers_new_agent(self):
        """First-time agent gets auto-registered in registry"""
        result = self.verifier.verify(agent_identifier="brand-new-agent")
        agent = self.registry.lookup("brand-new-agent")
        assert agent is not None
        assert agent["trust_score"] == 0.3  # new agent starts low

    @pytest.mark.unit
    def test_verify_known_agent_high_trust(self):
        """Known agent with many transactions has higher trust"""
        self.registry.register("trusted-agent", agent_type="visa_tap")
        self.registry.update_trust("trusted-agent", 0.9)
        for _ in range(50):
            self.registry.record_transaction("trusted-agent")
        result = self.verifier.verify(agent_identifier="trusted-agent")
        assert result["verified"] is True
        assert result["trust_score"] >= 0.8

    @pytest.mark.unit
    def test_verify_combines_signals(self):
        """Multiple valid signals increase trust"""
        import base64
        import time
        self.registry.register("multi-signal-agent", agent_type="stripe_acp")
        self.registry.update_trust("multi-signal-agent", 0.7)
        future_exp = int(time.time()) + 3600
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "multi-signal-agent"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(
            agent_identifier="multi-signal-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            token=token
        )
        assert result["verified"] is True
        # 3 signals: registry(0.7) + api_key(0.6) + jwt(0.7) = avg 0.667
        assert result["trust_score"] >= 0.65


class TestVerifyAgentIdentityImpl:
    """Tests for the verify_agent_identity_impl function"""

    @pytest.mark.unit
    def test_impl_returns_valid_result(self):
        from server import verify_agent_identity_impl
        result = verify_agent_identity_impl(
            agent_identifier="test-impl-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        )
        assert "verified" in result
        assert "trust_score" in result
        assert "verification_timestamp" in result

    @pytest.mark.unit
    def test_impl_no_identifier(self):
        from server import verify_agent_identity_impl
        result = verify_agent_identity_impl()
        assert result["verified"] is False
        assert "no_identifier" in result["warnings"]

    @pytest.mark.unit
    def test_impl_with_all_credentials(self):
        import base64
        import time
        from server import verify_agent_identity_impl
        future_exp = int(time.time()) + 3600
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "full-cred-agent"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = verify_agent_identity_impl(
            agent_identifier="full-cred-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            token=token
        )
        assert result["verified"] is True or result["trust_score"] > 0

    @pytest.mark.unit
    def test_impl_error_handling(self):
        from server import verify_agent_identity_impl
        # Should not crash on bad input
        result = verify_agent_identity_impl(
            agent_identifier=12345  # wrong type
        )
        assert "error" in result or "warnings" in result


class TestIdentityInRiskScoring:
    """Tests for identity score integration in generate_risk_score_impl"""

    @pytest.mark.unit
    def test_agent_with_identity_gets_identity_score(self):
        """Agent traffic with identifier includes identity score"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2026-02-21T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "test-identity-agent"},
        )
        assert result.get("traffic_source") == "agent"
        assert "identity" in result.get("component_scores", {})

    @pytest.mark.unit
    def test_agent_without_identity_no_identity_score(self):
        """Agent traffic without identifier has no identity component"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2026-02-21T12:00:00", "payment_method": "credit_card",
             "is_agent": True},
        )
        assert result.get("traffic_source") == "agent"
        # No agent_identifier, so identity verification is skipped
        assert "identity" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_human_traffic_no_identity_check(self):
        """Human traffic doesn't get identity verification"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2026-02-21T12:00:00", "payment_method": "credit_card",
             "is_agent": False},
        )
        assert "identity" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_verified_agent_reduces_risk(self):
        """Verified agent with high trust gets lower risk score"""
        from server import generate_risk_score_impl, agent_registry
        agent_registry.register("trusted-risk-agent", agent_type="stripe_acp")
        agent_registry.update_trust("trusted-risk-agent", 0.9)
        for _ in range(10):
            agent_registry.record_transaction("trusted-risk-agent")
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2026-02-21T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "trusted-risk-agent"},
        )
        assert "identity" in result.get("component_scores", {})
        # Identity score should reflect high trust
        identity_score = result["component_scores"]["identity"]
        assert identity_score <= 0.5  # Low risk for trusted agent

    @pytest.mark.unit
    def test_unverified_agent_increases_risk(self):
        """Unverified agent gets higher risk from identity component"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2026-02-21T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "totally-unknown-agent-xyz"},
        )
        assert "identity" in result.get("component_scores", {})
        identity_score = result["component_scores"]["identity"]
        assert identity_score >= 0.5  # High risk for unverified agent
