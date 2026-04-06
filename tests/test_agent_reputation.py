"""Tests for AgentReputationScorer class."""

import pytest


class TestAgentReputationScorer:
    """Test AgentReputationScorer reputation computation."""

    def test_import(self):
        """AgentReputationScorer is importable."""
        from server import AgentReputationScorer

        scorer = AgentReputationScorer()
        assert scorer is not None

    def test_singleton_exists(self):
        """Module-level reputation_scorer singleton exists."""
        from server import reputation_scorer

        assert reputation_scorer is not None

    def test_unknown_agent_returns_low_reputation(self, tmp_path):
        """Unknown agent gets a low default reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry.json"
        )
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("nonexistent-agent")
        assert result["reputation_score"] < 0.5
        assert result["history_length"] == 0
        assert result["transaction_count"] == 0

    def test_registered_agent_gets_baseline_reputation(self, tmp_path):
        """Registered agent with default trust gets moderate reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry2.json"
        )
        registry.register("test-agent", agent_type="stripe_acp")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("test-agent")
        assert result["reputation_score"] > 0.0
        assert result["transaction_count"] == 0

    def test_high_trust_agent_gets_high_reputation(self, tmp_path):
        """Agent with high trust score gets higher reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry3.json"
        )
        registry.register("trusted-agent")
        registry.update_trust("trusted-agent", 0.95)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("trusted-agent")
        assert result["reputation_score"] > 0.3

    def test_transaction_history_boosts_reputation(self, tmp_path):
        """More transactions improve the history component."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry4.json"
        )
        registry.register("active-agent")
        for _ in range(50):
            registry.record_transaction("active-agent")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("active-agent")
        assert result["transaction_count"] == 50
        assert result["components"]["history_factor"] > 0.0

    def test_result_has_required_fields(self, tmp_path):
        """score() result contains all required fields."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry5.json"
        )
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("any-agent")
        assert "reputation_score" in result
        assert "history_length" in result
        assert "transaction_count" in result
        assert "trust_score" in result
        assert "behavioral_consistency" in result
        assert "components" in result

    def test_reputation_score_clamped_0_to_1(self, tmp_path):
        """Reputation score is always between 0 and 1."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry6.json"
        )
        registry.register("edge-agent")
        registry.update_trust("edge-agent", 1.0)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("edge-agent")
        assert 0.0 <= result["reputation_score"] <= 1.0

    def test_components_sum_reflects_score(self, tmp_path):
        """Component weights sum to produce the reputation score."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry7.json"
        )
        registry.register("check-agent")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("check-agent")
        components = result["components"]
        # Weighted sum should equal reputation_score
        expected = (
            components["trust_score"] * 0.4
            + components["history_factor"] * 0.25
            + components["behavioral_consistency"] * 0.25
            + components["collusion_safety"] * 0.1
        )
        assert result["reputation_score"] == pytest.approx(expected, abs=0.01)

    def test_behavioral_consistency_with_baseline(self, tmp_path):
        """Agent with behavioral baseline gets consistency score."""
        from server import (
            AgentReputationScorer,
            AgentIdentityRegistry,
            AgentBehavioralFingerprint,
        )

        registry = AgentIdentityRegistry(
            registry_path=tmp_path / "test_rep_registry8.json"
        )
        registry.register("fp-agent")
        fingerprinter = AgentBehavioralFingerprint()
        # Record enough observations to build baseline
        for i in range(15):
            fingerprinter.record(
                agent_id="fp-agent",
                api_timing_ms=100.0 + i,
                decision_pattern="purchase",
                request_structure_hash="abc123",
            )
        scorer = AgentReputationScorer(registry=registry, fingerprinter=fingerprinter)
        result = scorer.score("fp-agent")
        assert result["behavioral_consistency"] > 0.0
        assert result["components"]["behavioral_consistency"] > 0.0
