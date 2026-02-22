"""Tests for AgentReputationScorer class."""

import os
import uuid
import pytest
from pathlib import Path

# Use sandbox-writable temp directory with unique suffix per test run
_TMPDIR = Path(os.environ.get("TMPDIR", "/Volumes/SSDRAID0/tmp/claude/claude-501"))
_RUN_ID = uuid.uuid4().hex[:8]


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

    def test_unknown_agent_returns_low_reputation(self):
        """Unknown agent gets a low default reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry_{_RUN_ID}.json"
        )
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("nonexistent-agent")
        assert result["reputation_score"] < 0.5
        assert result["history_length"] == 0
        assert result["transaction_count"] == 0

    def test_registered_agent_gets_baseline_reputation(self):
        """Registered agent with default trust gets moderate reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry2_{_RUN_ID}.json"
        )
        registry.register("test-agent", agent_type="stripe_acp")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("test-agent")
        assert result["reputation_score"] > 0.0
        assert result["transaction_count"] == 0

    def test_high_trust_agent_gets_high_reputation(self):
        """Agent with high trust score gets higher reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry3_{_RUN_ID}.json"
        )
        registry.register("trusted-agent")
        registry.update_trust("trusted-agent", 0.95)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("trusted-agent")
        assert result["reputation_score"] > 0.3

    def test_transaction_history_boosts_reputation(self):
        """More transactions improve the history component."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry4_{_RUN_ID}.json"
        )
        registry.register("active-agent")
        for _ in range(50):
            registry.record_transaction("active-agent")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("active-agent")
        assert result["transaction_count"] == 50
        assert result["components"]["history_factor"] > 0.0

    def test_result_has_required_fields(self):
        """score() result contains all required fields."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry5_{_RUN_ID}.json"
        )
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("any-agent")
        assert "reputation_score" in result
        assert "history_length" in result
        assert "transaction_count" in result
        assert "trust_score" in result
        assert "behavioral_consistency" in result
        assert "components" in result

    def test_reputation_score_clamped_0_to_1(self):
        """Reputation score is always between 0 and 1."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry6_{_RUN_ID}.json"
        )
        registry.register("edge-agent")
        registry.update_trust("edge-agent", 1.0)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("edge-agent")
        assert 0.0 <= result["reputation_score"] <= 1.0

    def test_components_sum_reflects_score(self):
        """Component weights sum to produce the reputation score."""
        from server import AgentReputationScorer, AgentIdentityRegistry

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry7_{_RUN_ID}.json"
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

    def test_behavioral_consistency_with_baseline(self):
        """Agent with behavioral baseline gets consistency score."""
        from server import (
            AgentReputationScorer,
            AgentIdentityRegistry,
            AgentBehavioralFingerprint,
        )

        registry = AgentIdentityRegistry(
            registry_path=_TMPDIR / f"test_rep_registry8_{_RUN_ID}.json"
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
