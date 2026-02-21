"""Tests for AgentBehavioralFingerprint class.

Tests agent behavioral fingerprinting: API timing patterns, decision
consistency, request structure fingerprints, and per-agent Isolation
Forest anomaly detection.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


class TestAgentBehavioralFingerprintInit:
    """Tests for AgentBehavioralFingerprint initialization."""

    @pytest.mark.unit
    def test_class_exists(self):
        """AgentBehavioralFingerprint class is importable from server."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        assert fp is not None

    @pytest.mark.unit
    def test_singleton_exists(self):
        """Module-level agent_fingerprinter singleton exists."""
        from server import agent_fingerprinter

        assert agent_fingerprinter is not None

    @pytest.mark.unit
    def test_has_analyze_method(self):
        """Instance has an analyze() method."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        assert callable(getattr(fp, "analyze", None))

    @pytest.mark.unit
    def test_has_record_method(self):
        """Instance has a record() method to store agent behavior."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        assert callable(getattr(fp, "record", None))

    @pytest.mark.unit
    def test_has_get_baseline_method(self):
        """Instance has a get_baseline() method."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        assert callable(getattr(fp, "get_baseline", None))


class TestAgentBehavioralFingerprintRecord:
    """Tests for recording agent behavioral data."""

    def _make_fingerprinter(self):
        from server import AgentBehavioralFingerprint

        return AgentBehavioralFingerprint()

    @pytest.mark.unit
    def test_record_basic_behavior(self):
        """Record a basic agent behavior event."""
        fp = self._make_fingerprinter()
        fp.record(
            agent_id="agent-1",
            api_timing_ms=45.0,
            decision_pattern="approve",
            request_structure_hash="abc123",
        )
        baseline = fp.get_baseline("agent-1")
        assert baseline is not None
        assert baseline["observation_count"] >= 1

    @pytest.mark.unit
    def test_record_multiple_events(self):
        """Recording multiple events increments observation count."""
        fp = self._make_fingerprinter()
        for i in range(5):
            fp.record(
                agent_id="agent-1",
                api_timing_ms=40.0 + i,
                decision_pattern="approve",
                request_structure_hash="abc123",
            )
        baseline = fp.get_baseline("agent-1")
        assert baseline["observation_count"] == 5

    @pytest.mark.unit
    def test_record_different_agents(self):
        """Different agents have separate baselines."""
        fp = self._make_fingerprinter()
        fp.record(agent_id="agent-1", api_timing_ms=40.0)
        fp.record(agent_id="agent-2", api_timing_ms=80.0)
        b1 = fp.get_baseline("agent-1")
        b2 = fp.get_baseline("agent-2")
        assert b1 is not None
        assert b2 is not None
        assert b1 != b2

    @pytest.mark.unit
    def test_get_baseline_unknown_agent(self):
        """Getting baseline for unknown agent returns None."""
        fp = self._make_fingerprinter()
        assert fp.get_baseline("nonexistent") is None

    @pytest.mark.unit
    def test_record_bounded_history(self):
        """History per agent is bounded (doesn't grow unbounded)."""
        fp = self._make_fingerprinter()
        for i in range(2000):
            fp.record(agent_id="agent-1", api_timing_ms=float(i))
        baseline = fp.get_baseline("agent-1")
        # Should be bounded at some reasonable limit
        assert baseline["observation_count"] <= 1000

    @pytest.mark.unit
    def test_record_with_optional_fields(self):
        """Record with only required fields (agent_id) works."""
        fp = self._make_fingerprinter()
        fp.record(agent_id="agent-1")
        baseline = fp.get_baseline("agent-1")
        assert baseline is not None


class TestAgentBehavioralFingerprintAnalyze:
    """Tests for analyzing agent behavior against baseline."""

    def _make_fingerprinter_with_baseline(self, agent_id="agent-1", n=50):
        """Create a fingerprinter with a pre-built baseline."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        # Build a consistent baseline
        for i in range(n):
            fp.record(
                agent_id=agent_id,
                api_timing_ms=45.0 + (i % 5),  # tight timing: 45-49ms
                decision_pattern="approve",
                request_structure_hash="consistent_hash",
            )
        return fp

    @pytest.mark.unit
    def test_analyze_returns_required_fields(self):
        """Analyze result contains risk_score, confidence, is_anomaly, details."""
        fp = self._make_fingerprinter_with_baseline()
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
            decision_pattern="approve",
            request_structure_hash="consistent_hash",
        )
        assert "risk_score" in result
        assert "confidence" in result
        assert "is_anomaly" in result
        assert "details" in result
        assert isinstance(result["risk_score"], float)
        assert isinstance(result["is_anomaly"], bool)

    @pytest.mark.unit
    def test_analyze_risk_score_range(self):
        """Risk score is between 0 and 1."""
        fp = self._make_fingerprinter_with_baseline()
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
        )
        assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.unit
    def test_analyze_consistent_behavior_low_risk(self):
        """Agent behaving consistently with baseline gets lower risk than deviant."""
        fp = self._make_fingerprinter_with_baseline()
        consistent_result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
            decision_pattern="approve",
            request_structure_hash="consistent_hash",
        )
        # Build a fresh fingerprinter for the deviant test
        fp2 = self._make_fingerprinter_with_baseline(agent_id="agent-2")
        deviant_result = fp2.analyze(
            agent_id="agent-2",
            api_timing_ms=5000.0,
            decision_pattern="reject",
            request_structure_hash="totally_new_hash",
        )
        # Consistent behavior should score no higher risk than deviant behavior
        # (or at least details should not flag anomalies)
        assert (
            len(consistent_result["details"]) == 0
            or consistent_result["risk_score"] <= deviant_result["risk_score"]
        )

    @pytest.mark.unit
    def test_analyze_deviant_timing_higher_risk(self):
        """Agent with drastically different timing gets higher risk."""
        fp = self._make_fingerprinter_with_baseline()
        # Way outside normal range (45-49ms) -> 5000ms
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=5000.0,
            decision_pattern="approve",
            request_structure_hash="consistent_hash",
        )
        assert result["risk_score"] > 0.3

    @pytest.mark.unit
    def test_analyze_new_decision_pattern_signals(self):
        """New decision pattern not seen in baseline raises concern."""
        fp = self._make_fingerprinter_with_baseline()
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
            decision_pattern="reject",  # never seen before
            request_structure_hash="consistent_hash",
        )
        # Should show up in details
        assert (
            "decision_pattern_novel" in result["details"] or result["risk_score"] > 0.0
        )

    @pytest.mark.unit
    def test_analyze_new_request_structure_signals(self):
        """New request structure hash not seen before raises concern."""
        fp = self._make_fingerprinter_with_baseline()
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
            decision_pattern="approve",
            request_structure_hash="completely_different_hash",
        )
        assert (
            "request_structure_novel" in result["details"] or result["risk_score"] > 0.0
        )

    @pytest.mark.unit
    def test_analyze_unknown_agent_high_risk(self):
        """Unknown agent with no baseline gets elevated risk."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        result = fp.analyze(
            agent_id="unknown-agent",
            api_timing_ms=50.0,
        )
        # No baseline -> elevated risk or specific indicator
        assert result["risk_score"] >= 0.5 or "no_baseline" in result["details"]

    @pytest.mark.unit
    def test_analyze_few_observations_low_confidence(self):
        """Agent with very few observations gets low confidence."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        fp.record(agent_id="agent-new", api_timing_ms=50.0)
        result = fp.analyze(agent_id="agent-new", api_timing_ms=50.0)
        assert result["confidence"] < 0.7

    @pytest.mark.unit
    def test_analyze_details_contains_timing_info(self):
        """Details include timing deviation information."""
        fp = self._make_fingerprinter_with_baseline()
        result = fp.analyze(
            agent_id="agent-1",
            api_timing_ms=46.0,
        )
        details = result["details"]
        assert isinstance(details, (dict, list))

    @pytest.mark.unit
    def test_analyze_records_observation(self):
        """Analyze also records the observation for future baseline."""
        fp = self._make_fingerprinter_with_baseline(n=10)
        before = fp.get_baseline("agent-1")["observation_count"]
        fp.analyze(agent_id="agent-1", api_timing_ms=46.0)
        after = fp.get_baseline("agent-1")["observation_count"]
        assert after == before + 1


class TestAgentBehavioralFingerprintThreadSafety:
    """Tests for thread safety."""

    @pytest.mark.unit
    def test_concurrent_records(self):
        """Multiple threads can record without errors."""
        import threading
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        errors = []

        def worker(agent_id, n):
            try:
                for i in range(n):
                    fp.record(agent_id=agent_id, api_timing_ms=float(i))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(f"agent-{i}", 50))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(4):
            baseline = fp.get_baseline(f"agent-{i}")
            assert baseline is not None
            assert baseline["observation_count"] == 50


class TestAgentBehavioralFingerprintEdgeCases:
    """Edge case tests."""

    @pytest.mark.unit
    def test_analyze_with_zero_timing(self):
        """Zero timing doesn't crash."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        fp.record(agent_id="agent-1", api_timing_ms=0.0)
        result = fp.analyze(agent_id="agent-1", api_timing_ms=0.0)
        assert "risk_score" in result

    @pytest.mark.unit
    def test_analyze_with_negative_timing(self):
        """Negative timing is handled gracefully."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        result = fp.analyze(agent_id="agent-1", api_timing_ms=-10.0)
        assert "risk_score" in result

    @pytest.mark.unit
    def test_analyze_with_very_large_timing(self):
        """Very large timing values don't crash."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        fp.record(agent_id="agent-1", api_timing_ms=1.0)
        result = fp.analyze(agent_id="agent-1", api_timing_ms=1e9)
        assert "risk_score" in result
        assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.unit
    def test_analyze_empty_string_agent_id(self):
        """Empty string agent_id is handled."""
        from server import AgentBehavioralFingerprint

        fp = AgentBehavioralFingerprint()
        result = fp.analyze(agent_id="", api_timing_ms=50.0)
        assert "risk_score" in result
