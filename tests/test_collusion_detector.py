"""Tests for CollusionDetector class."""

import pytest
from datetime import datetime, timedelta


class TestCollusionDetector:
    """Test CollusionDetector graph-based collusion detection."""

    def test_import(self):
        """CollusionDetector is importable."""
        from server import CollusionDetector

        detector = CollusionDetector()
        assert detector is not None

    def test_singleton_exists(self):
        """Module-level collusion_detector singleton exists."""
        from server import collusion_detector

        assert collusion_detector is not None

    def test_record_interaction(self):
        """record_interaction adds an edge to the graph."""
        from server import CollusionDetector

        detector = CollusionDetector()
        detector.record_interaction("agent_a", "agent_b", 100.0)
        assert detector.graph.has_edge("agent_a", "agent_b")

    def test_record_interaction_accumulates(self):
        """Multiple interactions accumulate on the same edge."""
        from server import CollusionDetector

        detector = CollusionDetector()
        detector.record_interaction("agent_a", "agent_b", 100.0)
        detector.record_interaction("agent_a", "agent_b", 200.0)
        edge = detector.graph["agent_a"]["agent_b"]
        assert edge["transaction_count"] == 2
        assert edge["total_amount"] == pytest.approx(300.0)

    def test_detect_no_agents_returns_safe(self):
        """detect() with no agents returns low collusion score."""
        from server import CollusionDetector

        detector = CollusionDetector()
        result = detector.detect([], window_seconds=3600)
        assert result["collusion_score"] == 0.0
        assert result["suspected_ring"] == []

    def test_detect_single_agent_returns_safe(self):
        """detect() with single agent returns low collusion score."""
        from server import CollusionDetector

        detector = CollusionDetector()
        detector.record_interaction("agent_a", "merchant_x", 100.0)
        result = detector.detect(["agent_a"], window_seconds=3600)
        assert result["collusion_score"] < 0.5

    def test_detect_circular_flow(self):
        """Circular flow A->B->C->A is detected."""
        from server import CollusionDetector

        detector = CollusionDetector()
        now = datetime.now()
        detector.record_interaction("agent_a", "agent_b", 100.0, timestamp=now)
        detector.record_interaction(
            "agent_b", "agent_c", 100.0, timestamp=now + timedelta(seconds=10)
        )
        detector.record_interaction(
            "agent_c", "agent_a", 100.0, timestamp=now + timedelta(seconds=20)
        )

        result = detector.detect(["agent_a", "agent_b", "agent_c"], window_seconds=3600)
        assert result["collusion_score"] > 0.0
        assert any("circular_flow" in e for e in result["evidence"])

    def test_detect_temporal_clustering(self):
        """Multiple agents transacting in burst are flagged."""
        from server import CollusionDetector

        detector = CollusionDetector()
        now = datetime.now()
        target = "merchant_x"
        # 5 agents all hit the same target within 10 seconds
        for i in range(5):
            detector.record_interaction(
                f"agent_{i}",
                target,
                100.0,
                timestamp=now + timedelta(seconds=i * 2),
            )

        agents = [f"agent_{i}" for i in range(5)]
        result = detector.detect(agents, window_seconds=60)
        assert result["collusion_score"] > 0.0
        assert any("temporal_cluster" in e for e in result["evidence"])

    def test_detect_volume_anomaly(self):
        """Sudden coordinated volume spike is detected."""
        from server import CollusionDetector

        detector = CollusionDetector()
        now = datetime.now()
        # Many transactions between two agents in short window
        for i in range(20):
            detector.record_interaction(
                "agent_a",
                "agent_b",
                50.0,
                timestamp=now + timedelta(seconds=i),
            )

        result = detector.detect(["agent_a", "agent_b"], window_seconds=60)
        assert result["collusion_score"] > 0.0
        assert any("volume_anomaly" in e for e in result["evidence"])

    def test_result_has_required_fields(self):
        """detect() result contains all required fields."""
        from server import CollusionDetector

        detector = CollusionDetector()
        result = detector.detect([], window_seconds=3600)
        assert "collusion_score" in result
        assert "suspected_ring" in result
        assert "evidence" in result
        assert "graph_metrics" in result

    def test_graph_bounded_memory(self):
        """Graph evicts old nodes when over MAX_NODES."""
        from server import CollusionDetector

        detector = CollusionDetector(max_nodes=10)
        for i in range(20):
            detector.record_interaction(f"src_{i}", f"dst_{i}", 10.0)
        assert len(detector.graph.nodes) <= 20  # max_nodes applies to src nodes

    def test_detect_returns_suspected_ring(self):
        """Agents in circular flow appear in suspected_ring."""
        from server import CollusionDetector

        detector = CollusionDetector()
        now = datetime.now()
        detector.record_interaction("a", "b", 100.0, timestamp=now)
        detector.record_interaction("b", "c", 100.0, timestamp=now)
        detector.record_interaction("c", "a", 100.0, timestamp=now)

        result = detector.detect(["a", "b", "c"], window_seconds=3600)
        ring = result["suspected_ring"]
        # All 3 should be in the suspected ring
        assert len(ring) >= 2

    def test_graph_metrics_included(self):
        """Graph metrics include node and edge counts."""
        from server import CollusionDetector

        detector = CollusionDetector()
        detector.record_interaction("a", "b", 100.0)
        result = detector.detect(["a", "b"], window_seconds=3600)
        metrics = result["graph_metrics"]
        assert "total_nodes" in metrics
        assert "total_edges" in metrics
        assert metrics["total_nodes"] >= 2
