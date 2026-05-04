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


class TestSuspectedRingIncludesAllInvolved:
    """Pre-2026-05-04 detect() returned suspected_ring as the queried
    subset — even if 5 outside agents formed a cycle, the result was
    suspected_ring=[] when the query didn't include them, despite
    collusion_score > 0. Now suspected_ring includes ALL involved
    agents and query_in_ring is the queried subset for back-compat.
    """

    def setup_method(self):
        from server import CollusionDetector
        self.det = CollusionDetector()

    def test_full_ring_returned_even_when_query_does_not_include_ring(self):
        from datetime import datetime
        # 3-cycle: A → B → C → A
        now = datetime.now()
        self.det.record_interaction("A", "B", 100, timestamp=now)
        self.det.record_interaction("B", "C", 100, timestamp=now)
        self.det.record_interaction("C", "A", 100, timestamp=now)

        # Query mentions only A — pre-fix would return suspected_ring=[]
        result = self.det.detect(["A", "B", "C"])
        assert "suspected_ring" in result
        assert "query_in_ring" in result  # back-compat field
        assert set(result["suspected_ring"]) == {"A", "B", "C"}

    def test_volume_anomaly_reports_all_distinct_pairs(self):
        """Pre-2026-05-04 the volume_anomaly check broke after the FIRST
        match. Now it reports every distinct (src, tgt) pair with a burst.
        """
        from datetime import datetime
        now = datetime.now()
        # Two distinct pairs each with a 12-burst
        for _ in range(12):
            self.det.record_interaction("A", "X", 100, timestamp=now)
        for _ in range(12):
            self.det.record_interaction("B", "Y", 100, timestamp=now)

        result = self.det.detect(["A", "B"])
        evidence_str = "\n".join(result["evidence"])
        assert "A -> X" in evidence_str, evidence_str
        assert "B -> Y" in evidence_str, evidence_str

    def test_volume_anomaly_deduplicates_repeated_interactions(self):
        """A single (src, tgt) pair appearing many times in `recent` must
        produce ONE volume_anomaly evidence line, not N copies."""
        from datetime import datetime
        now = datetime.now()
        for _ in range(15):
            self.det.record_interaction("A", "X", 100, timestamp=now)
        result = self.det.detect(["A"])
        volume_lines = [e for e in result["evidence"] if "volume_anomaly: A -> X" in e]
        assert len(volume_lines) == 1, (
            f"Expected exactly one volume_anomaly for A→X, got {len(volume_lines)}"
        )
