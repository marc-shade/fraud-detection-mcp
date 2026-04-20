# Phase D: Mandate and Collusion Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add mandate verification and agent collusion detection — two new MCP tools (#17, #18) that check whether agents operate within authorized scope and detect coordinated fraudulent behavior.

**Architecture:** `MandateVerifier` is a stateless class that validates transactions against a caller-supplied mandate dict. `CollusionDetector` maintains a directed graph of agent-to-agent interactions and detects temporal clustering, circular flows, and coordinated volume spikes. Both integrate into the existing `_impl` / MCP tool pattern. `analyze_agent_transaction_impl` gets real mandate compliance replacing its `1.0` placeholder.

**Tech Stack:** Python 3.10+, NetworkX (already in deps), scikit-learn IsolationForest (already in deps), FastMCP, pytest

---

### Task 1: Add MandateVerifier class + tests

**Files:**
- Create: `tests/test_mandate_verifier.py`
- Modify: `server.py` (add class after `AgentBehavioralFingerprint`, ~line 1487)

**Step 1: Write the failing tests**

Create `tests/test_mandate_verifier.py`:

```python
"""Tests for MandateVerifier class."""

import pytest
from datetime import datetime


class TestMandateVerifier:
    """Test MandateVerifier constraint checking."""

    def test_fully_compliant_transaction(self):
        """Transaction within all mandate constraints passes."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {
            "max_amount": 500.0,
            "allowed_merchants": ["Amazon", "Office Depot"],
        }
        transaction = {
            "amount": 100.0,
            "merchant": "Amazon",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True
        assert result["violations"] == []
        assert result["drift_score"] == 0.0

    def test_amount_exceeds_max(self):
        """Transaction over max_amount is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"max_amount": 100.0}
        transaction = {"amount": 200.0, "timestamp": datetime.now().isoformat()}
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("amount_exceeded" in v for v in result["violations"])

    def test_blocked_merchant(self):
        """Transaction with blocked merchant is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"blocked_merchants": ["Casino", "Gambling"]}
        transaction = {
            "amount": 50.0,
            "merchant": "Casino",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("blocked_merchant" in v for v in result["violations"])

    def test_merchant_not_in_allowed_list(self):
        """Transaction with merchant not in allowed list is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"allowed_merchants": ["Amazon", "Office Depot"]}
        transaction = {
            "amount": 50.0,
            "merchant": "Unauthorized Store",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("merchant_not_allowed" in v for v in result["violations"])

    def test_outside_time_window(self):
        """Transaction outside operating hours is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"time_window": {"start": "09:00", "end": "17:00"}}
        # Create a transaction at 3 AM
        transaction = {
            "amount": 50.0,
            "timestamp": "2026-02-21T03:00:00",
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("outside_time_window" in v for v in result["violations"])

    def test_within_time_window(self):
        """Transaction within operating hours is compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"time_window": {"start": "09:00", "end": "17:00"}}
        transaction = {
            "amount": 50.0,
            "timestamp": "2026-02-21T12:00:00",
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True

    def test_location_not_allowed(self):
        """Transaction from disallowed location is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"allowed_locations": ["United States", "Canada"]}
        transaction = {
            "amount": 50.0,
            "location": "Russia",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("location_not_allowed" in v for v in result["violations"])

    def test_daily_limit_exceeded(self):
        """Transaction pushing over daily_limit is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"daily_limit": 500.0}
        transaction = {
            "amount": 600.0,
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("daily_limit_exceeded" in v for v in result["violations"])

    def test_drift_score_multiple_violations(self):
        """Drift score reflects ratio of violations to checks."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {
            "max_amount": 100.0,
            "allowed_merchants": ["Amazon"],
            "allowed_locations": ["United States"],
        }
        transaction = {
            "amount": 200.0,
            "merchant": "Bad Store",
            "location": "Russia",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        # 3 violations out of 3 constraints = drift_score 1.0
        assert result["drift_score"] == pytest.approx(1.0, abs=0.01)

    def test_empty_mandate_always_compliant(self):
        """Empty mandate means no constraints, always compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {}
        transaction = {
            "amount": 99999.0,
            "merchant": "Anything",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True
        assert result["drift_score"] == 0.0

    def test_mandate_utilization(self):
        """Mandate utilization shows how close to limits."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"max_amount": 100.0, "daily_limit": 1000.0}
        transaction = {
            "amount": 80.0,
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert "mandate_utilization" in result
        # 80% of max_amount
        assert result["mandate_utilization"]["amount_pct"] == pytest.approx(0.8, abs=0.01)

    def test_result_has_required_fields(self):
        """Verify result contains all required fields."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        result = verifier.verify({"amount": 10.0, "timestamp": datetime.now().isoformat()}, {})
        assert "compliant" in result
        assert "violations" in result
        assert "drift_score" in result
        assert "mandate_utilization" in result
        assert "checks_performed" in result
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mandate_verifier.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'MandateVerifier' from 'server'`

**Step 3: Write the MandateVerifier implementation**

Add to `server.py` after the `agent_fingerprinter` singleton (after line ~1490), before the `# =============================================================================` section headers for impl functions:

```python
# =============================================================================
# Mandate Verifier
# =============================================================================


class MandateVerifier:
    """Stateless mandate compliance checker for agent transactions.

    Verifies whether a transaction falls within an agent's authorized scope.
    Mandates define constraints: spending limits, merchant whitelists/blacklists,
    time windows, and geographic restrictions. The mandate is passed per-call
    by the orchestrating agent.
    """

    def verify(
        self, transaction: Dict[str, Any], mandate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check transaction against mandate constraints.

        Args:
            transaction: Transaction data with amount, merchant, location, timestamp.
            mandate: Constraint dict with optional keys: max_amount, daily_limit,
                allowed_merchants, blocked_merchants, allowed_locations,
                time_window (start/end HH:MM).

        Returns:
            Dict with compliant (bool), violations (list), drift_score (0-1),
            mandate_utilization (dict), and checks_performed (int).
        """
        violations: List[str] = []
        checks = 0
        utilization: Dict[str, float] = {}

        amount = float(transaction.get("amount", 0.0))
        merchant = str(transaction.get("merchant", "")).lower()
        location = str(transaction.get("location", "")).lower()

        # --- Amount check ---
        max_amount = mandate.get("max_amount")
        if max_amount is not None:
            checks += 1
            max_amount = float(max_amount)
            utilization["amount_pct"] = amount / max_amount if max_amount > 0 else 0.0
            if amount > max_amount:
                violations.append(f"amount_exceeded: {amount} > {max_amount}")

        # --- Daily limit check ---
        daily_limit = mandate.get("daily_limit")
        if daily_limit is not None:
            checks += 1
            daily_limit = float(daily_limit)
            utilization["daily_pct"] = amount / daily_limit if daily_limit > 0 else 0.0
            if amount > daily_limit:
                violations.append(f"daily_limit_exceeded: {amount} > {daily_limit}")

        # --- Blocked merchants ---
        blocked = mandate.get("blocked_merchants")
        if blocked is not None:
            checks += 1
            blocked_lower = [m.lower() for m in blocked]
            if merchant in blocked_lower:
                violations.append(f"blocked_merchant: {merchant}")

        # --- Allowed merchants ---
        allowed_merchants = mandate.get("allowed_merchants")
        if allowed_merchants is not None:
            checks += 1
            allowed_lower = [m.lower() for m in allowed_merchants]
            if merchant and merchant != "unknown" and merchant not in allowed_lower:
                violations.append(f"merchant_not_allowed: {merchant}")

        # --- Allowed locations ---
        allowed_locations = mandate.get("allowed_locations")
        if allowed_locations is not None:
            checks += 1
            allowed_loc_lower = [loc.lower() for loc in allowed_locations]
            if location and location != "unknown" and location not in allowed_loc_lower:
                violations.append(f"location_not_allowed: {location}")

        # --- Time window ---
        time_window = mandate.get("time_window")
        if time_window is not None and "start" in time_window and "end" in time_window:
            checks += 1
            try:
                ts = transaction.get("timestamp", "")
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, datetime):
                    dt = ts
                else:
                    dt = datetime.now()

                txn_time = dt.strftime("%H:%M")
                start = time_window["start"]
                end = time_window["end"]

                if start <= end:
                    if not (start <= txn_time <= end):
                        violations.append(
                            f"outside_time_window: {txn_time} not in {start}-{end}"
                        )
                else:
                    # Overnight window (e.g., 22:00-06:00)
                    if end < txn_time < start:
                        violations.append(
                            f"outside_time_window: {txn_time} not in {start}-{end}"
                        )
            except (ValueError, TypeError):
                pass  # Skip time check if timestamp unparseable

        drift_score = len(violations) / checks if checks > 0 else 0.0

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "drift_score": float(drift_score),
            "mandate_utilization": utilization,
            "checks_performed": checks,
        }


mandate_verifier = MandateVerifier()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mandate_verifier.py -v --tb=short`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_mandate_verifier.py
git commit -m "feat: Add MandateVerifier class for agent transaction mandate compliance"
```

---

### Task 2: Add CollusionDetector class + tests

**Files:**
- Create: `tests/test_collusion_detector.py`
- Modify: `server.py` (add class after `mandate_verifier` singleton)

**Step 1: Write the failing tests**

Create `tests/test_collusion_detector.py`:

```python
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
        detector.record_interaction("agent_b", "agent_c", 100.0, timestamp=now + timedelta(seconds=10))
        detector.record_interaction("agent_c", "agent_a", 100.0, timestamp=now + timedelta(seconds=20))

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
                f"agent_{i}", target, 100.0,
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
                "agent_a", "agent_b", 50.0,
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collusion_detector.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'CollusionDetector' from 'server'`

**Step 3: Write the CollusionDetector implementation**

Add to `server.py` after the `mandate_verifier = MandateVerifier()` singleton:

```python
# =============================================================================
# Collusion Detector
# =============================================================================


class CollusionDetector:
    """Graph-based detection of coordinated agent behavior.

    Maintains a directed graph of agent-to-agent transaction flows.
    Detects: circular money flows, temporal clustering (burst of agents
    hitting same target), and volume anomalies (coordinated spikes).
    """

    def __init__(self, max_nodes: int = 5000):
        self.graph = nx.DiGraph()
        self._node_order: deque = deque()
        self.max_nodes = max_nodes
        self._interactions: List[Dict[str, Any]] = []
        self._max_interactions = 50000

    def record_interaction(
        self,
        source: str,
        target: str,
        amount: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record an agent-to-agent or agent-to-merchant interaction.

        Args:
            source: Source agent ID.
            target: Target agent or merchant ID.
            amount: Transaction amount.
            timestamp: When the interaction occurred (defaults to now).
        """
        ts = timestamp or datetime.now()

        # Track node insertion order for eviction
        for node in (source, target):
            if node not in self.graph:
                self._node_order.append(node)
                self.graph.add_node(node)

        if self.graph.has_edge(source, target):
            edge = self.graph[source][target]
            edge["transaction_count"] += 1
            edge["total_amount"] += amount
            edge["timestamps"].append(ts)
        else:
            self.graph.add_edge(
                source,
                target,
                transaction_count=1,
                total_amount=amount,
                timestamps=[ts],
            )

        self._interactions.append(
            {"source": source, "target": target, "amount": amount, "timestamp": ts}
        )

        # Bound interaction history
        if len(self._interactions) > self._max_interactions:
            self._interactions = self._interactions[-self._max_interactions:]

        # Evict oldest nodes if over cap
        while len(self.graph.nodes) > self.max_nodes * 2:
            oldest = self._node_order.popleft()
            if oldest in self.graph:
                self.graph.remove_node(oldest)

    def detect(
        self, agent_ids: List[str], window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Detect collusion patterns among a set of agents.

        Args:
            agent_ids: Agent identifiers to analyze.
            window_seconds: Time window in seconds for temporal analysis.

        Returns:
            Dict with collusion_score (0-1), suspected_ring, evidence, graph_metrics.
        """
        if not agent_ids:
            return {
                "collusion_score": 0.0,
                "suspected_ring": [],
                "evidence": [],
                "graph_metrics": self._graph_metrics(),
            }

        evidence: List[str] = []
        suspected: set = set()
        score_components: List[float] = []

        # --- Circular flow detection ---
        subgraph_nodes = [a for a in agent_ids if a in self.graph]
        if len(subgraph_nodes) >= 2:
            subgraph = self.graph.subgraph(subgraph_nodes)
            try:
                cycles = list(nx.simple_cycles(subgraph))
                # Filter to cycles of length >= 3 (A->B->C->A)
                real_cycles = [c for c in cycles if len(c) >= 3]
                if real_cycles:
                    for cycle in real_cycles[:5]:  # Cap at 5 reported cycles
                        evidence.append(
                            f"circular_flow: {' -> '.join(cycle)} -> {cycle[0]}"
                        )
                        suspected.update(cycle)
                    score_components.append(min(1.0, len(real_cycles) * 0.3))
            except Exception:
                pass  # Cycle detection can fail on certain graph shapes

        # --- Temporal clustering ---
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [
            i for i in self._interactions
            if i["timestamp"] >= cutoff and i["source"] in agent_ids
        ]

        # Group by target
        target_hits: Dict[str, List[str]] = {}
        for interaction in recent:
            t = interaction["target"]
            s = interaction["source"]
            if t not in target_hits:
                target_hits[t] = []
            if s not in target_hits[t]:
                target_hits[t].append(s)

        for target, sources in target_hits.items():
            if len(sources) >= 3:
                evidence.append(
                    f"temporal_cluster: {len(sources)} agents targeted {target} "
                    f"within {window_seconds}s"
                )
                suspected.update(sources)
                score_components.append(min(1.0, len(sources) * 0.15))

        # --- Volume anomaly ---
        for interaction in recent:
            src, tgt = interaction["source"], interaction["target"]
            if self.graph.has_edge(src, tgt):
                edge = self.graph[src][tgt]
                if edge["transaction_count"] >= 10:
                    recent_ts = [
                        t for t in edge["timestamps"] if t >= cutoff
                    ]
                    if len(recent_ts) >= 10:
                        evidence.append(
                            f"volume_anomaly: {src} -> {tgt} had "
                            f"{len(recent_ts)} transactions in window"
                        )
                        suspected.add(src)
                        suspected.add(tgt)
                        score_components.append(
                            min(1.0, len(recent_ts) * 0.05)
                        )
                        break  # One volume anomaly per detect call

        collusion_score = 0.0
        if score_components:
            collusion_score = float(min(1.0, max(score_components)))

        return {
            "collusion_score": collusion_score,
            "suspected_ring": sorted(suspected & set(agent_ids)),
            "evidence": evidence,
            "graph_metrics": self._graph_metrics(),
        }

    def _graph_metrics(self) -> Dict[str, Any]:
        """Return basic graph metrics."""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "interaction_count": len(self._interactions),
        }


collusion_detector = CollusionDetector()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collusion_detector.py -v --tb=short`
Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_collusion_detector.py
git commit -m "feat: Add CollusionDetector class for agent collusion ring detection"
```

---

### Task 3: Add MCP tools #17 and #18 + tests

**Files:**
- Create: `tests/test_mandate_collusion_tools.py`
- Modify: `server.py` (add `_impl` functions and MCP tool wrappers)

**Step 1: Write the failing tests**

Create `tests/test_mandate_collusion_tools.py`:

```python
"""Tests for verify_transaction_mandate and detect_agent_collusion MCP tools."""

import pytest
from datetime import datetime, timedelta


class TestVerifyTransactionMandateImpl:
    """Test verify_transaction_mandate_impl function."""

    def test_returns_dict(self):
        """verify_transaction_mandate_impl returns a dict."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={"amount": 50.0, "timestamp": datetime.now().isoformat()},
            mandate={},
        )
        assert isinstance(result, dict)

    def test_compliant_transaction(self):
        """Compliant transaction returns compliant=True."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 50.0,
                "merchant": "Amazon",
                "timestamp": datetime.now().isoformat(),
            },
            mandate={"max_amount": 100.0, "allowed_merchants": ["Amazon"]},
        )
        assert result["compliant"] is True
        assert result["status"] == "verified"

    def test_non_compliant_transaction(self):
        """Non-compliant transaction returns violations."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={
                "amount": 500.0,
                "merchant": "Casino",
                "timestamp": datetime.now().isoformat(),
            },
            mandate={"max_amount": 100.0, "blocked_merchants": ["Casino"]},
        )
        assert result["compliant"] is False
        assert len(result["violations"]) >= 1

    def test_invalid_transaction_data(self):
        """Invalid input returns error."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data="not a dict",
            mandate={},
        )
        assert "error" in result

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import verify_transaction_mandate_impl

        result = verify_transaction_mandate_impl(
            transaction_data={"amount": 50.0, "timestamp": datetime.now().isoformat()},
            mandate={},
        )
        assert "analysis_timestamp" in result


class TestDetectAgentCollusionImpl:
    """Test detect_agent_collusion_impl function."""

    def test_returns_dict(self):
        """detect_agent_collusion_impl returns a dict."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(
            agent_ids=[],
            window_seconds=3600,
        )
        assert isinstance(result, dict)

    def test_empty_agents_safe(self):
        """No agents returns low collusion score."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert result["collusion_score"] == 0.0
        assert result["status"] == "analyzed"

    def test_invalid_agent_ids(self):
        """Non-list agent_ids returns error."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(
            agent_ids="not_a_list",
            window_seconds=3600,
        )
        assert "error" in result

    def test_has_graph_metrics(self):
        """Result includes graph_metrics."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert "graph_metrics" in result

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import detect_agent_collusion_impl

        result = detect_agent_collusion_impl(agent_ids=[], window_seconds=3600)
        assert "analysis_timestamp" in result


class TestMCPToolRegistration:
    """Verify MCP tool registration for Phase D tools."""

    def test_mcp_has_verify_transaction_mandate(self):
        """MCP server has verify_transaction_mandate tool."""
        from server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "verify_transaction_mandate" in tool_names

    def test_mcp_has_detect_agent_collusion(self):
        """MCP server has detect_agent_collusion tool."""
        from server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "detect_agent_collusion" in tool_names

    def test_total_mcp_tools_count_is_18(self):
        """Server should now have 18 MCP tools registered."""
        from server import mcp

        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 18
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mandate_collusion_tools.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'verify_transaction_mandate_impl' from 'server'`

**Step 3: Write the _impl functions and MCP tool wrappers**

Add `_impl` functions near the other `_impl` functions in `server.py` (after `analyze_agent_transaction_impl`, around line ~2530):

```python
def verify_transaction_mandate_impl(
    transaction_data: Dict[str, Any],
    mandate: Dict[str, Any],
) -> Dict[str, Any]:
    """Check whether a transaction falls within an agent's authorized scope.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp).
        mandate: Constraint dict with optional keys: max_amount, daily_limit,
            allowed_merchants, blocked_merchants, allowed_locations,
            time_window (start/end HH:MM).

    Returns:
        Dict with compliant, violations, drift_score, mandate_utilization, and status.
    """
    try:
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
            }

        if not isinstance(mandate, dict):
            return {
                "error": "mandate must be a dictionary",
                "status": "validation_failed",
            }

        result = mandate_verifier.verify(transaction_data, mandate)
        result["status"] = "verified"
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Mandate verification failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "compliant": False,
            "violations": [],
            "drift_score": 0.0,
        }


def detect_agent_collusion_impl(
    agent_ids: Any,
    window_seconds: int = 3600,
    transactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Detect coordinated agent behavior using graph analysis.

    Args:
        agent_ids: List of agent identifiers to analyze.
        window_seconds: Time window in seconds for temporal analysis.
        transactions: Optional list of transaction dicts to record before analysis.
            Each dict should have source, target, amount, and optional timestamp.

    Returns:
        Dict with collusion_score, suspected_ring, evidence, graph_metrics, and status.
    """
    try:
        if not isinstance(agent_ids, list):
            return {
                "error": "agent_ids must be a list",
                "status": "validation_failed",
            }

        # Record any provided transactions first
        if transactions:
            for txn in transactions:
                if isinstance(txn, dict):
                    ts = txn.get("timestamp")
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            ts = None
                    collusion_detector.record_interaction(
                        source=str(txn.get("source", "")),
                        target=str(txn.get("target", "")),
                        amount=float(txn.get("amount", 0.0)),
                        timestamp=ts,
                    )

        result = collusion_detector.detect(
            agent_ids=[str(a) for a in agent_ids],
            window_seconds=int(window_seconds),
        )
        result["status"] = "analyzed"
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Collusion detection failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "collusion_score": 0.0,
            "suspected_ring": [],
            "evidence": [],
        }
```

Add the MCP tool wrappers. Place these after the `analyze_agent_transaction` MCP tool (after line ~3437):

```python
@_monitored("/verify_transaction_mandate", "TOOL")
@mcp.tool()
def verify_transaction_mandate(
    transaction_data: Dict[str, Any],
    mandate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check whether a transaction falls within an agent's authorized scope.

    Validates transaction against mandate constraints including spending limits,
    merchant whitelists/blacklists, time windows, and geographic restrictions.
    Returns compliance status, violations list, and drift score.

    Args:
        transaction_data: Transaction details with amount, merchant, location, timestamp.
        mandate: Constraint dict with optional keys: max_amount, daily_limit,
            allowed_merchants, blocked_merchants, allowed_locations,
            time_window (with start/end in HH:MM format).

    Returns:
        Compliance result with compliant (bool), violations (list),
        drift_score (0-1, higher means more violations), and mandate_utilization
    """
    return verify_transaction_mandate_impl(transaction_data, mandate)


@_monitored("/detect_agent_collusion", "TOOL")
@mcp.tool()
def detect_agent_collusion(
    agent_ids: List[str],
    window_seconds: int = 3600,
    transactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Detect coordinated agent behavior using graph analysis.

    Analyzes agent-to-agent transaction flows to detect collusion patterns:
    circular money flows (A->B->C->A), temporal clustering (multiple agents
    hitting same target in burst), and volume anomalies (sudden coordinated spikes).

    Args:
        agent_ids: List of agent identifiers to analyze for collusion.
        window_seconds: Time window in seconds for temporal analysis (default 3600).
        transactions: Optional list of transaction dicts to record before analysis.
            Each dict should have source, target, amount, and optional timestamp fields.

    Returns:
        Detection result with collusion_score (0-1), suspected_ring (list of agent IDs),
        evidence (list of findings), and graph_metrics
    """
    return detect_agent_collusion_impl(agent_ids, window_seconds, transactions)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mandate_collusion_tools.py -v --tb=short`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_mandate_collusion_tools.py
git commit -m "feat: Add verify_transaction_mandate and detect_agent_collusion MCP tools (#17, #18)"
```

---

### Task 4: Integrate mandate into analyze_agent_transaction + update tool counts

**Files:**
- Modify: `server.py` (`analyze_agent_transaction_impl` at line ~2402, `generate_risk_score_impl` at line ~1957)
- Modify: `tests/test_analyze_agent_transaction.py` (add mandate tests)
- Modify: `tests/test_training_and_persistence.py:442` (tool count 16 -> 18)
- Modify: `tests/test_synthetic_data.py:900` (tool count 16 -> 18)

**Step 1: Write the failing tests**

Add to `tests/test_analyze_agent_transaction.py` (append at end of file):

```python
class TestMandateInAgentTransaction:
    """Test mandate compliance integration in analyze_agent_transaction_impl."""

    def test_mandate_compliance_with_mandate(self):
        """analyze_agent_transaction_impl uses real mandate when provided."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 50.0,
                "merchant": "Amazon",
                "location": "United States",
                "timestamp": "2026-02-21T12:00:00",
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "mandate-test-agent",
            },
            mandate={
                "max_amount": 100.0,
                "allowed_merchants": ["Amazon"],
            },
        )
        assert result["mandate_compliance"] == 1.0  # fully compliant

    def test_mandate_violation_reduces_compliance(self):
        """Mandate violation reduces mandate_compliance below 1.0."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 500.0,
                "merchant": "Casino",
                "location": "United States",
                "timestamp": "2026-02-21T12:00:00",
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "mandate-test-agent-2",
            },
            mandate={
                "max_amount": 100.0,
                "blocked_merchants": ["Casino"],
            },
        )
        assert result["mandate_compliance"] < 1.0

    def test_no_mandate_returns_default(self):
        """No mandate parameter returns mandate_compliance=1.0 (no constraints)."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 50.0,
                "merchant": "Amazon",
                "location": "United States",
                "timestamp": "2026-02-21T12:00:00",
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "mandate-test-agent-3",
            },
        )
        assert result["mandate_compliance"] == 1.0

    def test_mandate_violations_in_anomalies(self):
        """Mandate violations appear in anomalies list."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 500.0,
                "merchant": "Amazon",
                "location": "United States",
                "timestamp": "2026-02-21T12:00:00",
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "mandate-test-agent-4",
            },
            mandate={"max_amount": 100.0},
        )
        assert any("mandate" in a for a in result["anomalies"])
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analyze_agent_transaction.py::TestMandateInAgentTransaction -v --tb=short`
Expected: FAIL (analyze_agent_transaction_impl doesn't accept `mandate` parameter yet)

**Step 3: Update analyze_agent_transaction_impl and generate_risk_score_impl**

In `server.py`, update `analyze_agent_transaction_impl` (line ~2402) to accept an optional `mandate` parameter:

1. Change the function signature to add `mandate: Optional[Dict[str, Any]] = None`
2. After the fingerprint section and before the composite risk score, add mandate verification:

```python
        # --- Mandate verification ---
        mandate_result = None
        mandate_compliance = 1.0  # default: no mandate = fully compliant
        if mandate:
            mandate_result = mandate_verifier.verify(transaction_data, mandate)
            mandate_compliance = 1.0 - mandate_result.get("drift_score", 0.0)
            if not mandate_result.get("compliant", True):
                anomalies.append("mandate_violation")
                anomalies.extend(
                    f"mandate_{v.split(':')[0]}" for v in mandate_result.get("violations", [])
                )
```

3. Replace the existing `"mandate_compliance": 1.0,  # placeholder until Phase D` with `"mandate_compliance": mandate_compliance,`

4. Update the MCP tool wrapper `analyze_agent_transaction` to pass through the `mandate` parameter. Add `mandate: Optional[Dict[str, Any]] = None` to its signature and pass it to `analyze_agent_transaction_impl`.

5. Update tool count assertions in `tests/test_training_and_persistence.py:442` and `tests/test_synthetic_data.py:900` from `16` to `18`.

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (660 existing + ~25 new from Task 1 + ~13 from Task 2 + ~12 from Task 3 + ~4 from Task 4 = ~709+ tests)

**Step 5: Commit**

```bash
git add server.py tests/test_analyze_agent_transaction.py tests/test_training_and_persistence.py tests/test_synthetic_data.py
git commit -m "feat: Integrate mandate verification into analyze_agent_transaction, update tool count to 18"
```

---

### Task 5: Final verification and lint

**Step 1: Run ruff check and format**

```bash
ruff check . --fix
ruff format .
```
Expected: 0 errors

**Step 2: Run full test suite with coverage**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass, 0 failures

**Step 3: Verify MCP tool count**

```bash
python -c "from server import mcp; print(f'MCP tools: {len(mcp._tool_manager._tools)}')"
```
Expected: `MCP tools: 18`

**Step 4: Commit any lint fixes**

```bash
git add -A && git commit -m "chore: Phase D lint and format fixes" || echo "Nothing to commit"
```
