# Phase C: Agent Behavioral Fingerprinting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `analyze_agent_transaction` MCP tool with agent behavioral fingerprinting that replaces human biometrics for agent traffic.

**Architecture:** Add `AgentBehavioralFingerprint` class to `server.py` that builds per-agent behavioral baselines using Isolation Forest. Tracks 3 signal categories: API timing patterns (inter-request intervals), decision consistency (amount/merchant pattern deviation), and request structure fingerprints (field presence/completeness). Expose as the 16th MCP tool. Integrate into `generate_risk_score_impl` so agent traffic uses behavioral fingerprinting instead of keystroke/mouse analysis.

**Tech Stack:** Python, FastMCP, scikit-learn (IsolationForest), numpy, collections.deque, pytest

---

### Task 1: Add AgentBehavioralFingerprint class

**Files:**
- Modify: `server.py` (add class after `AgentIdentityVerifier` and its singleton, before `_monitored`)

**Step 1: Write the test**

Create `tests/test_agent_behavioral.py`:

```python
"""Tests for AgentBehavioralFingerprint and analyze_agent_transaction MCP tool"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
import time
from server import AgentBehavioralFingerprint


class TestAgentBehavioralFingerprint:
    """Tests for the AgentBehavioralFingerprint class"""

    def setup_method(self):
        self.fingerprinter = AgentBehavioralFingerprint()

    @pytest.mark.unit
    def test_record_and_analyze_first_transaction(self):
        """First transaction for an agent establishes baseline"""
        result = self.fingerprinter.analyze(
            agent_id="agent-1",
            transaction_data={"amount": 100.0, "merchant": "Store A", "location": "NYC",
                              "payment_method": "credit_card", "timestamp": "2025-01-01T12:00:00"}
        )
        assert "risk_score" in result
        assert "confidence" in result
        assert "behavioral_consistency" in result
        assert result["risk_score"] >= 0.0
        assert result["confidence"] >= 0.0

    @pytest.mark.unit
    def test_consistent_behavior_low_risk(self):
        """Agent with consistent behavior gets low risk"""
        # Build baseline with consistent transactions
        for i in range(15):
            self.fingerprinter.analyze(
                agent_id="consistent-agent",
                transaction_data={"amount": 100.0 + i, "merchant": "Store A",
                                  "location": "NYC", "payment_method": "credit_card",
                                  "timestamp": f"2025-01-01T{12+i//60:02d}:{i%60:02d}:00"}
            )
        # Analyze a similar transaction
        result = self.fingerprinter.analyze(
            agent_id="consistent-agent",
            transaction_data={"amount": 105.0, "merchant": "Store A",
                              "location": "NYC", "payment_method": "credit_card",
                              "timestamp": "2025-01-01T13:00:00"}
        )
        # Consistent agent should have low risk
        assert result["risk_score"] <= 0.6

    @pytest.mark.unit
    def test_anomalous_behavior_higher_risk(self):
        """Agent deviating from baseline gets higher risk"""
        # Build baseline with small amounts
        for i in range(15):
            self.fingerprinter.analyze(
                agent_id="normal-agent",
                transaction_data={"amount": 50.0 + i, "merchant": "Store A",
                                  "location": "NYC", "payment_method": "credit_card",
                                  "timestamp": f"2025-01-01T12:{i:02d}:00"}
            )
        # Now send wildly different transaction
        result = self.fingerprinter.analyze(
            agent_id="normal-agent",
            transaction_data={"amount": 50000.0, "merchant": "Unknown Merchant",
                              "location": "Unknown", "payment_method": "crypto",
                              "timestamp": "2025-01-01T13:00:00"}
        )
        # Anomalous behavior should have higher risk than baseline
        assert result["risk_score"] > 0.0

    @pytest.mark.unit
    def test_unknown_agent_moderate_risk(self):
        """Unknown agent with no history gets moderate risk"""
        result = self.fingerprinter.analyze(
            agent_id="brand-new-agent",
            transaction_data={"amount": 100.0, "merchant": "Store", "location": "NYC",
                              "payment_method": "credit_card",
                              "timestamp": "2025-01-01T12:00:00"}
        )
        # New agent with no baseline
        assert 0.3 <= result["risk_score"] <= 0.7

    @pytest.mark.unit
    def test_result_structure(self):
        """Result contains all required fields"""
        result = self.fingerprinter.analyze(
            agent_id="test-agent",
            transaction_data={"amount": 100.0, "merchant": "Store",
                              "timestamp": "2025-01-01T12:00:00"}
        )
        assert "risk_score" in result
        assert "confidence" in result
        assert "behavioral_consistency" in result
        assert "is_anomaly" in result
        assert "features_analyzed" in result
        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 1.0
        assert isinstance(result["is_anomaly"], bool)

    @pytest.mark.unit
    def test_timing_pattern_tracked(self):
        """Inter-request timing is tracked per agent"""
        self.fingerprinter.analyze(
            agent_id="timing-agent",
            transaction_data={"amount": 100.0, "timestamp": "2025-01-01T12:00:00"}
        )
        time.sleep(0.01)  # Small delay
        self.fingerprinter.analyze(
            agent_id="timing-agent",
            transaction_data={"amount": 200.0, "timestamp": "2025-01-01T12:01:00"}
        )
        # Agent should have timing history
        assert "timing-agent" in self.fingerprinter._agent_histories

    @pytest.mark.unit
    def test_different_agents_separate_histories(self):
        """Each agent maintains its own behavioral history"""
        self.fingerprinter.analyze(
            agent_id="agent-A",
            transaction_data={"amount": 100.0, "timestamp": "2025-01-01T12:00:00"}
        )
        self.fingerprinter.analyze(
            agent_id="agent-B",
            transaction_data={"amount": 999.0, "timestamp": "2025-01-01T12:00:00"}
        )
        assert "agent-A" in self.fingerprinter._agent_histories
        assert "agent-B" in self.fingerprinter._agent_histories

    @pytest.mark.unit
    def test_history_capped(self):
        """Agent history doesn't grow unbounded"""
        for i in range(200):
            self.fingerprinter.analyze(
                agent_id="prolific-agent",
                transaction_data={"amount": float(i), "timestamp": f"2025-01-01T12:{i%60:02d}:00"}
            )
        history = self.fingerprinter._agent_histories["prolific-agent"]
        assert len(history["features"]) <= 100  # capped

    @pytest.mark.unit
    def test_empty_transaction_data(self):
        """Empty transaction data handled gracefully"""
        result = self.fingerprinter.analyze(agent_id="agent-1", transaction_data={})
        assert result["risk_score"] >= 0.0

    @pytest.mark.unit
    def test_missing_agent_id(self):
        """Missing agent_id handled gracefully"""
        result = self.fingerprinter.analyze(
            agent_id="",
            transaction_data={"amount": 100.0}
        )
        assert "risk_score" in result

    @pytest.mark.unit
    def test_request_structure_features(self):
        """Request structure (field completeness) is part of fingerprint"""
        # Full request
        result_full = self.fingerprinter.analyze(
            agent_id="struct-agent",
            transaction_data={"amount": 100.0, "merchant": "Store", "location": "NYC",
                              "payment_method": "credit_card", "timestamp": "2025-01-01T12:00:00",
                              "user_id": "user-1", "device_id": "device-1"}
        )
        assert result_full["features_analyzed"] > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_behavioral.py -v`
Expected: FAIL (AgentBehavioralFingerprint not defined)

**Step 3: Implement AgentBehavioralFingerprint**

In `server.py`, after the `agent_verifier` singleton (after `AgentIdentityVerifier` class and its singleton), add:

```python
# =============================================================================
# Agent Behavioral Fingerprinting
# =============================================================================

class AgentBehavioralFingerprint:
    """Behavioral fingerprinting for AI agent transactions.

    Replaces human BehavioralBiometrics for agent traffic. Tracks:
    - API timing patterns (inter-request intervals)
    - Decision consistency (amount/merchant deviation from baseline)
    - Request structure (field completeness and patterns)

    Builds per-agent baselines using Isolation Forest.
    """

    MAX_HISTORY = 100  # Max feature vectors per agent
    MIN_BASELINE = 10  # Min observations before anomaly detection activates

    # Fields used to compute request structure completeness
    STRUCTURE_FIELDS = [
        "amount", "merchant", "location", "payment_method", "timestamp",
        "user_id", "device_id", "ip_address", "merchant_category", "currency",
    ]

    def __init__(self):
        self._agent_histories: Dict[str, Dict[str, Any]] = {}
        self._agent_models: Dict[str, IsolationForest] = {}
        self._lock = threading.Lock()

    def _extract_features(self, transaction_data: Dict[str, Any],
                          timing_interval: Optional[float] = None) -> List[float]:
        """Extract behavioral features from a transaction.

        Features (8 total):
        0: log_amount
        1: payment_method_hash (0-1 normalized)
        2: merchant_hash (0-1 normalized)
        3: location_hash (0-1 normalized)
        4: hour_of_day (0-23 normalized to 0-1)
        5: field_completeness (0-1, fraction of STRUCTURE_FIELDS present)
        6: timing_interval (seconds since last request, log-scaled, 0 if first)
        7: amount_magnitude (order of magnitude: log10(amount+1))
        """
        features = []

        # Feature 0: log amount
        amount = float(transaction_data.get("amount", 0))
        features.append(math.log1p(max(0, amount)))

        # Feature 1: payment method hash
        pm = str(transaction_data.get("payment_method", ""))
        features.append((hash(pm) % 1000) / 1000.0)

        # Feature 2: merchant hash
        merchant = str(transaction_data.get("merchant", ""))
        features.append((hash(merchant) % 1000) / 1000.0)

        # Feature 3: location hash
        location = str(transaction_data.get("location", ""))
        features.append((hash(location) % 1000) / 1000.0)

        # Feature 4: hour of day
        ts = transaction_data.get("timestamp", "")
        hour = 12.0  # default
        if isinstance(ts, str) and len(ts) >= 13:
            try:
                hour = float(datetime.fromisoformat(ts.replace("Z", "+00:00")).hour)
            except (ValueError, AttributeError):
                pass
        features.append(hour / 24.0)

        # Feature 5: field completeness
        present = sum(1 for f in self.STRUCTURE_FIELDS if f in transaction_data)
        features.append(present / len(self.STRUCTURE_FIELDS))

        # Feature 6: timing interval (log-scaled)
        if timing_interval is not None and timing_interval > 0:
            features.append(math.log1p(timing_interval))
        else:
            features.append(0.0)

        # Feature 7: amount magnitude
        features.append(math.log10(amount + 1) if amount >= 0 else 0.0)

        return features

    def analyze(self, agent_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an agent transaction against its behavioral baseline.

        Args:
            agent_id: Agent identifier.
            transaction_data: Transaction data dict.

        Returns:
            Dict with risk_score, confidence, behavioral_consistency,
            is_anomaly, and features_analyzed.
        """
        try:
            import time as _time

            with self._lock:
                # Initialize history for new agents
                if agent_id not in self._agent_histories:
                    self._agent_histories[agent_id] = {
                        "features": [],
                        "last_seen": _time.monotonic(),
                        "transaction_count": 0,
                    }

                history = self._agent_histories[agent_id]

                # Calculate timing interval
                now = _time.monotonic()
                timing_interval = now - history["last_seen"]
                history["last_seen"] = now
                history["transaction_count"] += 1

            # Extract features
            features = self._extract_features(transaction_data, timing_interval)

            with self._lock:
                # Add to history (capped)
                history["features"].append(features)
                if len(history["features"]) > self.MAX_HISTORY:
                    history["features"] = history["features"][-self.MAX_HISTORY:]

                feature_count = len(history["features"])

            # Anomaly detection
            if feature_count >= self.MIN_BASELINE:
                return self._detect_anomaly(agent_id, features, feature_count)
            else:
                # Not enough history -- return moderate risk
                return {
                    "risk_score": 0.5,
                    "confidence": float(feature_count / self.MIN_BASELINE) * 0.5,
                    "behavioral_consistency": 0.5,
                    "is_anomaly": False,
                    "features_analyzed": len(features),
                    "baseline_status": "building",
                    "observations": feature_count,
                }

        except Exception as e:
            logger.error(f"Agent behavioral analysis error: {e}")
            return {
                "risk_score": 0.5,
                "confidence": 0.0,
                "behavioral_consistency": 0.0,
                "is_anomaly": False,
                "features_analyzed": 0,
                "error": str(e),
            }

    def _detect_anomaly(self, agent_id: str, features: List[float],
                        feature_count: int) -> Dict[str, Any]:
        """Run Isolation Forest anomaly detection against agent baseline."""
        with self._lock:
            history = self._agent_histories[agent_id]
            training_data = np.array(history["features"])

        # Train or re-train model (every 10 new observations)
        if agent_id not in self._agent_models or feature_count % 10 == 0:
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=50,
            )
            model.fit(training_data)
            with self._lock:
                self._agent_models[agent_id] = model

        model = self._agent_models[agent_id]
        features_arr = np.array([features])

        anomaly_score = model.decision_function(features_arr)[0]
        is_anomaly = model.predict(features_arr)[0] == -1

        # Convert to risk score: lower anomaly_score = higher risk
        risk_score = float(max(0, min(1, (0.5 - anomaly_score) * 2)))

        # Behavioral consistency: how well this transaction matches the baseline
        # Higher anomaly_score = more consistent
        consistency = float(max(0, min(1, (anomaly_score + 0.5))))

        return {
            "risk_score": risk_score,
            "confidence": min(0.9, 0.5 + (feature_count / 100) * 0.4),
            "behavioral_consistency": consistency,
            "is_anomaly": bool(is_anomaly),
            "features_analyzed": len(features),
            "baseline_status": "active",
            "observations": feature_count,
        }
```

Then add the singleton:

```python
agent_fingerprinter = AgentBehavioralFingerprint()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_behavioral.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_agent_behavioral.py
git commit -m "feat: Add AgentBehavioralFingerprint with Isolation Forest baselines"
```

---

### Task 2: Add analyze_agent_transaction MCP tool

**Files:**
- Modify: `server.py` (add `analyze_agent_transaction_impl` and `@mcp.tool()` wrapper)

**Step 1: Write the test**

Add to `tests/test_agent_behavioral.py`:

```python
class TestAnalyzeAgentTransactionImpl:
    """Tests for the analyze_agent_transaction_impl function"""

    @pytest.mark.unit
    def test_impl_returns_valid_result(self):
        from server import analyze_agent_transaction_impl
        result = analyze_agent_transaction_impl(
            transaction_data={"amount": 100.0, "merchant": "Store",
                              "timestamp": "2025-01-01T12:00:00"},
            agent_identifier="test-impl-agent"
        )
        assert "risk_score" in result
        assert "analysis_timestamp" in result
        assert "behavioral_consistency" in result

    @pytest.mark.unit
    def test_impl_no_agent_id(self):
        from server import analyze_agent_transaction_impl
        result = analyze_agent_transaction_impl(
            transaction_data={"amount": 100.0}
        )
        assert "error" in result

    @pytest.mark.unit
    def test_impl_invalid_transaction_data(self):
        from server import analyze_agent_transaction_impl
        result = analyze_agent_transaction_impl(
            transaction_data="not a dict",
            agent_identifier="agent-1"
        )
        assert "error" in result

    @pytest.mark.unit
    def test_impl_includes_identity_check(self):
        from server import analyze_agent_transaction_impl
        result = analyze_agent_transaction_impl(
            transaction_data={"amount": 100.0, "merchant": "Store",
                              "timestamp": "2025-01-01T12:00:00"},
            agent_identifier="identity-check-agent"
        )
        assert "agent_identity" in result

    @pytest.mark.unit
    def test_impl_with_behavioral_history(self):
        """Building up history then analyzing"""
        from server import analyze_agent_transaction_impl
        # Build baseline
        for i in range(12):
            analyze_agent_transaction_impl(
                transaction_data={"amount": 100.0 + i, "merchant": "Store A",
                                  "timestamp": f"2025-01-01T12:{i:02d}:00"},
                agent_identifier="history-agent"
            )
        # Analyze new transaction
        result = analyze_agent_transaction_impl(
            transaction_data={"amount": 105.0, "merchant": "Store A",
                              "timestamp": "2025-01-01T13:00:00"},
            agent_identifier="history-agent"
        )
        assert result["risk_score"] >= 0.0
        assert result["risk_score"] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_behavioral.py::TestAnalyzeAgentTransactionImpl -v`
Expected: FAIL

**Step 3: Implement analyze_agent_transaction_impl and MCP tool**

In `server.py`, after `verify_agent_identity_impl` (around line 1700), add:

```python
def analyze_agent_transaction_impl(
    transaction_data: Dict[str, Any],
    agent_identifier: Optional[str] = None,
) -> Dict[str, Any]:
    """Implementation of agent-specific transaction analysis.

    Combines agent behavioral fingerprinting with identity verification
    for a comprehensive agent transaction risk assessment.

    Args:
        transaction_data: Transaction data dict.
        agent_identifier: Agent identifier for behavioral baseline lookup.

    Returns:
        Analysis result with risk_score, behavioral_consistency,
        agent_identity verification, and anomaly details.
    """
    try:
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
                "risk_score": 0.5,
            }

        if not agent_identifier:
            return {
                "error": "agent_identifier is required for agent transaction analysis",
                "status": "validation_failed",
                "risk_score": 0.5,
            }

        # Agent behavioral fingerprinting
        behavioral_result = agent_fingerprinter.analyze(
            agent_id=str(agent_identifier),
            transaction_data=transaction_data,
        )

        # Agent identity verification
        identity_result = agent_verifier.verify(
            agent_identifier=str(agent_identifier),
            api_key=str(transaction_data.get("api_key", "")) or None,
            token=str(transaction_data.get("token", "")) or None,
        )

        # Combine results
        result = {
            "risk_score": behavioral_result.get("risk_score", 0.5),
            "confidence": behavioral_result.get("confidence", 0.0),
            "behavioral_consistency": behavioral_result.get("behavioral_consistency", 0.0),
            "is_anomaly": behavioral_result.get("is_anomaly", False),
            "features_analyzed": behavioral_result.get("features_analyzed", 0),
            "baseline_status": behavioral_result.get("baseline_status", "unknown"),
            "observations": behavioral_result.get("observations", 0),
            "agent_identity": {
                "verified": identity_result.get("verified", False),
                "trust_score": identity_result.get("trust_score", 0.0),
                "warnings": identity_result.get("warnings", []),
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return result

    except Exception as e:
        logger.error(f"Agent transaction analysis failed: {e}")
        return {
            "error": str(e),
            "risk_score": 0.5,
            "status": "analysis_failed",
        }
```

Then in the MCP tool definitions section, after `verify_agent_identity` tool, add:

```python
@_monitored("/analyze_agent_transaction", "TOOL")
@mcp.tool()
def analyze_agent_transaction(
    transaction_data: Dict[str, Any],
    agent_identifier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze an AI agent-initiated transaction using behavioral fingerprinting.

    Builds per-agent behavioral baselines tracking API timing patterns,
    decision consistency, and request structure. Uses Isolation Forest
    anomaly detection against the agent's historical profile. Also performs
    identity verification against the agent registry.

    This tool replaces behavioral biometrics (keystroke/mouse analysis)
    for agent-originated transactions where human biometrics don't exist.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp, etc.)
        agent_identifier: Agent identity string for baseline lookup

    Returns:
        Agent transaction analysis with risk score, behavioral consistency,
        anomaly detection, identity verification, and baseline status
    """
    return analyze_agent_transaction_impl(transaction_data, agent_identifier)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_behavioral.py -v`
Expected: All 17 tests PASS

**Step 5: Update MCP tool count tests**

Update any tests that assert MCP tool count (should now be 16).

**Step 6: Commit**

```bash
git add server.py tests/test_agent_behavioral.py
git commit -m "feat: Add analyze_agent_transaction MCP tool with behavioral fingerprinting"
```

---

### Task 3: Integrate behavioral fingerprint into generate_risk_score_impl

**Files:**
- Modify: `server.py` (`generate_risk_score_impl`)

**Step 1: Write the test**

Add to `tests/test_agent_behavioral.py`:

```python
class TestBehavioralFingerprintInRiskScoring:
    """Tests for behavioral fingerprint integration in generate_risk_score_impl"""

    @pytest.mark.unit
    def test_agent_traffic_gets_behavioral_fingerprint(self):
        """Agent traffic uses behavioral fingerprinting instead of biometrics"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "fingerprint-agent"},
        )
        assert result.get("traffic_source") == "agent"
        assert "agent_behavioral" in result.get("component_scores", {})

    @pytest.mark.unit
    def test_human_traffic_no_agent_fingerprint(self):
        """Human traffic doesn't get agent behavioral fingerprinting"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": False},
        )
        assert "agent_behavioral" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_agent_without_identifier_no_fingerprint(self):
        """Agent without identifier can't do fingerprinting"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True},
        )
        assert "agent_behavioral" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_agent_behavioral_score_in_overall(self):
        """Agent behavioral score contributes to overall risk score"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "overall-agent"},
        )
        assert result["overall_risk_score"] >= 0.0
        assert result["overall_risk_score"] <= 1.0
        assert "agent_behavioral" in result.get("component_scores", {})
        # Should have multiple analysis components
        assert len(result.get("analysis_components", [])) >= 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_behavioral.py::TestBehavioralFingerprintInRiskScoring -v`
Expected: FAIL (agent_behavioral not in component_scores)

**Step 3: Update generate_risk_score_impl**

In `server.py`, in `generate_risk_score_impl`, after the identity analysis block (around line 1607) and before the weighted score calculation (around line 1609), add:

```python
        # Agent behavioral fingerprinting (agent traffic with identifier only)
        if is_agent_traffic and transaction_data.get("agent_identifier"):
            agent_behavioral = agent_fingerprinter.analyze(
                agent_id=str(transaction_data["agent_identifier"]),
                transaction_data=transaction_data,
            )
            ab_score = agent_behavioral.get("risk_score", 0.5)
            comprehensive_result["component_scores"]["agent_behavioral"] = ab_score
            scores.append(ab_score)
            confidences.append(agent_behavioral.get("confidence", 0.3))

            if agent_behavioral.get("is_anomaly"):
                comprehensive_result["detected_anomalies"].append("agent_behavioral_anomaly")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_behavioral.py -v`
Expected: All 21 tests PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 636+ passed, 2 skipped

**Step 6: Commit**

```bash
git add server.py tests/test_agent_behavioral.py
git commit -m "feat: Integrate agent behavioral fingerprinting into risk scoring"
```

---

### Task 4: Final verification and lint

**Step 1: Run ruff check**

Run: `ruff check . --exclude=venv,test_data,.claude`
Expected: 0 errors

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 636+ passed, 2 skipped

**Step 3: Verify new MCP tool count**

Run: `python -c "from server import mcp; print(len(mcp._tool_manager._tools))"`
Expected: 16 (was 15, +1 analyze_agent_transaction)

**Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "chore: Phase C agent behavioral fingerprinting complete"
```
