# Phase A: Agent Traffic Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `classify_traffic_source` MCP tool and agent-aware risk scoring to fraud-detection-mcp.

**Architecture:** Add a `TrafficClassifier` class to `server.py` that uses heuristic signals (user_agent strings, explicit `is_agent` flags, timing patterns, absence of behavioral data) to classify traffic. Add `classify_traffic_source` as the 14th MCP tool. Update `generate_risk_score_impl` to apply different weights when agent traffic is detected. Add optional agent fields to `TransactionData` Pydantic model.

**Tech Stack:** Python, FastMCP, Pydantic, pytest

---

### Task 1: Add agent fields to Pydantic model

**Files:**
- Modify: `models_validation.py` (add fields to `TransactionData` class and new `TrafficSource` enum)

**Step 1: Add TrafficSource enum and agent fields**

In `models_validation.py`, after the `RiskLevel` enum (around line 32), add:

```python
class TrafficSource(str, Enum):
    """Traffic source classification"""
    HUMAN = "human"
    AGENT = "agent"
    UNKNOWN = "unknown"
```

In the `TransactionData` class, after `merchant_id` field (around line 119), add:

```python
    # Agent transaction fields (optional)
    is_agent: Optional[bool] = Field(
        None,
        description="Whether this transaction was initiated by an AI agent"
    )
    agent_identifier: Optional[str] = Field(
        None,
        max_length=200,
        description="Agent identity string (e.g., 'stripe-acp:agent-123')"
    )
    user_agent: Optional[str] = Field(
        None,
        max_length=500,
        description="HTTP User-Agent or agent protocol identifier"
    )
```

**Step 2: Run tests to verify no regressions**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 560 passed, 2 skipped (new optional fields shouldn't break anything)

**Step 3: Commit**

```bash
git add models_validation.py
git commit -m "feat: Add TrafficSource enum and agent fields to TransactionData model"
```

---

### Task 2: Add TrafficClassifier class to server.py

**Files:**
- Modify: `server.py` (add `TrafficClassifier` class and module-level instance)

**Step 1: Write the test**

Create `tests/test_traffic_classifier.py`:

```python
"""Tests for TrafficClassifier and classify_traffic_source MCP tool"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
from server import TrafficClassifier


class TestTrafficClassifier:
    """Tests for the TrafficClassifier class"""

    def setup_method(self):
        self.classifier = TrafficClassifier()

    @pytest.mark.unit
    def test_classify_explicit_agent_flag(self):
        """When is_agent=True is set, classify as agent"""
        result = self.classifier.classify({"is_agent": True})
        assert result["source"] == "agent"
        assert result["confidence"] >= 0.8
        assert "explicit_flag" in result["signals"]

    @pytest.mark.unit
    def test_classify_explicit_human_flag(self):
        """When is_agent=False is set, classify as human"""
        result = self.classifier.classify({"is_agent": False})
        assert result["source"] == "human"
        assert result["confidence"] >= 0.8
        assert "explicit_flag" in result["signals"]

    @pytest.mark.unit
    def test_classify_stripe_acp_user_agent(self):
        """Stripe ACP user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "Stripe-ACP/1.0 agent-id:abc123"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "stripe_acp"
        assert "user_agent_match" in result["signals"]

    @pytest.mark.unit
    def test_classify_visa_tap_user_agent(self):
        """Visa TAP user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "Visa-TAP/2.0 commerce-agent"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "visa_tap"

    @pytest.mark.unit
    def test_classify_openai_operator_user_agent(self):
        """OpenAI Operator user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "OpenAI-Operator/1.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "openai"

    @pytest.mark.unit
    def test_classify_browser_user_agent(self):
        """Standard browser user agent classified as human"""
        result = self.classifier.classify({
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0"
        })
        assert result["source"] == "human"
        assert "user_agent_match" in result["signals"]

    @pytest.mark.unit
    def test_classify_agent_identifier_present(self):
        """Presence of agent_identifier signals agent"""
        result = self.classifier.classify({
            "agent_identifier": "mastercard-agent-pay:agent-456"
        })
        assert result["source"] == "agent"
        assert "agent_identifier_present" in result["signals"]

    @pytest.mark.unit
    def test_classify_unknown_no_signals(self):
        """No signals results in unknown classification"""
        result = self.classifier.classify({
            "amount": 100.0,
            "merchant": "Store"
        })
        assert result["source"] == "unknown"
        assert result["confidence"] < 0.5

    @pytest.mark.unit
    def test_classify_returns_all_required_fields(self):
        """Result contains all required fields"""
        result = self.classifier.classify({"is_agent": True})
        assert "source" in result
        assert "confidence" in result
        assert "agent_type" in result
        assert "signals" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.unit
    def test_classify_empty_input(self):
        """Empty dict returns unknown"""
        result = self.classifier.classify({})
        assert result["source"] == "unknown"

    @pytest.mark.unit
    def test_classify_multiple_signals_boost_confidence(self):
        """Multiple agent signals increase confidence"""
        result = self.classifier.classify({
            "is_agent": True,
            "agent_identifier": "stripe-acp:agent-789",
            "user_agent": "Stripe-ACP/1.0"
        })
        assert result["source"] == "agent"
        assert result["confidence"] >= 0.95

    @pytest.mark.unit
    def test_classify_coinbase_x402(self):
        """Coinbase x402 protocol detected"""
        result = self.classifier.classify({
            "user_agent": "x402-client/1.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "x402"

    @pytest.mark.unit
    def test_classify_google_ap2(self):
        """Google AP2 protocol detected"""
        result = self.classifier.classify({
            "user_agent": "Google-AP2/1.0 agent"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "google_ap2"

    @pytest.mark.unit
    def test_classify_paypal_agent(self):
        """PayPal Agent Ready detected"""
        result = self.classifier.classify({
            "user_agent": "PayPal-Agent/2.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "paypal"

    @pytest.mark.unit
    def test_classify_anthropic_agent(self):
        """Anthropic Claude agent detected"""
        result = self.classifier.classify({
            "user_agent": "Anthropic-Agent/1.0 claude"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "anthropic"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_traffic_classifier.py -v`
Expected: FAIL (TrafficClassifier not yet defined)

**Step 3: Implement TrafficClassifier**

In `server.py`, after the `NetworkAnalyzer` class definition (which ends around line 900) and before the `_monitored` function (line 915), add:

```python
# =============================================================================
# Traffic Source Classifier
# =============================================================================

# Known AI agent User-Agent patterns
AGENT_USER_AGENT_PATTERNS = {
    "stripe_acp": ["stripe-acp", "stripe acp"],
    "visa_tap": ["visa-tap", "visa tap"],
    "mastercard_agent": ["mastercard-agent", "mastercard agent"],
    "openai": ["openai-operator", "openai operator", "openai-agent"],
    "anthropic": ["anthropic-agent", "anthropic agent", "claude-agent"],
    "google_ap2": ["google-ap2", "google ap2"],
    "paypal": ["paypal-agent", "paypal agent"],
    "x402": ["x402-client", "x402 client"],
    "coinbase": ["coinbase-agent", "coinbase agent", "agentkit"],
}

# Browser User-Agent patterns indicating human traffic
BROWSER_USER_AGENT_PATTERNS = [
    "mozilla/", "chrome/", "safari/", "firefox/", "edge/", "opera/",
]


class TrafficClassifier:
    """Classifies transaction traffic as human, agent, or unknown.

    Uses heuristic signals: explicit flags, user_agent patterns, agent
    identifiers, and absence/presence of behavioral data.
    """

    def classify(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a transaction's traffic source.

        Args:
            metadata: Transaction metadata including optional fields:
                is_agent, agent_identifier, user_agent, behavioral_data.

        Returns:
            Dict with source, confidence, agent_type, and signals.
        """
        signals = []
        agent_score = 0.0  # positive = agent, negative = human
        agent_type = None

        # Signal 1: Explicit is_agent flag (strongest signal)
        is_agent = metadata.get("is_agent")
        if is_agent is True:
            agent_score += 0.8
            signals.append("explicit_flag")
        elif is_agent is False:
            agent_score -= 0.8
            signals.append("explicit_flag")

        # Signal 2: Agent identifier present
        agent_id = metadata.get("agent_identifier")
        if agent_id and isinstance(agent_id, str) and len(agent_id.strip()) > 0:
            agent_score += 0.6
            signals.append("agent_identifier_present")
            # Try to extract agent type from identifier
            if not agent_type:
                agent_id_lower = agent_id.lower()
                for atype, patterns in AGENT_USER_AGENT_PATTERNS.items():
                    if any(p in agent_id_lower for p in patterns):
                        agent_type = atype
                        break

        # Signal 3: User-Agent string analysis
        user_agent = metadata.get("user_agent")
        if user_agent and isinstance(user_agent, str):
            ua_lower = user_agent.lower()

            # Check for known agent patterns
            for atype, patterns in AGENT_USER_AGENT_PATTERNS.items():
                if any(p in ua_lower for p in patterns):
                    agent_score += 0.7
                    signals.append("user_agent_match")
                    if not agent_type:
                        agent_type = atype
                    break
            else:
                # Check for browser patterns (human signal)
                if any(p in ua_lower for p in BROWSER_USER_AGENT_PATTERNS):
                    agent_score -= 0.5
                    signals.append("user_agent_match")

        # Clamp confidence to [0, 1]
        raw_confidence = min(abs(agent_score), 1.0)

        # Determine classification
        if agent_score > 0.3:
            source = "agent"
            confidence = raw_confidence
        elif agent_score < -0.3:
            source = "human"
            confidence = raw_confidence
        else:
            source = "unknown"
            confidence = 1.0 - raw_confidence  # low confidence in the unknown case

        return {
            "source": source,
            "confidence": float(confidence),
            "agent_type": agent_type,
            "signals": signals,
        }
```

Then, after the existing singleton initializations (around where `behavioral_analyzer`, `transaction_analyzer`, `network_analyzer` are created), add:

```python
traffic_classifier = TrafficClassifier()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_traffic_classifier.py -v`
Expected: All 16 tests PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 576+ passed, 2 skipped

**Step 6: Commit**

```bash
git add server.py tests/test_traffic_classifier.py
git commit -m "feat: Add TrafficClassifier for agent vs human traffic detection"
```

---

### Task 3: Add classify_traffic_source MCP tool

**Files:**
- Modify: `server.py` (add `classify_traffic_source_impl` and `@mcp.tool()` wrapper)

**Step 1: Write the test**

Add to `tests/test_traffic_classifier.py`:

```python
class TestClassifyTrafficSourceImpl:
    """Tests for the classify_traffic_source_impl function"""

    @pytest.mark.unit
    def test_impl_returns_valid_result(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({
            "user_agent": "Stripe-ACP/1.0",
            "amount": 100.0
        })
        assert result["source"] == "agent"
        assert "classification_timestamp" in result

    @pytest.mark.unit
    def test_impl_with_transaction_data_and_metadata(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl(
            {"amount": 100.0, "merchant": "Store"},
            {"user_agent": "Mozilla/5.0 Chrome/120", "is_agent": False}
        )
        assert result["source"] == "human"

    @pytest.mark.unit
    def test_impl_invalid_input(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl("not a dict")
        assert "error" in result

    @pytest.mark.unit
    def test_impl_metadata_in_transaction_data(self):
        """Agent fields in transaction_data itself should be detected"""
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({
            "amount": 100.0,
            "is_agent": True,
            "agent_identifier": "test-agent-1"
        })
        assert result["source"] == "agent"

    @pytest.mark.unit
    def test_impl_empty_input(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({})
        assert result["source"] == "unknown"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_traffic_classifier.py::TestClassifyTrafficSourceImpl -v`
Expected: FAIL (classify_traffic_source_impl not defined)

**Step 3: Implement classify_traffic_source_impl and MCP tool**

In `server.py`, after the `explain_decision_impl` function (around line 1430), add:

```python
def classify_traffic_source_impl(
    transaction_data: Dict[str, Any],
    request_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of traffic source classification.

    Determines whether a transaction originates from a human user,
    an AI agent, or an unknown source.

    Args:
        transaction_data: Transaction data (may contain agent fields).
        request_metadata: Optional additional metadata (user_agent, is_agent, etc.).

    Returns:
        Classification result with source, confidence, agent_type, and signals.
    """
    try:
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
                "source": "unknown",
                "confidence": 0.0,
            }

        # Merge transaction_data agent fields with request_metadata
        merged = {}
        for key in ("is_agent", "agent_identifier", "user_agent"):
            val = None
            if request_metadata and isinstance(request_metadata, dict):
                val = request_metadata.get(key)
            if val is None:
                val = transaction_data.get(key)
            if val is not None:
                merged[key] = val

        result = traffic_classifier.classify(merged)
        result["classification_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Traffic classification failed: {e}")
        return {
            "error": str(e),
            "source": "unknown",
            "confidence": 0.0,
            "status": "classification_failed",
        }
```

Then in the MCP tool definitions section (after the `explain_decision` tool, around line 2220), add:

```python
@_monitored("/classify_traffic_source", "TOOL")
@mcp.tool()
def classify_traffic_source(
    transaction_data: Dict[str, Any],
    request_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify whether a transaction originates from a human, AI agent, or unknown source.

    Analyzes transaction metadata, User-Agent strings, and explicit agent flags
    to determine traffic source. Recognizes major agent commerce protocols:
    Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal Agent Ready,
    Coinbase x402, OpenAI Operator, and Anthropic Claude agents.

    Args:
        transaction_data: Transaction details (may include is_agent, agent_identifier, user_agent fields)
        request_metadata: Optional request metadata (user_agent, is_agent flag, agent_identifier)

    Returns:
        Classification with source (human/agent/unknown), confidence, agent_type, and signals
    """
    return classify_traffic_source_impl(transaction_data, request_metadata)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_traffic_classifier.py -v`
Expected: All 21 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_traffic_classifier.py
git commit -m "feat: Add classify_traffic_source MCP tool"
```

---

### Task 4: Update generate_risk_score_impl for agent-aware weighting

**Files:**
- Modify: `server.py` (`generate_risk_score_impl` function)

**Step 1: Write the test**

Add to `tests/test_traffic_classifier.py`:

```python
class TestAgentAwareRiskScoring:
    """Tests for agent-aware risk score weighting"""

    @pytest.mark.unit
    def test_risk_score_human_uses_standard_weights(self):
        """Human traffic uses standard 50/30/20 weights"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card"},
            {"keystroke_dynamics": [{"key": "a", "dwell_time": 80, "flight_time": 120}] * 10},
            {"entity_id": "user123", "connections": [{"target": "user456", "type": "shared_device"}]}
        )
        assert "traffic_source" not in result or result.get("traffic_source") == "unknown"
        assert result["overall_risk_score"] >= 0.0

    @pytest.mark.unit
    def test_risk_score_agent_uses_agent_weights(self):
        """Agent traffic applies agent-specific weighting"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "stripe-acp:agent-1"},
            None,  # No behavioral data for agents
            {"entity_id": "agent-1", "connections": [{"target": "merchant-1", "type": "transaction"}]}
        )
        assert result.get("traffic_source") == "agent"
        assert "agent_classification" in result
        assert result["overall_risk_score"] >= 0.0

    @pytest.mark.unit
    def test_risk_score_agent_no_behavioral_skips_behavioral(self):
        """Agent traffic without behavioral data skips behavioral weight"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True},
        )
        assert result.get("traffic_source") == "agent"
        assert "behavioral" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_risk_score_agent_with_network_reweights(self):
        """Agent traffic with network data uses heavier network weight"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True},
            None,
            {"entity_id": "agent-1", "connections": []}
        )
        assert result.get("traffic_source") == "agent"
        assert result["overall_risk_score"] >= 0.0
        assert "analysis_components" in result

    @pytest.mark.unit
    def test_risk_score_unknown_traffic_uses_standard(self):
        """Unknown traffic source uses standard weights"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card"}
        )
        # No agent fields, should be unknown/standard
        assert result["overall_risk_score"] >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_traffic_classifier.py::TestAgentAwareRiskScoring -v`
Expected: Some tests FAIL (traffic_source not in result yet)

**Step 3: Update generate_risk_score_impl**

In `server.py`, modify `generate_risk_score_impl` (starts at line 1205). Add traffic classification at the start and adjust weighting:

After the validation check (line 1216), before "Perform all analyses" (line 1218), add:

```python
        # Classify traffic source
        classification = traffic_classifier.classify(transaction_data)
        traffic_source = classification["source"]
        is_agent_traffic = traffic_source == "agent"
```

Replace the existing weighted score calculation (lines 1267-1273):

```python
        # Calculate weighted overall score
        if len(scores) == 1:
            overall_score = scores[0]
        elif len(scores) == 2:
            overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            overall_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
```

With:

```python
        # Calculate weighted overall score
        # Agent traffic: heavier network weight, no behavioral biometrics
        # Human traffic: standard weights (transaction 50%, behavioral 30%, network 20%)
        if is_agent_traffic:
            if len(scores) == 1:
                overall_score = scores[0]
            elif len(scores) == 2:
                # Transaction + network (no behavioral for agents)
                overall_score = (scores[0] * 0.55 + scores[1] * 0.45)
            else:
                # All three present (unusual for agents but handle it)
                overall_score = (scores[0] * 0.4 + scores[1] * 0.2 + scores[2] * 0.4)
        else:
            if len(scores) == 1:
                overall_score = scores[0]
            elif len(scores) == 2:
                overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
            else:
                overall_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
```

Before the return statement (before line 1320), add traffic classification info:

```python
        comprehensive_result["traffic_source"] = traffic_source
        comprehensive_result["agent_classification"] = classification
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_traffic_classifier.py -v`
Expected: All 26 tests PASS

**Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 586+ passed, 2 skipped

**Step 6: Commit**

```bash
git add server.py tests/test_traffic_classifier.py
git commit -m "feat: Add agent-aware risk score weighting in generate_risk_score"
```

---

### Task 5: Final verification and lint

**Step 1: Run ruff check**

Run: `ruff check . --exclude=venv,test_data,.claude`
Expected: 0 errors

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 586+ passed, 2 skipped

**Step 3: Verify new MCP tool count**

Run: `python -c "from server import mcp; print(len(mcp._tool_manager._tools))"`
Expected: 14 (was 13, +1 classify_traffic_source)

**Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "chore: Phase A agent traffic classification complete"
```
