# Phase E: Reputation and Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add longitudinal agent reputation scoring and update explain_decision for agent-specific reasoning, completing the agent-to-agent transaction fraud detection roadmap.

**Architecture:** `AgentReputationScorer` pulls data from existing singletons (`agent_registry`, `agent_fingerprinter`, `collusion_detector`) to compute a weighted reputation score. `explain_decision_impl` is enhanced to recognize agent-specific components (identity, fingerprint, mandate, collusion, reputation) and produce meaningful explanations for agent traffic.

**Tech Stack:** Python 3.10+, FastMCP, pytest

---

### Task 1: Add AgentReputationScorer class + tests

**Files:**
- Create: `tests/test_agent_reputation.py`
- Modify: `server.py` (add class after `collusion_detector` singleton, ~line 1943)

**Step 1: Write the failing tests**

Create `tests/test_agent_reputation.py`:

```python
"""Tests for AgentReputationScorer class."""

import pytest
from datetime import datetime, timedelta


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
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry.json"))
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("nonexistent-agent")
        assert result["reputation_score"] < 0.5
        assert result["history_length"] == 0
        assert result["transaction_count"] == 0

    def test_registered_agent_gets_baseline_reputation(self):
        """Registered agent with default trust gets moderate reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry2.json"))
        registry.register("test-agent", agent_type="stripe_acp")
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("test-agent")
        assert result["reputation_score"] > 0.0
        assert result["transaction_count"] == 0

    def test_high_trust_agent_gets_high_reputation(self):
        """Agent with high trust score gets higher reputation."""
        from server import AgentReputationScorer, AgentIdentityRegistry
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry3.json"))
        registry.register("trusted-agent")
        registry.update_trust("trusted-agent", 0.95)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("trusted-agent")
        assert result["reputation_score"] > 0.3

    def test_transaction_history_boosts_reputation(self):
        """More transactions improve the history component."""
        from server import AgentReputationScorer, AgentIdentityRegistry
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry4.json"))
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
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry5.json"))
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
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry6.json"))
        registry.register("edge-agent")
        registry.update_trust("edge-agent", 1.0)
        scorer = AgentReputationScorer(registry=registry)
        result = scorer.score("edge-agent")
        assert 0.0 <= result["reputation_score"] <= 1.0

    def test_components_sum_reflects_score(self):
        """Component weights sum to produce the reputation score."""
        from server import AgentReputationScorer, AgentIdentityRegistry
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry7.json"))
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
        from server import AgentReputationScorer, AgentIdentityRegistry, AgentBehavioralFingerprint
        from pathlib import Path

        registry = AgentIdentityRegistry(registry_path=Path("/tmp/test_rep_registry8.json"))
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_reputation.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'AgentReputationScorer' from 'server'`

**Step 3: Write the AgentReputationScorer implementation**

Add to `server.py` after the `collusion_detector = CollusionDetector()` singleton (after line ~1943):

```python
# =============================================================================
# Agent Reputation Scorer
# =============================================================================


class AgentReputationScorer:
    """Longitudinal reputation scoring for AI agents.

    Computes a composite reputation from:
    - Trust score from identity registry (40%)
    - Transaction history length (25%)
    - Behavioral consistency from fingerprinter (25%)
    - Collusion safety from collusion detector (10%)
    """

    TRUST_WEIGHT = 0.4
    HISTORY_WEIGHT = 0.25
    CONSISTENCY_WEIGHT = 0.25
    COLLUSION_WEIGHT = 0.1
    HISTORY_CAP = 100  # transactions for full history credit

    def __init__(
        self,
        registry: Optional["AgentIdentityRegistry"] = None,
        fingerprinter: Optional["AgentBehavioralFingerprint"] = None,
        detector: Optional["CollusionDetector"] = None,
    ):
        self._registry = registry
        self._fingerprinter = fingerprinter
        self._detector = detector

    def score(self, agent_id: str) -> Dict[str, Any]:
        """Compute reputation score for an agent.

        Args:
            agent_id: Agent identifier to score.

        Returns:
            Dict with reputation_score (0-1), history_length, transaction_count,
            trust_score, behavioral_consistency, and components breakdown.
        """
        registry = self._registry or agent_registry
        fingerprinter = self._fingerprinter or agent_fingerprinter
        detector = self._detector or collusion_detector

        # --- Trust score from registry ---
        entry = registry.lookup(agent_id)
        if entry:
            trust = float(entry.get("trust_score", 0.5))
            txn_count = int(entry.get("transaction_count", 0))
            first_seen = entry.get("first_seen", "")
            last_seen = entry.get("last_seen", "")
        else:
            trust = 0.0
            txn_count = 0
            first_seen = ""
            last_seen = ""

        # --- History factor ---
        history_factor = min(1.0, txn_count / self.HISTORY_CAP) if self.HISTORY_CAP > 0 else 0.0

        # History length in days
        history_days = 0
        if first_seen and last_seen:
            try:
                fs = datetime.fromisoformat(first_seen)
                ls = datetime.fromisoformat(last_seen)
                history_days = max(0, (ls - fs).days)
            except (ValueError, TypeError):
                pass

        # --- Behavioral consistency ---
        consistency = 0.0
        baseline = fingerprinter.get_baseline(agent_id)
        if baseline and baseline.get("observation_count", 0) >= 10:
            # More observations = more consistent agent
            obs = baseline["observation_count"]
            # Low timing std relative to mean = consistent
            timing_std = baseline.get("timing_std", 0.0)
            timing_mean = baseline.get("timing_mean", 1.0)
            if timing_mean > 0:
                cv = timing_std / timing_mean  # coefficient of variation
                # CV < 0.3 = very consistent, CV > 1.0 = inconsistent
                consistency = max(0.0, min(1.0, 1.0 - cv))
            else:
                consistency = 0.5
            # Bonus for having many observations
            consistency = min(1.0, consistency * min(1.0, obs / 20))

        # --- Collusion safety ---
        collusion_safety = 1.0
        try:
            col_result = detector.detect([agent_id], window_seconds=86400)
            collusion_score = col_result.get("collusion_score", 0.0)
            collusion_safety = 1.0 - collusion_score
        except Exception:
            pass

        # --- Weighted composite ---
        reputation = (
            self.TRUST_WEIGHT * trust
            + self.HISTORY_WEIGHT * history_factor
            + self.CONSISTENCY_WEIGHT * consistency
            + self.COLLUSION_WEIGHT * collusion_safety
        )
        reputation = float(max(0.0, min(1.0, reputation)))

        return {
            "reputation_score": reputation,
            "history_length": history_days,
            "transaction_count": txn_count,
            "trust_score": trust,
            "behavioral_consistency": consistency,
            "components": {
                "trust_score": trust,
                "history_factor": history_factor,
                "behavioral_consistency": consistency,
                "collusion_safety": collusion_safety,
            },
        }


reputation_scorer = AgentReputationScorer()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_reputation.py -v --tb=short`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_agent_reputation.py
git commit -m "feat: Add AgentReputationScorer class for longitudinal agent reputation"
```

---

### Task 2: Add score_agent_reputation MCP tool (#19) + tests

**Files:**
- Create: `tests/test_score_agent_reputation_tool.py`
- Modify: `server.py` (add `_impl` function and MCP tool wrapper)

**Step 1: Write the failing tests**

Create `tests/test_score_agent_reputation_tool.py`:

```python
"""Tests for score_agent_reputation MCP tool."""

import pytest


class TestScoreAgentReputationImpl:
    """Test score_agent_reputation_impl function."""

    def test_returns_dict(self):
        """score_agent_reputation_impl returns a dict."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("test-agent")
        assert isinstance(result, dict)

    def test_has_reputation_score(self):
        """Result includes reputation_score."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("test-agent")
        assert "reputation_score" in result
        assert 0.0 <= result["reputation_score"] <= 1.0

    def test_has_status(self):
        """Result includes status field."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("test-agent")
        assert result["status"] == "scored"

    def test_has_timestamp(self):
        """Result includes analysis_timestamp."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("test-agent")
        assert "analysis_timestamp" in result

    def test_invalid_agent_id(self):
        """Empty agent_id returns error."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("")
        assert "error" in result

    def test_none_agent_id(self):
        """None agent_id returns error."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl(None)
        assert "error" in result

    def test_has_components(self):
        """Result includes components breakdown."""
        from server import score_agent_reputation_impl
        result = score_agent_reputation_impl("test-agent")
        assert "components" in result


class TestMCPToolRegistration:
    """Verify MCP tool registration for score_agent_reputation."""

    def test_mcp_has_score_agent_reputation(self):
        """MCP server has score_agent_reputation tool."""
        from server import mcp
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "score_agent_reputation" in tool_names

    def test_total_mcp_tools_count_is_19(self):
        """Server should now have 19 MCP tools registered."""
        from server import mcp
        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 19
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_score_agent_reputation_tool.py -v --tb=short`
Expected: FAIL with `ImportError: cannot import name 'score_agent_reputation_impl' from 'server'`

**Step 3: Write the _impl function and MCP tool wrapper**

Add `score_agent_reputation_impl` near the other `_impl` functions in `server.py` (after `detect_agent_collusion_impl`):

```python
def score_agent_reputation_impl(
    agent_id: Any,
    time_window_days: int = 30,
) -> Dict[str, Any]:
    """Compute longitudinal reputation score for an AI agent.

    Args:
        agent_id: Agent identifier to score.
        time_window_days: Time window in days for reputation analysis (default 30).

    Returns:
        Dict with reputation_score, history_length, transaction_count,
        trust_score, behavioral_consistency, components, and status.
    """
    try:
        if not agent_id or not isinstance(agent_id, str):
            return {
                "error": "agent_id must be a non-empty string",
                "status": "validation_failed",
            }

        result = reputation_scorer.score(str(agent_id))
        result["status"] = "scored"
        result["agent_id"] = str(agent_id)
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Reputation scoring failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "reputation_score": 0.0,
        }
```

Add the MCP tool wrapper after `detect_agent_collusion` (after line ~3918):

```python
@_monitored("/score_agent_reputation", "TOOL")
@mcp.tool()
def score_agent_reputation(
    agent_id: str,
    time_window_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute longitudinal reputation score for an AI agent.

    Aggregates trust score from identity verification, transaction history length,
    behavioral consistency from fingerprinting, and collusion safety into a single
    reputation score. Higher scores indicate more trustworthy agents.

    Args:
        agent_id: Agent identifier to score (e.g., 'stripe-acp:agent-123').
        time_window_days: Time window in days for analysis (default 30).

    Returns:
        Reputation result with reputation_score (0-1, higher is better),
        history_length (days), transaction_count, trust_score,
        behavioral_consistency, and per-component breakdown
    """
    return score_agent_reputation_impl(agent_id, time_window_days)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_score_agent_reputation_tool.py -v --tb=short`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_score_agent_reputation_tool.py
git commit -m "feat: Add score_agent_reputation MCP tool (#19)"
```

---

### Task 3: Update explain_decision_impl for agent-specific reasoning + tests

**Files:**
- Modify: `server.py:2465-2615` (`explain_decision_impl`)
- Modify: `tests/test_explainability.py` (add agent explanation tests)

**Step 1: Write the failing tests**

Append to `tests/test_explainability.py`:

```python
class TestAgentExplanations:
    """Test explain_decision_impl with agent-specific components."""

    def test_explains_identity_component(self):
        """Explanation includes identity component details."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.6,
            "risk_level": "HIGH",
            "detected_anomalies": ["unverified_agent_identity"],
            "component_scores": {
                "transaction": 0.3,
                "identity": 0.8,
                "behavioral_fingerprint": 0.4,
            },
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        contributions = result["algorithm_contributions"]
        assert "identity" in contributions
        assert "behavioral_fingerprint" in contributions

    def test_explains_mandate_violation(self):
        """Explanation includes mandate violation factor."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.7,
            "risk_level": "HIGH",
            "detected_anomalies": ["mandate_violation", "mandate_amount_exceeded"],
            "component_scores": {"transaction": 0.3, "mandate_compliance": 0.9},
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        factors = [f["factor"] for f in result["key_factors"]]
        assert "mandate_violation" in factors

    def test_explains_collusion_evidence(self):
        """Explanation includes collusion component."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.8,
            "risk_level": "CRITICAL",
            "detected_anomalies": ["agent_collusion_detected"],
            "component_scores": {"transaction": 0.3, "collusion": 0.9},
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        contributions = result["algorithm_contributions"]
        assert "collusion" in contributions

    def test_traffic_source_in_explanation(self):
        """Explanation includes traffic_source when present."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.3,
            "risk_level": "LOW",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.3},
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        assert result.get("traffic_source") == "agent"

    def test_agent_factor_descriptions(self):
        """Agent-specific anomalies get descriptive explanations."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.6,
            "risk_level": "HIGH",
            "detected_anomalies": [
                "unverified_agent_identity",
                "behavioral_fingerprint_anomaly",
                "mandate_violation",
            ],
            "component_scores": {"transaction": 0.5},
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        # All factors should have descriptions
        for factor in result["key_factors"]:
            assert "description" in factor
            assert len(factor["description"]) > 10

    def test_reputation_component_explained(self):
        """Reputation component is explained when present."""
        from server import explain_decision_impl

        analysis = {
            "overall_risk_score": 0.4,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.3, "reputation": 0.5},
            "traffic_source": "agent",
        }
        result = explain_decision_impl(analysis)
        contributions = result["algorithm_contributions"]
        assert "reputation" in contributions
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_explainability.py::TestAgentExplanations -v --tb=short`
Expected: Some tests FAIL (explain_decision_impl doesn't handle agent components)

**Step 3: Update explain_decision_impl**

In `server.py`, modify `explain_decision_impl` (starting at line ~2465). The key changes are in the `algorithm_contributions` section and factor descriptions.

Replace the existing `# Algorithm contributions` block (lines ~2513-2532) with:

```python
        # Algorithm contributions
        component_scores = analysis_result.get("component_scores", {})
        traffic_source = analysis_result.get("traffic_source")

        # Agent-specific component weight map
        _AGENT_COMPONENT_WEIGHTS = {
            "transaction": 0.20,
            "identity": 0.25,
            "behavioral_fingerprint": 0.25,
            "mandate_compliance": 0.15,
            "collusion": 0.15,
            "reputation": 0.10,
            "behavioral": 0.0,  # not used for agent traffic
            "network": 0.15,
        }
        _HUMAN_COMPONENT_WEIGHTS = {
            "transaction": 0.50,
            "behavioral": 0.30,
            "network": 0.20,
        }

        if component_scores:
            is_agent = traffic_source == "agent"
            weight_map = _AGENT_COMPONENT_WEIGHTS if is_agent else _HUMAN_COMPONENT_WEIGHTS
            for component, score in component_scores.items():
                weight = weight_map.get(component, 0.1)
                explanation["algorithm_contributions"][component] = {
                    "score": float(score),
                    "weight": float(weight),
                    "contribution": f"{weight * 100:.1f}% of final decision",
                }

        # Include traffic source in explanation
        if traffic_source:
            explanation["traffic_source"] = traffic_source
```

Also update the key_factors section to add agent-specific descriptions. Replace the existing `# Key contributing factors` block (lines ~2502-2511) with:

```python
        # Agent-specific factor descriptions
        _AGENT_FACTOR_DESCRIPTIONS = {
            "unverified_agent_identity": "Agent identity could not be verified through API key or JWT token validation",
            "behavioral_fingerprint_anomaly": "Agent behavior deviates significantly from established baseline patterns",
            "agent_behavioral_fingerprint_anomaly": "Agent behavioral fingerprint does not match historical patterns",
            "mandate_violation": "Transaction violates the agent's authorized scope or spending mandate",
            "mandate_amount_exceeded": "Transaction amount exceeds the agent's authorized spending limit",
            "mandate_blocked_merchant": "Transaction targets a merchant on the agent's blocked list",
            "mandate_merchant_not_allowed": "Transaction targets a merchant outside the agent's allowed list",
            "mandate_location_not_allowed": "Transaction originates from a location outside the agent's authorized regions",
            "mandate_outside_time_window": "Transaction occurred outside the agent's authorized operating hours",
            "mandate_daily_limit_exceeded": "Transaction would push the agent past its daily spending limit",
            "agent_collusion_detected": "Graph analysis detected coordinated behavior with other agents",
            "missing_agent_identifier": "Agent transaction lacks an identifier for verification",
        }

        # Key contributing factors
        if detected_anomalies:
            explanation["key_factors"] = [
                {
                    "factor": anomaly,
                    "impact": "high" if any(
                        k in anomaly for k in ("unverified", "collusion", "mandate_violation")
                    ) else "medium",
                    "description": _AGENT_FACTOR_DESCRIPTIONS.get(
                        anomaly,
                        f"Detected pattern: {anomaly.replace('_', ' ')}",
                    ),
                }
                for anomaly in detected_anomalies
            ]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_explainability.py -v --tb=short`
Expected: All tests PASS (existing + new agent explanation tests)

**Step 5: Commit**

```bash
git add server.py tests/test_explainability.py
git commit -m "feat: Update explain_decision for agent-specific reasoning"
```

---

### Task 4: Update tool count assertions + final verification

**Files:**
- Modify: `tests/test_training_and_persistence.py` (tool count 18 -> 19)
- Modify: `tests/test_synthetic_data.py` (tool count 18 -> 19)

**Step 1: Update tool count assertions**

In `tests/test_training_and_persistence.py`, change `assert tool_count == 18` to `assert tool_count == 19`.

In `tests/test_synthetic_data.py`, change `assert tool_count == 18` to `assert tool_count == 19`.

**Step 2: Run ruff check and format**

```bash
ruff check . --fix
ruff format .
```
Expected: 0 errors

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests pass (702 existing + ~10 Task 1 + ~9 Task 2 + ~6 Task 3 = ~727+ tests)

**Step 4: Verify MCP tool count**

```bash
python -c "from server import mcp; print(f'MCP tools: {len(mcp._tool_manager._tools)}')"
```
Expected: `MCP tools: 19`

**Step 5: Commit**

```bash
git add tests/test_training_and_persistence.py tests/test_synthetic_data.py
git add -A  # catch any lint fixes
git commit -m "chore: Phase E complete - update tool count to 19, lint fixes"
```
