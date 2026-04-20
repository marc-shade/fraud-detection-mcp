# Phase 1: Foundation Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix server.py correctness, establish the _impl testable pattern, fix CI to actually enforce quality.

**Architecture:** Refactor all @mcp.tool() functions into thin wrappers over _impl functions. Fix validation, dead code, memory leaks, and key consistency. Rewrite tests to call _impl directly. Fix CI to run tests and fail on errors.

**Tech Stack:** Python 3.10+, FastMCP, scikit-learn, XGBoost, NetworkX, pytest

---

### Task 1: Fix Input Validation — bool and nan/inf bypass

**Files:**
- Modify: `server.py:43-67` (validate_transaction_data)
- Test: `tests/test_validation.py`

**Step 1: Write the failing tests**

Add to `tests/test_validation.py`:

```python
import math
from server import validate_transaction_data

def test_boolean_amount_rejected():
    valid, msg = validate_transaction_data({"amount": True})
    assert not valid
    assert "boolean" in msg.lower() or "numeric" in msg.lower()

def test_nan_amount_rejected():
    valid, msg = validate_transaction_data({"amount": float('nan')})
    assert not valid

def test_inf_amount_rejected():
    valid, msg = validate_transaction_data({"amount": float('inf')})
    assert not valid

def test_negative_inf_amount_rejected():
    valid, msg = validate_transaction_data({"amount": float('-inf')})
    assert not valid
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation.py::test_boolean_amount_rejected tests/test_validation.py::test_nan_amount_rejected tests/test_validation.py::test_inf_amount_rejected tests/test_validation.py::test_negative_inf_amount_rejected -v`
Expected: FAIL (bool passes as int, nan/inf pass range checks)

**Step 3: Fix validate_transaction_data in server.py:49-56**

Replace lines 49-56:

```python
    if "amount" in data:
        amount = data["amount"]
        if isinstance(amount, bool):
            return False, "amount must be numeric, not boolean"
        if not isinstance(amount, (int, float)):
            return False, "amount must be numeric"
        if math.isnan(amount) or math.isinf(amount):
            return False, "amount must be a finite number"
        if amount < 0:
            return False, "amount cannot be negative"
        if amount > 1_000_000_000:  # 1 billion limit
            return False, "amount exceeds maximum allowed value"
```

Also add `import math` at the top of server.py (after line 9).

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server.py tests/test_validation.py
git commit -m "fix: reject bool, nan, and inf in amount validation"
```

---

### Task 2: Remove Dead Imports and Dead XGBoost Training

**Files:**
- Modify: `server.py:7-8,24-26,247-265`

**Step 1: Verify torch is unused**

Run: `grep -n 'torch\|nn\.' server.py | grep -v import | grep -v '#'`
Expected: No non-import usages of torch or nn in server.py

**Step 2: Remove dead imports and dead XGBoost training**

Remove lines 7-8 (unused asyncio, json):
```python
import asyncio
import json
```

Remove lines 24-26:
```python
# Deep learning for autoencoders
import torch
import torch.nn as nn
```

Replace `TransactionAnalyzer._initialize_models` (lines 247-265) with:

```python
    def _initialize_models(self):
        """Initialize transaction analysis models"""
        # Fit Isolation Forest with dummy transaction data
        # Transaction features: amount, log_amount, hour, weekday, day, location_hash, merchant_hash, payment_risk
        dummy_transaction_features = np.random.randn(100, 8) * 100 + 500
        self.isolation_forest.fit(dummy_transaction_features)
```

The `self.xgb_model = None` in `__init__` (line 244) stays as-is — it's already set to None before `_initialize_models` overwrites it.

Change line 244 to stay as `self.xgb_model = None` and don't reassign it in `_initialize_models`.

**Step 3: Also remove unused `from pathlib import Path`**

Check: `grep -n 'Path' server.py | grep -v import`
If no usages, remove `from pathlib import Path` (line 12).

**Step 4: Run existing tests to verify nothing breaks**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS (torch was never used, xgb_model was never called)

**Step 5: Commit**

```bash
git add server.py
git commit -m "fix: remove unused torch/json/asyncio imports, dead XGBoost training"
```

---

### Task 3: Fix Hash Encoding to Be Deterministic

**Files:**
- Modify: `server.py:298-338` (_extract_transaction_features)
- Test: `tests/test_transaction_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_transaction_analysis.py`:

```python
from server import TransactionAnalyzer

def test_feature_extraction_deterministic():
    """Features must be identical across calls (hash must be deterministic)."""
    analyzer = TransactionAnalyzer()
    txn = {"amount": 100, "location": "USA", "merchant": "Amazon", "payment_method": "credit_card"}
    f1 = analyzer._extract_transaction_features(txn)
    f2 = analyzer._extract_transaction_features(txn)
    assert f1 == f2, "Feature extraction must be deterministic"
```

**Step 2: Run to verify it might fail (depends on PYTHONHASHSEED)**

Run: `PYTHONHASHSEED=random python -m pytest tests/test_transaction_analysis.py::test_feature_extraction_deterministic -v`
Expected: May pass (same process = same hash seed), but the fix is still needed for cross-process consistency.

**Step 3: Replace hash() with hashlib.md5 in server.py**

Add `import hashlib` after the `import math` line.

Replace lines 319-325:

```python
        # Location-based features (deterministic hash)
        location = transaction.get('location', '')
        features.append(int(hashlib.md5(location.encode()).hexdigest(), 16) % 1000)

        # Merchant-based features (deterministic hash)
        merchant = transaction.get('merchant', '')
        features.append(int(hashlib.md5(merchant.encode()).hexdigest(), 16) % 1000)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_transaction_analysis.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server.py tests/test_transaction_analysis.py
git commit -m "fix: use deterministic hashlib.md5 for feature encoding"
```

---

### Task 4: Fix Geographic Risk — Config-Driven Exact Match

**Files:**
- Modify: `server.py:340-367` (_identify_risk_factors)
- Modify: `config.py:49-51` (HIGH_RISK_LOCATIONS default)
- Test: `tests/test_transaction_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_transaction_analysis.py`:

```python
from server import TransactionAnalyzer

def test_geographic_risk_exact_match_not_substring():
    """'China Palace Restaurant' should NOT trigger geographic risk."""
    analyzer = TransactionAnalyzer()
    txn = {"amount": 100, "location": "China Palace Restaurant, Seattle"}
    features = analyzer._extract_transaction_features(txn)
    risk_factors = analyzer._identify_risk_factors(txn, features)
    assert "high_risk_geographic_location" not in risk_factors

def test_geographic_risk_exact_match_triggers():
    """Exact match on 'unknown' should trigger."""
    analyzer = TransactionAnalyzer()
    txn = {"amount": 100, "location": "unknown"}
    features = analyzer._extract_transaction_features(txn)
    risk_factors = analyzer._identify_risk_factors(txn, features)
    assert "high_risk_geographic_location" in risk_factors
```

**Step 2: Run to verify first test fails**

Run: `python -m pytest tests/test_transaction_analysis.py::test_geographic_risk_exact_match_not_substring -v`
Expected: FAIL (substring match catches "china" in "China Palace")

**Step 3: Update config.py defaults**

Replace `config.py` line 49-51:

```python
    HIGH_RISK_LOCATIONS: list = Field(
        default_factory=lambda: ["unknown"]
    )
```

**Step 4: Fix _identify_risk_factors in server.py**

Add at top of server.py (after other imports):

```python
from config import get_config
```

Replace lines 361-365 in `_identify_risk_factors`:

```python
        # Geographic risk (config-driven, exact match)
        location = transaction.get('location', '').lower().strip()
        app_config = get_config()
        if location in app_config.HIGH_RISK_LOCATIONS:
            risk_factors.append("high_risk_geographic_location")
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_transaction_analysis.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add server.py config.py tests/test_transaction_analysis.py
git commit -m "fix: config-driven exact match for geographic risk locations"
```

---

### Task 5: Fix NetworkAnalyzer — Bare Except and Memory Leak

**Files:**
- Modify: `server.py:369-479` (NetworkAnalyzer class)
- Test: `tests/test_network_analysis.py`

**Step 1: Write the failing test for graph size cap**

Add to `tests/test_network_analysis.py`:

```python
from server import NetworkAnalyzer

def test_graph_size_capped():
    """Graph should not exceed MAX_GRAPH_NODES."""
    analyzer = NetworkAnalyzer()
    # Add more nodes than the cap
    for i in range(12000):
        analyzer._update_graph(f"entity_{i}", [])
    assert len(analyzer.transaction_graph.nodes) <= analyzer.MAX_GRAPH_NODES
```

**Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_network_analysis.py::test_graph_size_capped -v`
Expected: FAIL (no cap exists, graph grows to 12000)

**Step 3: Fix NetworkAnalyzer**

Replace the `NetworkAnalyzer` class (lines 369-479):

```python
class NetworkAnalyzer:
    """Graph-based network analysis for fraud ring detection"""

    MAX_GRAPH_NODES = 10000

    def __init__(self):
        self.transaction_graph = nx.Graph()
        self._node_order = deque()

    def analyze_network_risk(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network patterns for fraud ring detection"""
        try:
            entity_id = entity_data.get('entity_id')
            connections = entity_data.get('connections', [])

            if not entity_id:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "no_entity_id"}

            # Add entity and connections to graph
            self._update_graph(entity_id, connections)

            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(entity_id)

            # Detect suspicious patterns
            risk_patterns = self._detect_risk_patterns(entity_id, network_metrics)

            # Calculate network risk score
            risk_score = self._calculate_network_risk_score(network_metrics, risk_patterns)

            return {
                "risk_score": float(risk_score),
                "network_metrics": network_metrics,
                "risk_patterns": risk_patterns,
                "confidence": 0.82,
                "analysis_type": "network_analysis"
            }

        except Exception as e:
            logger.error(f"Network analysis error: {e}")
            return {"risk_score": 0.0, "confidence": 0.0, "status": "error", "error": str(e)}

    def _update_graph(self, entity_id: str, connections: List[Dict]):
        """Update the transaction graph with new entity and connections"""
        # Track insertion order for FIFO eviction
        if entity_id not in self.transaction_graph:
            self._node_order.append(entity_id)
        self.transaction_graph.add_node(entity_id)

        for connection in connections:
            connected_entity = connection.get('entity_id')
            if connected_entity:
                if connected_entity not in self.transaction_graph:
                    self._node_order.append(connected_entity)
                self.transaction_graph.add_edge(
                    entity_id,
                    connected_entity,
                    weight=connection.get('strength', 1.0),
                    transaction_count=connection.get('transaction_count', 1)
                )

        # Evict oldest nodes if over cap
        while len(self.transaction_graph.nodes) > self.MAX_GRAPH_NODES:
            oldest = self._node_order.popleft()
            if oldest in self.transaction_graph:
                self.transaction_graph.remove_node(oldest)

    def _calculate_network_metrics(self, entity_id: str) -> Dict[str, float]:
        """Calculate network centrality and connectivity metrics"""
        if entity_id not in self.transaction_graph:
            return {}

        # Basic metrics
        degree = self.transaction_graph.degree(entity_id)
        clustering = nx.clustering(self.transaction_graph, entity_id)

        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.transaction_graph).get(entity_id, 0)
            closeness = nx.closeness_centrality(self.transaction_graph).get(entity_id, 0)
        except Exception as e:
            logger.error(f"Centrality calculation error: {e}")
            betweenness = 0
            closeness = 0

        return {
            "degree": float(degree),
            "clustering_coefficient": float(clustering),
            "betweenness_centrality": float(betweenness),
            "closeness_centrality": float(closeness)
        }

    def _detect_risk_patterns(self, entity_id: str, metrics: Dict[str, float]) -> List[str]:
        """Detect suspicious network patterns"""
        patterns = []

        # High connectivity risk
        if metrics.get("degree", 0) > 50:
            patterns.append("unusually_high_connectivity")

        # Hub behavior (high betweenness centrality)
        if metrics.get("betweenness_centrality", 0) > 0.1:
            patterns.append("potential_fraud_hub")

        # Tight clustering (potential fraud ring)
        if metrics.get("clustering_coefficient", 0) > 0.8:
            patterns.append("tight_clustering_pattern")

        return patterns

    def _calculate_network_risk_score(self, metrics: Dict[str, float], patterns: List[str]) -> float:
        """Calculate overall network risk score"""
        base_score = 0.0

        # Degree risk
        degree = metrics.get("degree", 0)
        base_score += min(0.3, degree / 100)

        # Centrality risk
        betweenness = metrics.get("betweenness_centrality", 0)
        base_score += min(0.4, betweenness * 2)

        # Pattern risk
        pattern_score = len(patterns) * 0.2

        return min(1.0, base_score + pattern_score)
```

Also add `from collections import deque` at the top of server.py.

**Step 4: Run tests**

Run: `python -m pytest tests/test_network_analysis.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server.py tests/test_network_analysis.py
git commit -m "fix: cap network graph at 10K nodes with FIFO eviction, fix bare except"
```

---

### Task 6: Extract _impl Functions and Wire analyze_transaction MCP Tool

This is the largest task. We extract logic from all 4 existing `@mcp.tool()` functions into `_impl` functions, and add the missing `analyze_transaction` MCP tool wrapper.

**Files:**
- Modify: `server.py:487-872` (entire MCP tools section)

**Step 1: Replace the entire MCP tools section (lines 487-872)**

The existing `analyze_transaction_impl` (lines 491-575) stays as-is.

Extract the other 4 tools into `_impl` functions and make the `@mcp.tool()` decorators thin wrappers. Also standardize `all_detected_anomalies` -> `detected_anomalies` in `generate_risk_score_impl` and `explain_decision_impl`.

Replace everything from line 487 to 872 with:

```python
# =============================================================================
# Implementation Functions (testable, import these in tests)
# =============================================================================

def analyze_transaction_impl(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of comprehensive transaction fraud analysis"""
    try:
        # Validate inputs
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {"error": f"Invalid transaction data: {msg}", "status": "validation_failed"}

        if behavioral_data:
            valid, msg = validate_behavioral_data(behavioral_data)
            if not valid:
                return {"error": f"Invalid behavioral data: {msg}", "status": "validation_failed"}

        # Primary transaction analysis
        transaction_result = transaction_analyzer.analyze_transaction(transaction_data)

        results = {
            "transaction_analysis": transaction_result,
            "overall_risk_score": transaction_result.get("risk_score", 0.0),
            "risk_level": "LOW",
            "detected_anomalies": [],
            "explanations": [],
            "recommended_actions": []
        }

        # Add transaction risk factors
        risk_factors = transaction_result.get("risk_factors", [])
        results["detected_anomalies"].extend(risk_factors)

        # Behavioral analysis if requested
        if include_behavioral and behavioral_data:
            behavioral_result = {}

            if "keystroke_dynamics" in behavioral_data:
                keystroke_result = behavioral_analyzer.analyze_keystroke_dynamics(
                    behavioral_data["keystroke_dynamics"]
                )
                behavioral_result["keystroke"] = keystroke_result

                if keystroke_result.get("is_anomaly"):
                    results["detected_anomalies"].append("abnormal_keystroke_dynamics")
                    results["overall_risk_score"] = min(1.0, results["overall_risk_score"] + 0.2)

            results["behavioral_analysis"] = behavioral_result

        # Determine risk level
        risk_score = results["overall_risk_score"]
        if risk_score >= 0.8:
            results["risk_level"] = "CRITICAL"
            results["recommended_actions"] = ["block_transaction", "require_manual_review"]
        elif risk_score >= 0.6:
            results["risk_level"] = "HIGH"
            results["recommended_actions"] = ["require_additional_verification", "flag_for_review"]
        elif risk_score >= 0.4:
            results["risk_level"] = "MEDIUM"
            results["recommended_actions"] = ["monitor_closely", "collect_additional_data"]
        else:
            results["risk_level"] = "LOW"
            results["recommended_actions"] = ["allow_transaction"]

        # Generate explanation
        if results["detected_anomalies"]:
            explanation = f"Transaction flagged due to: {', '.join(results['detected_anomalies'])}"
        else:
            explanation = "Transaction appears normal with no significant risk factors detected"

        results["explanation"] = explanation
        results["analysis_timestamp"] = datetime.now().isoformat()
        results["model_version"] = "v2.1.0"

        return results

    except Exception as e:
        logger.error(f"Transaction analysis failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def detect_behavioral_anomaly_impl(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of behavioral biometrics anomaly detection"""
    try:
        # Validate input
        valid, msg = validate_behavioral_data(behavioral_data)
        if not valid:
            return {"error": f"Invalid behavioral data: {msg}", "status": "validation_failed"}

        results = {
            "overall_anomaly_score": 0.0,
            "behavioral_analyses": {},
            "detected_anomalies": [],
            "confidence": 0.0
        }

        total_confidence = 0.0
        analysis_count = 0

        # Keystroke dynamics analysis
        if "keystroke_dynamics" in behavioral_data:
            keystroke_result = behavioral_analyzer.analyze_keystroke_dynamics(
                behavioral_data["keystroke_dynamics"]
            )
            results["behavioral_analyses"]["keystroke"] = keystroke_result

            if keystroke_result.get("is_anomaly"):
                results["detected_anomalies"].append("keystroke_anomaly")

            results["overall_anomaly_score"] = max(
                results["overall_anomaly_score"],
                keystroke_result.get("risk_score", 0.0)
            )

            total_confidence += keystroke_result.get("confidence", 0.0)
            analysis_count += 1

        # Calculate average confidence
        if analysis_count > 0:
            results["confidence"] = total_confidence / analysis_count

        results["analysis_timestamp"] = datetime.now().isoformat()

        return results

    except Exception as e:
        logger.error(f"Behavioral analysis failed: {e}")
        return {
            "error": str(e),
            "overall_anomaly_score": 0.0,
            "status": "analysis_failed"
        }


def assess_network_risk_impl(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of network-based risk assessment"""
    return network_analyzer.analyze_network_risk(entity_data)


def generate_risk_score_impl(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of comprehensive risk score generation"""
    try:
        # Perform all analyses
        transaction_analysis = transaction_analyzer.analyze_transaction(transaction_data)

        # Initialize comprehensive results
        comprehensive_result = {
            "overall_risk_score": 0.0,
            "component_scores": {
                "transaction": transaction_analysis.get("risk_score", 0.0)
            },
            "risk_level": "LOW",
            "confidence": 0.0,
            "detected_anomalies": [],
            "comprehensive_explanation": "",
            "recommended_actions": []
        }

        scores = [transaction_analysis.get("risk_score", 0.0)]
        confidences = [transaction_analysis.get("confidence", 0.0)]

        # Add transaction anomalies
        comprehensive_result["detected_anomalies"].extend(
            transaction_analysis.get("risk_factors", [])
        )

        # Behavioral analysis
        if behavioral_data:
            behavioral_analysis = behavioral_analyzer.analyze_keystroke_dynamics(
                behavioral_data.get("keystroke_dynamics", [])
            )
            behavioral_score = behavioral_analysis.get("risk_score", 0.0)
            comprehensive_result["component_scores"]["behavioral"] = behavioral_score
            scores.append(behavioral_score)
            confidences.append(behavioral_analysis.get("confidence", 0.0))

            if behavioral_analysis.get("is_anomaly"):
                comprehensive_result["detected_anomalies"].append("behavioral_anomaly")

        # Network analysis
        if network_data:
            network_analysis = network_analyzer.analyze_network_risk(network_data)
            network_score = network_analysis.get("risk_score", 0.0)
            comprehensive_result["component_scores"]["network"] = network_score
            scores.append(network_score)
            confidences.append(network_analysis.get("confidence", 0.0))

            comprehensive_result["detected_anomalies"].extend(
                network_analysis.get("risk_patterns", [])
            )

        # Calculate weighted overall score
        if len(scores) == 1:
            overall_score = scores[0]
        elif len(scores) == 2:
            overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            overall_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)

        comprehensive_result["overall_risk_score"] = float(overall_score)
        comprehensive_result["confidence"] = float(np.mean(confidences))

        # Determine risk level and actions
        if overall_score >= 0.8:
            comprehensive_result["risk_level"] = "CRITICAL"
            comprehensive_result["recommended_actions"] = [
                "block_transaction",
                "require_manual_review",
                "investigate_account"
            ]
        elif overall_score >= 0.6:
            comprehensive_result["risk_level"] = "HIGH"
            comprehensive_result["recommended_actions"] = [
                "require_additional_verification",
                "flag_for_review",
                "monitor_account"
            ]
        elif overall_score >= 0.4:
            comprehensive_result["risk_level"] = "MEDIUM"
            comprehensive_result["recommended_actions"] = [
                "monitor_closely",
                "collect_additional_data"
            ]
        else:
            comprehensive_result["risk_level"] = "LOW"
            comprehensive_result["recommended_actions"] = ["allow_transaction"]

        # Generate comprehensive explanation
        if comprehensive_result["detected_anomalies"]:
            explanation = (
                f"Risk assessment detected {len(comprehensive_result['detected_anomalies'])} "
                f"anomalies: {', '.join(comprehensive_result['detected_anomalies'])}. "
                f"Combined analysis suggests {comprehensive_result['risk_level']} risk level."
            )
        else:
            explanation = (
                f"Comprehensive analysis found no significant anomalies. "
                f"Risk level assessed as {comprehensive_result['risk_level']}."
            )

        comprehensive_result["comprehensive_explanation"] = explanation
        comprehensive_result["analysis_timestamp"] = datetime.now().isoformat()
        comprehensive_result["analysis_components"] = list(comprehensive_result["component_scores"].keys())

        return comprehensive_result

    except Exception as e:
        logger.error(f"Comprehensive risk assessment failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def explain_decision_impl(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of explainable AI reasoning for fraud decisions"""
    try:
        explanation = {
            "decision_summary": "",
            "key_factors": [],
            "algorithm_contributions": {},
            "confidence_breakdown": {},
            "alternative_scenarios": [],
            "explanation_timestamp": datetime.now().isoformat()
        }

        risk_score = analysis_result.get("overall_risk_score", 0.0)
        risk_level = analysis_result.get("risk_level", "UNKNOWN")
        detected_anomalies = analysis_result.get("detected_anomalies", [])

        # Decision summary
        explanation["decision_summary"] = (
            f"The fraud detection system assessed this case as {risk_level} risk "
            f"with a confidence score of {risk_score:.2f}. "
            f"This decision was based on analysis of {len(detected_anomalies)} risk factors."
        )

        # Key contributing factors
        if detected_anomalies:
            explanation["key_factors"] = [
                {
                    "factor": anomaly,
                    "impact": "high" if "high" in anomaly else "medium",
                    "description": f"Detected pattern: {anomaly.replace('_', ' ')}"
                }
                for anomaly in detected_anomalies
            ]

        # Algorithm contributions
        component_scores = analysis_result.get("component_scores", {})
        if component_scores:
            for component, score in component_scores.items():
                if component == "transaction":
                    weight = 0.6 if len(component_scores) > 1 else 1.0
                elif component == "behavioral":
                    weight = 0.3 if len(component_scores) == 3 else 0.4
                else:  # network
                    weight = 0.2 if len(component_scores) == 3 else 0.4

                explanation["algorithm_contributions"][component] = {
                    "score": float(score),
                    "weight": float(weight),
                    "contribution": f"{weight * 100:.1f}% of final decision"
                }

        # Confidence breakdown
        explanation["confidence_breakdown"] = {
            "model_confidence": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low",
            "data_quality": "Good" if len(detected_anomalies) > 0 else "Limited",
            "recommendation_strength": "Strong" if risk_score > 0.8 or risk_score < 0.2 else "Moderate"
        }

        # Alternative scenarios
        if risk_score > 0.5:
            explanation["alternative_scenarios"].append(
                "If behavioral patterns were more consistent with user profile, "
                "risk score could be reduced by 0.2-0.3 points"
            )

        if "high_amount_transaction" in detected_anomalies:
            explanation["alternative_scenarios"].append(
                "For smaller transaction amounts, this would likely be classified as low risk"
            )

        return explanation

    except Exception as e:
        logger.error(f"Decision explanation failed: {e}")
        return {
            "error": str(e),
            "decision_summary": "Unable to generate explanation",
            "status": "explanation_failed"
        }


# =============================================================================
# MCP Tool Wrappers (thin delegates to _impl functions)
# =============================================================================

@mcp.tool()
def analyze_transaction(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive transaction fraud analysis with optional behavioral biometrics.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp, etc.)
        include_behavioral: Whether to include behavioral analysis
        behavioral_data: Behavioral biometrics data (keystroke dynamics, mouse movements)

    Returns:
        Fraud analysis results with risk score, level, anomalies, and recommendations
    """
    return analyze_transaction_impl(transaction_data, include_behavioral, behavioral_data)


@mcp.tool()
def detect_behavioral_anomaly(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze behavioral biometrics for anomaly detection.

    Args:
        behavioral_data: Behavioral patterns (keystroke dynamics, mouse movements, etc.)

    Returns:
        Behavioral anomaly analysis results
    """
    return detect_behavioral_anomaly_impl(behavioral_data)


@mcp.tool()
def assess_network_risk(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze network patterns for fraud ring detection.

    Args:
        entity_data: Entity information and network connections

    Returns:
        Network-based risk assessment
    """
    return assess_network_risk_impl(entity_data)


@mcp.tool()
def generate_risk_score(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive risk score combining all analysis methods.

    Args:
        transaction_data: Transaction details
        behavioral_data: Behavioral biometrics data
        network_data: Network connection data

    Returns:
        Comprehensive risk assessment with detailed scoring
    """
    return generate_risk_score_impl(transaction_data, behavioral_data, network_data)


@mcp.tool()
def explain_decision(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide explainable AI reasoning for fraud detection decisions.

    Args:
        analysis_result: Previous analysis result to explain

    Returns:
        Detailed explanation of the decision-making process
    """
    return explain_decision_impl(analysis_result)


if __name__ == "__main__":
    mcp.run()
```

**Step 2: Run existing tests (they will fail because they import from server_wrapper)**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | head -30`
Expected: ImportError or failures related to server_wrapper imports. This is expected — Task 7 fixes the tests.

**Step 3: Commit server.py changes**

```bash
git add server.py
git commit -m "refactor: extract _impl functions, wire analyze_transaction MCP tool, standardize detected_anomalies key"
```

---

### Task 7: Refactor Tests — Delete server_wrapper, Import _impl Directly

**Files:**
- Delete: `tests/server_wrapper.py`
- Modify: `tests/test_mcp_tools.py`
- Modify: `tests/test_integration.py`
- Modify: `tests/test_error_handling.py`
- Modify: `tests/test_keystroke_analysis.py`
- Modify: `tests/test_transaction_analysis.py`
- Modify: `tests/test_network_analysis.py`
- Modify: `tests/test_validation.py`
- Modify: `tests/conftest.py:317-323` (fix reset_ml_models)

**Step 1: Delete server_wrapper.py**

```bash
rm tests/server_wrapper.py
```

**Step 2: Update all test imports**

In every test file, replace:
```python
from tests.server_wrapper import analyze_transaction
# or
from server_wrapper import analyze_transaction
# or
sys.path.insert(0, ...)
from server_wrapper import ...
```

With direct imports from server:
```python
from server import (
    analyze_transaction_impl,
    detect_behavioral_anomaly_impl,
    assess_network_risk_impl,
    generate_risk_score_impl,
    explain_decision_impl,
    validate_transaction_data,
    validate_behavioral_data,
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer,
    behavioral_analyzer,
    transaction_analyzer,
    network_analyzer,
)
```

Then replace all function call references:
- `analyze_transaction(` -> `analyze_transaction_impl(`
- `detect_behavioral_anomaly(` -> `detect_behavioral_anomaly_impl(`
- `assess_network_risk(` -> `assess_network_risk_impl(`
- `generate_risk_score(` -> `generate_risk_score_impl(`
- `explain_decision(` -> `explain_decision_impl(`

Also update any references to `all_detected_anomalies` in test assertions to `detected_anomalies` (matching the key standardization from Task 6).

**Step 3: Fix reset_ml_models in conftest.py**

Replace the `reset_ml_models` fixture (conftest.py lines 317-323):

```python
@pytest.fixture(autouse=True)
def reset_ml_models():
    """Reset ML models before each test to ensure isolation"""
    from server import network_analyzer
    import networkx as nx
    from collections import deque
    yield
    # Reset network graph state after each test
    network_analyzer.transaction_graph = nx.Graph()
    network_analyzer._node_order = deque()
```

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add -A tests/
git commit -m "refactor: delete server_wrapper, test _impl functions directly, fix test isolation"
```

---

### Task 8: Fix CI Pipeline

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Replace ci.yml entirely**

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short --cov=server --cov-report=term-missing --cov-fail-under=60

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install ruff
      - run: ruff check . --output-format=github
      - run: ruff format --check .

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install bandit
      - run: bandit -r . -x ./tests,./venv -ll

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: |
          pip install mypy
          pip install -r requirements.txt
      - run: mypy server.py --ignore-missing-imports
```

**Step 2: Verify locally that tests pass**

Run: `python -m pytest tests/ -v --tb=short --cov=server --cov-report=term-missing --cov-fail-under=60`
Expected: PASS with coverage >= 60%

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "fix: CI now runs tests, enforces coverage, removes || true, adds Python matrix"
```

---

### Task 9: Final Verification

**Step 1: Run full test suite with coverage**

Run: `python -m pytest tests/ -v --cov=server --cov-report=term-missing`

**Step 2: Verify all 5 MCP tools are registered**

Run: `python -c "from server import mcp; print([t.name for t in mcp._tool_manager._tools.values()])"`
Expected: List containing `analyze_transaction`, `detect_behavioral_anomaly`, `assess_network_risk`, `generate_risk_score`, `explain_decision`

**Step 3: Verify no bare except remains**

Run: `grep -n 'except:' server.py`
Expected: No output (no bare except statements)

**Step 4: Verify no torch imports**

Run: `grep -n 'torch' server.py`
Expected: No output

**Step 5: Commit any final adjustments, then tag**

```bash
git tag -a v1.1.0 -m "Phase 1: Foundation fixes complete"
```
