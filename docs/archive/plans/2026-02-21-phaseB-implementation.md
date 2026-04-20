# Phase B: Agent Identity Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `verify_agent_identity` MCP tool with agent credential validation, a local identity registry, and identity-aware risk scoring.

**Architecture:** Add `AgentIdentityRegistry` (JSON-file-backed, thread-safe) and `AgentIdentityVerifier` (validates API keys, JWT tokens, and X.509 certs) to `server.py`. Expose as the 15th MCP tool. When agent traffic is detected in `generate_risk_score_impl`, perform identity verification and apply a trust bonus/penalty.

**Tech Stack:** Python, FastMCP, json, threading, base64, jwt (optional), pytest

---

### Task 1: Add AgentIdentityRegistry class

**Files:**
- Modify: `server.py` (add class after `TrafficClassifier`, before `_monitored`)

**Step 1: Write the test**

Create `tests/test_agent_identity.py`:

```python
"""Tests for AgentIdentityRegistry and AgentIdentityVerifier"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pytest
import tempfile
from pathlib import Path
from server import AgentIdentityRegistry


class TestAgentIdentityRegistry:
    """Tests for the JSON-backed agent identity registry"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        self.registry = AgentIdentityRegistry(self.registry_path)

    @pytest.mark.unit
    def test_register_new_agent(self):
        """Register a new agent and retrieve it"""
        self.registry.register("stripe-acp:agent-1", agent_type="stripe_acp")
        agent = self.registry.lookup("stripe-acp:agent-1")
        assert agent is not None
        assert agent["agent_type"] == "stripe_acp"
        assert agent["transaction_count"] == 0
        assert agent["trust_score"] == 0.5  # default neutral trust

    @pytest.mark.unit
    def test_lookup_missing_agent(self):
        """Lookup non-existent agent returns None"""
        assert self.registry.lookup("nonexistent") is None

    @pytest.mark.unit
    def test_update_agent_transaction_count(self):
        """Recording a transaction increments count and updates last_seen"""
        self.registry.register("agent-1", agent_type="visa_tap")
        self.registry.record_transaction("agent-1")
        agent = self.registry.lookup("agent-1")
        assert agent["transaction_count"] == 1

    @pytest.mark.unit
    def test_update_trust_score(self):
        """Trust score can be updated"""
        self.registry.register("agent-1")
        self.registry.update_trust("agent-1", 0.9)
        agent = self.registry.lookup("agent-1")
        assert agent["trust_score"] == 0.9

    @pytest.mark.unit
    def test_trust_score_clamped(self):
        """Trust score clamped to [0, 1]"""
        self.registry.register("agent-1")
        self.registry.update_trust("agent-1", 1.5)
        assert self.registry.lookup("agent-1")["trust_score"] == 1.0
        self.registry.update_trust("agent-1", -0.5)
        assert self.registry.lookup("agent-1")["trust_score"] == 0.0

    @pytest.mark.unit
    def test_persistence(self):
        """Registry persists to disk and loads back"""
        self.registry.register("agent-1", agent_type="openai")
        self.registry.record_transaction("agent-1")
        # Create new registry from same path
        registry2 = AgentIdentityRegistry(self.registry_path)
        agent = registry2.lookup("agent-1")
        assert agent is not None
        assert agent["agent_type"] == "openai"
        assert agent["transaction_count"] == 1

    @pytest.mark.unit
    def test_registry_creates_file(self):
        """Registry creates file on first write"""
        assert not self.registry_path.exists()
        self.registry.register("agent-1")
        assert self.registry_path.exists()

    @pytest.mark.unit
    def test_register_duplicate_no_overwrite(self):
        """Registering same agent twice doesn't reset data"""
        self.registry.register("agent-1", agent_type="stripe_acp")
        self.registry.record_transaction("agent-1")
        self.registry.register("agent-1", agent_type="stripe_acp")
        agent = self.registry.lookup("agent-1")
        assert agent["transaction_count"] == 1  # not reset

    @pytest.mark.unit
    def test_list_agents(self):
        """List all registered agents"""
        self.registry.register("agent-1")
        self.registry.register("agent-2")
        agents = self.registry.list_agents()
        assert len(agents) == 2
        assert "agent-1" in agents
        assert "agent-2" in agents
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_identity.py::TestAgentIdentityRegistry -v`
Expected: FAIL (AgentIdentityRegistry not defined)

**Step 3: Implement AgentIdentityRegistry**

In `server.py`, after `traffic_classifier = TrafficClassifier()` (line 1021) and before `_monitored` (line 1024), add:

```python
# =============================================================================
# Agent Identity Registry
# =============================================================================

class AgentIdentityRegistry:
    """Thread-safe JSON-backed registry of known AI agent identities.

    Tracks agent identifiers, types, trust scores, and transaction history.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or Path("data/agent_registry.json")
        self._lock = threading.Lock()
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load registry from disk if it exists."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._agents = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._agents = {}

    def _save(self):
        """Persist registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._agents, f, indent=2, default=str)

    def register(self, agent_id: str, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Register a new agent or return existing entry."""
        with self._lock:
            if agent_id in self._agents:
                return self._agents[agent_id]
            entry = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "transaction_count": 0,
                "trust_score": 0.5,
            }
            self._agents[agent_id] = entry
            self._save()
            return entry

    def lookup(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Look up an agent by identifier."""
        with self._lock:
            return self._agents.get(agent_id)

    def record_transaction(self, agent_id: str):
        """Record a transaction for an agent, incrementing count."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["transaction_count"] += 1
                self._agents[agent_id]["last_seen"] = datetime.now().isoformat()
                self._save()

    def update_trust(self, agent_id: str, trust_score: float):
        """Update an agent's trust score (clamped to [0, 1])."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["trust_score"] = max(0.0, min(1.0, trust_score))
                self._save()

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered agents."""
        with self._lock:
            return dict(self._agents)
```

Add `import json` to the imports at the top of `server.py` if not already present. (It is already imported inside `_generate_cache_key` -- move to module level.)

Then add the singleton after the class:

```python
agent_registry = AgentIdentityRegistry()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_identity.py::TestAgentIdentityRegistry -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_agent_identity.py
git commit -m "feat: Add AgentIdentityRegistry with JSON persistence"
```

---

### Task 2: Add AgentIdentityVerifier class

**Files:**
- Modify: `server.py` (add class after `AgentIdentityRegistry`)

**Step 1: Write the test**

Add to `tests/test_agent_identity.py`:

```python
from server import AgentIdentityVerifier, AgentIdentityRegistry


class TestAgentIdentityVerifier:
    """Tests for agent credential validation"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmpdir) / "test_registry.json"
        self.registry = AgentIdentityRegistry(self.registry_path)
        self.verifier = AgentIdentityVerifier(self.registry)

    @pytest.mark.unit
    def test_verify_with_valid_agent_id(self):
        """Known agent in registry is verified"""
        self.registry.register("stripe-acp:agent-1", agent_type="stripe_acp")
        self.registry.update_trust("stripe-acp:agent-1", 0.8)
        result = self.verifier.verify(agent_identifier="stripe-acp:agent-1")
        assert result["verified"] is True
        assert result["trust_score"] >= 0.5
        assert result["identity"]["agent_id"] == "stripe-acp:agent-1"

    @pytest.mark.unit
    def test_verify_unknown_agent(self):
        """Unknown agent not in registry gets low trust"""
        result = self.verifier.verify(agent_identifier="unknown-agent")
        assert result["verified"] is False
        assert result["trust_score"] < 0.5
        assert "not_in_registry" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_api_key_valid_format(self):
        """Valid API key format passes basic validation"""
        result = self.verifier.verify(
            agent_identifier="test-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        )
        assert "invalid_key_format" not in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_api_key_too_short(self):
        """API key that's too short gets a warning"""
        result = self.verifier.verify(
            agent_identifier="test-agent",
            api_key="short"
        )
        assert "invalid_key_format" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_jwt_expired(self):
        """Expired JWT token gets a warning"""
        import base64
        # Create a fake expired JWT (header.payload.signature)
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload = base64.urlsafe_b64encode(b'{"exp":1000000000,"sub":"agent-1"}').rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(agent_identifier="agent-1", token=token)
        assert "token_expired" in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_jwt_valid_expiry(self):
        """Non-expired JWT doesn't get expiry warning"""
        import base64
        import time
        future_exp = int(time.time()) + 3600  # 1 hour from now
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "agent-1"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(agent_identifier="agent-1", token=token)
        assert "token_expired" not in result["warnings"]

    @pytest.mark.unit
    def test_verify_with_malformed_token(self):
        """Malformed token gets a warning"""
        result = self.verifier.verify(agent_identifier="agent-1", token="not-a-jwt")
        assert "token_parse_error" in result["warnings"]

    @pytest.mark.unit
    def test_verify_result_structure(self):
        """Result contains all required fields"""
        result = self.verifier.verify(agent_identifier="test-agent")
        assert "verified" in result
        assert "identity" in result
        assert "trust_score" in result
        assert "warnings" in result
        assert isinstance(result["verified"], bool)
        assert isinstance(result["trust_score"], float)
        assert isinstance(result["warnings"], list)

    @pytest.mark.unit
    def test_verify_no_identifier(self):
        """No identifier at all returns unverified"""
        result = self.verifier.verify()
        assert result["verified"] is False
        assert "no_identifier" in result["warnings"]

    @pytest.mark.unit
    def test_verify_auto_registers_new_agent(self):
        """First-time agent gets auto-registered in registry"""
        result = self.verifier.verify(agent_identifier="brand-new-agent")
        agent = self.registry.lookup("brand-new-agent")
        assert agent is not None
        assert agent["trust_score"] == 0.3  # new agent starts low

    @pytest.mark.unit
    def test_verify_known_agent_high_trust(self):
        """Known agent with many transactions has higher trust"""
        self.registry.register("trusted-agent", agent_type="visa_tap")
        self.registry.update_trust("trusted-agent", 0.9)
        for _ in range(50):
            self.registry.record_transaction("trusted-agent")
        result = self.verifier.verify(agent_identifier="trusted-agent")
        assert result["verified"] is True
        assert result["trust_score"] >= 0.8

    @pytest.mark.unit
    def test_verify_combines_signals(self):
        """Multiple valid signals increase trust"""
        import base64
        import time
        self.registry.register("multi-signal-agent", agent_type="stripe_acp")
        self.registry.update_trust("multi-signal-agent", 0.7)
        future_exp = int(time.time()) + 3600
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "multi-signal-agent"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = self.verifier.verify(
            agent_identifier="multi-signal-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            token=token
        )
        assert result["verified"] is True
        assert result["trust_score"] >= 0.7
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_identity.py::TestAgentIdentityVerifier -v`
Expected: FAIL (AgentIdentityVerifier not defined)

**Step 3: Implement AgentIdentityVerifier**

In `server.py`, after the `AgentIdentityRegistry` class and its singleton, add:

```python
class AgentIdentityVerifier:
    """Validates agent credentials and computes trust scores.

    Checks API key format, JWT token expiry, and registry membership.
    """

    # Minimum API key length for basic format validation
    MIN_KEY_LENGTH = 16

    def __init__(self, registry: AgentIdentityRegistry):
        self._registry = registry

    def verify(
        self,
        agent_identifier: Optional[str] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify agent identity using available credentials.

        Args:
            agent_identifier: Agent identifier string.
            api_key: API key credential.
            token: JWT-style bearer token.

        Returns:
            Dict with verified, identity, trust_score, warnings.
        """
        warnings: List[str] = []
        trust_signals: List[float] = []
        identity: Dict[str, Any] = {}

        # Check for identifier
        if not agent_identifier:
            return {
                "verified": False,
                "identity": {},
                "trust_score": 0.0,
                "warnings": ["no_identifier"],
            }

        identity["agent_id"] = agent_identifier

        # Signal 1: Registry lookup
        registry_entry = self._registry.lookup(agent_identifier)
        if registry_entry:
            trust_signals.append(registry_entry["trust_score"])
            identity.update(registry_entry)
        else:
            warnings.append("not_in_registry")
            # Auto-register with low initial trust
            self._registry.register(agent_identifier)
            self._registry.update_trust(agent_identifier, 0.3)
            trust_signals.append(0.3)

        # Signal 2: API key format validation
        if api_key:
            if isinstance(api_key, str) and len(api_key) >= self.MIN_KEY_LENGTH:
                trust_signals.append(0.6)  # key present and reasonable format
            else:
                warnings.append("invalid_key_format")
                trust_signals.append(0.1)

        # Signal 3: JWT token validation (expiry check only)
        if token:
            token_trust = self._validate_token(token, warnings)
            trust_signals.append(token_trust)

        # Compute final trust score
        if trust_signals:
            trust_score = float(sum(trust_signals) / len(trust_signals))
        else:
            trust_score = 0.0

        # Verified if trust >= 0.5 and no critical warnings
        critical_warnings = {"no_identifier", "token_expired"}
        has_critical = bool(critical_warnings & set(warnings))
        verified = trust_score >= 0.5 and not has_critical

        return {
            "verified": verified,
            "identity": identity,
            "trust_score": trust_score,
            "warnings": warnings,
        }

    def _validate_token(self, token: str, warnings: List[str]) -> float:
        """Validate JWT token expiry. Returns trust signal."""
        import base64 as _b64
        import time as _time

        try:
            parts = token.split(".")
            if len(parts) != 3:
                warnings.append("token_parse_error")
                return 0.1

            # Decode payload (second part)
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_bytes = _b64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes)

            # Check expiry
            exp = payload.get("exp")
            if exp and isinstance(exp, (int, float)):
                if exp < _time.time():
                    warnings.append("token_expired")
                    return 0.1
                else:
                    return 0.7  # valid expiry
            return 0.5  # no expiry claim, neutral

        except Exception:
            warnings.append("token_parse_error")
            return 0.1
```

Then add singleton:

```python
agent_verifier = AgentIdentityVerifier(agent_registry)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_identity.py -v`
Expected: All 22 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_agent_identity.py
git commit -m "feat: Add AgentIdentityVerifier with credential validation"
```

---

### Task 3: Add verify_agent_identity MCP tool

**Files:**
- Modify: `server.py` (add `verify_agent_identity_impl` and `@mcp.tool()` wrapper)

**Step 1: Write the test**

Add to `tests/test_agent_identity.py`:

```python
class TestVerifyAgentIdentityImpl:
    """Tests for the verify_agent_identity_impl function"""

    @pytest.mark.unit
    def test_impl_returns_valid_result(self):
        from server import verify_agent_identity_impl
        result = verify_agent_identity_impl(
            agent_identifier="test-impl-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        )
        assert "verified" in result
        assert "trust_score" in result
        assert "verification_timestamp" in result

    @pytest.mark.unit
    def test_impl_no_identifier(self):
        from server import verify_agent_identity_impl
        result = verify_agent_identity_impl()
        assert result["verified"] is False
        assert "no_identifier" in result["warnings"]

    @pytest.mark.unit
    def test_impl_with_all_credentials(self):
        import base64
        import time
        from server import verify_agent_identity_impl
        future_exp = int(time.time()) + 3600
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
        payload_bytes = json.dumps({"exp": future_exp, "sub": "full-cred-agent"}).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b'=').decode()
        sig = base64.urlsafe_b64encode(b'fakesig').rstrip(b'=').decode()
        token = f"{header}.{payload}.{sig}"
        result = verify_agent_identity_impl(
            agent_identifier="full-cred-agent",
            api_key="sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            token=token
        )
        assert result["verified"] is True or result["trust_score"] > 0

    @pytest.mark.unit
    def test_impl_error_handling(self):
        from server import verify_agent_identity_impl
        # Should not crash on bad input
        result = verify_agent_identity_impl(
            agent_identifier=12345  # wrong type
        )
        assert "error" in result or "warnings" in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_identity.py::TestVerifyAgentIdentityImpl -v`
Expected: FAIL

**Step 3: Implement verify_agent_identity_impl and MCP tool**

In `server.py`, after `classify_traffic_source_impl` (around line 1660), add:

```python
def verify_agent_identity_impl(
    agent_identifier: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Implementation of agent identity verification.

    Validates agent credentials against the identity registry and
    checks API key format and JWT token expiry.

    Args:
        agent_identifier: Agent identifier string.
        api_key: API key credential.
        token: JWT-style bearer token.

    Returns:
        Verification result with verified status, identity, trust_score, warnings.
    """
    try:
        result = agent_verifier.verify(
            agent_identifier=str(agent_identifier) if agent_identifier is not None else None,
            api_key=str(api_key) if api_key is not None else None,
            token=str(token) if token is not None else None,
        )
        result["verification_timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Agent identity verification failed: {e}")
        return {
            "error": str(e),
            "verified": False,
            "trust_score": 0.0,
            "warnings": ["verification_error"],
            "status": "verification_failed",
        }
```

Then in the MCP tool definitions section, after the `classify_traffic_source` tool, add:

```python
@_monitored("/verify_agent_identity", "TOOL")
@mcp.tool()
def verify_agent_identity(
    agent_identifier: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify an AI agent's identity using available credentials.

    Validates agent credentials against the identity registry, checks API key
    format, and verifies JWT token expiry. Supports Stripe ACP, Visa TAP,
    Mastercard Agent Pay, Google AP2, and other agent commerce protocols.

    Args:
        agent_identifier: Agent identifier (e.g., 'stripe-acp:agent-123')
        api_key: API key credential for format validation
        token: JWT-style bearer token for expiry verification

    Returns:
        Verification result with verified status, identity details, trust score, and warnings
    """
    return verify_agent_identity_impl(agent_identifier, api_key, token)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_identity.py -v`
Expected: All 26 tests PASS

**Step 5: Commit**

```bash
git add server.py tests/test_agent_identity.py
git commit -m "feat: Add verify_agent_identity MCP tool"
```

---

### Task 4: Integrate identity score into generate_risk_score_impl

**Files:**
- Modify: `server.py` (`generate_risk_score_impl`)

**Step 1: Write the test**

Add to `tests/test_agent_identity.py`:

```python
class TestIdentityInRiskScoring:
    """Tests for identity score integration in generate_risk_score_impl"""

    @pytest.mark.unit
    def test_agent_with_identity_gets_identity_score(self):
        """Agent traffic with identifier includes identity score"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "test-identity-agent"},
        )
        assert result.get("traffic_source") == "agent"
        assert "identity" in result.get("component_scores", {})

    @pytest.mark.unit
    def test_agent_without_identity_no_identity_score(self):
        """Agent traffic without identifier has no identity component"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True},
        )
        assert result.get("traffic_source") == "agent"
        # No agent_identifier, so identity verification is skipped
        assert "identity" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_human_traffic_no_identity_check(self):
        """Human traffic doesn't get identity verification"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": False},
        )
        assert "identity" not in result.get("component_scores", {})

    @pytest.mark.unit
    def test_verified_agent_reduces_risk(self):
        """Verified agent with high trust gets lower risk score"""
        from server import generate_risk_score_impl, agent_registry
        agent_registry.register("trusted-risk-agent", agent_type="stripe_acp")
        agent_registry.update_trust("trusted-risk-agent", 0.9)
        for _ in range(10):
            agent_registry.record_transaction("trusted-risk-agent")
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "trusted-risk-agent"},
        )
        assert "identity" in result.get("component_scores", {})
        # Identity score should reflect high trust
        identity_score = result["component_scores"]["identity"]
        assert identity_score <= 0.5  # Low risk for trusted agent

    @pytest.mark.unit
    def test_unverified_agent_increases_risk(self):
        """Unverified agent gets higher risk from identity component"""
        from server import generate_risk_score_impl
        result = generate_risk_score_impl(
            {"amount": 100.0, "merchant": "Store", "location": "NYC",
             "timestamp": "2025-01-01T12:00:00", "payment_method": "credit_card",
             "is_agent": True, "agent_identifier": "totally-unknown-agent-xyz"},
        )
        assert "identity" in result.get("component_scores", {})
        identity_score = result["component_scores"]["identity"]
        assert identity_score >= 0.5  # High risk for unverified agent
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_identity.py::TestIdentityInRiskScoring -v`
Expected: Most FAIL (identity not in component_scores)

**Step 3: Update generate_risk_score_impl**

In `server.py`, in `generate_risk_score_impl`, after the traffic classification (line 1330) and before "Perform all analyses" (line 1332), add agent identity verification:

```python
        # Agent identity verification (only for agent traffic with identifier)
        identity_verification = None
        if is_agent_traffic and transaction_data.get("agent_identifier"):
            identity_verification = agent_verifier.verify(
                agent_identifier=str(transaction_data["agent_identifier"]),
                api_key=str(transaction_data.get("api_key", "")) or None,
                token=str(transaction_data.get("token", "")) or None,
            )
```

After the network analysis section (around line 1379), before the weighted score calculation, add:

```python
        # Identity analysis (agent traffic only)
        if identity_verification:
            # Convert trust_score to risk_score: high trust = low risk
            id_trust = identity_verification.get("trust_score", 0.5)
            identity_risk_score = 1.0 - id_trust
            comprehensive_result["component_scores"]["identity"] = identity_risk_score
            scores.append(identity_risk_score)
            confidences.append(0.7 if identity_verification.get("verified") else 0.4)

            if not identity_verification.get("verified"):
                comprehensive_result["detected_anomalies"].append("unverified_agent_identity")

            # Record transaction in registry
            agent_id = transaction_data.get("agent_identifier")
            if agent_id:
                agent_registry.record_transaction(str(agent_id))
```

Update the agent weighting block to handle the additional identity score. Replace the agent branch:

```python
        if is_agent_traffic:
            if len(scores) == 1:
                overall_score = scores[0]
            elif len(scores) == 2:
                # Transaction + network (no behavioral for agents)
                overall_score = (scores[0] * 0.55 + scores[1] * 0.45)
            else:
                # All three present (unusual for agents but handle it)
                overall_score = (scores[0] * 0.4 + scores[1] * 0.2 + scores[2] * 0.4)
```

With:

```python
        if is_agent_traffic:
            if identity_verification and len(scores) >= 3:
                # Transaction + identity + network (or more)
                # Weights: transaction 35%, identity 35%, remaining 30%
                w_txn = 0.35
                w_id = 0.35
                w_rest = 0.30
                overall_score = scores[0] * w_txn + scores[-1 if identity_verification else 1] * w_id
                remaining = [s for i, s in enumerate(scores) if i != 0 and i != (len(scores) - 1 if not identity_verification else len(scores) - 2)]
                if remaining:
                    overall_score += sum(remaining) / len(remaining) * w_rest
                else:
                    overall_score = (overall_score / (w_txn + w_id)) * 1.0
            elif len(scores) == 2:
                if identity_verification:
                    # Transaction + identity
                    overall_score = (scores[0] * 0.5 + scores[1] * 0.5)
                else:
                    # Transaction + network
                    overall_score = (scores[0] * 0.55 + scores[1] * 0.45)
            elif len(scores) == 1:
                overall_score = scores[0]
            else:
                # Fallback: equal weights
                overall_score = sum(scores) / len(scores)
```

Actually, this is getting complex. Let's simplify. The scores list is built in order: [transaction, (behavioral), (identity), (network)]. For agents the simpler approach:

Replace the ENTIRE agent weighting block with:

```python
        if is_agent_traffic:
            n = len(scores)
            if n == 1:
                overall_score = scores[0]
            else:
                # For agent traffic, use equal weighting across all available components
                # This naturally adapts as more components (identity, network) are added
                overall_score = sum(scores) / n
```

This is cleaner and more forward-compatible. As more agent-specific analyzers come in future phases, they just add to the scores list.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_identity.py -v`
Expected: All 31 tests PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 616+ passed, 2 skipped

**Step 6: Commit**

```bash
git add server.py tests/test_agent_identity.py
git commit -m "feat: Integrate agent identity verification into risk scoring"
```

---

### Task 5: Final verification and lint

**Step 1: Run ruff check**

Run: `ruff check . --exclude=venv,test_data,.claude`
Expected: 0 errors

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 616+ passed, 2 skipped

**Step 3: Verify new MCP tool count**

Run: `python -c "from server import mcp; print(len(mcp._tool_manager._tools))"`
Expected: 15 (was 14, +1 verify_agent_identity)

**Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "chore: Phase B agent identity layer complete"
```
