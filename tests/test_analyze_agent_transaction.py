"""Tests for analyze_agent_transaction MCP tool and impl function.

Tests the specialized agent transaction analysis that combines traffic
classification, identity verification, and behavioral fingerprinting
into a single agent-aware analysis pipeline.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
from datetime import datetime


class TestAnalyzeAgentTransactionImpl:
    """Tests for analyze_agent_transaction_impl function."""

    @pytest.mark.unit
    def test_impl_exists(self):
        """analyze_agent_transaction_impl is importable from server."""
        from server import analyze_agent_transaction_impl

        assert callable(analyze_agent_transaction_impl)

    @pytest.mark.unit
    def test_basic_agent_transaction(self):
        """Basic agent transaction analysis returns required fields."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 100.0,
                "merchant": "TestStore",
                "location": "NYC",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "stripe-acp:agent-test-1",
            }
        )
        assert "risk_score" in result
        assert "anomalies" in result
        assert "fingerprint_match" in result
        assert "mandate_compliance" in result
        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.unit
    def test_with_agent_behavior_data(self):
        """Agent transaction with behavioral data includes fingerprint score."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 50.0,
                "merchant": "Store",
                "location": "LA",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "visa-tap:agent-2",
            },
            agent_behavior={
                "api_timing_ms": 45.0,
                "decision_pattern": "approve",
                "request_structure_hash": "abc123",
            },
        )
        assert "fingerprint_match" in result
        assert isinstance(result["fingerprint_match"], float)

    @pytest.mark.unit
    def test_without_agent_identifier(self):
        """Transaction without agent_identifier still works."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 200.0,
                "merchant": "Store",
                "location": "SF",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "crypto",
                "is_agent": True,
            }
        )
        assert "risk_score" in result
        # No agent_identifier -> identity not verified
        assert result.get("identity_verified") is False or "identity" not in result.get(
            "component_scores", {}
        )

    @pytest.mark.unit
    def test_with_api_key_and_token(self):
        """Agent transaction with API key and token includes identity check."""
        import base64
        import json
        import time

        from server import analyze_agent_transaction_impl

        future_exp = int(time.time()) + 3600
        header = (
            base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}')
            .rstrip(b"=")
            .decode()
        )
        payload_bytes = json.dumps(
            {"exp": future_exp, "sub": "agent-txn-test"}
        ).encode()
        payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig}"

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 100.0,
                "merchant": "Store",
                "location": "NYC",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "test-identity-txn-agent",
                "api_key": "sk_agent_a1b2c3d4e5f6g7h8i9j0k1l2",
                "token": token,
            }
        )
        assert "risk_score" in result
        assert "identity_verified" in result

    @pytest.mark.unit
    def test_invalid_transaction_data(self):
        """Invalid transaction data returns error."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(transaction_data="not_a_dict")
        assert "error" in result

    @pytest.mark.unit
    def test_negative_amount(self):
        """Negative amount is caught by validation."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": -100.0,
                "merchant": "Store",
                "is_agent": True,
            }
        )
        assert "error" in result or result.get("status") == "validation_failed"

    @pytest.mark.unit
    def test_result_contains_traffic_source(self):
        """Result includes traffic source classification."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 100.0,
                "merchant": "Store",
                "location": "NYC",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "test-traffic-agent",
            }
        )
        assert "traffic_source" in result

    @pytest.mark.unit
    def test_result_contains_analysis_timestamp(self):
        """Result has an analysis timestamp."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 100.0,
                "merchant": "Store",
                "location": "NYC",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "test-ts-agent",
            }
        )
        assert "analysis_timestamp" in result

    @pytest.mark.unit
    def test_exception_handling(self):
        """Function handles unexpected errors gracefully."""
        from server import analyze_agent_transaction_impl

        # Empty dict should not crash
        result = analyze_agent_transaction_impl(transaction_data={})
        assert "risk_score" in result or "error" in result

    @pytest.mark.unit
    def test_high_amount_increases_risk(self):
        """Very high transaction amount contributes to risk."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 999999.0,
                "merchant": "Unknown Merchant",
                "location": "Unknown",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "crypto",
                "is_agent": True,
                "agent_identifier": "high-amount-agent",
            }
        )
        assert result.get("risk_score", 0) > 0


class TestAnalyzeAgentTransactionMCPTool:
    """Tests for the MCP tool registration and wrapper."""

    @pytest.mark.unit
    def test_tool_registered(self):
        """analyze_agent_transaction is registered as an MCP tool."""
        import server

        tools = list(server.mcp._tool_manager._tools.keys())
        assert "analyze_agent_transaction" in tools

    @pytest.mark.unit
    def test_tool_callable(self):
        """analyze_agent_transaction MCP tool is callable."""
        from server import analyze_agent_transaction

        assert callable(analyze_agent_transaction)

    @pytest.mark.unit
    def test_tool_returns_result_via_impl(self):
        """Impl function returns valid result (MCP tool delegates to impl)."""
        from server import analyze_agent_transaction_impl

        result = analyze_agent_transaction_impl(
            transaction_data={
                "amount": 100.0,
                "merchant": "Store",
                "location": "NYC",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
                "is_agent": True,
                "agent_identifier": "mcp-tool-test-agent",
            }
        )
        assert "risk_score" in result
        assert "anomalies" in result
        assert "fingerprint_match" in result


class TestMandateInAgentTransaction:
    """Test mandate compliance integration in analyze_agent_transaction_impl."""

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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
