"""Tests for the Tier 0 agent commerce features added to server.py.

Covers:
  - 12-feature behavioral fingerprint (T3 / Tier 0.1)
  - TrafficClassifier claimed_protocol vs verified_protocol (T4 / Tier 0.5)
  - 3 new MCP tools: verify_agent_signature, check_idempotency_key,
    validate_nonce (T9)
"""
from __future__ import annotations

import time
from typing import Any, Dict

import pytest

import server
from server import (
    AgentBehavioralFingerprint,
    TrafficClassifier,
    check_idempotency_key_impl,
    validate_nonce_impl,
    verify_agent_signature_impl,
)


pytestmark = [pytest.mark.unit, pytest.mark.signature]


# ---------------------------------------------------------------------------
# Tier 0.1 — 12-feature behavioral fingerprint
# ---------------------------------------------------------------------------


class TestBehavioralFingerprint12Features:
    def test_feature_dim_is_12(self):
        fp = AgentBehavioralFingerprint()
        assert fp.FEATURE_DIM == 12

    def test_extract_features_no_transaction_returns_12_dims(self):
        fp = AgentBehavioralFingerprint()
        v = fp._extract_features(
            api_timing_ms=100.0,
            decision_pattern="approve",
            request_structure_hash="h1",
            baseline=None,
            transaction=None,
        )
        assert v.shape == (1, 12)
        # Indices 6-11 (transaction features) must be zero when no txn provided
        assert all(v[0, i] == 0.0 for i in range(6, 12))

    def test_extract_features_with_transaction_populates_indices_6_to_11(self):
        fp = AgentBehavioralFingerprint()
        txn = {
            "amount": 99.50,
            "merchant": "Starbucks",
            "location": "Seattle",
            "payment_method": "card",
            "timestamp": "2026-05-03T14:30:00Z",
            "currency": "USD",
        }
        v = fp._extract_features(100.0, "approve", "h1", None, transaction=txn)
        assert v.shape == (1, 12)
        # log_amount should be log1p(99.50)
        import math
        assert abs(v[0, 6] - math.log1p(99.50)) < 1e-6
        # payment/merchant/location hashes are in [0, 1)
        for i in (7, 8, 9):
            assert 0.0 <= v[0, i] < 1.0
        # hour_of_day extracted from timestamp
        assert v[0, 10] == 14.0
        # field_completeness: all 6 expected fields present
        assert v[0, 11] == 1.0

    def test_partial_transaction_lowers_completeness(self):
        fp = AgentBehavioralFingerprint()
        txn = {"amount": 50.0, "merchant": "x"}  # 2 of 6 fields present
        v = fp._extract_features(0.0, None, None, None, transaction=txn)
        # 2 / 6 = 0.333...
        assert abs(v[0, 11] - (2.0 / 6.0)) < 1e-6

    def test_stolen_token_replay_detected_by_payment_features(self):
        """Build a baseline of consistent transactions, then replay against a
        radically different merchant/amount/location distribution. Even with
        identical timing, the 12-feature model must flag it as anomalous —
        which the old 6-feature model could NOT do.
        """
        fp = AgentBehavioralFingerprint()
        agent_id = "stolen-token-victim"
        consistent_txn = {
            "amount": 25.0,
            "merchant": "Starbucks",
            "location": "Seattle",
            "payment_method": "card",
            "timestamp": "2026-05-03T08:00:00Z",
            "currency": "USD",
        }
        # Build 20-observation baseline — all very similar
        for i in range(20):
            fp.analyze(
                agent_id=agent_id,
                api_timing_ms=100.0 + (i % 3),  # tight timing
                decision_pattern="approve",
                request_structure_hash="checkout",
                transaction=consistent_txn,
            )
        # Now replay with same timing/decision/structure but different txn
        attack_txn = {
            "amount": 9999.99,
            "merchant": "AdversaryCasino",
            "location": "Macau",
            "payment_method": "crypto",
            "timestamp": "2026-05-03T03:00:00Z",
            "currency": "USDT",
        }
        result = fp.analyze(
            agent_id=agent_id,
            api_timing_ms=101.0,             # matches baseline timing
            decision_pattern="approve",      # in baseline
            request_structure_hash="checkout",  # in baseline
            transaction=attack_txn,
        )
        # The Isolation Forest trained on 12 features should still flag this.
        # (Cannot guarantee high score across all random seeds, but anomaly
        # behaviour should be present.)
        assert result["risk_score"] > 0.4

    def test_record_with_transaction_persists_features(self):
        fp = AgentBehavioralFingerprint()
        txn = {
            "amount": 10.0, "merchant": "X", "location": "Y",
            "payment_method": "card", "timestamp": "2026-01-01T12:00:00Z",
        }
        fp.record("agent-1", api_timing_ms=50.0, transaction=txn)
        # Re-extracting features for the recorded obs should populate indices 6-11
        with fp._lock:
            obs = fp._history["agent-1"][0]
        assert obs["transaction"] is not None
        assert obs["transaction"]["amount"] == 10.0


# ---------------------------------------------------------------------------
# Tier 0.5 — TrafficClassifier claimed vs verified
# ---------------------------------------------------------------------------


class TestTrafficClassifierClaimedVsVerified:
    def setup_method(self):
        self.tc = TrafficClassifier()

    def test_claimed_protocol_set_when_user_agent_matches(self):
        result = self.tc.classify({
            "is_agent": True,
            "user_agent": "stripe-acp/1.0",
        })
        assert result["claimed_protocol"] == "stripe_acp"
        assert result["verified_protocol"] is None
        assert result["verification_status"] in {"unverified", "no_signature_provided"}

    def test_no_signature_status(self):
        result = self.tc.classify({
            "is_agent": True,
            "agent_identifier": "stripe-acp:agent-7",
        })
        # Without a signature_headers field, status must be 'unverified'
        # (claimed protocol present) or 'no_signature_provided' (no claim).
        assert result["verification_status"] == "unverified"
        assert "signature_absent" in result["signals"]

    def test_invalid_signature_marks_failed(self):
        now = int(time.time())
        result = self.tc.classify({
            "is_agent": True,
            "agent_identifier": "visa:agent-99",
            "user_agent": "visa-tap/1.0",
            "signature_headers": {
                "Signature-Input": (
                    f'sig1=("@method" "@authority" "@path");'
                    f'keyid="visa:agent-99";alg="EdDSA";created={now};'
                    f'nonce="abc";tag="agent-payer-auth"'
                ),
                "Signature": "sig1=:Zm9vYmFy:",  # not a real signature
            },
            "http_method": "POST",
            "http_path": "/pay",
            "http_authority": "merchant.example.com",
            "expected_signature_tag": "agent-payer-auth",
        })
        assert result["verification_status"] == "verification_failed"
        assert result["verified_protocol"] is None
        assert "signature_failed" in result["signals"]

    def test_no_claim_no_signature_human_user_agent(self):
        result = self.tc.classify({
            "user_agent": "Mozilla/5.0 Chrome/120",
        })
        assert result["source"] == "human"
        assert result["claimed_protocol"] is None
        assert result["verification_status"] == "no_signature_provided"

    def test_verification_status_field_present_always(self):
        for meta in [
            {},
            {"is_agent": True},
            {"user_agent": "stripe-acp/1.0"},
            {"agent_identifier": "x402-client:agent-1"},
        ]:
            result = self.tc.classify(meta)
            assert "verification_status" in result
            assert "claimed_protocol" in result
            assert "verified_protocol" in result


# ---------------------------------------------------------------------------
# Tier 0 MCP tools
# ---------------------------------------------------------------------------


class TestVerifyAgentSignatureImpl:
    def test_missing_signature_headers_returns_unverified(self):
        result = verify_agent_signature_impl(headers={"Content-Type": "application/json"})
        assert result["verified"] is False
        assert "missing" in result["reason"]

    def test_passes_through_to_acp_signatures(self):
        # Wrap test_acp_signatures' freshness check by passing an old created
        old = int(time.time()) - 3600
        result = verify_agent_signature_impl(
            headers={
                "Signature-Input": (
                    f'sig1=("@method");keyid="visa:k";alg="EdDSA";created={old}'
                ),
                "Signature": "sig1=:Zm9v:",
            },
            method="POST",
        )
        assert result["verified"] is False
        # Should fail freshness rather than crypto
        assert "freshness_failed" in result["reason"]


class TestCheckIdempotencyKeyImpl:
    def test_miss_then_store_then_hit(self):
        # Use a unique key per test
        body = {"amount": 50, "merchant": "test-store"}
        r1 = check_idempotency_key_impl(
            "test-key-1", "agent-7", request_payload=body,
        )
        assert r1["status"] == "miss"

        r2 = check_idempotency_key_impl(
            "test-key-1", "agent-7", request_payload=body,
            cache_result={"order_id": "o-99"},
        )
        assert r2["status"] == "stored"

        r3 = check_idempotency_key_impl(
            "test-key-1", "agent-7", request_payload=body,
        )
        assert r3["status"] == "hit"
        assert r3["result"] == {"order_id": "o-99"}

    def test_conflict_on_different_body(self):
        body1 = {"amount": 100}
        body2 = {"amount": 200}
        check_idempotency_key_impl(
            "test-key-2", "agent-x", request_payload=body1,
            cache_result={"order_id": "o-1"},
        )
        r = check_idempotency_key_impl(
            "test-key-2", "agent-x", request_payload=body2,
        )
        assert r["status"] == "conflict"


class TestValidateNonceImpl:
    def test_unseen_nonce_returns_seen_false_and_records(self):
        # Use unique keyid to avoid collision with other tests
        keyid = f"validate-nonce-test-{time.time_ns()}"
        r1 = validate_nonce_impl(keyid, "nonce-x", record_seen=True)
        assert r1["seen"] is False
        assert r1["recorded"] is True
        # Second call with same keyid+nonce must be seen=True
        r2 = validate_nonce_impl(keyid, "nonce-x", record_seen=True)
        assert r2["seen"] is True

    def test_query_only_does_not_mutate(self):
        keyid = f"query-only-test-{time.time_ns()}"
        r1 = validate_nonce_impl(keyid, "nonce-q", record_seen=False)
        assert r1["seen"] is False
        assert r1["recorded"] is False
        # Second call without recording should still see it as unseen
        r2 = validate_nonce_impl(keyid, "nonce-q", record_seen=False)
        assert r2["seen"] is False


class TestNewToolsRegisteredInMCP:
    def test_three_new_tools_present(self):
        import asyncio
        from server import mcp

        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert "verify_agent_signature" in names
        assert "check_idempotency_key" in names
        assert "validate_nonce" in names
