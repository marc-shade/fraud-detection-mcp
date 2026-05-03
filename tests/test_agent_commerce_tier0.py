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
    consume_nonce_impl,
    validate_nonce_impl,
    verify_agent_signature_impl,
)


pytestmark = [pytest.mark.unit, pytest.mark.signature]


# ---------------------------------------------------------------------------
# Tier 0.1 — 12-feature behavioral fingerprint
# ---------------------------------------------------------------------------


class TestBehavioralFingerprint12Features:
    def test_feature_dim_is_13(self):
        fp = AgentBehavioralFingerprint()
        assert fp.FEATURE_DIM == 13

    def test_extract_features_no_transaction_returns_13_dims(self):
        fp = AgentBehavioralFingerprint()
        v = fp._extract_features(
            api_timing_ms=100.0,
            decision_pattern="approve",
            request_structure_hash="h1",
            baseline=None,
            transaction=None,
        )
        assert v.shape == (1, 13)
        # Indices 6-9 + 12 (txn features minus hour_cos) must be zero when no txn provided.
        # Index 10 (hour_sin) = 0 (sin(0)=0). Index 11 (hour_cos) = 1 (cos(0)=1, neutral).
        for i in (6, 7, 8, 9, 10, 12):
            assert v[0, i] == 0.0, f"index {i} should be zero with no txn"
        assert v[0, 11] == 1.0  # cos(0) baseline

    def test_extract_features_with_transaction_populates_txn_indices(self):
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
        assert v.shape == (1, 13)
        import math
        # log_amount should be log1p(99.50)
        assert abs(v[0, 6] - math.log1p(99.50)) < 1e-6
        # payment/merchant/location hashes are in [0, 1)
        for i in (7, 8, 9):
            assert 0.0 <= v[0, i] < 1.0
        # Cyclical hour: 14:00 → angle 14 * 2π/24
        angle = 2.0 * math.pi * (14.0 / 24.0)
        assert abs(v[0, 10] - math.sin(angle)) < 1e-6
        assert abs(v[0, 11] - math.cos(angle)) < 1e-6
        # field_completeness: all 6 expected fields present
        assert v[0, 12] == 1.0

    def test_cyclical_hour_wraparound_is_smooth(self):
        """23:00 and 00:00 should be close in feature space, not 23 units apart."""
        import math
        fp = AgentBehavioralFingerprint()
        txn_23 = {"timestamp": "2026-05-03T23:00:00Z", "amount": 1.0}
        txn_00 = {"timestamp": "2026-05-04T00:00:00Z", "amount": 1.0}
        v23 = fp._extract_features(0, None, None, None, transaction=txn_23)
        v00 = fp._extract_features(0, None, None, None, transaction=txn_00)
        # Euclidean distance in (sin, cos) plane between 23:00 and 00:00
        d = math.sqrt(
            (v23[0, 10] - v00[0, 10]) ** 2 + (v23[0, 11] - v00[0, 11]) ** 2
        )
        # Should be ~0.26 (chord length for π/12 angle), much less than the
        # 23.0 a raw integer would imply
        assert d < 0.5

    def test_partial_transaction_lowers_completeness(self):
        fp = AgentBehavioralFingerprint()
        txn = {"amount": 50.0, "merchant": "x"}  # 2 of 6 fields present
        v = fp._extract_features(0.0, None, None, None, transaction=txn)
        # 2 / 6 = 0.333...; field_completeness lives at index 12 now
        assert abs(v[0, 12] - (2.0 / 6.0)) < 1e-6

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
    """validate_nonce is now SAFE BY DEFAULT — peeks without consuming."""

    def test_default_is_peek_only_does_not_mutate(self):
        keyid = f"peek-default-test-{time.time_ns()}"
        r1 = validate_nonce_impl(keyid, "nonce-default")
        assert r1["seen"] is False
        assert r1["recorded"] is False
        # Repeated peeks return False; never mutates
        r2 = validate_nonce_impl(keyid, "nonce-default")
        assert r2["seen"] is False
        assert r2["recorded"] is False

    def test_record_seen_true_still_works(self):
        keyid = f"record-explicit-test-{time.time_ns()}"
        r1 = validate_nonce_impl(keyid, "nonce-r", record_seen=True)
        assert r1["seen"] is False
        assert r1["recorded"] is True
        r2 = validate_nonce_impl(keyid, "nonce-r")
        assert r2["seen"] is True


class TestConsumeNonceImpl:
    """consume_nonce is the explicit atomic check+record operation."""

    def test_first_consume_accepts(self):
        keyid = f"consume-test-{time.time_ns()}"
        r = consume_nonce_impl(keyid, "nonce-c1")
        assert r["accepted"] is True
        assert r["replayed"] is False

    def test_second_consume_replays(self):
        keyid = f"consume-replay-test-{time.time_ns()}"
        consume_nonce_impl(keyid, "nonce-r1")
        r = consume_nonce_impl(keyid, "nonce-r1")
        assert r["accepted"] is False
        assert r["replayed"] is True

    def test_consume_then_validate_sees(self):
        keyid = f"consume-then-validate-{time.time_ns()}"
        consume_nonce_impl(keyid, "n1")
        # validate (peek) should see it without mutating
        r = validate_nonce_impl(keyid, "n1")
        assert r["seen"] is True
        assert r["recorded"] is False


class TestNewToolsRegisteredInMCP:
    def test_four_new_tools_present(self):
        import asyncio
        from server import mcp

        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert "verify_agent_signature" in names
        assert "check_idempotency_key" in names
        assert "validate_nonce" in names
        assert "consume_nonce" in names


# ---------------------------------------------------------------------------
# F1 wiring — analyze_agent_transaction_impl with signature_headers
# ---------------------------------------------------------------------------


class TestAnalyzeAgentTransactionWiresSignatureVerification:
    """F1 — verify that signature_headers in transaction_data flows through
    analyze_agent_transaction_impl and produces verification_status +
    anomaly + identity_trust adjustments."""

    def _ed25519_keypair(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        import base64

        priv = Ed25519PrivateKey.generate()
        raw_pub = priv.public_key().public_bytes_raw()
        x_b64 = base64.urlsafe_b64encode(raw_pub).rstrip(b"=").decode("ascii")
        kid = "wired-test-key-1"
        return {
            "public_jwk": {"kty": "OKP", "crv": "Ed25519", "x": x_b64,
                           "kid": kid, "alg": "EdDSA"},
            "sign": lambda data: priv.sign(data),
            "kid": kid,
        }

    def _signed_headers(
        self, kp, method, path, authority, nonce="wire-test-nonce"
    ):
        from acp_signatures import (
            parse_signature_input, build_signature_base,
        )
        import base64

        created = int(time.time())
        sig_input_value = (
            f'sig1=("@method" "@authority" "@path");'
            f'keyid="{kp["kid"]}";alg="EdDSA";created={created};'
            f'nonce="{nonce}";tag="agent-payer-auth"'
        )
        sig_input = parse_signature_input(sig_input_value)
        base = build_signature_base(
            sig_input, headers={}, method=method, path=path, authority=authority,
        )
        sig = kp["sign"](base.encode("utf-8"))
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
        return {
            "Signature-Input": sig_input_value,
            "Signature": f"sig1=:{sig_b64}:",
        }

    def setup_method(self):
        # Inject test issuer + key into the real shared resolver
        from acp_signatures import jwks_resolver
        self.kp = self._ed25519_keypair()
        jwks_resolver._issuers["wired_test_issuer"] = "memory://"
        jwks_resolver._cache["wired_test_issuer"] = {
            "jwks_url": "memory://",
            "expires_at": time.time() + 3600,
            "keys": {self.kp["kid"]: self.kp["public_jwk"]},
        }

    def test_verified_signature_in_pipeline_boosts_trust(self):
        from server import analyze_agent_transaction_impl

        headers = self._signed_headers(
            self.kp, "POST", "/checkout", "merchant.example.com",
            nonce=f"wire-pos-{time.time_ns()}",
        )
        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "payment_method": "card",
            "timestamp": "2026-05-03T12:00:00Z",
            "currency": "USD",
            "is_agent": True,
            "agent_identifier": "wired-test-agent",
            "user_agent": "visa-tap/1.0",
            "signature_headers": headers,
            "http_method": "POST",
            "http_path": "/checkout",
            "http_authority": "merchant.example.com",
            "expected_issuer": "wired_test_issuer",
            "expected_signature_tag": "agent-payer-auth",
        }
        result = analyze_agent_transaction_impl(txn)
        assert result.get("verification_status") == "verified", result
        assert result.get("identity_verified") is True
        # No signature anomaly
        assert "signature_verification_failed" not in result.get("anomalies", [])

    def test_failed_signature_in_pipeline_adds_anomaly_and_drops_trust(self):
        from server import analyze_agent_transaction_impl

        # Tamper: pass a path different from what was signed
        headers = self._signed_headers(
            self.kp, "POST", "/checkout", "merchant.example.com",
            nonce=f"wire-neg-{time.time_ns()}",
        )
        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "payment_method": "card",
            "timestamp": "2026-05-03T12:00:00Z",
            "currency": "USD",
            "is_agent": True,
            "agent_identifier": "wired-test-agent",
            "user_agent": "visa-tap/1.0",
            "signature_headers": headers,
            "http_method": "POST",
            "http_path": "/admin",  # different path → crypto fails
            "http_authority": "merchant.example.com",
            "expected_issuer": "wired_test_issuer",
        }
        result = analyze_agent_transaction_impl(txn)
        assert result.get("verification_status") == "verification_failed", result
        assert "signature_verification_failed" in result.get("anomalies", [])
        # Identity trust should be lowered vs the verified case
        assert result.get("identity_trust_score", 0.0) < 0.85

    def test_no_signature_headers_yields_unverified_status(self):
        from server import analyze_agent_transaction_impl

        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "payment_method": "card",
            "timestamp": "2026-05-03T12:00:00Z",
            "is_agent": True,
            "agent_identifier": "wired-test-agent-2",
            "user_agent": "visa-tap/1.0",
        }
        result = analyze_agent_transaction_impl(txn)
        assert result.get("verification_status") == "unverified"
        assert "agent_protocol_claimed_but_unsigned" in result.get("anomalies", [])

    def test_verified_protocol_uses_protocol_enum_not_raw_issuer(self):
        """F4 — verified_protocol should report 'visa_tap', not 'visa'."""
        from server import analyze_agent_transaction_impl, ISSUER_TO_PROTOCOL
        from acp_signatures import jwks_resolver

        # Re-key under "visa" issuer so the result reports a real mapping
        jwks_resolver._issuers["visa"] = "memory://"
        jwks_resolver._cache["visa"] = {
            "jwks_url": "memory://",
            "expires_at": time.time() + 3600,
            "keys": {self.kp["kid"]: self.kp["public_jwk"]},
        }
        headers = self._signed_headers(
            self.kp, "POST", "/checkout", "merchant.example.com",
            nonce=f"wire-prot-{time.time_ns()}",
        )
        txn = {
            "amount": 50.0, "merchant": "M",
            "is_agent": True,
            "agent_identifier": "visa:agent-99",
            "user_agent": "visa-tap/1.0",
            "signature_headers": headers,
            "http_method": "POST",
            "http_path": "/checkout",
            "http_authority": "merchant.example.com",
            "expected_issuer": "visa",
            "expected_signature_tag": "agent-payer-auth",
        }
        result = analyze_agent_transaction_impl(txn)
        assert result.get("verification_status") == "verified", result
        # Mapped via ISSUER_TO_PROTOCOL
        assert result.get("verified_protocol") == ISSUER_TO_PROTOCOL["visa"] == "visa_tap"
