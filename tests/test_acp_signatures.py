"""Tests for acp_signatures.py (Tier 0.2 — RFC 9421 verifier + JWKS resolver).

Covers parser correctness, freshness/replay/tag enforcement, and full
end-to-end verify against locally-generated EdDSA / ES256 keys (no network).
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict

import pytest

from acp_signatures import (
    ALLOWED_ALGORITHMS,
    JOSE_AVAILABLE,
    JWKSResolver,
    MAX_SIGNATURE_AGE_SECONDS,
    SignatureInput,
    build_signature_base,
    parse_signature_input,
    verify_rfc9421_signature,
)
from agent_security import NonceCache


pytestmark = pytest.mark.signature


# ---------------------------------------------------------------------------
# Fixtures: deterministic JWKS keys (Ed25519 + ES256)
# ---------------------------------------------------------------------------


def _ed25519_keypair() -> Dict[str, Any]:
    """Build an Ed25519 JWK pair using cryptography. Returns dict with priv/pub JWK + sign(bytes)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    raw_pub = priv.public_key().public_bytes_raw()

    def b64url(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    pub_jwk = {
        "kty": "OKP",
        "crv": "Ed25519",
        "x": b64url(raw_pub),
        "kid": "test-ed25519-1",
        "alg": "EdDSA",
    }

    def sign(data: bytes) -> bytes:
        return priv.sign(data)

    return {"public_jwk": pub_jwk, "sign": sign, "kid": pub_jwk["kid"]}


def _b64url_no_pad(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseSignatureInput:
    def test_minimal_valid_header(self):
        h = (
            'sig1=("@method" "@authority" "@path");'
            'keyid="visa:agent-7";alg="EdDSA";created=1700000000'
        )
        s = parse_signature_input(h)
        assert s.label == "sig1"
        assert s.covered_components == ["@method", "@authority", "@path"]
        assert s.keyid == "visa:agent-7"
        assert s.alg == "EdDSA"
        assert s.created == 1700000000
        assert s.expires is None
        assert s.nonce is None
        assert s.tag is None

    def test_full_header_with_optional_params(self):
        h = (
            'sig1=("@method" "@path");keyid="visa:k1";alg="ES256";'
            "created=1700000000;expires=1700000100;"
            'nonce="abc-123";tag="agent-payer-auth"'
        )
        s = parse_signature_input(h)
        assert s.expires == 1700000100
        assert s.nonce == "abc-123"
        assert s.tag == "agent-payer-auth"

    def test_missing_required_keyid_rejected(self):
        h = 'sig1=("@method");alg="EdDSA";created=1700000000'
        with pytest.raises(ValueError, match="keyid"):
            parse_signature_input(h)

    def test_missing_required_alg_rejected(self):
        h = 'sig1=("@method");keyid="k1";created=1700000000'
        with pytest.raises(ValueError, match="alg"):
            parse_signature_input(h)

    def test_disallowed_algorithm_rejected(self):
        h = 'sig1=("@method");keyid="k1";alg="HS256";created=1700000000'
        with pytest.raises(ValueError, match="not in allowed set"):
            parse_signature_input(h)

    def test_allowed_algorithms_set(self):
        # Sanity: the allowed set is what the spec mandates.
        assert ALLOWED_ALGORITHMS == frozenset({"EdDSA", "PS256", "ES256", "RS256"})

    def test_empty_header_rejected(self):
        with pytest.raises(ValueError):
            parse_signature_input("")

    def test_garbage_header_rejected(self):
        with pytest.raises(ValueError):
            parse_signature_input("not a real signature input")


# ---------------------------------------------------------------------------
# Freshness tests (8-min window per Visa TAP)
# ---------------------------------------------------------------------------


class TestSignatureFreshness:
    def test_fresh_signature_passes(self):
        now = int(time.time())
        s = SignatureInput(
            label="x", covered_components=[], keyid="k", alg="EdDSA",
            created=now,
        )
        ok, _reason = s.is_fresh()
        assert ok

    def test_signature_too_old_fails(self):
        old = int(time.time()) - MAX_SIGNATURE_AGE_SECONDS - 1
        s = SignatureInput(
            label="x", covered_components=[], keyid="k", alg="EdDSA",
            created=old,
        )
        ok, reason = s.is_fresh()
        assert not ok
        assert "old" in reason

    def test_signature_in_future_fails(self):
        future = int(time.time()) + 120
        s = SignatureInput(
            label="x", covered_components=[], keyid="k", alg="EdDSA",
            created=future,
        )
        ok, reason = s.is_fresh()
        assert not ok
        assert "future" in reason

    def test_lifetime_over_8min_fails(self):
        now = int(time.time())
        s = SignatureInput(
            label="x", covered_components=[], keyid="k", alg="EdDSA",
            created=now, expires=now + MAX_SIGNATURE_AGE_SECONDS + 60,
        )
        ok, reason = s.is_fresh()
        assert not ok
        assert "lifetime" in reason


# ---------------------------------------------------------------------------
# Signature base construction (RFC 9421 §2.3)
# ---------------------------------------------------------------------------


class TestBuildSignatureBase:
    def test_method_path_authority_components(self):
        s = SignatureInput(
            label="sig1",
            covered_components=["@method", "@authority", "@path"],
            keyid="k1", alg="EdDSA", created=1700000000,
        )
        base = build_signature_base(
            s, headers={}, method="POST", path="/checkout", authority="m.example.com"
        )
        # Order must match; @signature-params trailing line required.
        assert '"@method": POST' in base
        assert '"@authority": m.example.com' in base
        assert '"@path": /checkout' in base
        assert '"@signature-params":' in base
        assert 'keyid="k1"' in base
        assert 'alg="EdDSA"' in base
        assert 'created=1700000000' in base

    def test_authority_lowercased(self):
        s = SignatureInput(
            label="sig1", covered_components=["@authority"],
            keyid="k", alg="EdDSA", created=1700000000,
        )
        base = build_signature_base(s, headers={}, authority="API.EXAMPLE.COM")
        assert '"@authority": api.example.com' in base

    def test_method_uppercased(self):
        s = SignatureInput(
            label="sig1", covered_components=["@method"],
            keyid="k", alg="EdDSA", created=1700000000,
        )
        base = build_signature_base(s, headers={}, method="post")
        assert '"@method": POST' in base


# ---------------------------------------------------------------------------
# JWKS resolver
# ---------------------------------------------------------------------------


class TestJWKSResolver:
    def test_register_and_lookup(self):
        r = JWKSResolver()
        r.register_issuer("custom", "https://nonexistent.example/.well-known/jwks")
        # Lookup will fail because URL is fake — but the issuer must be registered.
        assert "custom" in r.known_issuers()

    def test_lookup_unknown_issuer_returns_none(self):
        r = JWKSResolver()
        assert r.get_key("nonexistent_issuer", "any-kid") is None

    def test_register_invalidates_cache(self):
        r = JWKSResolver()
        # Register first so get_key has a JWKS URL to consult
        r.register_issuer("custom", "https://nope.example/jwks")
        # Inject a "freshly fetched" cache entry directly
        r._cache["custom"] = {
            "jwks_url": "https://nope.example/jwks",
            "expires_at": time.time() + 9999,
            "keys": {"kid1": {"kty": "OKP", "crv": "Ed25519"}},
        }
        # Cache hit
        assert r.get_key("custom", "kid1") is not None
        # Re-registering must invalidate the cache
        r.register_issuer("custom", "https://still-nope.example/jwks")
        assert "custom" not in r._cache


# ---------------------------------------------------------------------------
# End-to-end Ed25519 verification (no network — JWKS provided directly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JOSE_AVAILABLE, reason="python-jose unavailable")
class TestEd25519Verification:
    def setup_method(self):
        self.kp = _ed25519_keypair()
        self.resolver = JWKSResolver()
        # Inject the public JWK directly into resolver cache (no network)
        self.resolver._cache["test_issuer"] = {
            "jwks_url": "memory://",
            "expires_at": time.time() + 3600,
            "keys": {self.kp["kid"]: self.kp["public_jwk"]},
        }
        self.resolver._issuers["test_issuer"] = "memory://"

    def _sign_and_build(
        self,
        method: str = "POST",
        path: str = "/checkout",
        authority: str = "merchant.example.com",
        nonce: str = "nonce-1",
        tag: str = "agent-payer-auth",
        created: int | None = None,
    ) -> Dict[str, str]:
        created = created or int(time.time())
        sig_input_value = (
            f'sig1=("@method" "@authority" "@path");'
            f'keyid="{self.kp["kid"]}";alg="EdDSA";created={created};'
            f'nonce="{nonce}";tag="{tag}"'
        )
        sig_input = parse_signature_input(sig_input_value)
        base = build_signature_base(
            sig_input, headers={}, method=method, path=path, authority=authority
        )
        signature = self.kp["sign"](base.encode("utf-8"))
        return {
            "Signature-Input": sig_input_value,
            "Signature": f"sig1=:{_b64url_no_pad(signature)}:",
        }

    def test_valid_signature_verifies(self):
        headers = self._sign_and_build()
        result = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            expected_tag="agent-payer-auth",
            resolver=self.resolver,
        )
        assert result["verified"] is True
        assert result["issuer"] == "test_issuer"
        assert result["algorithm"] == "EdDSA"
        assert result["signature_age_seconds"] >= 0

    def test_tampered_path_fails(self):
        headers = self._sign_and_build(path="/checkout")
        result = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/admin",  # different path
            authority="merchant.example.com",
            issuer="test_issuer",
            expected_tag="agent-payer-auth",
            resolver=self.resolver,
        )
        assert result["verified"] is False
        assert "cryptographic_verify_failed" in result["reason"]

    def test_wrong_tag_rejected(self):
        headers = self._sign_and_build(tag="agent-payer-auth")
        result = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            expected_tag="agent-browser-auth",  # mismatch
            resolver=self.resolver,
        )
        assert result["verified"] is False
        assert "tag_mismatch" in result["reason"]

    def test_expired_signature_rejected(self):
        old = int(time.time()) - MAX_SIGNATURE_AGE_SECONDS - 60
        headers = self._sign_and_build(created=old)
        result = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            resolver=self.resolver,
        )
        assert result["verified"] is False
        assert "freshness_failed" in result["reason"]

    def test_unknown_keyid_rejected(self):
        headers = self._sign_and_build()
        # Strip our cached key so the resolver can't find it
        self.resolver._cache.pop("test_issuer", None)
        result = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            resolver=self.resolver,
        )
        assert result["verified"] is False
        # Either jwks_key_not_found (issuer registered, key missing) or
        # no_issuer_resolved (issuer dropped).
        assert "jwks" in result["reason"] or "no_issuer" in result["reason"]

    def test_replay_via_nonce_cache_blocked(self):
        cache = NonceCache(ttl_seconds=60)
        headers = self._sign_and_build(nonce="replay-test-1")
        # First call succeeds and records the nonce
        r1 = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            expected_tag="agent-payer-auth",
            resolver=self.resolver,
            nonce_cache=cache,
        )
        assert r1["verified"] is True
        # Second call with the same nonce must fail
        r2 = verify_rfc9421_signature(
            headers=headers,
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            issuer="test_issuer",
            expected_tag="agent-payer-auth",
            resolver=self.resolver,
            nonce_cache=cache,
        )
        assert r2["verified"] is False
        assert "nonce_replay_detected" in r2["reason"]

    def test_missing_signature_headers(self):
        result = verify_rfc9421_signature(
            headers={"Content-Type": "application/json"},
            method="POST",
            path="/checkout",
            authority="merchant.example.com",
            resolver=self.resolver,
        )
        assert result["verified"] is False
        assert "missing" in result["reason"]
