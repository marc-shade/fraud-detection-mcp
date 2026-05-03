"""Tests for agent_security.py (Tier 0.3 NonceCache + Tier 0.4 IdempotencyStore)."""
from __future__ import annotations

import time

import pytest

from agent_security import (
    DEFAULT_IDEMPOTENCY_TTL_SECONDS,
    DEFAULT_NONCE_TTL_SECONDS,
    IdempotencyStore,
    NonceCache,
    canonical_request_fingerprint,
)


pytestmark = pytest.mark.signature


# ---------------------------------------------------------------------------
# NonceCache (Visa TAP-compatible 8-min window)
# ---------------------------------------------------------------------------


class TestNonceCache:
    def test_default_ttl_is_8_min(self):
        assert DEFAULT_NONCE_TTL_SECONDS == 8 * 60

    def test_unseen_nonce_returns_false(self):
        c = NonceCache()
        assert c.seen("agent-1", "nonce-fresh") is False

    def test_seen_after_add(self):
        c = NonceCache()
        c.add("agent-1", "n1")
        assert c.seen("agent-1", "n1") is True

    def test_different_keyid_independent(self):
        c = NonceCache()
        c.add("agent-1", "shared-nonce")
        assert c.seen("agent-1", "shared-nonce") is True
        assert c.seen("agent-2", "shared-nonce") is False

    def test_empty_nonce_never_seen(self):
        c = NonceCache()
        c.add("agent-1", "")  # no-op
        assert c.seen("agent-1", "") is False

    def test_expired_nonce_returns_false_and_evicts(self):
        c = NonceCache(ttl_seconds=60)
        # Inject expired entry by adding with negative TTL
        c.add("agent-1", "old", now=time.time() - 120)  # expired ~60s ago
        assert c.seen("agent-1", "old") is False

    def test_clear_resets(self):
        c = NonceCache()
        c.add("agent-1", "n1")
        c.add("agent-2", "n2")
        c.clear()
        assert c.stats()["size"] == 0

    def test_max_entries_eviction(self):
        c = NonceCache(ttl_seconds=600, max_entries=10)
        for i in range(20):
            c.add(f"agent-{i}", f"nonce-{i}")
        assert c.stats()["size"] <= 10

    def test_concurrent_adds_thread_safe(self):
        import threading

        c = NonceCache()

        def add_many(prefix: str) -> None:
            for i in range(100):
                c.add(prefix, f"nonce-{i}")

        threads = [threading.Thread(target=add_many, args=(f"agent-{t}",)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.stats()["size"] == 8 * 100


# ---------------------------------------------------------------------------
# IdempotencyStore (Stripe ACP-compatible)
# ---------------------------------------------------------------------------


class TestIdempotencyStore:
    def test_default_ttl_is_24h(self):
        assert DEFAULT_IDEMPOTENCY_TTL_SECONDS == 24 * 60 * 60

    def test_miss_when_empty(self):
        s = IdempotencyStore()
        r = s.lookup("idem-1", "agent-7")
        assert r["status"] == "miss"

    def test_hit_after_store(self):
        s = IdempotencyStore()
        s.store("idem-1", "agent-7", {"order_id": "o-1"})
        r = s.lookup("idem-1", "agent-7")
        assert r["status"] == "hit"
        assert r["result"] == {"order_id": "o-1"}

    def test_different_agent_does_not_collide(self):
        s = IdempotencyStore()
        s.store("idem-1", "agent-A", {"agent": "A"})
        s.store("idem-1", "agent-B", {"agent": "B"})
        assert s.lookup("idem-1", "agent-A")["result"] == {"agent": "A"}
        assert s.lookup("idem-1", "agent-B")["result"] == {"agent": "B"}

    def test_fingerprint_match_returns_hit(self):
        s = IdempotencyStore()
        body = {"amount": 100, "merchant": "x"}
        fp = canonical_request_fingerprint(body)
        s.store("idem-1", "agent-7", {"ok": True}, request_fingerprint=fp)

        # Same body → hit
        r = s.lookup(
            "idem-1", "agent-7", request_fingerprint=canonical_request_fingerprint(body)
        )
        assert r["status"] == "hit"

    def test_fingerprint_mismatch_returns_conflict(self):
        s = IdempotencyStore()
        body1 = {"amount": 100}
        body2 = {"amount": 200}  # different body, same key
        s.store(
            "idem-1", "agent-7", {"ok": True},
            request_fingerprint=canonical_request_fingerprint(body1),
        )
        r = s.lookup(
            "idem-1", "agent-7",
            request_fingerprint=canonical_request_fingerprint(body2),
        )
        assert r["status"] == "conflict"
        assert "fingerprint_mismatch" in r["reason"]

    def test_expired_entry_returns_miss(self):
        s = IdempotencyStore(ttl_seconds=60)
        # Store with `now` in the past so the entry is already expired
        s.store(
            "idem-1", "agent-7", {"stale": True},
            now=time.time() - 120,  # entry expires 60s ago
        )
        r = s.lookup("idem-1", "agent-7")
        assert r["status"] == "miss"

    def test_no_idempotency_key_is_miss(self):
        s = IdempotencyStore()
        assert s.lookup("", "agent-7")["status"] == "miss"

    def test_store_with_empty_key_is_noop(self):
        s = IdempotencyStore()
        s.store("", "agent-7", {"x": 1})
        assert s.stats()["size"] == 0


# ---------------------------------------------------------------------------
# canonical_request_fingerprint
# ---------------------------------------------------------------------------


class TestRequestFingerprint:
    def test_dict_order_irrelevant(self):
        fp1 = canonical_request_fingerprint({"a": 1, "b": 2})
        fp2 = canonical_request_fingerprint({"b": 2, "a": 1})
        assert fp1 == fp2

    def test_different_values_differ(self):
        assert canonical_request_fingerprint({"a": 1}) != canonical_request_fingerprint(
            {"a": 2}
        )

    def test_returns_hex_sha256(self):
        fp = canonical_request_fingerprint({"a": 1})
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_handles_non_serializable(self):
        # repr fallback should still produce a stable hash
        class X:
            def __repr__(self) -> str:
                return "X()"

        fp1 = canonical_request_fingerprint(X())
        fp2 = canonical_request_fingerprint(X())
        # Falls back to repr; same repr -> same hash
        assert fp1 == fp2
