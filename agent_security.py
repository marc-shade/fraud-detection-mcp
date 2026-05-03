"""Agent commerce replay-protection primitives.

Two thread-safe in-memory stores with TTL-driven eviction:

  - ``NonceCache`` tracks ``(keyid, nonce)`` tuples for an 8-minute window
    (Visa TAP-compatible). Used by ``acp_signatures.verify_rfc9421_signature``
    to reject replayed signatures.

  - ``IdempotencyStore`` caches the *result* of an operation keyed by
    ``(idempotency_key, agent_id)`` so retries with the same key return the
    identical response (Stripe ACP requires this).

Both classes are intentionally process-local (no Redis dep). For a multi-
process deployment, run an external store or pin client routing.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

# Visa TAP mandates an 8-minute signature/nonce window. Match that here.
DEFAULT_NONCE_TTL_SECONDS = 8 * 60

# Stripe ACP doesn't pin an idempotency window in spec; 24h is the standard
# Stripe API convention for Idempotency-Key TTL.
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60

# Cap on entries before forced eviction (memory safety).
DEFAULT_MAX_ENTRIES = 100_000


class NonceCache:
    """Thread-safe ``(keyid, nonce)`` replay cache with TTL eviction.

    Entries expire after ``ttl_seconds``. A bounded ``max_entries`` cap
    triggers oldest-first eviction to defend against memory exhaustion from
    forged-keyid floods.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._lock = threading.Lock()
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        # (keyid, nonce) -> expires_at
        self._entries: Dict[Tuple[str, str], float] = {}
        # Counter for periodic sweeps (avoid sweeping every call).
        self._calls_since_sweep = 0

    def seen(self, keyid: str, nonce: str, now: Optional[float] = None) -> bool:
        """Return True if ``(keyid, nonce)`` is in cache and not yet expired."""
        if not nonce:
            return False
        now = now or time.time()
        with self._lock:
            self._maybe_sweep_unlocked(now)
            expires = self._entries.get((keyid, nonce))
            if expires is None:
                return False
            if expires < now:
                # expired entry — treat as not seen + clean it
                self._entries.pop((keyid, nonce), None)
                return False
            return True

    def add(self, keyid: str, nonce: str, now: Optional[float] = None) -> None:
        """Mark ``(keyid, nonce)`` as seen with ``ttl_seconds`` expiration."""
        if not nonce:
            return
        now = now or time.time()
        with self._lock:
            self._entries[(keyid, nonce)] = now + self._ttl
            self._maybe_sweep_unlocked(now)
            if len(self._entries) > self._max:
                # Evict the soonest-to-expire entries until under cap.
                excess = len(self._entries) - self._max
                ordered = sorted(self._entries.items(), key=lambda kv: kv[1])
                for k, _ in ordered[:excess]:
                    self._entries.pop(k, None)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics (size, ttl, max_entries)."""
        with self._lock:
            return {
                "size": len(self._entries),
                "ttl_seconds": self._ttl,
                "max_entries": self._max,
            }

    def clear(self) -> None:
        """Remove all entries (testing helper)."""
        with self._lock:
            self._entries.clear()
            self._calls_since_sweep = 0

    def _maybe_sweep_unlocked(self, now: float) -> None:
        """Periodically purge expired entries. Caller must hold the lock."""
        self._calls_since_sweep += 1
        if self._calls_since_sweep < 1024:
            return
        self._calls_since_sweep = 0
        expired = [k for k, exp in self._entries.items() if exp < now]
        for k in expired:
            self._entries.pop(k, None)


class IdempotencyStore:
    """Thread-safe ``(idempotency_key, agent_id)`` -> result cache.

    Stripe ACP requires the merchant to honour the ``Idempotency-Key`` header:
    a retry with the same key + same body MUST return the original response.
    This store caches the response payload so callers can short-circuit
    repeated work and avoid double-spend on transient network failures.

    The store also records a ``request_fingerprint`` (hash of the canonical
    request body) so retries with a *different* body but the same key can be
    flagged as a conflict — Stripe returns HTTP 409 in that case.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_IDEMPOTENCY_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._lock = threading.Lock()
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        # (key, agent_id) -> {request_fingerprint, result, expires_at, created_at}
        self._entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._calls_since_sweep = 0

    def lookup(
        self,
        idempotency_key: str,
        agent_id: str,
        request_fingerprint: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Look up a prior result for ``(key, agent_id)``.

        Returns a dict with ``status`` one of:
          - ``"miss"``     — no prior request; caller should proceed.
          - ``"hit"``      — same request body; ``result`` field carries cached value.
          - ``"conflict"`` — same key but different body; caller should reject (HTTP 409).
        """
        if not idempotency_key:
            return {"status": "miss", "reason": "no_idempotency_key"}
        now = now or time.time()
        with self._lock:
            self._maybe_sweep_unlocked(now)
            entry = self._entries.get((idempotency_key, agent_id))
            if entry is None or entry["expires_at"] < now:
                if entry is not None:
                    self._entries.pop((idempotency_key, agent_id), None)
                return {"status": "miss"}
            if (
                request_fingerprint is not None
                and entry.get("request_fingerprint") is not None
                and request_fingerprint != entry["request_fingerprint"]
            ):
                return {
                    "status": "conflict",
                    "reason": "fingerprint_mismatch",
                    "original_fingerprint": entry["request_fingerprint"],
                    "original_created_at": entry["created_at"],
                }
            return {
                "status": "hit",
                "result": entry["result"],
                "created_at": entry["created_at"],
                "age_seconds": now - entry["created_at"],
            }

    def store(
        self,
        idempotency_key: str,
        agent_id: str,
        result: Any,
        request_fingerprint: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """Cache ``result`` against ``(key, agent_id)``."""
        if not idempotency_key:
            return
        now = now or time.time()
        with self._lock:
            self._entries[(idempotency_key, agent_id)] = {
                "result": result,
                "request_fingerprint": request_fingerprint,
                "expires_at": now + self._ttl,
                "created_at": now,
            }
            self._maybe_sweep_unlocked(now)
            if len(self._entries) > self._max:
                excess = len(self._entries) - self._max
                ordered = sorted(
                    self._entries.items(), key=lambda kv: kv[1]["expires_at"]
                )
                for k, _ in ordered[:excess]:
                    self._entries.pop(k, None)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._entries),
                "ttl_seconds": self._ttl,
                "max_entries": self._max,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._calls_since_sweep = 0

    def _maybe_sweep_unlocked(self, now: float) -> None:
        self._calls_since_sweep += 1
        if self._calls_since_sweep < 1024:
            return
        self._calls_since_sweep = 0
        expired = [k for k, e in self._entries.items() if e["expires_at"] < now]
        for k in expired:
            self._entries.pop(k, None)


def canonical_request_fingerprint(payload: Any) -> str:
    """Produce a stable fingerprint for an idempotency-key payload.

    Uses sorted-key JSON + SHA-256 to give the same fingerprint regardless of
    dict ordering. Falls back to ``repr`` for non-JSON-serializable inputs.
    """
    import hashlib

    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = repr(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Module-level singletons matching the rest of fraud-detection-mcp's pattern
nonce_cache = NonceCache()
idempotency_store = IdempotencyStore()
