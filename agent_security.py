"""Agent commerce replay-protection primitives — pluggable backends.

Two thread-safe stores with TTL-driven eviction:

  - ``NonceCache`` tracks ``(keyid, nonce)`` tuples for an 8-minute window
    (Visa TAP-compatible). Used by ``acp_signatures.verify_rfc9421_signature``
    to reject replayed signatures.
  - ``IdempotencyStore`` caches the *result* of an operation keyed by
    ``(idempotency_key, agent_id)`` so retries with the same key return the
    identical response (Stripe ACP requires this).

Both classes delegate to a pluggable backend:

  - ``InMemoryNonceBackend`` / ``InMemoryIdempotencyBackend`` (default) —
    process-local, fast, zero-config. Use for single-process deployments
    or when distributed replay protection is not required.
  - ``SQLiteNonceBackend`` / ``SQLiteIdempotencyBackend`` — file-backed
    via SQLite WAL mode + busy_timeout. Multi-process safe: workers
    sharing the same SQLite file see consistent replay state. Zero infra
    dependency (sqlite3 ships with Python).

Backend selection is governed by ``config.ACP_BACKEND`` (``in_memory`` or
``sqlite``) plus ``config.ACP_SQLITE_PATH``. The module-level singletons
``nonce_cache`` and ``idempotency_store`` are wired up at import time
according to that config.
"""
from __future__ import annotations

import abc
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Visa TAP mandates an 8-minute signature/nonce window. Match that here.
DEFAULT_NONCE_TTL_SECONDS = 8 * 60

# Stripe ACP doesn't pin an idempotency window in spec; 24h is the standard
# Stripe API convention for Idempotency-Key TTL.
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60

# Cap on entries before forced eviction (memory safety).
DEFAULT_MAX_ENTRIES = 100_000


# ============================================================================
# Backend interfaces
# ============================================================================


class NonceBackend(abc.ABC):
    """Storage backend for ``NonceCache``.

    Implementations MUST be safe to call from multiple threads. Implementations
    MAY also be safe to call from multiple processes (the SQLite backend is;
    the in-memory backend is not).
    """

    @abc.abstractmethod
    def seen(self, keyid: str, nonce: str, now: float) -> bool: ...

    @abc.abstractmethod
    def add(self, keyid: str, nonce: str, expires_at: float) -> None: ...

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def clear(self) -> None: ...


class IdempotencyBackend(abc.ABC):
    """Storage backend for ``IdempotencyStore``."""

    @abc.abstractmethod
    def lookup(
        self, idempotency_key: str, agent_id: str, now: float
    ) -> Optional[Dict[str, Any]]:
        """Return the entry dict (with `result`, `request_fingerprint`,
        `created_at`, `expires_at`) or ``None`` if absent or expired."""

    @abc.abstractmethod
    def store(
        self,
        idempotency_key: str,
        agent_id: str,
        result: Any,
        request_fingerprint: Optional[str],
        expires_at: float,
        created_at: float,
    ) -> None: ...

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def clear(self) -> None: ...


# ============================================================================
# In-memory backends
# ============================================================================


class InMemoryNonceBackend(NonceBackend):
    """Process-local thread-safe nonce store. Default backend."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._lock = threading.Lock()
        self._max = int(max_entries)
        self._entries: Dict[Tuple[str, str], float] = {}
        self._calls_since_sweep = 0

    def seen(self, keyid: str, nonce: str, now: float) -> bool:
        with self._lock:
            self._maybe_sweep_unlocked(now)
            expires = self._entries.get((keyid, nonce))
            if expires is None:
                return False
            if expires < now:
                self._entries.pop((keyid, nonce), None)
                return False
            return True

    def add(self, keyid: str, nonce: str, expires_at: float) -> None:
        with self._lock:
            self._entries[(keyid, nonce)] = expires_at
            now = time.time()
            self._maybe_sweep_unlocked(now)
            if len(self._entries) > self._max:
                excess = len(self._entries) - self._max
                ordered = sorted(self._entries.items(), key=lambda kv: kv[1])
                for k, _ in ordered[:excess]:
                    self._entries.pop(k, None)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._entries),
                "max_entries": self._max,
                "backend": "in_memory",
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
        expired = [k for k, exp in self._entries.items() if exp < now]
        for k in expired:
            self._entries.pop(k, None)


class InMemoryIdempotencyBackend(IdempotencyBackend):
    """Process-local thread-safe idempotency store. Default backend."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._lock = threading.Lock()
        self._max = int(max_entries)
        self._entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._calls_since_sweep = 0

    def lookup(
        self, idempotency_key: str, agent_id: str, now: float
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._maybe_sweep_unlocked(now)
            entry = self._entries.get((idempotency_key, agent_id))
            if entry is None:
                return None
            if entry["expires_at"] < now:
                self._entries.pop((idempotency_key, agent_id), None)
                return None
            # Return a copy so caller can't mutate stored state
            return dict(entry)

    def store(
        self,
        idempotency_key: str,
        agent_id: str,
        result: Any,
        request_fingerprint: Optional[str],
        expires_at: float,
        created_at: float,
    ) -> None:
        with self._lock:
            self._entries[(idempotency_key, agent_id)] = {
                "result": result,
                "request_fingerprint": request_fingerprint,
                "expires_at": expires_at,
                "created_at": created_at,
            }
            now = time.time()
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
                "max_entries": self._max,
                "backend": "in_memory",
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


# ============================================================================
# SQLite backends — multi-process safe via WAL mode + busy_timeout
# ============================================================================


_SQLITE_PRAGMAS = (
    "PRAGMA journal_mode=WAL;",   # multi-reader + single-writer concurrency
    "PRAGMA synchronous=NORMAL;",  # fsync on commit, not every write
    "PRAGMA busy_timeout=5000;",  # block 5s on lock contention
    "PRAGMA temp_store=MEMORY;",
)


def _open_sqlite(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode + busy_timeout configured.

    Each process should call this independently — connections are NOT
    shareable across processes. A `check_same_thread=False` allows pooling
    inside one process.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        timeout=5.0,
        isolation_level=None,  # autocommit
        check_same_thread=False,
    )
    for pragma in _SQLITE_PRAGMAS:
        conn.execute(pragma)
    return conn


class SQLiteNonceBackend(NonceBackend):
    """SQLite-backed nonce store.

    Multi-process safe: WAL mode lets multiple writers in the same node
    serialise via SQLite's locking, and ``busy_timeout`` blocks rather than
    failing fast on contention. Suitable for fleets with N workers sharing a
    filesystem (typical container-with-volume or single-host multi-process).
    For cross-host distribution, swap in a Redis backend.

    Schema: single table ``nonces(keyid TEXT, nonce TEXT, expires_at REAL,
    PRIMARY KEY(keyid, nonce))``. Eviction sweeps run periodically and are
    bounded by ``max_entries``.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS nonces (
        keyid TEXT NOT NULL,
        nonce TEXT NOT NULL,
        expires_at REAL NOT NULL,
        PRIMARY KEY (keyid, nonce)
    );
    CREATE INDEX IF NOT EXISTS idx_nonces_expires ON nonces(expires_at);
    """

    def __init__(self, path: Path, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._path = Path(path)
        self._max = int(max_entries)
        self._lock = threading.Lock()  # protects per-process connection
        self._conn = _open_sqlite(self._path)
        self._conn.executescript(self._SCHEMA)
        self._calls_since_sweep = 0

    def seen(self, keyid: str, nonce: str, now: float) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT expires_at FROM nonces WHERE keyid = ? AND nonce = ?",
                (keyid, nonce),
            ).fetchone()
            if row is None:
                self._maybe_sweep_unlocked(now)
                return False
            expires_at = row[0]
            if expires_at < now:
                # Expired — evict + report unseen
                self._conn.execute(
                    "DELETE FROM nonces WHERE keyid = ? AND nonce = ?",
                    (keyid, nonce),
                )
                return False
            return True

    def add(self, keyid: str, nonce: str, expires_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO nonces(keyid, nonce, expires_at) "
                "VALUES(?, ?, ?)",
                (keyid, nonce, float(expires_at)),
            )
            now = time.time()
            self._maybe_sweep_unlocked(now)
            self._maybe_evict_unlocked()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            (size,) = self._conn.execute("SELECT COUNT(*) FROM nonces").fetchone()
            return {
                "size": int(size),
                "max_entries": self._max,
                "backend": "sqlite",
                "path": str(self._path),
            }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM nonces")
            self._calls_since_sweep = 0

    def _maybe_sweep_unlocked(self, now: float) -> None:
        self._calls_since_sweep += 1
        if self._calls_since_sweep < 256:
            return
        self._calls_since_sweep = 0
        self._conn.execute("DELETE FROM nonces WHERE expires_at < ?", (now,))

    def _maybe_evict_unlocked(self) -> None:
        (size,) = self._conn.execute("SELECT COUNT(*) FROM nonces").fetchone()
        if size <= self._max:
            return
        excess = size - self._max
        # Evict the soonest-to-expire entries
        self._conn.execute(
            "DELETE FROM nonces WHERE rowid IN ("
            "  SELECT rowid FROM nonces ORDER BY expires_at ASC LIMIT ?"
            ")",
            (excess,),
        )


class SQLiteIdempotencyBackend(IdempotencyBackend):
    """SQLite-backed idempotency store.

    Same multi-process semantics as ``SQLiteNonceBackend``. Stores the
    cached result as JSON when serialisable, otherwise as ``repr()``
    (which trades round-trip fidelity for "always stores something" —
    callers caching non-JSON-serialisable types should be aware).

    Schema: ``idempotency(idempotency_key TEXT, agent_id TEXT,
    result_json TEXT, request_fingerprint TEXT, expires_at REAL,
    created_at REAL, PRIMARY KEY(idempotency_key, agent_id))``.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS idempotency (
        idempotency_key TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        result_json TEXT NOT NULL,
        request_fingerprint TEXT,
        expires_at REAL NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (idempotency_key, agent_id)
    );
    CREATE INDEX IF NOT EXISTS idx_idem_expires ON idempotency(expires_at);
    """

    def __init__(self, path: Path, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._path = Path(path)
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._conn = _open_sqlite(self._path)
        self._conn.executescript(self._SCHEMA)
        self._calls_since_sweep = 0

    def lookup(
        self, idempotency_key: str, agent_id: str, now: float
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json, request_fingerprint, expires_at, created_at "
                "FROM idempotency WHERE idempotency_key = ? AND agent_id = ?",
                (idempotency_key, agent_id),
            ).fetchone()
            if row is None:
                self._maybe_sweep_unlocked(now)
                return None
            result_json, request_fingerprint, expires_at, created_at = row
            if expires_at < now:
                self._conn.execute(
                    "DELETE FROM idempotency WHERE idempotency_key = ? AND agent_id = ?",
                    (idempotency_key, agent_id),
                )
                return None
            try:
                result = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                # Non-JSON-serialisable was stored as repr; return as string.
                result = result_json
            return {
                "result": result,
                "request_fingerprint": request_fingerprint,
                "expires_at": expires_at,
                "created_at": created_at,
            }

    def store(
        self,
        idempotency_key: str,
        agent_id: str,
        result: Any,
        request_fingerprint: Optional[str],
        expires_at: float,
        created_at: float,
    ) -> None:
        try:
            result_json = json.dumps(result, default=str, sort_keys=True)
        except (TypeError, ValueError):
            result_json = repr(result)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO idempotency"
                "(idempotency_key, agent_id, result_json, request_fingerprint, expires_at, created_at)"
                " VALUES(?, ?, ?, ?, ?, ?)",
                (
                    idempotency_key,
                    agent_id,
                    result_json,
                    request_fingerprint,
                    float(expires_at),
                    float(created_at),
                ),
            )
            now = time.time()
            self._maybe_sweep_unlocked(now)
            self._maybe_evict_unlocked()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            (size,) = self._conn.execute(
                "SELECT COUNT(*) FROM idempotency"
            ).fetchone()
            return {
                "size": int(size),
                "max_entries": self._max,
                "backend": "sqlite",
                "path": str(self._path),
            }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM idempotency")
            self._calls_since_sweep = 0

    def _maybe_sweep_unlocked(self, now: float) -> None:
        self._calls_since_sweep += 1
        if self._calls_since_sweep < 256:
            return
        self._calls_since_sweep = 0
        self._conn.execute(
            "DELETE FROM idempotency WHERE expires_at < ?", (now,)
        )

    def _maybe_evict_unlocked(self) -> None:
        (size,) = self._conn.execute("SELECT COUNT(*) FROM idempotency").fetchone()
        if size <= self._max:
            return
        excess = size - self._max
        self._conn.execute(
            "DELETE FROM idempotency WHERE rowid IN ("
            "  SELECT rowid FROM idempotency ORDER BY expires_at ASC LIMIT ?"
            ")",
            (excess,),
        )


# ============================================================================
# Public-API wrappers — delegate to whichever backend was selected
# ============================================================================


class NonceCache:
    """Visa-TAP-compatible (keyid, nonce) replay cache with TTL eviction.

    Wraps a ``NonceBackend``. Default backend is ``InMemoryNonceBackend`` for
    zero-config single-process use; pass a ``SQLiteNonceBackend`` for
    multi-process deployments.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        backend: Optional[NonceBackend] = None,
    ):
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._backend: NonceBackend = backend or InMemoryNonceBackend(max_entries=max_entries)

    def seen(self, keyid: str, nonce: str, now: Optional[float] = None) -> bool:
        if not nonce:
            return False
        return self._backend.seen(keyid, nonce, now or time.time())

    def add(self, keyid: str, nonce: str, now: Optional[float] = None) -> None:
        if not nonce:
            return
        n = now or time.time()
        self._backend.add(keyid, nonce, n + self._ttl)

    def stats(self) -> Dict[str, Any]:
        s = self._backend.stats()
        s["ttl_seconds"] = self._ttl
        return s

    def clear(self) -> None:
        self._backend.clear()


class IdempotencyStore:
    """Stripe-ACP-compatible (idempotency_key, agent_id) → result cache.

    Wraps an ``IdempotencyBackend``. Records ``request_fingerprint`` so a
    retry with the same key but different body is reported as a conflict
    (HTTP 409 in Stripe semantics).
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_IDEMPOTENCY_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        backend: Optional[IdempotencyBackend] = None,
    ):
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._backend: IdempotencyBackend = backend or InMemoryIdempotencyBackend(
            max_entries=max_entries
        )

    def lookup(
        self,
        idempotency_key: str,
        agent_id: str,
        request_fingerprint: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not idempotency_key:
            return {"status": "miss", "reason": "no_idempotency_key"}
        n = now or time.time()
        entry = self._backend.lookup(idempotency_key, agent_id, n)
        if entry is None:
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
            "age_seconds": n - entry["created_at"],
        }

    def store(
        self,
        idempotency_key: str,
        agent_id: str,
        result: Any,
        request_fingerprint: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        if not idempotency_key:
            return
        n = now or time.time()
        self._backend.store(
            idempotency_key=idempotency_key,
            agent_id=agent_id,
            result=result,
            request_fingerprint=request_fingerprint,
            expires_at=n + self._ttl,
            created_at=n,
        )

    def stats(self) -> Dict[str, Any]:
        s = self._backend.stats()
        s["ttl_seconds"] = self._ttl
        return s

    def clear(self) -> None:
        self._backend.clear()


def canonical_request_fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint for an idempotency-key payload.

    Sorted-key JSON serialisation gives the same fingerprint regardless of
    dict ordering. Falls back to ``repr`` for non-JSON-serialisable inputs.
    """
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError):
        canonical = repr(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================================
# Module-level singletons — wired to backends per config.ACP_BACKEND
# ============================================================================


def _make_nonce_backend_from_config() -> NonceBackend:
    """Construct a NonceBackend per the active config."""
    try:
        from config import config as _cfg

        backend_kind = (_cfg.ACP_BACKEND or "in_memory").lower()
        max_entries = int(getattr(_cfg, "ACP_REPLAY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES))
        if backend_kind == "sqlite":
            path = _cfg.ACP_SQLITE_PATH or (Path(_cfg.DATA_DIR) / "agent_security.sqlite3")
            return SQLiteNonceBackend(path=Path(path), max_entries=max_entries)
        return InMemoryNonceBackend(max_entries=max_entries)
    except Exception as e:  # noqa: BLE001
        logger.warning("Falling back to in-memory NonceBackend: %s", e)
        return InMemoryNonceBackend(max_entries=DEFAULT_MAX_ENTRIES)


def _make_idempotency_backend_from_config() -> IdempotencyBackend:
    try:
        from config import config as _cfg

        backend_kind = (_cfg.ACP_BACKEND or "in_memory").lower()
        max_entries = int(getattr(_cfg, "ACP_REPLAY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES))
        if backend_kind == "sqlite":
            path = _cfg.ACP_SQLITE_PATH or (Path(_cfg.DATA_DIR) / "agent_security.sqlite3")
            return SQLiteIdempotencyBackend(path=Path(path), max_entries=max_entries)
        return InMemoryIdempotencyBackend(max_entries=max_entries)
    except Exception as e:  # noqa: BLE001
        logger.warning("Falling back to in-memory IdempotencyBackend: %s", e)
        return InMemoryIdempotencyBackend(max_entries=DEFAULT_MAX_ENTRIES)


def _read_ttl(name: str, default: int) -> int:
    try:
        from config import config as _cfg

        return int(getattr(_cfg, name, default))
    except Exception:  # noqa: BLE001
        return default


nonce_cache = NonceCache(
    ttl_seconds=_read_ttl("ACP_NONCE_TTL_SECONDS", DEFAULT_NONCE_TTL_SECONDS),
    max_entries=_read_ttl("ACP_REPLAY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES),
    backend=_make_nonce_backend_from_config(),
)

idempotency_store = IdempotencyStore(
    ttl_seconds=_read_ttl("ACP_IDEMPOTENCY_TTL_SECONDS", DEFAULT_IDEMPOTENCY_TTL_SECONDS),
    max_entries=_read_ttl("ACP_REPLAY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES),
    backend=_make_idempotency_backend_from_config(),
)


__all__ = [
    "DEFAULT_NONCE_TTL_SECONDS",
    "DEFAULT_IDEMPOTENCY_TTL_SECONDS",
    "DEFAULT_MAX_ENTRIES",
    "NonceBackend",
    "IdempotencyBackend",
    "InMemoryNonceBackend",
    "InMemoryIdempotencyBackend",
    "SQLiteNonceBackend",
    "SQLiteIdempotencyBackend",
    "NonceCache",
    "IdempotencyStore",
    "canonical_request_fingerprint",
    "nonce_cache",
    "idempotency_store",
]
