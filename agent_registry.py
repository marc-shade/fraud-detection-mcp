"""AgentIdentityRegistry — JSON-backed, thread- and process-safe.

Extracted from ``server.py`` 2026-05-04 so subprocess-based concurrency
tests can import it without paying the full server import cost
(model fits, MCP setup, etc.). The class itself depends only on stdlib
modules; importing this file should be near-instant.

The exported singleton ``agent_registry`` mirrors the one previously
exposed at ``server.agent_registry`` — server.py now re-exports it from
here for backward compatibility with existing tests and callers.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# fcntl is POSIX-only; on Windows the registry falls back to thread-only
# locking (no cross-process lock) and the operator should know about it.
try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — Windows path
    fcntl = None  # type: ignore[assignment]


class AgentIdentityRegistry:
    """Thread- and process-safe JSON-backed registry of known AI agent identities.

    Tracks agent identifiers, types, trust scores, and transaction history.

    **Concurrency model**: every mutation acquires an exclusive ``fcntl``
    advisory lock on a sibling ``.lock`` file, re-reads the JSON under
    the lock, applies the change, and writes back via ``tempfile +
    os.replace`` (atomic rename). This means concurrent processes see
    each other's writes; the older "open + json.dump" pattern was
    demonstrated to lose 87% of registrations under 8-process
    contention (see ``tests/test_agent_identity.py``).
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or Path("data/agent_registry.json")
        self._thread_lock = threading.Lock()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._load_unlocked()

    def _load_unlocked(self) -> None:
        """Load registry from disk into ``self._agents``. Used inside the
        critical section after acquiring the file lock — and once at __init__
        before any other process is accessing this instance."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = f.read()
                if data.strip():
                    self._agents = json.loads(data)
                else:
                    self._agents = {}
            except (json.JSONDecodeError, OSError):
                # Corrupted JSON or read error: treat as empty so we can
                # rebuild rather than refuse-to-start.
                self._agents = {}
        else:
            self._agents = {}

    def _atomic_write_unlocked(self) -> None:
        """Write ``self._agents`` to disk via tempfile + ``os.replace``.

        ``os.replace`` is atomic on POSIX and Windows (NTFS) — readers
        see either the old file or the new file, never a half-written
        one. Caller MUST hold both ``_thread_lock`` and the file lock.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile as _tempfile
        fd: Optional[int] = None
        tmp_path: Optional[Path] = None
        try:
            fd, tmp_path_str = _tempfile.mkstemp(
                prefix=".agent_registry.",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            tmp_path = Path(tmp_path_str)
            with os.fdopen(fd, "w") as f:
                fd = None  # ownership transferred to file object
                json.dump(self._agents, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
            tmp_path = None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @contextlib.contextmanager
    def _file_lock(self):
        """Cross-process exclusive lock via ``fcntl.flock`` on the
        sibling ``.lock`` file. POSIX-only; on platforms without
        ``fcntl`` (Windows) this degrades to thread-local locking only.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            yield
            return
        lock_fd = os.open(
            str(self._lock_path),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def register(
        self, agent_id: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new agent or return existing entry. Multi-process safe."""
        with self._thread_lock, self._file_lock():
            self._load_unlocked()
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
            self._atomic_write_unlocked()
            return entry

    def lookup(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Look up an agent. Reads in-memory cache (no file lock); for
        cross-process freshness call ``refresh()`` first."""
        with self._thread_lock:
            return self._agents.get(agent_id)

    def refresh(self) -> None:
        """Re-read the registry file under the file lock. Use this when a
        caller needs to see writes made by other processes since this
        instance's last mutation."""
        with self._thread_lock, self._file_lock():
            self._load_unlocked()

    def record_transaction(self, agent_id: str) -> None:
        """Record a transaction for an agent, incrementing count. MP-safe."""
        with self._thread_lock, self._file_lock():
            self._load_unlocked()
            if agent_id in self._agents:
                self._agents[agent_id]["transaction_count"] += 1
                self._agents[agent_id]["last_seen"] = datetime.now().isoformat()
                self._atomic_write_unlocked()

    def update_trust(self, agent_id: str, trust_score: float) -> None:
        """Update an agent's trust score (clamped to [0, 1]). MP-safe."""
        with self._thread_lock, self._file_lock():
            self._load_unlocked()
            if agent_id in self._agents:
                self._agents[agent_id]["trust_score"] = max(0.0, min(1.0, trust_score))
                self._atomic_write_unlocked()

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered agents (in-memory snapshot)."""
        with self._thread_lock:
            return dict(self._agents)


# Module-level singleton — server.py re-exports for backward compatibility.
agent_registry = AgentIdentityRegistry()


__all__ = ["AgentIdentityRegistry", "agent_registry"]
