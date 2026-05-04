"""Tests for the pluggable backends in ``agent_security`` (Phase C).

Covers:
  - InMemoryNonceBackend / InMemoryIdempotencyBackend (current default).
  - SQLiteNonceBackend / SQLiteIdempotencyBackend (multi-process safe via
    SQLite WAL mode + busy_timeout).
  - End-to-end multi-process safety: spawn two child processes, share a
    SQLite file, confirm a nonce consumed by process A is rejected by
    process B (real proof, not just claim).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from agent_security import (
    DEFAULT_MAX_ENTRIES,
    IdempotencyStore,
    InMemoryIdempotencyBackend,
    InMemoryNonceBackend,
    NonceCache,
    SQLiteIdempotencyBackend,
    SQLiteNonceBackend,
    canonical_request_fingerprint,
)


pytestmark = pytest.mark.signature


# ---------------------------------------------------------------------------
# Per-backend behavioural parity — the same tests should pass under either
# backend. Parametrize so SQLite and in-memory share the contract.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["in_memory", "sqlite"])
def nonce_cache_with_backend(request, tmp_path):
    if request.param == "in_memory":
        backend = InMemoryNonceBackend(max_entries=1000)
    else:
        backend = SQLiteNonceBackend(
            path=tmp_path / "nonces.sqlite3", max_entries=1000
        )
    yield NonceCache(ttl_seconds=60, backend=backend)


@pytest.fixture(params=["in_memory", "sqlite"])
def idempotency_store_with_backend(request, tmp_path):
    if request.param == "in_memory":
        backend = InMemoryIdempotencyBackend(max_entries=1000)
    else:
        backend = SQLiteIdempotencyBackend(
            path=tmp_path / "idem.sqlite3", max_entries=1000
        )
    yield IdempotencyStore(ttl_seconds=120, backend=backend)


class TestNonceBackendsParity:
    def test_unseen_returns_false(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        assert c.seen("agent-1", "n1") is False

    def test_add_then_seen(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        c.add("agent-1", "n1")
        assert c.seen("agent-1", "n1") is True

    def test_independent_keyids(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        c.add("agent-1", "shared")
        assert c.seen("agent-1", "shared") is True
        assert c.seen("agent-2", "shared") is False

    def test_expired_evicts(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        c.add("agent-1", "old", now=time.time() - 1000)
        assert c.seen("agent-1", "old") is False

    def test_clear(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        c.add("a", "n1")
        c.add("b", "n2")
        c.clear()
        assert c.stats()["size"] == 0

    def test_consume_first_call_accepts_second_replays(self, nonce_cache_with_backend):
        c = nonce_cache_with_backend
        assert c.consume("agent-1", "fresh-nonce") is True
        assert c.consume("agent-1", "fresh-nonce") is False  # replay

    def test_consume_thread_race_exactly_one_winner(self, nonce_cache_with_backend):
        """All N threads racing to consume the same nonce — exactly one wins."""
        import threading

        c = nonce_cache_with_backend
        N = 50
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(N)

        def race():
            barrier.wait()
            r = c.consume("race-key", "thread-race-nonce")
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=race) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        accepted = sum(1 for r in results if r)
        replayed = sum(1 for r in results if not r)
        assert accepted == 1, (
            f"Expected exactly 1 acceptance under thread race, got "
            f"{accepted} accepted / {replayed} replayed. consume() not atomic."
        )
        assert replayed == N - 1


class TestIdempotencyBackendsParity:
    def test_miss_then_store_then_hit(self, idempotency_store_with_backend):
        s = idempotency_store_with_backend
        body = {"amount": 100}
        fp = canonical_request_fingerprint(body)
        assert s.lookup("k1", "agent-A")["status"] == "miss"
        s.store("k1", "agent-A", {"order_id": "o-99"}, request_fingerprint=fp)
        r = s.lookup("k1", "agent-A", request_fingerprint=fp)
        assert r["status"] == "hit"
        assert r["result"] == {"order_id": "o-99"}

    def test_conflict_on_different_body(self, idempotency_store_with_backend):
        s = idempotency_store_with_backend
        s.store(
            "k1", "agent-A", {"ok": True},
            request_fingerprint=canonical_request_fingerprint({"x": 1}),
        )
        r = s.lookup(
            "k1", "agent-A",
            request_fingerprint=canonical_request_fingerprint({"x": 2}),
        )
        assert r["status"] == "conflict"

    def test_independent_agent_ids(self, idempotency_store_with_backend):
        s = idempotency_store_with_backend
        s.store("k1", "agent-A", {"who": "A"})
        s.store("k1", "agent-B", {"who": "B"})
        a = s.lookup("k1", "agent-A")
        b = s.lookup("k1", "agent-B")
        assert a["result"] == {"who": "A"}
        assert b["result"] == {"who": "B"}


# ---------------------------------------------------------------------------
# SQLite-specific: multi-process subprocess test
# ---------------------------------------------------------------------------


# Use %s substitution (not .format()) to avoid having to escape every JSON
# brace in the child script.
_CHILD_SCRIPT_TEMPLATE = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, %r)
from agent_security import NonceCache, SQLiteNonceBackend

action = sys.argv[1]
db_path = sys.argv[2]
keyid = sys.argv[3]
nonce = sys.argv[4]
ready_file = sys.argv[5] if len(sys.argv) > 5 else None
go_file = sys.argv[6] if len(sys.argv) > 6 else None

cache = NonceCache(ttl_seconds=60,
                   backend=SQLiteNonceBackend(path=Path(db_path)))

if action == "consume":
    # Atomic check-and-add. This is the production code path —
    # see agent_security.NonceCache.consume() and the SQLite backend's
    # BEGIN IMMEDIATE + INSERT ... ON CONFLICT DO NOTHING.
    accepted = cache.consume(keyid, nonce)
    print(json.dumps({"accepted": accepted, "replayed": not accepted}))
elif action == "peek":
    print(json.dumps({"seen": cache.seen(keyid, nonce)}))
elif action == "consume_with_barrier":
    # Cross-process race test: signal ready, then wait for go-signal,
    # then race to consume the SAME nonce. Exactly one process should
    # win (accepted=True); the rest must lose (accepted=False).
    if ready_file:
        Path(ready_file).touch()
    if go_file:
        deadline = time.time() + 10.0
        while not Path(go_file).exists() and time.time() < deadline:
            time.sleep(0.005)
    accepted = cache.consume(keyid, nonce)
    print(json.dumps({"accepted": accepted, "replayed": not accepted}))
"""


def _run_child(action: str, db_path: str, keyid: str, nonce: str) -> dict:
    """Run a child Python process that uses SQLiteNonceBackend on db_path."""
    import json

    project_root = str(Path(__file__).resolve().parent.parent)
    script = _CHILD_SCRIPT_TEMPLATE % project_root
    out = subprocess.run(
        [sys.executable, "-c", script, action, db_path, keyid, nonce],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"child failed (rc={out.returncode}): stdout={out.stdout} "
            f"stderr={out.stderr}"
        )
    # Child may print log lines before the JSON; extract the last line.
    last = [ln for ln in out.stdout.strip().splitlines() if ln.startswith("{")][-1]
    return json.loads(last)


class TestSQLiteMultiProcessSafety:
    """Real proof: two SUBPROCESSES sharing the same SQLite file see each
    other's writes — replay defended across the process boundary."""

    def test_nonce_consumed_by_process_a_is_replayed_by_process_b(self, tmp_path):
        db_path = str(tmp_path / "shared.sqlite3")
        keyid = "shared-keyid"
        nonce = "shared-nonce-1"

        # Process A consumes
        r_a = _run_child("consume", db_path, keyid, nonce)
        assert r_a == {"accepted": True, "replayed": False}

        # Process B sees the nonce as already-consumed
        r_b = _run_child("consume", db_path, keyid, nonce)
        assert r_b == {"accepted": False, "replayed": True}

    def test_nonce_unknown_to_process_b_is_unknown_in_peek(self, tmp_path):
        db_path = str(tmp_path / "shared2.sqlite3")
        # Initialize file via process A peek (no consume)
        r_a = _run_child("peek", db_path, "k", "never-added")
        assert r_a == {"seen": False}

        # Process B agrees — empty cache
        r_b = _run_child("peek", db_path, "k", "never-added")
        assert r_b == {"seen": False}

    def test_concurrent_consume_atomic_across_processes(self, tmp_path):
        """REAL race test: spawn N child processes that all race to consume
        the same nonce simultaneously. Exactly ONE must win (accepted=True);
        the rest must lose (replayed=True).

        This caught a TOCTOU race in the pre-atomic ``seen() then add()``
        pattern where 30 concurrent processes ALL got accepted=True.
        """
        import json
        import time

        db_path = str(tmp_path / "race.sqlite3")
        keyid = "race-key"
        nonce = "race-nonce-only-one-winner"

        # Initialize the schema once in the parent so children can race
        # without colliding on schema creation.
        from agent_security import NonceCache, SQLiteNonceBackend
        NonceCache(backend=SQLiteNonceBackend(path=Path(db_path)))

        N_PROCS = 8
        ready_dir = tmp_path / "ready"
        ready_dir.mkdir()
        go_file = tmp_path / "go.flag"

        project_root = str(Path(__file__).resolve().parent.parent)
        script = _CHILD_SCRIPT_TEMPLATE % project_root

        procs = []
        for i in range(N_PROCS):
            ready_file = ready_dir / f"ready_{i}"
            p = subprocess.Popen(
                [
                    sys.executable, "-c", script, "consume_with_barrier",
                    db_path, keyid, nonce, str(ready_file), str(go_file),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            procs.append(p)

        # Wait until all children signal ready
        deadline = time.time() + 15.0
        while time.time() < deadline:
            ready = sum(1 for f in ready_dir.iterdir() if f.is_file())
            if ready >= N_PROCS:
                break
            time.sleep(0.01)
        else:
            for p in procs:
                p.kill()
            raise AssertionError("children did not all become ready in time")

        # Drop the go-flag — children unblock simultaneously
        go_file.touch()

        # Collect results
        accepted_count = 0
        replayed_count = 0
        for p in procs:
            stdout, stderr = p.communicate(timeout=20)
            assert p.returncode == 0, (
                f"child failed: rc={p.returncode} stdout={stdout!r} stderr={stderr!r}"
            )
            last = [ln for ln in stdout.strip().splitlines() if ln.startswith("{")][-1]
            r = json.loads(last)
            if r["accepted"]:
                accepted_count += 1
            else:
                replayed_count += 1

        assert accepted_count == 1, (
            f"Expected exactly 1 acceptance under cross-process race, got "
            f"{accepted_count} accepted / {replayed_count} replayed. "
            f"This means consume() is NOT atomic across processes — replay "
            f"protection is broken."
        )
        assert replayed_count == N_PROCS - 1


# ---------------------------------------------------------------------------
# Module-level singletons follow config
# ---------------------------------------------------------------------------


class TestModuleSingletonsFollowConfig:
    def test_default_is_in_memory(self):
        from agent_security import nonce_cache, idempotency_store

        assert nonce_cache.stats()["backend"] == "in_memory"
        assert idempotency_store.stats()["backend"] == "in_memory"
