# Phase 12: User Transaction History & Velocity Features

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add in-memory per-user transaction history tracking to detect velocity anomalies, amount deviations, geographic impossibilities, and merchant diversity spikes — enriching risk factors with real behavioral signals.

**Architecture:** Create a `UserTransactionHistory` class in `server.py` (near `TransactionAnalyzer`) that maintains a bounded deque of recent transactions per user. `TransactionAnalyzer._identify_risk_factors` calls new velocity/deviation methods on the history tracker. A module-level `user_history` singleton is used by `analyze_transaction_impl`. Health check reports history stats.

**Tech Stack:** Pure Python stdlib (collections.deque, threading.Lock, time, math). No new dependencies.

---

### Task 1: Create UserTransactionHistory class

**Files:**
- Modify: `server.py` (insert new class after `TransactionAnalyzer` class, around line 540)

**Step 1: Write the class**

Insert the following class after the `TransactionAnalyzer` class (before `class NetworkAnalyzer`). The class tracks recent transactions per user with bounded deques and provides velocity analysis methods.

```python
class UserTransactionHistory:
    """In-memory per-user transaction history for velocity analysis.

    Thread-safe, bounded deque per user. No external dependencies.
    """

    def __init__(self, max_history: int = 100, max_users: int = 10000):
        self.max_history = max_history
        self.max_users = max_users
        self._history: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Record a transaction for a user."""
        import time as _time
        entry = {
            "amount": float(transaction.get("amount", 0)),
            "merchant": str(transaction.get("merchant", "")),
            "location": str(transaction.get("location", "")),
            "timestamp": transaction.get("timestamp", ""),
            "recorded_at": _time.monotonic(),
        }
        with self._lock:
            if user_id not in self._history:
                # Evict oldest user if at capacity
                if len(self._history) >= self.max_users:
                    oldest_key = next(iter(self._history))
                    del self._history[oldest_key]
                self._history[user_id] = deque(maxlen=self.max_history)
            self._history[user_id].append(entry)

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get transaction history for a user."""
        with self._lock:
            if user_id in self._history:
                return list(self._history[user_id])
            return []

    def check_velocity(self, user_id: str, window_seconds: int = 3600) -> Dict[str, Any]:
        """Check transaction velocity (count in time window).

        Args:
            user_id: User identifier.
            window_seconds: Lookback window in seconds (default: 1 hour).

        Returns:
            Dict with count, window, and is_suspicious flag.
        """
        import time as _time
        cutoff = _time.monotonic() - window_seconds
        history = self.get_history(user_id)
        recent = [h for h in history if h["recorded_at"] > cutoff]
        count = len(recent)
        return {
            "transaction_count": count,
            "window_seconds": window_seconds,
            "is_suspicious": count >= 10,
        }

    def check_amount_deviation(self, user_id: str, current_amount: float) -> Dict[str, Any]:
        """Check if current amount deviates from user's historical pattern.

        Returns:
            Dict with mean, std, z_score, and is_suspicious flag.
        """
        history = self.get_history(user_id)
        amounts = [h["amount"] for h in history]
        if len(amounts) < 3:
            return {
                "mean": 0.0,
                "std": 0.0,
                "z_score": 0.0,
                "is_suspicious": False,
                "insufficient_history": True,
            }
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts))
        if std_amt < 1e-6:
            z_score = 0.0
        else:
            z_score = (current_amount - mean_amt) / std_amt
        return {
            "mean": round(mean_amt, 2),
            "std": round(std_amt, 2),
            "z_score": round(z_score, 2),
            "is_suspicious": abs(z_score) > 3.0,
            "insufficient_history": False,
        }

    def check_geographic_velocity(self, user_id: str) -> Dict[str, Any]:
        """Detect impossible travel (different locations in rapid succession).

        Returns:
            Dict with location_changes, time_between, and is_suspicious flag.
        """
        history = self.get_history(user_id)
        if len(history) < 2:
            return {
                "location_changes": 0,
                "is_suspicious": False,
                "insufficient_history": True,
            }
        last = history[-1]
        prev = history[-2]
        same_location = last["location"].lower().strip() == prev["location"].lower().strip()
        time_between = last["recorded_at"] - prev["recorded_at"]
        return {
            "location_changes": 0 if same_location else 1,
            "time_between_seconds": round(time_between, 2),
            "is_suspicious": not same_location and time_between < 300,
            "insufficient_history": False,
        }

    def check_merchant_diversity(self, user_id: str, window_seconds: int = 3600) -> Dict[str, Any]:
        """Check merchant diversity in time window (card testing signal).

        Returns:
            Dict with unique_merchants, total, and is_suspicious flag.
        """
        import time as _time
        cutoff = _time.monotonic() - window_seconds
        history = self.get_history(user_id)
        recent = [h for h in history if h["recorded_at"] > cutoff]
        merchants = set(h["merchant"] for h in recent if h["merchant"])
        return {
            "unique_merchants": len(merchants),
            "total_transactions": len(recent),
            "window_seconds": window_seconds,
            "is_suspicious": len(merchants) >= 5 and len(recent) >= 5,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get history tracker statistics."""
        with self._lock:
            total_entries = sum(len(d) for d in self._history.values())
            return {
                "tracked_users": len(self._history),
                "max_users": self.max_users,
                "total_entries": total_entries,
                "max_history_per_user": self.max_history,
            }

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset history for a user or all users."""
        with self._lock:
            if user_id is not None:
                self._history.pop(user_id, None)
            else:
                self._history.clear()
```

**Step 2: Add `import threading` if not already present**

Check imports at top of `server.py`. Add `import threading` if missing.

**Step 3: Initialize the singleton**

After the `rate_limiter` / `sanitizer` initialization block (around line 756), add:

```python
# Initialize user transaction history tracker
user_history = UserTransactionHistory(max_history=100, max_users=10000)
```

**Step 4: Verify server imports cleanly**

Run: `python -c "import server; print('user_history:', server.user_history is not None)"`
Expected: `user_history: True`

**Step 5: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 505 passed, 2 skipped

**Step 6: Commit**

```bash
git add server.py
git commit -m "feat: Add UserTransactionHistory class with velocity analysis methods"
```

---

### Task 2: Integrate history tracking into analyze_transaction_impl

**Files:**
- Modify: `server.py` — `analyze_transaction_impl` (around line 839)

**Step 1: Record transaction in history**

In `analyze_transaction_impl`, after the primary transaction analysis line (`transaction_result = transaction_analyzer.analyze_transaction(transaction_data)` at ~line 840), add:

```python
        # Record transaction in user history for velocity analysis
        user_id = str(transaction_data.get("user_id", "anonymous"))
        user_history.record(user_id, transaction_data)
```

**Step 2: Add velocity risk factors**

After recording, add velocity checks that append to the existing `risk_factors`:

```python
        # Velocity-based risk factors
        velocity_info = user_history.check_velocity(user_id)
        if velocity_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "high_transaction_velocity"
            )

        amount_info = user_history.check_amount_deviation(
            user_id, float(transaction_data.get("amount", 0))
        )
        if amount_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "unusual_amount_deviation"
            )

        geo_info = user_history.check_geographic_velocity(user_id)
        if geo_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "impossible_travel_detected"
            )

        merchant_info = user_history.check_merchant_diversity(user_id)
        if merchant_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "high_merchant_diversity"
            )

        # Attach velocity details to results
        velocity_analysis = {
            "velocity": velocity_info,
            "amount_deviation": amount_info,
            "geographic": geo_info,
            "merchant_diversity": merchant_info,
        }
```

**Step 3: Include velocity_analysis in results**

After the `results` dict is constructed (around line 860), add:

```python
        results["velocity_analysis"] = velocity_analysis
```

**Step 4: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 505 passed, 2 skipped

**Step 5: Commit**

```bash
git add server.py
git commit -m "feat: Integrate velocity analysis into analyze_transaction_impl"
```

---

### Task 3: Add user history stats to health_check_impl

**Files:**
- Modify: `server.py` — `health_check_impl` (around line 1395)

**Step 1: Add user_history section to health check**

After the `"security_utils"` section in `health_check_impl`, add:

```python
        "user_history": user_history.get_stats(),
```

**Step 2: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 505 passed, 2 skipped

**Step 3: Commit**

```bash
git add server.py
git commit -m "feat: Add user_history stats to health_check_impl"
```

---

### Task 4: Write comprehensive tests

**Files:**
- Create: `tests/test_user_history.py`
- Modify: `pytest.ini` (add `velocity` marker)

**Step 1: Write test file**

```python
"""Tests for Phase 12: User Transaction History & Velocity Features"""

import time
import threading
import pytest


# ---------------------------------------------------------------------------
# UserTransactionHistory unit tests
# ---------------------------------------------------------------------------
class TestUserTransactionHistory:
    """Test UserTransactionHistory class."""

    def test_record_and_retrieve(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("user1", {"amount": 100, "merchant": "Amazon", "location": "US", "timestamp": ""})
        history = h.get_history("user1")
        assert len(history) == 1
        assert history[0]["amount"] == 100.0

    def test_empty_history(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        assert h.get_history("nonexistent") == []

    def test_max_history_per_user(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory(max_history=5)
        for i in range(10):
            h.record("user1", {"amount": i, "merchant": "M", "location": "US", "timestamp": ""})
        history = h.get_history("user1")
        assert len(history) == 5
        # Oldest entries evicted; most recent kept
        assert history[0]["amount"] == 5.0

    def test_max_users_eviction(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory(max_users=3)
        h.record("user1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("user2", {"amount": 20, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("user3", {"amount": 30, "merchant": "M", "location": "US", "timestamp": ""})
        # Adding a 4th user evicts the oldest (user1)
        h.record("user4", {"amount": 40, "merchant": "M", "location": "US", "timestamp": ""})
        assert h.get_history("user1") == []
        assert len(h.get_history("user4")) == 1

    def test_get_stats(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory(max_history=50, max_users=1000)
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        stats = h.get_stats()
        assert stats["tracked_users"] == 2
        assert stats["total_entries"] == 2
        assert stats["max_users"] == 1000

    def test_reset_specific_user(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        h.reset("a")
        assert h.get_history("a") == []
        assert len(h.get_history("b")) == 1

    def test_reset_all(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        h.reset()
        assert h.get_stats()["tracked_users"] == 0

    def test_thread_safety(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory(max_history=10000)
        errors = []

        def add_records(uid):
            try:
                for i in range(100):
                    h.record(uid, {"amount": i, "merchant": "M", "location": "US", "timestamp": ""})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_records, args=(f"user{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert h.get_stats()["tracked_users"] == 5


# ---------------------------------------------------------------------------
# Velocity checks
# ---------------------------------------------------------------------------
class TestVelocityChecks:
    """Test velocity analysis methods."""

    def test_velocity_under_limit(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        for i in range(3):
            h.record("u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_velocity("u1")
        assert result["transaction_count"] == 3
        assert not result["is_suspicious"]

    def test_velocity_over_limit(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        for i in range(12):
            h.record("u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_velocity("u1")
        assert result["transaction_count"] == 12
        assert result["is_suspicious"]

    def test_velocity_empty_user(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        result = h.check_velocity("nobody")
        assert result["transaction_count"] == 0
        assert not result["is_suspicious"]

    def test_amount_deviation_normal(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        for i in range(10):
            h.record("u1", {"amount": 50 + i, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_amount_deviation("u1", 55.0)
        assert not result["is_suspicious"]
        assert not result["insufficient_history"]
        assert abs(result["z_score"]) < 3.0

    def test_amount_deviation_suspicious(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        for i in range(10):
            h.record("u1", {"amount": 20, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_amount_deviation("u1", 50000.0)
        assert result["is_suspicious"]
        assert result["z_score"] > 3.0

    def test_amount_deviation_insufficient_history(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("u1", {"amount": 100, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_amount_deviation("u1", 500.0)
        assert result["insufficient_history"]
        assert not result["is_suspicious"]

    def test_geographic_velocity_same_location(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("u1", {"amount": 10, "merchant": "M", "location": "United States", "timestamp": ""})
        h.record("u1", {"amount": 20, "merchant": "M", "location": "United States", "timestamp": ""})
        result = h.check_geographic_velocity("u1")
        assert result["location_changes"] == 0
        assert not result["is_suspicious"]

    def test_geographic_velocity_different_location_fast(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("u1", {"amount": 10, "merchant": "M", "location": "United States", "timestamp": ""})
        # Recorded immediately after — within 300s window
        h.record("u1", {"amount": 20, "merchant": "M", "location": "Japan", "timestamp": ""})
        result = h.check_geographic_velocity("u1")
        assert result["location_changes"] == 1
        assert result["is_suspicious"]

    def test_geographic_velocity_insufficient_history(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        h.record("u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""})
        result = h.check_geographic_velocity("u1")
        assert result["insufficient_history"]

    def test_merchant_diversity_low(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        for i in range(3):
            h.record("u1", {"amount": 10, "merchant": "Amazon", "location": "US", "timestamp": ""})
        result = h.check_merchant_diversity("u1")
        assert result["unique_merchants"] == 1
        assert not result["is_suspicious"]

    def test_merchant_diversity_suspicious(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        merchants = ["Amazon", "Walmart", "Target", "BestBuy", "Costco", "Apple", "Google"]
        for m in merchants:
            h.record("u1", {"amount": 10, "merchant": m, "location": "US", "timestamp": ""})
        result = h.check_merchant_diversity("u1")
        assert result["unique_merchants"] == 7
        assert result["is_suspicious"]


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------
class TestHistoryServerIntegration:
    """Test history integration in server.py."""

    def test_user_history_exists(self):
        import server
        assert hasattr(server, "user_history")
        assert server.user_history is not None

    def test_user_history_class_available(self):
        from server import UserTransactionHistory
        h = UserTransactionHistory()
        assert h.get_stats()["tracked_users"] == 0

    def test_health_check_has_user_history(self):
        from server import health_check_impl
        result = health_check_impl()
        assert "user_history" in result
        assert "tracked_users" in result["user_history"]
        assert "max_users" in result["user_history"]

    def test_analyze_transaction_returns_velocity(self):
        import server
        txn = {
            "transaction_id": "vel-test-001",
            "user_id": "velocity-user",
            "amount": 100.0,
            "merchant": "TestStore",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn)
        assert "velocity_analysis" in result
        va = result["velocity_analysis"]
        assert "velocity" in va
        assert "amount_deviation" in va
        assert "geographic" in va
        assert "merchant_diversity" in va

    def test_velocity_accumulates_across_calls(self):
        import server
        # Reset history for clean test
        server.user_history.reset("velocity-accum-user")
        for i in range(3):
            txn = {
                "transaction_id": f"vel-accum-{i}",
                "user_id": "velocity-accum-user",
                "amount": 50.0,
                "merchant": "Store",
                "location": "United States",
                "timestamp": "2026-01-15T14:30:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        va = result["velocity_analysis"]
        assert va["velocity"]["transaction_count"] >= 3

    def test_high_velocity_adds_risk_factor(self):
        import server
        server.user_history.reset("burst-user")
        # Flood 12 transactions to trigger velocity flag
        for i in range(12):
            txn = {
                "transaction_id": f"burst-{i}",
                "user_id": "burst-user",
                "amount": 25.0,
                "merchant": "Store",
                "location": "United States",
                "timestamp": "2026-01-15T14:30:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        # After 12 rapid transactions, velocity should be flagged
        anomalies = result.get("detected_anomalies", [])
        assert "high_transaction_velocity" in anomalies

    def test_amount_deviation_adds_risk_factor(self):
        import server
        server.user_history.reset("deviation-user")
        # Build history with low amounts
        for i in range(5):
            txn = {
                "transaction_id": f"dev-small-{i}",
                "user_id": "deviation-user",
                "amount": 20.0,
                "merchant": "Store",
                "location": "United States",
                "timestamp": "2026-01-15T14:30:00Z",
                "payment_method": "credit_card",
            }
            server.analyze_transaction_impl(txn)
        # Now send a very large transaction
        big_txn = {
            "transaction_id": "dev-big",
            "user_id": "deviation-user",
            "amount": 99999.0,
            "merchant": "Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(big_txn)
        anomalies = result.get("detected_anomalies", [])
        assert "unusual_amount_deviation" in anomalies

    def test_impossible_travel_adds_risk_factor(self):
        import server
        server.user_history.reset("travel-user")
        txn1 = {
            "transaction_id": "travel-1",
            "user_id": "travel-user",
            "amount": 50.0,
            "merchant": "Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        server.analyze_transaction_impl(txn1)
        # Immediately from a different country
        txn2 = {
            "transaction_id": "travel-2",
            "user_id": "travel-user",
            "amount": 50.0,
            "merchant": "Store",
            "location": "Japan",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn2)
        anomalies = result.get("detected_anomalies", [])
        assert "impossible_travel_detected" in anomalies

    def test_merchant_diversity_adds_risk_factor(self):
        import server
        server.user_history.reset("merchant-user")
        merchants = ["Amazon", "Walmart", "Target", "BestBuy", "Costco", "Apple"]
        for i, m in enumerate(merchants):
            txn = {
                "transaction_id": f"merch-{i}",
                "user_id": "merchant-user",
                "amount": 10.0,
                "merchant": m,
                "location": "United States",
                "timestamp": "2026-01-15T14:30:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        anomalies = result.get("detected_anomalies", [])
        assert "high_merchant_diversity" in anomalies
```

**Step 2: Add velocity marker to pytest.ini**

Add `velocity: Tests for user transaction history and velocity analysis` to the markers list.

**Step 3: Run new tests**

Run: `python -m pytest tests/test_user_history.py -v --tb=short`
Expected: All tests pass

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: ~535+ passed, 2 skipped

**Step 5: Commit**

```bash
git add tests/test_user_history.py pytest.ini
git commit -m "test: Add Phase 12 user history and velocity tests"
```

---

### Task 5: Final verification and lint

**Step 1: Run ruff**

Run: `ruff check server.py tests/test_user_history.py`
Expected: Clean (fix any issues)

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: ~535+ passed, 2 skipped

**Step 3: Commit any fixes**

```bash
git add server.py tests/test_user_history.py
git commit -m "docs: Phase 12 velocity features complete, lint clean"
```
