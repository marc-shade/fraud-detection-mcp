# Phase 11: Lightweight Security Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add input sanitization and in-memory rate limiting to harden the MCP server, with zero new external dependencies.

**Architecture:** Create `security_utils.py` with `InputSanitizer` (extracted from `security.py`) and `InMemoryRateLimiter` (new, dict-based sliding window). Import in `server.py` with `SECURITY_UTILS_AVAILABLE` flag. Apply sanitization to user-facing tools and rate limiting as a decorator.

**Tech Stack:** Pure Python stdlib (re, time, collections, threading)

---

### Task 1: Create security_utils.py module

**Files:**
- Create: `security_utils.py`

**Step 1: Write the module**

Create `security_utils.py` with two classes:

```python
#!/usr/bin/env python3
"""
Lightweight security utilities for Fraud Detection MCP Server.
Pure Python — no external dependencies (no FastAPI, Redis, bcrypt, etc.)
"""

import re
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Input sanitization to prevent injection attacks.

    Extracted from security.py OWASP A03 mitigations.
    Pure Python, no external dependencies.
    """

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input.

        - Removes null bytes
        - Strips control characters (keeps newline, tab, CR)
        - Truncates to max_length
        - Strips whitespace
        """
        if not isinstance(value, str):
            return str(value)

        # Remove null bytes
        value = value.replace("\x00", "")

        # Remove control characters except \n, \t, \r
        value = "".join(
            ch for ch in value if ch.isprintable() or ch in "\n\t\r"
        )

        # Truncate
        value = value[:max_length]

        # Strip
        return value.strip()

    @staticmethod
    def sanitize_dict(
        data: Dict[str, Any],
        allowed_keys: Optional[List[str]] = None,
        max_string_length: int = 1000,
    ) -> Dict[str, Any]:
        """Sanitize a dictionary, optionally whitelisting keys.

        Recursively sanitizes string values. If allowed_keys is provided,
        only those keys are kept (mass-assignment protection).
        """
        if allowed_keys is not None:
            data = {k: v for k, v in data.items() if k in allowed_keys}

        sanitized = {}
        for key, value in data.items():
            key = InputSanitizer.sanitize_string(str(key), max_length=255)
            if isinstance(value, str):
                sanitized[key] = InputSanitizer.sanitize_string(
                    value, max_length=max_string_length
                )
            elif isinstance(value, dict):
                sanitized[key] = InputSanitizer.sanitize_dict(
                    value, max_string_length=max_string_length
                )
            elif isinstance(value, list):
                sanitized[key] = InputSanitizer.sanitize_list(
                    value, max_string_length=max_string_length
                )
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def sanitize_list(
        data: list,
        max_string_length: int = 1000,
    ) -> list:
        """Sanitize a list, recursively sanitizing elements."""
        sanitized = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(
                    InputSanitizer.sanitize_string(item, max_length=max_string_length)
                )
            elif isinstance(item, dict):
                sanitized.append(
                    InputSanitizer.sanitize_dict(item, max_string_length=max_string_length)
                )
            elif isinstance(item, list):
                sanitized.append(
                    InputSanitizer.sanitize_list(item, max_string_length=max_string_length)
                )
            else:
                sanitized.append(item)
        return sanitized

    @staticmethod
    def contains_suspicious_patterns(value: str) -> Tuple[bool, List[str]]:
        """Check for common injection patterns.

        Returns (is_suspicious, list_of_matched_patterns).
        """
        patterns = {
            "sql_injection": r"(?i)(\b(union|select|insert|update|delete|drop|alter|exec|execute)\b.*\b(from|into|table|where|set)\b)",
            "script_injection": r"<script[^>]*>",
            "path_traversal": r"\.\./|\.\.\\",
            "null_byte": r"\x00",
            "command_injection": r"[;&|`$]",
        }

        matched = []
        for name, pattern in patterns.items():
            if re.search(pattern, value):
                matched.append(name)

        return len(matched) > 0, matched


class InMemoryRateLimiter:
    """In-memory sliding window rate limiter.

    Thread-safe, no external dependencies (no Redis).
    Uses a per-key sliding window counter.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Window duration in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._total_allowed = 0
        self._total_denied = 0

    def check(self, key: str = "global") -> Tuple[bool, Dict[str, Any]]:
        """Check if a request is allowed.

        Args:
            key: Identifier (e.g., tool name, user ID).

        Returns:
            Tuple of (allowed, info_dict).
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Prune expired entries
            self._requests[key] = [
                t for t in self._requests[key] if t > cutoff
            ]

            current_count = len(self._requests[key])

            if current_count >= self.max_requests:
                self._total_denied += 1
                oldest = self._requests[key][0] if self._requests[key] else now
                reset_in = self.window_seconds - (now - oldest)
                return False, {
                    "allowed": False,
                    "limit": self.max_requests,
                    "remaining": 0,
                    "reset_in_seconds": round(max(0, reset_in), 1),
                }

            # Record this request
            self._requests[key].append(now)
            self._total_allowed += 1

            return True, {
                "allowed": True,
                "limit": self.max_requests,
                "remaining": self.max_requests - current_count - 1,
                "reset_in_seconds": round(self.window_seconds, 1),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            active_keys = len(self._requests)
            total_tracked = sum(len(v) for v in self._requests.values())

        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_keys": active_keys,
            "tracked_requests": total_tracked,
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
        }

    def reset(self, key: Optional[str] = None):
        """Reset rate limit state.

        Args:
            key: Specific key to reset, or None to reset all.
        """
        with self._lock:
            if key is not None:
                self._requests.pop(key, None)
            else:
                self._requests.clear()
```

**Step 2: Verify module imports cleanly**

Run: `python -c "from security_utils import InputSanitizer, InMemoryRateLimiter; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add security_utils.py
git commit -m "feat: Add security_utils.py with InputSanitizer and InMemoryRateLimiter"
```

---

### Task 2: Import security_utils in server.py

**Files:**
- Modify: `server.py` (imports section, after the synthetic data import block)

**Step 1: Add the import block**

After the `SYNTHETIC_DATA_AVAILABLE` import block, add:

```python
# Security utilities (graceful degradation)
try:
    from security_utils import InputSanitizer, InMemoryRateLimiter
    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    InputSanitizer = None
    InMemoryRateLimiter = None
```

**Step 2: Initialize rate limiter singleton**

After the `synthetic_data_integration` initialization block (around line 733), add:

```python
# Initialize rate limiter
if SECURITY_UTILS_AVAILABLE and InMemoryRateLimiter is not None:
    rate_limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60)
else:
    rate_limiter = None
```

**Step 3: Verify server starts**

Run: `python -c "import server; print('SECURITY_UTILS_AVAILABLE:', server.SECURITY_UTILS_AVAILABLE)"`
Expected: `SECURITY_UTILS_AVAILABLE: True`

**Step 4: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 476 passed, 2 skipped

**Step 5: Commit**

```bash
git add server.py
git commit -m "feat: Add SECURITY_UTILS_AVAILABLE flag and rate limiter initialization"
```

---

### Task 3: Apply input sanitization to user-facing tools

**Files:**
- Modify: `server.py` — `analyze_transaction_impl` (~line 768) and `detect_behavioral_anomaly_impl` (~line 915)

**Step 1: Add sanitization to analyze_transaction_impl**

At the start of `analyze_transaction_impl`, after the `_start = _time.monotonic()` line and before the `try:` block (or at the top of the try block before validation), add:

```python
        # Sanitize inputs
        if SECURITY_UTILS_AVAILABLE and InputSanitizer is not None:
            transaction_data = InputSanitizer.sanitize_dict(transaction_data)
            if behavioral_data is not None:
                behavioral_data = InputSanitizer.sanitize_dict(behavioral_data)
```

**Step 2: Add sanitization to detect_behavioral_anomaly_impl**

At the start of `detect_behavioral_anomaly_impl`, before the validation call, add:

```python
        # Sanitize inputs
        if SECURITY_UTILS_AVAILABLE and InputSanitizer is not None:
            behavioral_data = InputSanitizer.sanitize_dict(behavioral_data)
```

**Step 3: Add rate limiting to analyze_transaction_impl**

After the sanitization block in `analyze_transaction_impl`, add:

```python
        # Rate limiting
        if rate_limiter is not None:
            allowed, limit_info = rate_limiter.check("analyze_transaction")
            if not allowed:
                return {
                    "error": "Rate limit exceeded",
                    "status": "rate_limited",
                    "rate_limit": limit_info,
                }
```

**Step 4: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 476 passed, 2 skipped

**Step 5: Commit**

```bash
git add server.py
git commit -m "feat: Apply input sanitization and rate limiting to user-facing tools"
```

---

### Task 4: Add security status to health check and model status

**Files:**
- Modify: `server.py` — `health_check_impl` and `get_model_status_impl`

**Step 1: Add security section to health_check_impl**

In `health_check_impl`, after the `"synthetic_data"` section, add:

```python
    result["security"] = {
        "sanitization_available": SECURITY_UTILS_AVAILABLE,
        "rate_limiter_active": rate_limiter is not None,
        "rate_limiter_stats": rate_limiter.get_stats() if rate_limiter is not None else None,
    }
```

**Step 2: Run existing tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 476 passed, 2 skipped

**Step 3: Commit**

```bash
git add server.py
git commit -m "feat: Add security status to health check"
```

---

### Task 5: Write comprehensive tests

**Files:**
- Create: `tests/test_security_utils.py`

**Step 1: Write test file**

```python
"""Tests for Phase 11: Lightweight Security Layer"""

import pytest
import time
import threading


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------
class TestInputSanitizer:
    """Test InputSanitizer methods."""

    def test_sanitize_string_basic(self):
        from security_utils import InputSanitizer
        assert InputSanitizer.sanitize_string("hello") == "hello"

    def test_sanitize_string_null_bytes(self):
        from security_utils import InputSanitizer
        assert InputSanitizer.sanitize_string("hel\x00lo") == "hello"

    def test_sanitize_string_control_chars(self):
        from security_utils import InputSanitizer
        result = InputSanitizer.sanitize_string("hello\x01\x02world")
        assert "\x01" not in result
        assert "\x02" not in result
        assert "hello" in result

    def test_sanitize_string_preserves_newlines(self):
        from security_utils import InputSanitizer
        result = InputSanitizer.sanitize_string("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_string_max_length(self):
        from security_utils import InputSanitizer
        long_str = "a" * 2000
        result = InputSanitizer.sanitize_string(long_str, max_length=100)
        assert len(result) == 100

    def test_sanitize_string_strips_whitespace(self):
        from security_utils import InputSanitizer
        assert InputSanitizer.sanitize_string("  hello  ") == "hello"

    def test_sanitize_string_non_string_input(self):
        from security_utils import InputSanitizer
        result = InputSanitizer.sanitize_string(12345)
        assert result == "12345"

    def test_sanitize_dict_basic(self):
        from security_utils import InputSanitizer
        data = {"name": "test", "amount": 100.0}
        result = InputSanitizer.sanitize_dict(data)
        assert result["name"] == "test"
        assert result["amount"] == 100.0

    def test_sanitize_dict_null_bytes_in_values(self):
        from security_utils import InputSanitizer
        data = {"name": "test\x00inject", "value": "ok"}
        result = InputSanitizer.sanitize_dict(data)
        assert "\x00" not in result["name"]

    def test_sanitize_dict_allowed_keys(self):
        from security_utils import InputSanitizer
        data = {"name": "test", "secret": "hidden", "amount": 100}
        result = InputSanitizer.sanitize_dict(data, allowed_keys=["name", "amount"])
        assert "name" in result
        assert "amount" in result
        assert "secret" not in result

    def test_sanitize_dict_nested(self):
        from security_utils import InputSanitizer
        data = {"outer": {"inner": "val\x00ue"}}
        result = InputSanitizer.sanitize_dict(data)
        assert "\x00" not in result["outer"]["inner"]

    def test_sanitize_list(self):
        from security_utils import InputSanitizer
        data = ["hello\x00", {"key": "val\x00"}, 42]
        result = InputSanitizer.sanitize_list(data)
        assert "\x00" not in result[0]
        assert "\x00" not in result[1]["key"]
        assert result[2] == 42

    def test_contains_suspicious_sql(self):
        from security_utils import InputSanitizer
        suspicious, patterns = InputSanitizer.contains_suspicious_patterns(
            "SELECT * FROM users WHERE 1=1"
        )
        assert suspicious
        assert "sql_injection" in patterns

    def test_contains_suspicious_script(self):
        from security_utils import InputSanitizer
        suspicious, patterns = InputSanitizer.contains_suspicious_patterns(
            '<script>alert("xss")</script>'
        )
        assert suspicious
        assert "script_injection" in patterns

    def test_contains_suspicious_path_traversal(self):
        from security_utils import InputSanitizer
        suspicious, patterns = InputSanitizer.contains_suspicious_patterns(
            "../../etc/passwd"
        )
        assert suspicious
        assert "path_traversal" in patterns

    def test_clean_input_not_suspicious(self):
        from security_utils import InputSanitizer
        suspicious, patterns = InputSanitizer.contains_suspicious_patterns(
            "Normal transaction at Walmart for $50.00"
        )
        assert not suspicious
        assert len(patterns) == 0


# ---------------------------------------------------------------------------
# InMemoryRateLimiter
# ---------------------------------------------------------------------------
class TestInMemoryRateLimiter:
    """Test InMemoryRateLimiter."""

    def test_allows_within_limit(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        allowed, info = limiter.check("test")
        assert allowed
        assert info["remaining"] == 4

    def test_denies_over_limit(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("test")
        allowed, info = limiter.check("test")
        assert not allowed
        assert info["remaining"] == 0

    def test_separate_keys(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        limiter.check("key1")
        limiter.check("key1")
        # key1 is at limit
        allowed1, _ = limiter.check("key1")
        assert not allowed1
        # key2 should still be allowed
        allowed2, _ = limiter.check("key2")
        assert allowed2

    def test_get_stats(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60)
        limiter.check("a")
        limiter.check("b")
        stats = limiter.get_stats()
        assert stats["active_keys"] == 2
        assert stats["total_allowed"] == 2
        assert stats["total_denied"] == 0

    def test_reset_specific_key(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        limiter.check("test")
        allowed, _ = limiter.check("test")
        assert not allowed
        limiter.reset("test")
        allowed, _ = limiter.check("test")
        assert allowed

    def test_reset_all(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        limiter.check("a")
        limiter.check("b")
        limiter.reset()
        stats = limiter.get_stats()
        assert stats["active_keys"] == 0

    def test_thread_safety(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)
        results = []

        def make_requests():
            for _ in range(100):
                allowed, _ = limiter.check("shared")
                results.append(allowed)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = limiter.get_stats()
        assert stats["total_allowed"] + stats["total_denied"] == 500

    def test_stats_after_denials(self):
        from security_utils import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        limiter.check("x")
        limiter.check("x")
        limiter.check("x")  # denied
        stats = limiter.get_stats()
        assert stats["total_allowed"] == 2
        assert stats["total_denied"] == 1


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------
class TestSecurityServerIntegration:
    """Test security integration in server.py."""

    def test_security_utils_available(self):
        import server
        assert hasattr(server, "SECURITY_UTILS_AVAILABLE")
        assert isinstance(server.SECURITY_UTILS_AVAILABLE, bool)

    def test_rate_limiter_initialized(self):
        import server
        if server.SECURITY_UTILS_AVAILABLE:
            assert server.rate_limiter is not None

    def test_health_check_has_security(self):
        import server
        result = server.health_check_impl()
        assert "security" in result
        assert "sanitization_available" in result["security"]
        assert "rate_limiter_active" in result["security"]

    def test_health_check_rate_limiter_stats(self):
        import server
        result = server.health_check_impl()
        if server.rate_limiter is not None:
            stats = result["security"]["rate_limiter_stats"]
            assert "max_requests" in stats
            assert "total_allowed" in stats


# ---------------------------------------------------------------------------
# Sanitization applied to tools
# ---------------------------------------------------------------------------
class TestSanitizationInTools:
    """Verify sanitization is applied to user-facing tools."""

    def test_analyze_transaction_sanitizes_null_bytes(self):
        import server
        txn = {
            "transaction_id": "test\x00-001",
            "user_id": "user\x00-001",
            "amount": 100.0,
            "merchant": "Test\x00Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn)
        # Should not error — sanitization handles bad input
        assert "risk_score" in result or "error" in result

    def test_behavioral_sanitizes_null_bytes(self):
        import server
        data = {
            "keystroke_dynamics": {
                "key_events": [
                    {"key": "a\x00", "press_time": 100, "release_time": 150}
                ]
            }
        }
        result = server.detect_behavioral_anomaly_impl(data)
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_analyze_transaction_handles_long_strings(self):
        import server
        txn = {
            "transaction_id": "x" * 5000,
            "user_id": "user-001",
            "amount": 100.0,
            "merchant": "Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn)
        assert isinstance(result, dict)
```

**Step 2: Add security marker to pytest.ini**

Add `security: Security tests` to the markers list.

**Step 3: Run tests**

Run: `python -m pytest tests/test_security_utils.py -v --tb=short`
Expected: All tests pass

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: ~510+ passed, 2 skipped

**Step 4: Commit**

```bash
git add tests/test_security_utils.py pytest.ini
git commit -m "test: Add Phase 11 security layer tests"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: ~510+ passed, 2 skipped

**Step 2: Run linting**

Run: `ruff check server.py security_utils.py tests/test_security_utils.py`
Expected: Clean

**Step 3: Commit**

```bash
git add server.py security_utils.py
git commit -m "feat: Phase 11 lightweight security layer complete"
```
