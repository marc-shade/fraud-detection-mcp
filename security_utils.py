"""
Lightweight security utilities for the fraud-detection MCP server.

Provides:
- InputSanitizer  -- strip dangerous characters / HTML from untrusted strings
- InMemoryRateLimiter -- sliding-window per-key rate limiter (no external deps)
"""

import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Input Sanitiser
# ---------------------------------------------------------------------------

class InputSanitizer:
    """Sanitise untrusted input before it reaches the analysis pipeline."""

    # Pre-compiled patterns
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @classmethod
    def strip_html_tags(cls, value: str) -> str:
        """Remove HTML / XML tags from *value*."""
        return cls._HTML_TAG_RE.sub("", value)

    @classmethod
    def sanitize_string(cls, value: str, *, max_length: int = 10_000) -> str:
        """Sanitise a single string value.

        1. Strip HTML tags
        2. Remove ASCII control characters (except ``\\t``, ``\\n``, ``\\r``)
        3. Truncate to *max_length*
        """
        value = cls.strip_html_tags(value)
        value = cls._CONTROL_CHAR_RE.sub("", value)
        return value[:max_length]

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], *, max_depth: int = 10) -> Dict[str, Any]:
        """Recursively sanitise every string value inside *data*.

        Non-string leaves (int, float, bool, None) are passed through
        unchanged.  Lists are traversed element-wise.  Recursion is
        capped at *max_depth* to prevent stack overflow on adversarial
        input.
        """
        if max_depth <= 0:
            return data
        out: Dict[str, Any] = {}
        for key, val in data.items():
            skey = cls.sanitize_string(str(key)) if not isinstance(key, str) else cls.sanitize_string(key)
            out[skey] = cls._sanitize_value(val, max_depth=max_depth - 1)
        return out

    @classmethod
    def _sanitize_value(cls, val: Any, *, max_depth: int) -> Any:
        if isinstance(val, str):
            return cls.sanitize_string(val)
        if isinstance(val, dict):
            if max_depth <= 0:
                return val
            return {
                cls.sanitize_string(str(k)): cls._sanitize_value(v, max_depth=max_depth - 1)
                for k, v in val.items()
            }
        if isinstance(val, list):
            return [cls._sanitize_value(item, max_depth=max_depth - 1) for item in val]
        # int, float, bool, None -- pass through
        return val


# ---------------------------------------------------------------------------
# In-Memory Rate Limiter (sliding window)
# ---------------------------------------------------------------------------

class InMemoryRateLimiter:
    """Simple sliding-window rate limiter backed by an in-memory dict.

    Parameters
    ----------
    max_requests : int
        Maximum number of allowed requests inside *window_seconds*.
    window_seconds : float
        Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def check_rate_limit(self, key: str) -> Dict[str, Any]:
        """Check (and record) a request for *key*.

        Returns a dict with:
        - ``allowed`` (bool)
        - ``remaining`` (int)  -- requests left in the current window
        - ``retry_after`` (float | None) -- seconds until the oldest
          request expires (only present when blocked)
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        # Prune expired timestamps
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > cutoff]
        timestamps = self._requests[key]

        if len(timestamps) >= self.max_requests:
            retry_after = round(timestamps[0] - cutoff, 2)
            return {
                "allowed": False,
                "remaining": 0,
                "retry_after": max(retry_after, 0.01),
            }

        timestamps.append(now)
        return {
            "allowed": True,
            "remaining": self.max_requests - len(timestamps),
            "retry_after": None,
        }

    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the limiter state (for health checks)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        active_keys = 0
        total_requests = 0
        for key, timestamps in self._requests.items():
            active = [t for t in timestamps if t > cutoff]
            if active:
                active_keys += 1
                total_requests += len(active)
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "active_keys": active_keys,
            "total_active_requests": total_requests,
        }

    # ------------------------------------------------------------------
    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate-limit state for *key*, or all keys when *key* is ``None``."""
        if key is None:
            self._requests.clear()
        else:
            self._requests.pop(key, None)
