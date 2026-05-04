#!/usr/bin/env python3
"""LRU cache used by the production prediction pipeline.

This module previously also exposed ``AsyncInferenceEngine``,
``AsyncFraudDetector`` and ``create_inference_engine``. None of those
were ever wired into ``server.py`` — they were aspirational scaffolding
that diverged from the actual MCP dispatch path. They were removed
2026-05-04 to keep the module honest: anything imported here is what
the production server actually uses.

If you need an async inference layer in the future, build it against
the live MCP entry points (``analyze_transaction_impl`` etc.) rather
than re-introducing a parallel engine that drifts out of sync.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional


class LRUCache:
    """Bounded LRU cache with optional TTL.

    Pre-2026-05-04 the cache was TTL-less — entries lived forever until
    LRU eviction. That made every cached result a potential indefinite
    stale-state risk. ``ttl_seconds=None`` preserves the old behaviour
    for callers that explicitly opt in (the defaults pin a sensible
    bound — 5 minutes for the production prediction cache).
    """

    def __init__(self, capacity: int = 1000, ttl_seconds: Optional[float] = None):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        # Per-key insertion timestamps for TTL enforcement. Kept in a
        # parallel dict so the cached values themselves are unmodified.
        self._inserted_at: Dict[str, float] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, evicting and returning None if expired."""
        if key not in self.cache:
            return None
        if self._ttl is not None:
            import time as _time
            inserted = self._inserted_at.get(key)
            if inserted is None or (_time.monotonic() - inserted) > self._ttl:
                # Expired — evict
                self.cache.pop(key, None)
                self._inserted_at.pop(key, None)
                return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache, recording the insertion timestamp."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if self._ttl is not None:
            import time as _time
            self._inserted_at[key] = _time.monotonic()
        if len(self.cache) > self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            self._inserted_at.pop(evicted_key, None)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self._inserted_at.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


__all__ = ["LRUCache"]
