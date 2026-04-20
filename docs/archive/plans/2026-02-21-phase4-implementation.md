# Phase 4: Prediction Caching & Batch Inference — Implementation Plan

> **Goal:** Integrate `LRUCache` from `async_inference.py` into `server.py` for prediction caching, add batch prediction and inference statistics MCP tools.

**Architecture:** LRUCache provides O(1) cache lookup for repeated transaction analyses. A cache key is derived from transaction data fields. Batch prediction processes lists of transactions. Inference stats expose cache hit rate and performance metrics.

**Tech Stack:** Python, pytest, sklearn, numpy, OrderedDict (LRUCache)

---

### Task 1: Add LRUCache import and caching infrastructure

Add the LRUCache class import and initialize prediction cache and statistics tracking.

### Task 2: Wire cache into analyze_transaction_impl

Add cache key generation, cache lookup before analysis, and cache storage after analysis.

### Task 3: Add batch prediction and inference stats

Add `analyze_batch_impl` and `get_inference_stats_impl` functions plus MCP tool wrappers.

### Task 4: Add LRU cache unit tests

Add comprehensive tests for caching behavior, batch prediction, and inference stats.

### Task 5: Final verification

Run full test suite, verify MCP tools registered (7 total), verify coverage >= 80%.
