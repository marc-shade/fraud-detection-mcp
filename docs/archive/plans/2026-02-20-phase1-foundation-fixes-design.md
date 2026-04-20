# Phase 1: Foundation Fixes Design

Date: 2026-02-20

## Goal

Make server.py correct, testable, and CI-enforced. No new features — fix what's broken.

## Execution Order

Server fixes first, then test refactor, then CI as the capstone.

## 1. _impl Pattern Refactor (server.py)

Every `@mcp.tool()` becomes a thin wrapper delegating to a plain `_impl` function.

The existing `analyze_transaction_impl` gets a `@mcp.tool()` wrapper (currently missing — this is the primary documented tool that doesn't exist). The other 4 tools (`detect_behavioral_anomaly`, `assess_network_risk`, `generate_risk_score`, `explain_decision`) get their logic extracted into `*_impl` functions, with the decorated functions becoming 1-3 line delegates.

This enables direct testing of `_impl` functions without FastMCP decorator interference.

## 2. Input Validation Fixes (server.py)

`validate_transaction_data`:
- Reject `bool` before the `(int, float)` check (`isinstance(amount, bool)`)
- Reject `nan`/`inf` after type check (`math.isnan`, `math.isinf`)

`detect_behavioral_anomaly_impl`:
- Add `validate_behavioral_data()` call at entry (currently bypasses all validation)

## 3. Dead Code & Import Cleanup (server.py)

- Remove unused `import torch` and `import torch.nn as nn`
- Remove dummy XGBoost training in `TransactionAnalyzer._initialize_models` (model is trained on random labels but never called)
- Keep `self.xgb_model = None` as placeholder for Phase 2

## 4. NetworkAnalyzer Fixes (server.py)

- Replace bare `except:` with `except Exception as e: logger.error(...)`
- Add `MAX_GRAPH_NODES = 10000` cap with FIFO eviction via `collections.deque` tracking insertion order
- Prevents unbounded memory growth and keeps betweenness_centrality bounded

## 5. Geographic Risk Fix (server.py + config.py)

- Use `config.HIGH_RISK_LOCATIONS` (already partially defined in AppConfig) instead of hardcoded inline list
- Change from substring matching (`any(loc in location ...)`) to exact match on `location.lower().strip()`
- Update config.py defaults to generic terms: `["unknown"]` — specific country lists are a deployment-time config decision

## 6. Hash Encoding Fix (server.py)

Replace `hash(location) % 1000` with `hashlib.md5(location.encode()).hexdigest()` modulo 1000 for deterministic cross-process feature encoding. Same for merchant.

## 7. Consistency Keys Fix (server.py)

Standardize anomaly list key to `detected_anomalies` everywhere. Currently `generate_risk_score` uses `all_detected_anomalies` while `analyze_transaction_impl` uses `detected_anomalies`. `explain_decision` reads `all_detected_anomalies`, silently returning empty results for `analyze_transaction` output.

Fix: `generate_risk_score_impl` and `explain_decision_impl` both use `detected_anomalies`.

## 8. Test Refactor (tests/)

- Delete `tests/server_wrapper.py`
- All test files import `*_impl` functions directly from `server`
- Fix `reset_ml_models` fixture to reset `network_analyzer.transaction_graph = nx.Graph()` and reset `network_analyzer._node_order` (the new FIFO deque)
- Verify all existing tests pass

## 9. CI Fix (.github/workflows/ci.yml)

- Remove all `|| true` suffixes so failures are real failures
- Add `test` job: `pytest tests/ -v --cov=server --cov-fail-under=60`
- Add Python version matrix: `[3.10, 3.11, 3.12]`
- Fix bandit exclusion path: `./venv` (not `./.venv`)
- Add pip caching

## Out of Scope (Phase 2+)

- Real trained models (keep dummy data for now)
- server_v2.py integration
- Async inference
- Security/monitoring modules
- Docker
- pyproject.toml migration
