# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server built with FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, Autoencoders), graph-based fraud ring detection (NetworkX), SHAP explainability, AI agent-to-agent transaction protection, defense compliance modules, RFC 9421 HTTP Message Signature verification with Content-Digest body coverage and @query/@query-param derived components (Visa TAP / Mastercard Web Bot Auth / Stripe ACP), and a training/benchmarking pipeline into a unified MCP tool interface with **28 exposed tools (19 core + 5 compliance + 4 agent commerce Tier 0)**. XGBoost is available optionally via the training pipeline but is not in the default detection path.

See `docs/roadmap/agentic_commerce_2026.md` for the full Stripe ACP / Visa TAP / Mastercard Verifiable Intent / Google AP2 / Coinbase x402 feature roadmap. Tier 0 — RFC 9421 verifier (incl. Content-Digest + @query + JWKS retry-with-backoff), nonce cache (peek/consume split), idempotency store, **13-feature behavioral fingerprint with cyclical hour encoding**, claimed-vs-verified traffic classification **fully wired through `analyze_agent_transaction_impl`**, issuer-to-protocol mapping, and real JWT signature verification — is implemented and live. Tiers 1 and 2 are open work.

## Development Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run MCP server
python server.py

# Run tests (with coverage, CI requires 60% minimum)
python run_tests.py

# Run tests directly
python -m pytest tests/ -v --tb=short --cov=server --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_transaction_analysis.py -v

# Run tests by marker
python -m pytest -m unit           # Unit tests only
python -m pytest -m integration    # Integration tests
python -m pytest -m behavioral     # Behavioral biometrics tests
python -m pytest -m network        # Network/graph analysis tests
python -m pytest -m transaction    # Transaction analysis tests
python -m pytest -m explainability # SHAP explainability tests
python -m pytest -m synthetic      # Synthetic data generation tests
python -m pytest -m benchmark      # Benchmark tests
python -m pytest -m security       # Security utility tests
python -m pytest -m velocity       # User history / velocity tests
python -m pytest -m signature      # RFC 9421 / JWS / JWT signature verification

# Linting (CI uses ruff)
ruff check . --output-format=github
ruff format --check .

# Security scanning
bandit -r . -x ./tests,./.venv -ll

# Type checking
mypy server.py --ignore-missing-imports
```

## Architecture

### Server Entry Point

`server.py` is the active MCP server. It contains all core analyzer classes, agent transaction protection classes, validation functions, and 27 MCP tool definitions (19 core + 5 defense compliance + 3 agent commerce Tier 0). Everything runs from this single file with graceful-degradation imports for optional dependencies. Cryptographic signature verification lives in `acp_signatures.py`; replay/idempotency primitives in `agent_security.py`; both are imported by `server.py` with try/except so the server starts even when their crypto deps are unavailable.

### Core Analyzer Classes (server.py)

1. **`BehavioralBiometrics`** -- Three live analyzers: keystroke dynamics via Isolation Forest (10 features: 5 dwell + 5 flight), mouse movement+click patterns via One-Class SVM (5 features: velocity / click-rate / idle / linearity / log-event-count), touch screen patterns via LOF (5 features: pressure / area / swipe-velocity / tap-swipe-ratio / idle). Pre-2026-05-04 the mouse and touch models were initialized at startup but never invoked — the `detect_behavioral_anomaly` MCP tool only routed `keystroke_dynamics` so any `mouse_patterns` or `touch_patterns` field was silently ignored despite README + docs/API.md claiming the feature. Now wired through `analyze_mouse_dynamics` / `analyze_touch_dynamics` and routed by `detect_behavioral_anomaly_impl`. All three confidence scores are derived from sample size (saturating exponentially) + decision-boundary margin, capped at 0.85 because the underlying models bootstrap from synthetic gaussian data (replace via real-data retraining for higher justified confidence).

2. **`TransactionAnalyzer`** -- 46-feature extraction via `FeatureEngineer`, Isolation Forest + Autoencoder ensemble scoring with configurable weights (default 60/40). Supports model persistence (`save_models`/`load_models` to `models/saved/`), hot-reload after training, and falls back to synthetic-data initialization when no saved models exist. Reported `confidence` is derived from decision-boundary margin (60%) + IF/AE ensemble agreement (40%) + a small uplift when models came from real-data training rather than the synthetic bootstrap; capped at 0.95.

3. **`UserTransactionHistory`** -- Thread-safe, bounded per-user transaction history for velocity analysis. Uses `collections.deque` with LRU eviction. Provides `record()`, `get_history()`, `check_velocity()`, and `get_user_stats()` methods.

4. **`NetworkAnalyzer`** -- Builds a NetworkX graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality for fraud ring detection. Reported `confidence` is derived from total graph size + per-entity local connectivity (each saturating exponentially) and capped at 0.9 — small graphs return low confidence, dense neighborhoods on a 50+ node graph saturate near the cap.

### Agent Transaction Protection Classes (server.py)

5. **`TrafficClassifier`** -- Classifies transactions as human, agent, or unknown. Recognizes 9 agent protocols (Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal, Coinbase, OpenAI, Anthropic, x402) via user_agent patterns, explicit flags, and agent identifiers. **Reports BOTH `claimed_protocol` (string-match only) AND `verified_protocol` (passes RFC 9421 signature verification)**, plus `verification_status` enum {`verified`, `unverified`, `verification_failed`, `no_signature_provided`, `verification_unavailable`}. Downstream code MUST distinguish the two: a User-Agent string is not a security guarantee.

6. **`AgentIdentityRegistry`** -- Thread- and process-safe JSON-backed registry at `data/agent_registry.json`. Tracks agent_id, agent_type, first_seen, last_seen, transaction_count, trust_score. Methods: `register()`, `lookup()`, `record_transaction()`, `update_trust()`, `list_agents()`, `refresh()`. Multi-process safety: every mutation acquires an exclusive `fcntl.flock` advisory lock on `data/agent_registry.json.lock`, re-reads the file under the lock, applies the change, and writes back via `tempfile + os.replace` (atomic rename). Pre-fix this used `open + json.dump` (truncate-then-write) and lost ~87% of registrations under 8-process contention; the fix is verified by `tests/test_agent_identity.py::TestAgentRegistryMultiProcessSafety::test_concurrent_registrations_no_loss` which spawns 4 children each registering 25 agents and asserts all 100 are present in the final registry. On Windows where `fcntl` is unavailable, the registry degrades to thread-only locking.

7. **`AgentIdentityVerifier`** -- Validates agent credentials. Three signals: registry membership, API key format (min 16 chars), JWT token (3-stage validation: parse → exp claim → cryptographic signature verification against issuer's JWKS via `acp_signatures.jwks_resolver` when an `iss` claim is present). Trust = average of signals. Verified = trust >= 0.5 and no critical warnings. A *crypto-verified* JWT contributes 0.85 (vs 0.7 for exp-only and 0.1 for forged). Auto-registers unknown agents with trust=0.3.

8. **`AgentBehavioralFingerprint`** -- Per-agent Isolation Forest baselines using **13 features**: timing/decision/structure (api_timing_ms, timing_z_score, log_timing, decision_pattern_novel, request_structure_novel, timing_ratio) + transaction-shape (log_amount, payment_method_hash, merchant_hash, location_hash, **hour_sin = sin(2π·h/24), hour_cos = cos(2π·h/24)**, field_completeness). The transaction-shape features are required for **stolen-token replay detection** — a stolen token replayed at matching API timing distribution but against a different merchant/amount distribution will diverge here. Hour-of-day uses cyclical sin/cos encoding so 23:00→00:00 is a small distance in feature space, matching `feature_engineering.py` convention. `analyze()` and `record()` accept an optional `transaction` dict; when omitted, txn indices are zero-filled (timing-only mode). Thread-safe, bounded memory (max 1000 observations/agent). MIN_OBSERVATIONS_FOR_MODEL=10 before the IsolationForest activates. Models auto-retrain when `FEATURE_DIM` changes (e.g. a model trained at 12 features is invalidated when the codebase upgrades to 13).

9. **`MandateVerifier`** -- Stateless mandate compliance checker. Validates transactions against caller-supplied mandate dict: max_amount, daily_limit, allowed_merchants, blocked_merchants, allowed_locations, time_window (start/end HH:MM). `verify()` accepts an optional `history` parameter (list of prior transaction dicts with `recorded_at` + `amount`) — when provided, `daily_limit` checks the **24h cumulative spend** rather than the single-transaction amount. Without history, daily_limit degrades to single-transaction comparison and the result includes a `daily_limit_no_history` warning. `analyze_agent_transaction_impl` automatically passes the agent's `user_history.get_history(agent_id)` so the cumulative path is the production default. Returns compliance status, violations, drift_score, utilization (incl. `daily_total_prior` + `daily_total_projected`), and warnings.

10. **`CollusionDetector`** -- Directed graph of agent interactions. Detects circular flows (`nx.simple_cycles`), temporal clustering (3+ agents targeting same entity), and volume anomalies (10+ transactions between pair in window). Memory-bounded with LRU eviction.

11. **`AgentReputationScorer`** -- Longitudinal reputation from existing singletons: trust score (40%), transaction history (25%), behavioral consistency (25%), collusion safety (10%). History caps at 100 transactions for full credit.

### Agent Commerce Replay-Protection Modules

12. **`acp_signatures.py`** -- RFC 9421 HTTP Message Signature verifier compatible with Visa TAP, Mastercard Web Bot Auth (Cloudflare), Stripe ACP signature header. Public API: `parse_signature_input()`, `build_signature_base()`, `verify_rfc9421_signature()`, `compute_content_digest()`, `verify_content_digest()`, `JWKSResolver`. Supports algorithms `EdDSA` (via `cryptography` directly — python-jose has limited Ed25519 support), `PS256`, `ES256`, `RS256`. Supported derived components: `@method`, `@authority`, `@path`, `@query`, `@query-param;name="..."`, `@signature-params`. Body coverage via RFC 9530 Content-Digest (sha-256 + sha-512). Default JWKS URL for `visa` issuer is `https://mcp.visa.com/.well-known/jwks` — verified live, RSA, sandbox CA. **Verified gap (2026-05-03):** Mastercard, Stripe ACP, OpenAI, Anthropic, Google AP2, Coinbase x402 do NOT publish JWKS endpoints today (DID-based, on-chain, bilateral, or not yet documented). Override via `jwks_resolver.register_issuer()` when a URL becomes available. Enforces 8-min freshness window, nonce-replay protection (auto-consumes nonce on successful verify), optional tag binding (Visa TAP `agent-browser-auth`/`agent-payer-auth`). JWKS fetch retries with exponential backoff (3 attempts, 0.5/1/2s).

13. **`agent_security.py`** -- `NonceCache` (Visa TAP-compatible 8-min TTL keyed by `(keyid, nonce)`) and `IdempotencyStore` (Stripe-ACP-compatible 24h TTL keyed by `(idempotency_key, agent_id)` with `request_fingerprint` for HTTP 409 conflict detection). Replay-protection commit operation is `NonceCache.consume(keyid, nonce)` — atomic check-and-add, safe under concurrent threads (in-memory backend) AND processes (SQLite backend, via `BEGIN IMMEDIATE` + `INSERT ... ON CONFLICT DO NOTHING` + `rowcount`). The earlier `seen() then add()` two-step pattern was raceable across processes (caught + fixed 2026-05-04) and remains in the API only as a non-mutating peek; never use it to "commit" a nonce. Module-level singletons `nonce_cache` and `idempotency_store` are imported by `server.py`.

### MCP Tools (28 total: 19 core + 5 compliance + 4 agent commerce Tier 0)

| Tool | Function |
|------|----------|
| `analyze_transaction` | Full transaction fraud analysis (46-feature pipeline) |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection |
| `assess_network_risk` | Graph-based fraud ring detection |
| `generate_risk_score` | Weighted composite score (agent-aware: equal weighting; human: 50/30/20) |
| `explain_decision` | SHAP-based explainable AI with agent-specific reasoning |
| `classify_traffic_source` | Detect human vs AI agent traffic |
| `verify_agent_identity` | Validate agent credentials (API keys, JWT, registry) |
| `analyze_agent_transaction` | Full agent-aware pipeline (identity + fingerprint + mandate + transaction) |
| `verify_transaction_mandate` | Check transactions against agent spending mandates |
| `detect_agent_collusion` | Graph-based coordinated agent behavior detection |
| `score_agent_reputation` | Longitudinal reputation from trust, history, consistency |
| `analyze_batch` | Batch transaction analysis with aggregated statistics |
| `get_inference_stats` | LRU cache hit rates and inference performance metrics |
| `health_check` | System health with model status, feature counts, capabilities |
| `get_model_status` | Detailed model info: source, paths, training availability |
| `train_models` | Train models from CSV/JSON with SMOTE and Optuna support |
| `generate_synthetic_dataset` | Generate labeled fraud datasets (CSV/JSON) for evaluation |
| `analyze_dataset` | Analyze stored datasets for fraud patterns and risk distribution |
| `run_benchmark` | Performance benchmark with throughput, latency percentiles, accuracy |
| `assess_insider_threat` | Insider threat assessment (28 NITTF behavioral indicators) |
| `generate_siem_events` | Export events in CEF/LEEF/Syslog with MITRE ATT&CK enrichment |
| `evaluate_cleared_personnel` | SEAD 4/6 cleared personnel analytics and CE checks |
| `get_compliance_dashboard` | NITTF maturity, KRIs, compliance posture, executive summary |
| `generate_threat_referral` | Formal case referral or personnel security action report |
| `verify_agent_signature` | RFC 9421 HTTP Message Signature verification (Visa TAP / Mastercard Web Bot Auth / Stripe ACP) — freshness + replay + tag binding + Content-Digest body coverage + @query/@query-param + JWKS-resolved crypto verify |
| `check_idempotency_key` | Stripe-ACP-compatible Idempotency-Key check; returns miss / hit / conflict / stored; defends against double-spend on retry |
| `validate_nonce` | Visa-TAP-compatible nonce replay PEEK (safe by default — non-mutating; pass record_seen=True or use consume_nonce to commit) |
| `consume_nonce` | Atomic check + record nonce (single-shot replay defence); returns accepted/replayed; call AFTER successful crypto verify |

### Risk Scoring

**Human traffic**: Transaction 50%, Behavioral 30%, Network 20%.

**Agent traffic**: Equal weighting across all available components (transaction, identity, behavioral fingerprint, mandate, collusion, network). Automatically adapts as components are added.

Thresholds: CRITICAL >= 0.8, HIGH >= 0.6, MEDIUM >= 0.4, LOW < 0.4.

### Graceful Degradation

The server uses try/except imports so it starts even when optional dependencies are missing:

| Module | Flag | Required For |
|--------|------|-------------|
| `monitoring.py` | `MONITORING_AVAILABLE` | Prometheus metrics, structured logging |
| `training_pipeline.py` | `TRAINING_AVAILABLE` | `train_models` tool |
| `models/autoencoder.py` | `AUTOENCODER_AVAILABLE` | Autoencoder ensemble member |
| `explainability.py` | `EXPLAINABILITY_AVAILABLE` | SHAP-based explanations |
| `integration.py` | `SYNTHETIC_DATA_AVAILABLE` | `generate_synthetic_dataset`, `run_benchmark` |
| `security_utils.py` | `SECURITY_UTILS_AVAILABLE` | Input sanitization, rate limiting |
| `acp_signatures.py` | `_ACP_SIGNATURES_AVAILABLE` | RFC 9421 signature verification, JWT crypto verify, JWKS resolution |
| `agent_security.py` | `_AGENT_SECURITY_AVAILABLE` | NonceCache, IdempotencyStore for replay-protection |

### Compliance Modules

The 5 defense compliance tools are backed by modules under `compliance/`:
- `insider_threat.py` -- 28 NITTF behavioral indicators (`assess_insider_threat`)
- `siem_integration.py` -- CEF/LEEF/Syslog export with MITRE ATT&CK enrichment (`generate_siem_events`)
- `cleared_personnel.py` -- SEAD 4/6 cleared personnel analytics (`evaluate_cleared_personnel`)
- `dashboard_metrics.py` -- NITTF maturity, KRIs, executive summary (`get_compliance_dashboard`)
- Referral generation (`generate_threat_referral`) is composed from the above

### Testing Architecture

**967 tests across 33 test files** (including `test_compliance_modules.py`, `test_coverage_gaps.py`, `test_acp_signatures.py`, `test_agent_security.py`, `test_agent_security_backends.py`, `test_agent_commerce_tier0.py`, `test_calibration_provenance.py`). Tests import from `tests/conftest.py` for fixtures and sample data.

Available pytest markers: `unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`, `explainability`, `synthetic`, `benchmark`, `error`, `security`, `velocity`, `signature`.

Test files map to functionality areas:
- `test_transaction_analysis.py` -- TransactionAnalyzer, 46-feature pipeline
- `test_keystroke_analysis.py` -- BehavioralBiometrics
- `test_network_analysis.py` -- NetworkAnalyzer, graph centrality
- `test_explainability.py` -- SHAP explanations, agent-specific reasoning
- `test_autoencoder_ensemble.py` -- Autoencoder training, ensemble scoring
- `test_cache_and_batch.py` -- LRU cache, batch analysis
- `test_user_history.py` -- UserTransactionHistory, velocity checks
- `test_training_and_persistence.py` -- Model save/load, training pipeline
- `test_synthetic_data.py` -- Synthetic dataset generation, dataset analysis
- `test_benchmark.py` -- Benchmark tool, latency percentiles
- `test_monitoring.py` -- Prometheus metrics, structured logging
- `test_security_utils.py` -- Input sanitization, rate limiting
- `test_validation.py` -- Pydantic model validation
- `test_mcp_tools.py` -- MCP tool wrappers
- `test_error_handling.py` -- Edge cases, malformed inputs
- `test_integration.py` -- End-to-end workflows
- `test_traffic_classifier.py` -- TrafficClassifier, agent-aware risk scoring
- `test_agent_identity.py` -- AgentIdentityRegistry, AgentIdentityVerifier, identity in risk scoring
- `test_agent_behavioral_fingerprint.py` -- AgentBehavioralFingerprint baselines and anomaly detection
- `test_analyze_agent_transaction.py` -- analyze_agent_transaction_impl pipeline, mandate integration
- `test_mandate_verifier.py` -- MandateVerifier constraint checking
- `test_collusion_detector.py` -- CollusionDetector graph analysis
- `test_mandate_collusion_tools.py` -- MCP tool wrappers for mandate and collusion
- `test_agent_reputation.py` -- AgentReputationScorer composite scoring
- `test_score_agent_reputation_tool.py` -- score_agent_reputation MCP tool
- `test_acp_signatures.py` -- RFC 9421 parser, freshness, replay, JWKS resolver, end-to-end Ed25519 verification, Content-Digest sha-256/sha-512 verify (covered+missing+tampered), @query/@query-param derived components
- `test_agent_security.py` -- NonceCache (8-min TTL, replay block) and IdempotencyStore (Stripe-ACP miss/hit/conflict semantics)
- `test_agent_security_backends.py` -- Backend parity (in_memory + sqlite share contract via parametrized fixtures), `consume()` thread-race test (50 threads racing → exactly 1 winner), **`test_concurrent_consume_atomic_across_processes` spawns 8 child processes that race to consume the same nonce simultaneously and asserts exactly one accepted — real proof of cross-process atomicity, not just sequential persistence**
- `test_agent_commerce_tier0.py` -- 13-feature behavioral fingerprint with cyclical hour encoding (wraparound test, **strong stolen-token replay test asserting risk_score >= 0.7 with 60-obs baseline**), TrafficClassifier claimed-vs-verified, validate_nonce peek default, consume_nonce atomic, F1 wiring tests, F4 issuer-to-protocol mapping
- `test_calibration_provenance.py` -- Drift-detector: runs `scripts/calibrate_agent_thresholds.py` and asserts F1 >= 0.70 + optimal threshold in [0.40, 0.70] + tampered_signature detection rate >= 95%

### Advanced Modules

| Module | Purpose |
|--------|---------|
| `feature_engineering.py` | 46-feature extraction with cyclical encoding, z-scores, velocity features |
| `training_pipeline.py` | Full ML pipeline: SMOTE resampling, cross-validation, Optuna hyperparameter tuning, MLflow tracking |
| `async_inference.py` | LRU cache for inference results with configurable TTL and max size |
| `explainability.py` | SHAP-based explanations with graceful fallback when SHAP unavailable |
| `security_utils.py` | Input sanitization (XSS/SQLi prevention) and in-memory rate limiter |
| `monitoring.py` | Prometheus metrics, structlog, health checks, Grafana dashboard config |
| `models/autoencoder.py` | PyTorch autoencoder for anomaly detection (reconstruction error scoring) |
| `models/gnn_fraud_detector.py` | Graph Neural Network fraud detector |
| `integration.py` | Synthetic data generation pipeline with configurable fraud patterns |
| `benchmarks.py` | Standalone performance benchmarking suite |
| `models_validation.py` | Pydantic v2 validation models including `TrafficSource` enum and agent fields |
| `config.py` | Pydantic-settings based configuration with `.env` file support |
| `acp_signatures.py` | RFC 9421 HTTP Message Signature verifier (Visa TAP, Mastercard Web Bot Auth, Stripe ACP). EdDSA via `cryptography` direct, PS256/ES256/RS256 via `python-jose`. Includes `JWKSResolver` with TTL cache + 3-attempt retry-with-backoff. Supports `@method`, `@authority`, `@path`, `@query`, `@query-param`, RFC 9530 Content-Digest body coverage. |
| `agent_security.py` | `NonceCache` + `IdempotencyStore` with **pluggable backends**: `InMemoryNonceBackend` / `InMemoryIdempotencyBackend` (default, process-local) and `SQLiteNonceBackend` / `SQLiteIdempotencyBackend` (file-backed via SQLite WAL mode + busy_timeout, **multi-process safe across Python workers sharing one file**). Backend selection via `config.ACP_BACKEND` (`in_memory` or `sqlite`) + `config.ACP_SQLITE_PATH`. |
| `scripts/calibrate_agent_thresholds.py` | Synthetic-data calibration of the agent commerce verification thresholds. Generates labeled balanced agent transactions, sweeps thresholds, reports F1/precision/recall + per-attack-type detection rate, writes provenance to `docs/calibration/agent_thresholds_<date>.md`. Run periodically and on every release. |

### Configuration

`config.py` uses Pydantic v2 `BaseSettings` with `ConfigDict` and `.env` file support. Key settings: model hyperparameters, risk thresholds, database URLs, rate limits, JWT config. Copy `.env.example` to `.env` for local configuration.

### Server Entry

- **`server.py`** -- The MCP server. Contains all core analyzers, agent protection classes, 27 MCP tools (19 core + 5 compliance + 3 agent commerce Tier 0), and production logic.

### Supporting Docs

See also at repo root: `README.md`, `QUICK_START.md`, `TESTING.md`, `CONTRIBUTING.md`, `docs/API.md`, `docs/plans/`, `docs/roadmap/`. Runtime/state directories: `models/saved/` (persisted models), `data/agent_registry.json` (agent registry), `test_data/` (fixtures), `examples/` (usage samples), `logs/`, `cache/`.

## Key Patterns

- All MCP tools follow the `_impl` pattern: `analyze_transaction_impl()` is the testable function, `analyze_transaction` is the `@mcp.tool()` wrapper. Tests import and call the `_impl` functions directly.
- Models are initialized with synthetic training data at import time. Replace with real trained models via the `train_models` MCP tool.
- Input validation happens in two layers: manual `validate_transaction_data()`/`validate_behavioral_data()` in `server.py`, and Pydantic models in `models_validation.py`.
- All analysis functions return dicts with `risk_score` (0-1 float), `confidence`, `is_anomaly` boolean, and domain-specific details.
- Agent traffic is automatically classified by `TrafficClassifier` and routed through agent-specific analysis (identity verification, behavioral fingerprinting, mandate compliance).
- The `@_monitored` decorator wraps MCP tools with optional Prometheus metrics when monitoring is available. **Decorator order matters**: `@mcp.tool()` MUST be the OUTERMOST decorator and `@_monitored(...)` directly above the function — i.e. `@_monitored` runs first, wraps the function with monitoring, then `@mcp.tool()` registers the monitoring-wrapped callable. Pre-2026-05-04 the order was reversed, which made `@mcp.tool()` register the bare function and the monitoring wrapper sat above as dead code that MCP never invoked. `tests/test_monitoring.py::TestMonitoringActuallyFiresOnMcpToolCall` is a regression gate that asserts the Prometheus counter actually moves on a real `mcp.call_tool()`.
- Thread-safe singletons: `agent_registry`, `agent_verifier`, `agent_fingerprinter`, `mandate_verifier`, `collusion_detector`, `reputation_scorer`, `acp_signatures.jwks_resolver`, `agent_security.nonce_cache`, `agent_security.idempotency_store`.

## Agent Commerce Verification Status (Tier 0 — production-grade 2026-05-03)

**Every threshold/delta is now config-tunable.** All `ACP_*` settings live in `config.AppConfig` and are env-overridable (e.g. `ACP_VERIFIED_CONFIDENCE_BOOST=0.20`). Defaults come from `scripts/calibrate_agent_thresholds.py` synthetic-data calibration (provenance: `docs/calibration/agent_thresholds_<date>.md`). Test `test_calibration_provenance.py` is a drift-detector — fires if anyone changes defaults in ways that materially degrade F1.

**Replay protection is multi-process-safe.** Set `ACP_BACKEND=sqlite` and provide `ACP_SQLITE_PATH` to share NonceCache + IdempotencyStore across Python workers on the same node. SQLite WAL mode + 5s busy_timeout serialises writes. The replay-defence operation is `NonceCache.consume(keyid, nonce)`, which uses `BEGIN IMMEDIATE` + `INSERT ... ON CONFLICT DO NOTHING` (rowcount distinguishes "first claim" from "replay"). This is the path used by `consume_nonce` MCP tool, by `analyze_agent_transaction_impl`, and inside `acp_signatures.verify_rfc9421_signature` after successful crypto. Tests prove it under contention: `test_concurrent_consume_atomic_across_processes` spawns 8 children that race to consume the same nonce simultaneously and asserts exactly one accepted; `test_consume_thread_race_exactly_one_winner` does the same with 50 threads. The earlier non-atomic `seen() then add()` pattern was demonstrated broken (30 cross-process callers all accepted the same nonce) and replaced by `consume()` on 2026-05-04. Default backend remains in-memory for zero-config single-process use.



When extending agent-fraud features, **always distinguish "claimed" from "verified"**:
- A `User-Agent: stripe-acp/1.0` header is a *claim*, not a security guarantee. `TrafficClassifier.classify()` reports both `claimed_protocol` and `verified_protocol` separately; downstream risk scoring already treats unverified claims as lower confidence than verified ones.
- **`analyze_agent_transaction_impl` now wires verification end-to-end**: pass `signature_headers`, `http_method`, `http_path`, `http_authority`, `expected_issuer`, `expected_signature_tag` (and optionally `http_query` + `http_body` for query/Content-Digest coverage) inside `transaction_data` and the verification status flows into anomalies + identity_trust + the returned `verified_protocol` field. A failed signature on a claimed-agent request adds `signature_verification_failed` to anomalies AND drops identity_trust by 0.30. A verified signature boosts identity_trust by 0.15 and overrides identity_verified to True.
- `verify_agent_signature` MCP tool is the standalone direct entry point for signature verification; the same code path as above.
- `verified_protocol` reports the **protocol enum** (e.g. `visa_tap`), not the raw issuer (e.g. `visa`). Mapping lives in `ISSUER_TO_PROTOCOL` near `AGENT_USER_AGENT_PATTERNS`. When you add a new issuer to `jwks_resolver`, add it to `ISSUER_TO_PROTOCOL` too.
- `validate_nonce` is a **safe peek by default** (no mutation). `consume_nonce` is the atomic check+record. Use `consume_nonce` (not `validate_nonce` with `record_seen=True`) anywhere you mean to commit a nonce post-verification.
- `check_idempotency_key` enforces ACP's Idempotency-Key contract. NonceCache + IdempotencyStore are process-local — for multi-process deployments, externalize the stores or pin client routing.
- The `AgentBehavioralFingerprint` 13-feature vector includes 7 transaction-shape features (amount/merchant/location/payment_method/hour_sin/hour_cos/completeness). Hour uses cyclical (sin/cos) encoding so 23:00 and 00:00 are close in feature space. Callers MUST pass `transaction=transaction_data` to `analyze()` for the txn features to fire; omitting it falls back to timing-only mode.
- **JWKS landscape (verified 2026-05-03)**: only Visa publishes a public JWKS (`https://mcp.visa.com/.well-known/jwks`). Mastercard/Stripe ACP/OpenAI/Anthropic/Google AP2/Coinbase x402 either don't publish JWKS by design (DID-based, on-chain, bilateral) or haven't yet. When a URL becomes available, register via `jwks_resolver.register_issuer(name, url)` AND add to `ISSUER_TO_PROTOCOL`.
- Tier 1 features (AP2 mandate verification, Stripe SPT allowance schema, Mastercard VI L1/L2/L3, streaming-payment burn-rate, x402 inspector, prompt-injection precursor signal, 2-phase mandate reserve/settle, ACP risk_signals export) are documented in `docs/roadmap/agentic_commerce_2026.md` but not yet implemented.
