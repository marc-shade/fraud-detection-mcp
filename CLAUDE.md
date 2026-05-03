# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server built with FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, Autoencoders), graph-based fraud ring detection (NetworkX), SHAP explainability, AI agent-to-agent transaction protection, defense compliance modules, RFC 9421 HTTP Message Signature verification (Visa TAP / Mastercard Web Bot Auth / Stripe ACP), and a training/benchmarking pipeline into a unified MCP tool interface with **27 exposed tools (19 core + 5 compliance + 3 agent commerce Tier 0)**. XGBoost is available optionally via the training pipeline but is not in the default detection path.

See `docs/roadmap/agentic_commerce_2026.md` for the full Stripe ACP / Visa TAP / Mastercard Verifiable Intent / Google AP2 / Coinbase x402 feature roadmap. Tier 0 (signature verification + nonce cache + idempotency + 12-feature behavioral fingerprint + claimed-vs-verified traffic classification + real JWT signature verification) is implemented; Tiers 1 and 2 are open work.

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

1. **`BehavioralBiometrics`** -- Keystroke dynamics via Isolation Forest, mouse movement patterns via One-Class SVM, touch screen patterns via LOF. Extracts 10 statistical features (5 dwell time + 5 flight time).

2. **`TransactionAnalyzer`** -- 46-feature extraction via `FeatureEngineer`, Isolation Forest + Autoencoder ensemble scoring with configurable weights (default 60/40). Supports model persistence (`save_models`/`load_models` to `models/saved/`), hot-reload after training, and falls back to synthetic-data initialization when no saved models exist.

3. **`UserTransactionHistory`** -- Thread-safe, bounded per-user transaction history for velocity analysis. Uses `collections.deque` with LRU eviction. Provides `record()`, `get_history()`, `check_velocity()`, and `get_user_stats()` methods.

4. **`NetworkAnalyzer`** -- Builds a NetworkX graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality for fraud ring detection.

### Agent Transaction Protection Classes (server.py)

5. **`TrafficClassifier`** -- Classifies transactions as human, agent, or unknown. Recognizes 9 agent protocols (Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal, Coinbase, OpenAI, Anthropic, x402) via user_agent patterns, explicit flags, and agent identifiers. **Reports BOTH `claimed_protocol` (string-match only) AND `verified_protocol` (passes RFC 9421 signature verification)**, plus `verification_status` enum {`verified`, `unverified`, `verification_failed`, `no_signature_provided`, `verification_unavailable`}. Downstream code MUST distinguish the two: a User-Agent string is not a security guarantee.

6. **`AgentIdentityRegistry`** -- Thread-safe JSON-backed registry at `data/agent_registry.json`. Tracks agent_id, agent_type, first_seen, last_seen, transaction_count, trust_score. Methods: `register()`, `lookup()`, `record_transaction()`, `update_trust()`, `list_agents()`.

7. **`AgentIdentityVerifier`** -- Validates agent credentials. Three signals: registry membership, API key format (min 16 chars), JWT token (3-stage validation: parse → exp claim → cryptographic signature verification against issuer's JWKS via `acp_signatures.jwks_resolver` when an `iss` claim is present). Trust = average of signals. Verified = trust >= 0.5 and no critical warnings. A *crypto-verified* JWT contributes 0.85 (vs 0.7 for exp-only and 0.1 for forged). Auto-registers unknown agents with trust=0.3.

8. **`AgentBehavioralFingerprint`** -- Per-agent Isolation Forest baselines using **12 features**: timing/decision/structure (api_timing_ms, timing_z_score, log_timing, decision_pattern_novel, request_structure_novel, timing_ratio) + transaction-shape (log_amount, payment_method_hash, merchant_hash, location_hash, hour_of_day, field_completeness). The transaction-shape features are required for **stolen-token replay detection** — a stolen token replayed at matching API timing distribution but against a different merchant/amount distribution will diverge here. `analyze()` and `record()` accept an optional `transaction` dict; when omitted, indices 6-11 are zero-filled (timing-only mode). Thread-safe, bounded memory (max 1000 observations/agent). MIN_OBSERVATIONS_FOR_MODEL=10 before the IsolationForest activates. Models auto-retrain when `FEATURE_DIM` changes (e.g. a model trained at 6 features is invalidated when the codebase upgrades to 12).

9. **`MandateVerifier`** -- Stateless mandate compliance checker. Validates transactions against caller-supplied mandate dict: max_amount, daily_limit, allowed_merchants, blocked_merchants, allowed_locations, time_window (start/end HH:MM). Returns compliance status, violations, drift_score, and utilization.

10. **`CollusionDetector`** -- Directed graph of agent interactions. Detects circular flows (`nx.simple_cycles`), temporal clustering (3+ agents targeting same entity), and volume anomalies (10+ transactions between pair in window). Memory-bounded with LRU eviction.

11. **`AgentReputationScorer`** -- Longitudinal reputation from existing singletons: trust score (40%), transaction history (25%), behavioral consistency (25%), collusion safety (10%). History caps at 100 transactions for full credit.

### Agent Commerce Replay-Protection Modules

12. **`acp_signatures.py`** -- RFC 9421 HTTP Message Signature verifier compatible with Visa TAP, Mastercard Web Bot Auth (Cloudflare), Stripe ACP signature header. Public API: `parse_signature_input()`, `build_signature_base()`, `verify_rfc9421_signature()`, `JWKSResolver`. Supports algorithms `EdDSA` (via `cryptography` directly — python-jose has limited Ed25519 support), `PS256`, `ES256`, `RS256`. Default JWKS URL for `visa` issuer is `https://mcp.visa.com/.well-known/jwks` (override via `jwks_resolver.register_issuer()`). Enforces 8-min freshness window, optional nonce-replay check, optional tag binding.

13. **`agent_security.py`** -- `NonceCache` (Visa TAP-compatible 8-min TTL keyed by `(keyid, nonce)`) and `IdempotencyStore` (Stripe-ACP-compatible 24h TTL keyed by `(idempotency_key, agent_id)` with `request_fingerprint` for HTTP 409 conflict detection). Both are thread-safe and process-local; for multi-process deployment run an external store. Module-level singletons `nonce_cache` and `idempotency_store` are imported by `server.py`.

### MCP Tools (27 total: 19 core + 5 compliance + 3 agent commerce Tier 0)

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
| `verify_agent_signature` | RFC 9421 HTTP Message Signature verification (Visa TAP / Mastercard Web Bot Auth / Stripe ACP) — freshness + replay + tag binding + JWKS-resolved crypto verify |
| `check_idempotency_key` | Stripe-ACP-compatible Idempotency-Key check; returns miss / hit / conflict / stored; defends against double-spend on retry |
| `validate_nonce` | Visa-TAP-compatible nonce replay check (8-min rolling window); defends against signature replay |

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

**896 tests across 31 test files** (including `test_compliance_modules.py`, `test_coverage_gaps.py`, `test_acp_signatures.py`, `test_agent_security.py`, `test_agent_commerce_tier0.py`). Tests import from `tests/conftest.py` for fixtures and sample data.

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
- `test_acp_signatures.py` -- RFC 9421 parser, freshness, replay, JWKS resolver, end-to-end Ed25519 verification
- `test_agent_security.py` -- NonceCache (8-min TTL, replay block) and IdempotencyStore (Stripe-ACP miss/hit/conflict semantics)
- `test_agent_commerce_tier0.py` -- 12-feature behavioral fingerprint (incl. stolen-token replay test), TrafficClassifier claimed-vs-verified, 3 new MCP tools

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
| `acp_signatures.py` | RFC 9421 HTTP Message Signature verifier (Visa TAP, Mastercard Web Bot Auth, Stripe ACP). EdDSA via `cryptography` direct, PS256/ES256/RS256 via `python-jose`. Includes `JWKSResolver` with TTL cache. |
| `agent_security.py` | `NonceCache` (Visa-TAP 8-min replay window) + `IdempotencyStore` (Stripe-ACP 24h replay cache with body-fingerprint conflict detection) |

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
- The `@_monitored` decorator wraps MCP tools with optional Prometheus metrics when monitoring is available.
- Thread-safe singletons: `agent_registry`, `agent_verifier`, `agent_fingerprinter`, `mandate_verifier`, `collusion_detector`, `reputation_scorer`, `acp_signatures.jwks_resolver`, `agent_security.nonce_cache`, `agent_security.idempotency_store`.

## Agent Commerce Verification Status (Tier 0 — implemented 2026-05-03)

When extending agent-fraud features, **always distinguish "claimed" from "verified"**:
- A `User-Agent: stripe-acp/1.0` header is a *claim*, not a security guarantee. `TrafficClassifier.classify()` reports both `claimed_protocol` and `verified_protocol` separately; downstream risk scoring should treat unverified claims as lower confidence than verified ones.
- `verify_agent_signature` is the only tool that produces a verified protocol attribution; it requires the caller to pass `signature_headers` (RFC 9421) and the JWKS for the issuer must be reachable (or pre-loaded into `jwks_resolver._cache`).
- `validate_nonce` enforces an 8-minute replay window per Visa TAP. `check_idempotency_key` enforces ACP's Idempotency-Key contract. Both are process-local — for multi-process deployments, externalize the stores or pin client routing.
- The `AgentBehavioralFingerprint` 12-feature vector includes 6 transaction-shape features (amount/merchant/location/payment_method/hour/completeness) that are required to detect stolen-token replay. Callers MUST pass `transaction=transaction_data` to `analyze()` for these features to fire; omitting it falls back to timing-only mode.
- Tier 1 features (AP2 mandate verification, Stripe SPT allowance schema, Mastercard VI L1/L2/L3, streaming-payment burn-rate, x402 inspector, prompt-injection precursor signal, 2-phase mandate reserve/settle, ACP risk_signals export) are documented in `docs/roadmap/agentic_commerce_2026.md` but not yet implemented.
