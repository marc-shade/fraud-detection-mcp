# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server built with FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, XGBoost, Autoencoders), graph-based fraud ring detection (NetworkX), SHAP explainability, AI agent-to-agent transaction protection, and a full training/benchmarking pipeline into a unified MCP tool interface with 19 exposed tools.

## Development Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run MCP server
python server.py

# Run tests (with coverage, requires 80% minimum)
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

`server.py` is the active MCP server. It contains all core analyzer classes, agent transaction protection classes, validation functions, and 19 MCP tool definitions. Everything runs from this single file with graceful-degradation imports for optional dependencies.

### Core Analyzer Classes (server.py)

1. **`BehavioralBiometrics`** -- Keystroke dynamics via Isolation Forest, mouse movement patterns via One-Class SVM, touch screen patterns via LOF. Extracts 10 statistical features (5 dwell time + 5 flight time).

2. **`TransactionAnalyzer`** -- 46-feature extraction via `FeatureEngineer`, Isolation Forest + Autoencoder ensemble scoring with configurable weights (default 60/40). Supports model persistence (`save_models`/`load_models` to `models/saved/`), hot-reload after training, and falls back to synthetic-data initialization when no saved models exist.

3. **`UserTransactionHistory`** -- Thread-safe, bounded per-user transaction history for velocity analysis. Uses `collections.deque` with LRU eviction. Provides `record()`, `get_history()`, `check_velocity()`, and `get_user_stats()` methods.

4. **`NetworkAnalyzer`** -- Builds a NetworkX graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality for fraud ring detection.

### Agent Transaction Protection Classes (server.py)

5. **`TrafficClassifier`** -- Classifies transactions as human, agent, or unknown. Recognizes 9 agent protocols (Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal, Coinbase, OpenAI, Anthropic, x402) via user_agent patterns, explicit flags, and agent identifiers.

6. **`AgentIdentityRegistry`** -- Thread-safe JSON-backed registry at `data/agent_registry.json`. Tracks agent_id, agent_type, first_seen, last_seen, transaction_count, trust_score. Methods: `register()`, `lookup()`, `record_transaction()`, `update_trust()`, `list_agents()`.

7. **`AgentIdentityVerifier`** -- Validates agent credentials. Three signals: registry membership, API key format (min 16 chars), JWT token expiry (base64 decode payload, check `exp` claim). Trust = average of signals. Verified = trust >= 0.5 and no critical warnings. Auto-registers unknown agents with trust=0.3.

8. **`AgentBehavioralFingerprint`** -- Per-agent Isolation Forest baselines using 8 features (log_amount, payment_method_hash, merchant_hash, location_hash, hour_of_day, field_completeness, timing_interval, amount_magnitude). Thread-safe, bounded memory (max 1000 observations/agent). MIN_BASELINE=10 observations before anomaly detection activates.

9. **`MandateVerifier`** -- Stateless mandate compliance checker. Validates transactions against caller-supplied mandate dict: max_amount, daily_limit, allowed_merchants, blocked_merchants, allowed_locations, time_window (start/end HH:MM). Returns compliance status, violations, drift_score, and utilization.

10. **`CollusionDetector`** -- Directed graph of agent interactions. Detects circular flows (`nx.simple_cycles`), temporal clustering (3+ agents targeting same entity), and volume anomalies (10+ transactions between pair in window). Memory-bounded with LRU eviction.

11. **`AgentReputationScorer`** -- Longitudinal reputation from existing singletons: trust score (40%), transaction history (25%), behavioral consistency (25%), collusion safety (10%). History caps at 100 transactions for full credit.

### MCP Tools (19 total)

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

### Testing Architecture

727 tests across 22 test files. Tests import from `tests/conftest.py` for fixtures and sample data.

Available pytest markers: `unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`, `explainability`, `synthetic`, `benchmark`, `error`, `security`, `velocity`.

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

### Advanced Modules

| Module | Purpose |
|--------|---------|
| `feature_engineering.py` | 46-feature extraction with cyclical encoding, z-scores, velocity features |
| `training_pipeline.py` | Full ML pipeline: SMOTE resampling, cross-validation, Optuna hyperparameter tuning, MLflow tracking |
| `async_inference.py` | LRU cache for inference results with configurable TTL and max size |
| `explainability.py` | SHAP-based explanations with graceful fallback when SHAP unavailable |
| `security_utils.py` | Input sanitization (XSS/SQLi prevention) and in-memory rate limiter |
| `security.py` | OWASP security layer: JWT auth, RBAC, rate limiting |
| `monitoring.py` | Prometheus metrics, structlog, health checks, Grafana dashboard config |
| `models/autoencoder.py` | PyTorch autoencoder for anomaly detection (reconstruction error scoring) |
| `models/gnn_fraud_detector.py` | Graph Neural Network fraud detector |
| `integration.py` | Synthetic data generation pipeline with configurable fraud patterns |
| `benchmarks.py` | Standalone performance benchmarking suite |
| `cli.py` | CLI for analyzing stored datasets (CSV/JSON) |
| `models_validation.py` | Pydantic v2 validation models including `TrafficSource` enum and agent fields |
| `config.py` | Pydantic-settings based configuration with `.env` file support |

### Configuration

`config.py` uses Pydantic v2 `BaseSettings` with `ConfigDict` and `.env` file support. Key settings: model hyperparameters, risk thresholds, database URLs, rate limits, JWT config. Copy `.env.example` to `.env` for local configuration.

### Two Server Versions

- **`server.py`** -- Active MCP server. Contains all core analyzers, agent protection classes, 19 MCP tools, and production logic.
- **`server_v2.py`** -- Extended version integrating additional Pydantic models and the full security layer. Not the active entrypoint.

## Key Patterns

- All MCP tools follow the `_impl` pattern: `analyze_transaction_impl()` is the testable function, `analyze_transaction` is the `@mcp.tool()` wrapper. Tests import and call the `_impl` functions directly.
- Models are initialized with synthetic training data at import time. Replace with real trained models via the `train_models` MCP tool.
- Input validation happens in two layers: manual `validate_transaction_data()`/`validate_behavioral_data()` in `server.py`, and Pydantic models in `models_validation.py`.
- All analysis functions return dicts with `risk_score` (0-1 float), `confidence`, `is_anomaly` boolean, and domain-specific details.
- Agent traffic is automatically classified by `TrafficClassifier` and routed through agent-specific analysis (identity verification, behavioral fingerprinting, mandate compliance).
- The `@_monitored` decorator wraps MCP tools with optional Prometheus metrics when monitoring is available.
- Thread-safe singletons: `agent_registry`, `agent_verifier`, `agent_fingerprinter`, `mandate_verifier`, `collusion_detector`, `reputation_scorer`.
