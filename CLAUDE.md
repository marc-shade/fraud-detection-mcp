# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server (v2.3.0) built with FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, XGBoost, Autoencoders), graph-based fraud ring detection (NetworkX), SHAP explainability, and a full training/benchmarking pipeline into a unified MCP tool interface with 13 exposed tools.

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

`server.py` (~2380 lines) is the active MCP server. It contains four core analyzer classes, validation functions, and 13 MCP tool definitions. Everything runs from this single file with graceful-degradation imports for optional dependencies.

### Core Analyzer Classes (server.py)

1. **`BehavioralBiometrics`** -- Keystroke dynamics via Isolation Forest, mouse movement patterns via One-Class SVM, touch screen patterns via LOF. Extracts 10 statistical features (5 dwell time + 5 flight time).

2. **`TransactionAnalyzer`** -- 46-feature extraction via `FeatureEngineer`, Isolation Forest + Autoencoder ensemble scoring with configurable weights (default 60/40). Supports model persistence (`save_models`/`load_models` to `models/saved/`), hot-reload after training, and falls back to synthetic-data initialization when no saved models exist.

3. **`UserTransactionHistory`** -- Thread-safe, bounded per-user transaction history for velocity analysis. Uses `collections.deque` with LRU eviction. Provides `record()`, `get_history()`, `check_velocity()`, and `get_user_stats()` methods.

4. **`NetworkAnalyzer`** -- Builds a NetworkX graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality for fraud ring detection.

### MCP Tools (13 total)

| Tool | Function | Added |
|------|----------|-------|
| `analyze_transaction` | Full transaction fraud analysis (46-feature pipeline) | Phase 1 |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection | Phase 1 |
| `assess_network_risk` | Graph-based fraud ring detection | Phase 1 |
| `generate_risk_score` | Weighted composite score (50% txn, 30% behavioral, 20% network) | Phase 1 |
| `explain_decision` | SHAP-based explainable AI reasoning for any analysis result | Phase 1 |
| `analyze_batch` | Batch transaction analysis with aggregated statistics | Phase 8 |
| `get_inference_stats` | LRU cache hit rates and inference performance metrics | Phase 8 |
| `health_check` | System health with model status, feature counts, capabilities | Phase 8 |
| `get_model_status` | Detailed model info: source, paths, training availability | Phase 9 |
| `train_models` | Train models from CSV/JSON with SMOTE and Optuna support | Phase 9 |
| `generate_synthetic_dataset` | Generate labeled fraud datasets (CSV/JSON) for evaluation | Phase 10 |
| `analyze_dataset` | Analyze stored datasets for fraud patterns and risk distribution | Phase 11 |
| `run_benchmark` | Performance benchmark with throughput, latency percentiles, accuracy | Phase 12 |

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

### Feature Engineering Pipeline

`feature_engineering.py` extracts 46 features per transaction:
- Amount features: raw, log-transformed, z-score, percentile rank
- Temporal features: hour, day-of-week, is-weekend, cyclical sin/cos encodings
- Location/merchant: hash-based encoding, frequency-based features
- Payment risk: method risk scoring, high-risk flag
- Velocity features: per-user transaction count, amount statistics, time-since-last
- Cross-field: amount-hour interaction, amount-payment interaction

### Risk Scoring

Thresholds defined in both `server.py` and `config.py` (`AppConfig`):
- CRITICAL >= 0.8
- HIGH >= 0.6
- MEDIUM >= 0.4
- LOW < 0.4

Risk score conversion from Isolation Forest: `max(0, min(1, (0.5 - anomaly_score) * 2))`.

Ensemble scoring (when autoencoder available): weighted combination of Isolation Forest (60%) and Autoencoder reconstruction error (40%).

### Testing Architecture

533 tests across 17 test files. Tests import from `tests/conftest.py` for fixtures and sample data.

Available pytest markers: `unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`, `explainability`, `synthetic`, `benchmark`, `error`, `security`, `velocity`.

Test files map to functionality areas:
- `test_transaction_analysis.py` -- TransactionAnalyzer, 46-feature pipeline
- `test_keystroke_analysis.py` -- BehavioralBiometrics
- `test_network_analysis.py` -- NetworkAnalyzer, graph centrality
- `test_explainability.py` -- SHAP explanations, fallback behavior
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
| `models_validation.py` | Pydantic v2 validation models for all input/output types |
| `config.py` | Pydantic-settings based configuration with `.env` file support |

### Configuration

`config.py` uses Pydantic v2 `BaseSettings` with `ConfigDict` and `.env` file support. Key settings: model hyperparameters, risk thresholds, database URLs, rate limits, JWT config. Field names map directly to environment variables when `case_sensitive=True`. Copy `.env.example` to `.env` for local configuration.

### Two Server Versions

- **`server.py`** -- Active MCP server (v2.3.0). Contains all core analyzers, 13 MCP tools, and production logic.
- **`server_v2.py`** -- Extended version integrating additional Pydantic models and the full security layer. Not the active entrypoint.

## Key Patterns

- Models are initialized with synthetic training data at import time. Replace with real trained models via the `train_models` MCP tool or the training pipeline directly.
- Input validation happens in two layers: manual `validate_transaction_data()`/`validate_behavioral_data()` in `server.py`, and Pydantic models in `models_validation.py`.
- All analysis functions return dicts with `risk_score` (0-1 float), `confidence`, `is_anomaly` boolean, and domain-specific details.
- The `TransactionAnalyzer` supports model persistence: `save_models()` serializes to `models/saved/`, `load_models()` restores them, and `train_models` triggers hot-reload after training.
- `UserTransactionHistory` is thread-safe with bounded memory. It tracks per-user velocity and is used automatically during transaction analysis.
- The LRU cache (`async_inference.py`) caches inference results keyed by transaction data hash.
