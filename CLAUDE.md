# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server using FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, XGBoost, Autoencoders), and graph-based fraud ring detection (NetworkX) into a unified MCP tool interface.

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

# Linting (CI uses ruff)
ruff check . --output-format=github
ruff format --check .

# Security scanning
bandit -r . -x ./tests,./.venv -ll

# Type checking
mypy server.py --ignore-missing-imports
```

## Architecture

### Two Server Versions

- **`server.py`** - Active MCP server. Contains all core analyzers (`BehavioralBiometrics`, `TransactionAnalyzer`, `NetworkAnalyzer`), validation functions, and MCP tool definitions. This is what runs in production.
- **`server_v2.py`** - Extended version integrating advanced modules (Pydantic models, training pipeline, autoencoders, GNN, security layer). Not the active entrypoint.

### Core Analysis Pipeline (server.py)

Three analyzer classes are instantiated as module-level singletons:

1. **`BehavioralBiometrics`** - Keystroke dynamics via Isolation Forest, mouse patterns via One-Class SVM, touch patterns via LOF. Extracts 10 statistical features (5 dwell time + 5 flight time).
2. **`TransactionAnalyzer`** - Extracts 8 features (amount, log_amount, hour, weekday, day, location_hash, merchant_hash, payment_risk). Uses Isolation Forest + XGBoost ensemble.
3. **`NetworkAnalyzer`** - Builds a NetworkX graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality.

### MCP Tools Exposed

| Tool | Function |
|------|----------|
| `analyze_transaction` | Transaction fraud analysis (defined via `analyze_transaction_impl`) |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection |
| `assess_network_risk` | Graph-based fraud ring detection |
| `generate_risk_score` | Weighted composite score (transaction 50%, behavioral 30%, network 20%) |
| `explain_decision` | Explainable AI reasoning for any analysis result |

### Risk Scoring

Thresholds: CRITICAL >= 0.8, HIGH >= 0.6, MEDIUM >= 0.4, LOW < 0.4. Defined in both `server.py` and `config.py` (`AppConfig`).

### Testing Architecture

Tests import from `tests/server_wrapper.py`, which re-exposes server functions as plain callables (bypassing FastMCP's `@mcp.tool()` decorator). Tests use fixtures from `tests/conftest.py` for sample data generation.

Available pytest markers: `unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`, `error`.

### Advanced Modules (not in active server)

| Module | Purpose |
|--------|---------|
| `feature_engineering.py` | 40+ feature extraction with cyclical encoding, uses Pydantic models from `models_validation.py` |
| `training_pipeline.py` | Full ML pipeline: SMOTE resampling, cross-validation, Optuna hyperparameter tuning, MLflow tracking |
| `async_inference.py` | Async prediction engine with LRU cache and batch support |
| `explainability.py` | SHAP-based explanations with graceful fallback when SHAP unavailable |
| `security.py` | OWASP security layer: JWT auth, RBAC, rate limiting, input sanitization |
| `monitoring.py` | Prometheus metrics, structlog, health checks, Grafana dashboard config |
| `models/autoencoder.py` | PyTorch autoencoder for anomaly detection |
| `models/gnn_fraud_detector.py` | Graph Neural Network fraud detector |
| `integration.py` | Synthetic data generation pipeline for testing |
| `benchmarks.py` | Performance benchmarking suite |
| `cli.py` | CLI for analyzing stored datasets (CSV/JSON) |

### Configuration

`config.py` uses Pydantic `BaseSettings` with `.env` file support. Key settings: model hyperparameters, risk thresholds, database URLs, rate limits, JWT config. Copy `.env.example` to `.env` for local configuration.

## Key Patterns

- Models are initialized with dummy training data at import time (random synthetic data). Replace with real trained models via the training pipeline for production use.
- Input validation happens in two layers: manual `validate_transaction_data()`/`validate_behavioral_data()` in `server.py`, and Pydantic models in `models_validation.py` (used by v2).
- All analysis functions return dicts with `risk_score` (0-1 float), `confidence`, `is_anomaly` boolean, and domain-specific details.
- Risk score conversion from Isolation Forest: `max(0, min(1, (0.5 - anomaly_score) * 2))`.
