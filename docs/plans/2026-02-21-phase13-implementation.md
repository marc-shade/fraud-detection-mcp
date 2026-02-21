# Phase 13: Code Quality Polish

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all 41 Pydantic v2 deprecation warnings, align version strings to 2.3.0, and update CLAUDE.md to reflect Phases 8-12.

**Architecture:** Fix deprecated Pydantic patterns in `config.py` and `models_validation.py` (class Config -> model_config = ConfigDict, max_items -> max_length, env= -> validation_alias, json_encoders -> custom serializers). Bump version in server.py health check, model_version, and MonitoringManager. Rewrite CLAUDE.md to document all 13 MCP tools and integrated modules.

**Tech Stack:** Pydantic v2, pydantic-settings

---

### Task 1: Fix Pydantic v2 deprecation warnings in config.py

**Files:**
- Modify: `config.py`

**Step 1: Replace `class Config` with `model_config = ConfigDict`**

In `config.py`, the `AppConfig` class (line 18) has a nested `class Config` at line 111-114:

```python
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
```

Replace with:

```python
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
```

Add `ConfigDict` to the pydantic-settings import. Change line 11 from:

```python
from pydantic_settings import BaseSettings
```

to:

```python
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
```

Wait — `ConfigDict` comes from `pydantic`, not `pydantic_settings`. Update the pydantic import at line 10 from:

```python
from pydantic import Field, field_validator
```

to:

```python
from pydantic import ConfigDict, Field, field_validator
```

**Step 2: Replace deprecated `env=` kwargs with `alias=`**

In Pydantic v2 with pydantic-settings, environment variable binding uses `alias` or `validation_alias` rather than `env=` as extra kwargs. For `BaseSettings`, the field name IS the env var name (case_sensitive=True), so the `env=` kwargs on fields that match the field name are redundant. Remove them:

Line 24: `Field(default="development", env="ENVIRONMENT")` → `Field(default="development")`
Line 25: `Field(default=False, env="DEBUG")` → `Field(default=False)`
Line 69: `Field(default=None, env="DATABASE_URL")` → `Field(default=None)`
Line 70: `Field(default="redis://localhost:6379", env="REDIS_URL")` → `Field(default="redis://localhost:6379")`
Line 74: `Field(default=None, env="JWT_SECRET_KEY")` → `Field(default=None)`
Line 84: `Field(default=True, env="ENABLE_METRICS")` → `Field(default=True)`
Line 85: `Field(default=9090, env="METRICS_PORT")` → `Field(default=9090)`
Line 86: `Field(default="INFO", env="LOG_LEVEL")` → `Field(default="INFO")`
Line 89-91: `Field(default=None, env="MLFLOW_TRACKING_URI")` → `Field(default=None)`

**Step 3: Update APP_VERSION**

Line 23: `APP_VERSION: str = "2.0.0"` → `APP_VERSION: str = "2.3.0"`

**Step 4: Verify config still loads**

Run: `python -c "from config import get_config; c = get_config(); print(c.APP_VERSION, c.ENVIRONMENT)"`
Expected: `2.3.0 development`

**Step 5: Run tests to check no warnings from config.py**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | grep "config.py" | wc -l`
Expected: `0` (no config.py warnings)

**Step 6: Commit**

```bash
git add config.py
git commit -m "fix: Eliminate Pydantic v2 deprecation warnings in config.py"
```

---

### Task 2: Fix Pydantic v2 deprecation warnings in models_validation.py

**Files:**
- Modify: `models_validation.py`

**Step 1: Add ConfigDict import**

Add `ConfigDict` to the pydantic import at line 7:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
```

**Step 2: Replace `class Config` in TransactionData (line 178-183)**

Replace:

```python
    class Config:
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

With:

```python
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
```

Note: `json_encoders` is removed entirely — Pydantic v2 uses custom serializers via `@field_serializer` instead, but we don't need it since the server already calls `.isoformat()` where needed.

**Step 3: Replace `class Config` in AnalysisRequest (line 292-293)**

Replace:

```python
    class Config:
        validate_assignment = True
```

With:

```python
    model_config = ConfigDict(validate_assignment=True)
```

**Step 4: Replace `class Config` in AnalysisResponse (line 317-321)**

Replace:

```python
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

With:

```python
    model_config = ConfigDict(use_enum_values=True)
```

**Step 5: Replace deprecated `max_items` with `max_length`**

Line 191: `max_items=10000` → `max_length=10000`
Line 196: `max_items=10000` → `max_length=10000`
Line 250: `max_items=10000` → `max_length=10000`

**Step 6: Replace deprecated `min_items` / `max_items` in BatchAnalysisRequest (line 369-370)**

Replace:

```python
        min_items=1,
        max_items=10000,
```

With:

```python
        min_length=1,
        max_length=10000,
```

**Step 7: Update model_version default in AnalysisResponse (line 306)**

`model_version: str = Field(default="2.0.0")` → `model_version: str = Field(default="2.3.0")`

**Step 8: Verify models still load**

Run: `python -c "from models_validation import TransactionData, BehavioralData, NetworkData; print('OK')"`
Expected: `OK`

**Step 9: Run tests to check no warnings from models_validation.py**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | grep "models_validation.py" | wc -l`
Expected: `0`

**Step 10: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 533 passed, 2 skipped, significantly fewer warnings

**Step 11: Commit**

```bash
git add models_validation.py
git commit -m "fix: Eliminate Pydantic v2 deprecation warnings in models_validation.py"
```

---

### Task 3: Align version strings in server.py

**Files:**
- Modify: `server.py`

**Step 1: Update version in health_check_impl**

Find `"version": "2.2.0"` in `health_check_impl` and change to `"version": "2.3.0"`.

**Step 2: Update model_version in analyze_transaction_impl**

Find `results["model_version"] = "v2.1.0"` and change to `results["model_version"] = "v2.3.0"`.

**Step 3: Update MonitoringManager version**

Find `MonitoringManager(app_name="fraud-detection-mcp", version="2.1.0")` and change to `MonitoringManager(app_name="fraud-detection-mcp", version="2.3.0")`.

**Step 4: Run tests**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 533 passed, 2 skipped

**Step 5: Commit**

```bash
git add server.py
git commit -m "chore: Bump all version strings to 2.3.0"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Rewrite CLAUDE.md**

Replace the entire contents with the updated version below. Key changes:
- TransactionAnalyzer now uses 46-feature FeatureEngineer (not 8 features)
- 13 MCP tools listed (was 5)
- Integrated modules section (was "Advanced Modules not in active server")
- New markers listed
- Graceful degradation pattern documented
- UserTransactionHistory documented
- Version 2.3.0

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced Fraud Detection MCP server using FastMCP. Combines behavioral biometrics (keystroke dynamics, mouse patterns), ML-based anomaly detection (Isolation Forest, XGBoost, Autoencoders), SHAP-based explainability, and graph-based fraud ring detection (NetworkX) into a unified MCP tool interface. Version 2.3.0.

## Development Commands

` ``bash
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
python -m pytest -m synthetic      # Synthetic data tests
python -m pytest -m benchmark      # Benchmark tests
python -m pytest -m security       # Security tests
python -m pytest -m velocity       # Velocity analysis tests

# Linting (CI uses ruff)
ruff check . --output-format=github
ruff format --check .

# Security scanning
bandit -r . -x ./tests,./.venv -ll

# Type checking
mypy server.py --ignore-missing-imports
` ``

## Architecture

### Server (server.py)

The active MCP server. Contains all core analyzers, integrated modules, validation functions, and 13 MCP tool definitions.

### Core Analysis Pipeline

Four analyzer/tracker classes instantiated as module-level singletons:

1. **`BehavioralBiometrics`** — Keystroke dynamics via Isolation Forest, mouse patterns via One-Class SVM, touch patterns via LOF. Extracts 10 statistical features (5 dwell time + 5 flight time).
2. **`TransactionAnalyzer`** — Uses `FeatureEngineer` to extract 46 features (amount, cyclical time encoding, categorical, location, merchant, behavioral, network, derived). Isolation Forest + optional Autoencoder ensemble scoring.
3. **`NetworkAnalyzer`** — Builds a NetworkX directed graph of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality.
4. **`UserTransactionHistory`** — In-memory per-user transaction tracker. Provides velocity analysis (burst detection), amount deviation (z-score), geographic velocity (impossible travel), and merchant diversity (card testing).

### MCP Tools (13)

| Tool | Function |
|------|----------|
| `analyze_transaction` | Full transaction fraud analysis with velocity tracking |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection |
| `assess_network_risk` | Graph-based fraud ring detection |
| `generate_risk_score` | Weighted composite score (transaction 50%, behavioral 30%, network 20%) |
| `explain_decision` | Explainable AI with optional SHAP-based feature explanations |
| `analyze_batch` | Batch transaction analysis with aggregated stats |
| `get_inference_stats` | Prediction cache and inference performance metrics |
| `health_check` | System health including all module statuses |
| `get_model_status` | Detailed model and feature information |
| `train_models` | Trigger model training pipeline |
| `generate_synthetic_dataset` | Generate fraud test datasets with configurable patterns |
| `analyze_dataset` | Analyze stored CSV/JSON datasets with performance metrics |
| `run_benchmark` | Pipeline benchmark (throughput, latency, accuracy) |

### Graceful Degradation Pattern

Optional modules use try/except imports with `*_AVAILABLE` flags:

| Flag | Module | Feature |
|------|--------|---------|
| `MONITORING_AVAILABLE` | `monitoring.py` | Prometheus metrics, health checks |
| `TRAINING_AVAILABLE` | `training_pipeline.py` | SMOTE, cross-validation, Optuna, MLflow |
| `AUTOENCODER_AVAILABLE` | `models/autoencoder.py` | PyTorch autoencoder ensemble member |
| `EXPLAINABILITY_AVAILABLE` | `explainability.py` | SHAP-based feature explanations |
| `SYNTHETIC_DATA_AVAILABLE` | `integration.py` | Synthetic fraud dataset generation |
| `SECURITY_UTILS_AVAILABLE` | `security_utils.py` | Input sanitization, rate limiting |

Hard dependencies (always imported): `config.py`, `models_validation.py`, `feature_engineering.py`, `async_inference.py` (LRUCache).

### Risk Scoring

Thresholds: CRITICAL >= 0.8, HIGH >= 0.6, MEDIUM >= 0.4, LOW < 0.4. Defined in `config.py` (`AppConfig`).

Ensemble: `0.6 * IF_score + 0.4 * AE_score` when autoencoder available, IF-only otherwise. Risk factor multiplier: `1 + len(risk_factors) * 0.1`.

### Testing Architecture

Tests import `_impl` functions directly from `server` module (e.g., `server.analyze_transaction_impl`). Some tests use `tests/server_wrapper.py` for legacy compatibility. Fixtures in `tests/conftest.py`.

Available pytest markers: `unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`, `explainability`, `synthetic`, `benchmark`, `security`, `velocity`, `error`.

### Configuration

`config.py` uses Pydantic `BaseSettings` with `.env` file support. Key settings: model hyperparameters, risk thresholds, high-risk locations/payment methods. Copy `.env.example` to `.env` for local configuration.

## Key Patterns

- `_impl` suffix: All MCP tools delegate to `*_impl` functions for testability.
- Models initialize with synthetic training data at import time. Replace with real trained models via `train_models` tool.
- Input validation: manual `validate_transaction_data()`/`validate_behavioral_data()` in server.py, plus Pydantic models in `models_validation.py` for feature engineering.
- Risk score from Isolation Forest: `max(0, min(1, 0.5 - anomaly_score))`.
- Velocity features recorded per-call in `UserTransactionHistory`, checked for: burst (10+ txns/hour), z-score deviation (>3.0), impossible travel (<300s between locations), merchant diversity (5+ unique in 5+ txns).
```

NOTE: The triple-backtick code fence for "Development Commands" must be properly closed. Make sure to use actual backticks, not the escaped versions shown above.

**Step 2: Run tests to verify nothing broke**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 533 passed, 2 skipped

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: Update CLAUDE.md with Phases 8-12 additions, 13 MCP tools"
```

---

### Task 5: Final verification

**Step 1: Count remaining warnings**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | grep -c "PydanticDeprecatedSince20"`
Expected: `0` (all Pydantic deprecation warnings eliminated)

**Step 2: Run ruff on modified files**

Run: `ruff check config.py models_validation.py server.py`
Expected: Clean

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: 533 passed, 2 skipped, minimal warnings (only the `datetime.utcnow()` one from monitoring.py remains)

**Step 4: Verify version consistency**

Run: `python -c "from config import get_config; import server; print('config:', get_config().APP_VERSION); hc = server.health_check_impl(); print('health:', hc['version'])"`
Expected: Both print `2.3.0`

**Step 5: Commit any remaining fixes**

```bash
git add config.py models_validation.py server.py CLAUDE.md
git commit -m "chore: Phase 13 code quality polish complete"
```
