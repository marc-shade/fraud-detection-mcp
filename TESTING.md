# Testing Guide

## Quick Smoke Test

```bash
python -c "from config import get_config; print('config OK')"
python -c "from models_validation import TransactionData; print('validation OK')"
```

## Install Dependencies

```bash
pip install -r requirements.txt        # core
pip install -r requirements-dev.txt    # dev / test tooling
pip install -r requirements-optional.txt  # SHAP, PyTorch, MLflow, etc.
```

## Run the Full Test Suite

```bash
# All tests with coverage (CI requires 60% minimum).
python run_tests.py

# Direct pytest invocation with verbose output.
python -m pytest tests/ -v --tb=short --cov=server --cov-report=term-missing

# A single file.
python -m pytest tests/test_transaction_analysis.py -v
```

## Run by Marker

Available markers (see `pytest.ini` for the authoritative list):
`unit`, `integration`, `slow`, `network`, `behavioral`, `transaction`,
`explainability`, `synthetic`, `benchmark`, `error`, `security`, `velocity`.

```bash
python -m pytest -m unit
python -m pytest -m "integration and not slow"
python -m pytest -m behavioral
```

## Benchmarks

```bash
python benchmarks.py
```

Reports throughput, latency percentiles, and compares against the claimed
detection rate / false-positive rate / latency / throughput targets.

## Optional Dependencies

The server uses graceful-degradation imports. These are optional:

- `shap` — SHAP explanations; falls back to feature importance
- `torch` — autoencoder ensemble member; falls back to Isolation Forest only
- `torch_geometric` — GNN fraud detector; fallback available
- `mlflow` — training pipeline tracking
- `redis` — distributed rate limiting; in-memory fallback used if absent

## Linting and Type Checking

```bash
ruff check . --output-format=github
ruff format --check .
mypy server.py --ignore-missing-imports
bandit -r . -x ./tests,./.venv -ll
```

## Pre-deployment Checklist

1. Install all dependencies (`requirements*.txt`).
2. `python run_tests.py` passes with coverage >= 60%.
3. `python benchmarks.py` meets published targets.
4. `ruff check` and `ruff format --check` clean.
5. `bandit` shows no high-severity findings.
6. Trained models present in `models/saved/` (or accept the synthetic-data
   cold start at import time).
