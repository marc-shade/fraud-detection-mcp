# Phase 6: Training Pipeline Integration & Model Persistence

## Overview
Integrate the `ModelTrainer` from `training_pipeline.py` into the active MCP server (`server.py`), adding model training via MCP tools and model persistence via joblib.

## Tasks

### Task 1: Add joblib import and TRAINING_AVAILABLE flag
- Add `import joblib` to server.py imports
- Add try/except import of `ModelTrainer` from `training_pipeline` with `TRAINING_AVAILABLE` flag
- Add `_model_source` attribute to `TransactionAnalyzer.__init__` (default: `"synthetic"`)
- Commit: `feat: Add joblib import and TRAINING_AVAILABLE flag`

### Task 2: Add model persistence to TransactionAnalyzer
- Add `save_models(path)` method: saves isolation_forest + feature_engineer via joblib
- Add `load_models(path)` method: loads from disk, sets `_model_source = "saved"`
- Attempt to load saved models in `__init__` before falling back to synthetic data
- Set `_model_source = "none"` initially, `"saved"` on load, `"synthetic"` on fallback
- Commit: `feat: Add model persistence with save/load to TransactionAnalyzer`

### Task 3: Add train_models MCP tool
- Create `train_models_impl(data_path, **kwargs)` function
- If `TRAINING_AVAILABLE`: instantiate `ModelTrainer`, call `train_all_models()`
- If not available: return `{"error": "Training dependencies not available", ...}`
- Expose via `@mcp.tool()` decorator
- Commit: `feat: Add train_models MCP tool for model training`

### Task 4: Add get_model_status MCP tool
- Create `get_model_status_impl()` returning model_source, training_available, model details
- Include feature count, estimator count, saved model paths
- Expose via `@mcp.tool()` decorator
- Commit: `feat: Add get_model_status MCP tool`

### Task 5: Write comprehensive tests
- Test model persistence (save/load round-trip)
- Test train_models_impl (with and without training deps)
- Test get_model_status_impl
- Test _model_source tracking through lifecycle
- Test TRAINING_AVAILABLE flag behavior
- Commit: `test: Add Phase 6 training pipeline and model persistence tests`

### Task 6: Final verification
- Update health_check_impl to include model_source
- Run full test suite (275+ existing + new tests)
- Run linting (ruff) and type checks (mypy)
- Commit: `feat: Add model_source to health check, Phase 6 complete`
