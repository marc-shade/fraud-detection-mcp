# Phase 8: SHAP-Based Explainability Integration

## Overview
Integrate the `FraudExplainer` from `explainability.py` into the active MCP server (`server.py`), adding graceful degradation, on-demand SHAP-based feature explanations in `explain_decision`, and explainability status in health check and model status endpoints.

## Tasks

### Task 1: Add EXPLAINABILITY_AVAILABLE flag with graceful degradation
- Convert direct `from explainability import FraudExplainer` to try/except with `EXPLAINABILITY_AVAILABLE` flag
- Import `SHAP_AVAILABLE` flag from explainability module
- Guard `fraud_explainer` initialization with `EXPLAINABILITY_AVAILABLE` check
- Guard `fraud_explainer` usage in `analyze_transaction_impl` with `is not None` check
- Commit: `feat: Add EXPLAINABILITY_AVAILABLE flag with graceful degradation`

### Task 2: Enhance explain_decision_impl with SHAP-based explanations
- Add optional `transaction_data` parameter to `explain_decision_impl`
- When `transaction_data` is provided and `fraud_explainer` is available, extract features and call `fraud_explainer.explain_prediction()` for on-demand SHAP-based explanations
- Integrate `fraud_explainer.generate_summary()` for human-readable summaries
- Add `explainability_method` field to track method used (SHAP, Feature Importance, rule_based)
- Update `explain_decision` MCP tool wrapper to pass through new parameter
- Commit: `feat: Enhance explain_decision_impl with SHAP-based explanations`

### Task 3: Add explainability status to health_check and get_model_status
- Add `explainability` section to `health_check_impl` with `available`, `shap_available`, `explainer_loaded`, `fallback_mode`
- Add `explainer` details to `get_model_status_impl` models section with `loaded`, `available`, `shap_available`, `fallback_mode`, `method`
- Commit: `feat: Add explainability status to health_check and get_model_status`

### Task 4: Write comprehensive explainability tests
- Test `EXPLAINABILITY_AVAILABLE` flag and `FraudExplainer` initialization
- Test `FraudExplainer` unit tests (explain_prediction, generate_summary, batch_explain)
- Test `explain_decision_impl` with `transaction_data` parameter
- Test feature explanation integration in `analyze_transaction_impl`
- Test health check and model status explainability fields
- Test graceful degradation when explainability unavailable (mock `fraud_explainer=None`)
- Test end-to-end explainability flow (analyze then explain)
- Add `explainability` pytest marker to `pytest.ini`
- Commit: `test: Add Phase 8 explainability integration tests (55 tests)`

### Task 5: Final verification
- Run ruff check on modified files, fix any lint errors
- Run full test suite (390 tests passing)
- Create plan document
- Commit: `docs: Add Phase 8 plan document, fix lint errors`

## Results
- **Tests before**: 335 passed, 2 skipped
- **Tests after**: 390 passed, 2 skipped (+55 new tests)
- **New features**: SHAP-based on-demand feature explanations, human-readable summaries, explainability status in health check and model status
- **Graceful degradation**: Full backward compatibility when explainability module unavailable
