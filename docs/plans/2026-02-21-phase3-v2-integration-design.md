# Phase 3: V2 Module Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement the corresponding implementation plan task-by-task.

**Goal:** Wire `feature_engineering.py`, `explainability.py`, and `models_validation.py` into `server.py` to upgrade from 8→46 features, add real SHAP explanations, and replace manual validation with Pydantic models.

**Architecture:** Incremental integration — each module wires in independently. Tests stay green after each step. No new MCP tools; existing tools get better internals.

## 1. Pydantic Validation

Replace manual `validate_transaction_data()` and `validate_behavioral_data()` with Pydantic models from `models_validation.py`.

**What changes:**
- `_impl` functions wrap input in `TransactionData(**data)` / `BehavioralData(**data)`
- Catch `ValidationError` → return structured error dict with field-level messages
- Remove manual validation functions after Pydantic is wired in

**Affects:** `analyze_transaction_impl`, `detect_behavioral_anomaly_impl`, `generate_risk_score_impl`

## 2. Feature Engineering (8→46 features)

Replace 8-feature manual extraction in `TransactionAnalyzer._extract_features()` with `FeatureEngineer.transform()`.

**What changes:**
- Initialize `FeatureEngineer` at module level alongside existing analyzers
- Retrain Isolation Forest on 46-feature synthetic data
- `_extract_features()` delegates to `FeatureEngineer.transform()`
- Feature names tracked for explainability

**Risk:** All thresholds shift with new feature space. Tests asserting specific risk levels need recalibration.

## 3. Explainability (real SHAP explanations)

Replace hollow `explain_decision_impl()` with `FraudExplainer.explain_prediction()`.

**What changes:**
- Initialize `FraudExplainer` with trained Isolation Forest + feature names
- `explain_decision_impl` calls `FraudExplainer.explain_prediction()` for transaction component
- SHAP is optional (graceful fallback to feature importance)

## 4. Testing Strategy

- Update existing tests for Pydantic validation error format
- Add tests for new 46-feature extraction
- Add tests for explainability output structure (top_features, human_summary)
- Recalibrate risk score assertions after model retraining
- Target: all tests pass, coverage >= 60%

## Out of Scope

- `server_v2.py` replacement (cherry-pick modules only)
- `async_inference.py` (synchronous is fine for MCP)
- `security.py` / `monitoring.py` (separate phases)
- `models/autoencoder.py`, `models/gnn_fraud_detector.py` (deep learning)
- `training_pipeline.py` (keep synthetic init data)
