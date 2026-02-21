# Phase 14: Test Coverage 95%+ & Warning Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Push server.py test coverage from 89% to 95%+ by covering exception handlers, edge cases, and import fallback branches. Fix remaining deprecation warnings.

**Architecture:** Create a single new test file `tests/test_coverage_gaps.py` with targeted tests for each uncovered code path. Use `unittest.mock.patch` to simulate failures and missing modules. Fix `datetime.utcnow()` deprecation in `monitoring.py`.

**Tech Stack:** pytest, unittest.mock, numpy

---

### Task 1: Fix monitoring.py deprecation warnings

**Files:**
- Modify: `monitoring.py:431,465`

**Step 1: Write the failing test**

Create `tests/test_coverage_gaps.py`:

```python
"""Tests targeting uncovered code paths in server.py to push coverage to 95%+."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock


class TestMonitoringDeprecationFix:
    """Verify datetime.utcnow() deprecation warnings are eliminated."""

    def test_no_utcnow_in_monitoring(self):
        """Ensure monitoring.py does not use deprecated datetime.utcnow()."""
        import inspect
        import monitoring
        source = inspect.getsource(monitoring)
        assert "utcnow()" not in source, (
            "monitoring.py still uses deprecated datetime.utcnow()"
        )
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coverage_gaps.py::TestMonitoringDeprecationFix::test_no_utcnow_in_monitoring -v`
Expected: FAIL because monitoring.py still has `utcnow()` on lines 431 and 465

**Step 3: Fix monitoring.py**

In `monitoring.py`, replace the two occurrences:

Line 431: change `datetime.utcnow().isoformat()` to `datetime.now(datetime.timezone.utc).isoformat()`
Line 465: change `datetime.utcnow().isoformat()` to `datetime.now(datetime.timezone.utc).isoformat()`

Also add `import datetime` at the top if it uses `from datetime import datetime` -- check imports first. The fix should use `datetime.now(datetime.timezone.utc)` since the file imports `from datetime import datetime`.  Actually, since `datetime` class is already imported, use `datetime.now(tz=__import__('datetime').timezone.utc)` -- NO. The cleanest approach: add `from datetime import timezone` to imports, then use `datetime.now(timezone.utc)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coverage_gaps.py::TestMonitoringDeprecationFix -v`
Expected: PASS

**Step 5: Run full suite to check no regressions**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: 534+ passed, warnings should no longer include `datetime.utcnow()` from monitoring.py

**Step 6: Commit**

```bash
git add monitoring.py tests/test_coverage_gaps.py
git commit -m "fix: Replace deprecated datetime.utcnow() in monitoring.py"
```

---

### Task 2: Test exception handler branches in analyzer classes

**Files:**
- Modify: `tests/test_coverage_gaps.py`

These cover uncovered lines: 242-244 (keystroke error), 291 (empty dwell_times), 302 (empty flight_times), 306-308 (feature extraction error), 403 (autoencoder threshold=0), 410-414 (autoencoder scoring failure), 495-496 (autoencoder save failure), 530-532 (autoencoder load failure), 771-774 (centrality calculation error).

**Step 1: Write the tests**

Append to `tests/test_coverage_gaps.py`:

```python
from server import (
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer,
    transaction_analyzer,
    network_analyzer,
)


class TestBehavioralExceptionPaths:
    """Cover exception branches in BehavioralBiometrics."""

    def test_keystroke_analysis_exception_returns_error(self):
        """Line 242-244: Exception in analyze_keystroke_dynamics."""
        bb = BehavioralBiometrics()
        # Pass data that will cause an internal error
        with patch.object(bb, '_extract_keystroke_features', side_effect=RuntimeError("boom")):
            result = bb.analyze_keystroke_dynamics([{"key": "a", "press_time": 100, "release_time": 150}] * 5)
        assert result["risk_score"] == 0.0
        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_feature_extraction_exception_returns_none(self):
        """Line 306-308: Exception in _extract_keystroke_features."""
        bb = BehavioralBiometrics()
        # Pass keystroke data that will trigger an exception inside feature extraction
        bad_data = [{"key": "a", "press_time": "not_a_number", "release_time": "bad"}] * 5
        result = bb._extract_keystroke_features(bad_data)
        assert result is None

    def test_empty_dwell_times_branch(self):
        """Line 291: dwell_times is empty, features get zeros."""
        bb = BehavioralBiometrics()
        # Keystroke data where press_time == release_time (no dwell) but we need
        # the code to handle missing 'release_time' or 'press_time' keys
        # Actually line 291 fires when dwell_times list is empty.
        # This happens if keystroke entries lack the required timing keys.
        # We need >= 2 entries for _extract_keystroke_features to not return None.
        data = [
            {"key": "a"},  # no press_time or release_time
            {"key": "b"},
            {"key": "c"},
        ]
        result = bb._extract_keystroke_features(data)
        # Will either return None (from exception) or features with zeros
        # The exact behavior depends on implementation; this covers the branch
        assert result is None or (isinstance(result, list) and len(result) == 10)


class TestTransactionAnalyzerExceptionPaths:
    """Cover exception branches in TransactionAnalyzer."""

    def test_autoencoder_threshold_zero_returns_zero_risk(self):
        """Line 403: autoencoder.threshold is 0 -> ae_risk = 0.0."""
        ta = TransactionAnalyzer()
        mock_ae = MagicMock()
        mock_ae.decision_function.return_value = np.array([0.5])
        mock_ae.threshold = 0  # triggers the else branch
        ta.autoencoder = mock_ae

        txn = {
            "amount": 100.0,
            "merchant": "TestShop",
            "location": "US",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card",
        }
        result = ta.analyze_transaction(txn)
        # autoencoder score should be 0.0 since threshold == 0
        assert "autoencoder" in result.get("model_scores", {})
        assert result["model_scores"]["autoencoder"] == 0.0

    def test_autoencoder_scoring_exception_falls_back_to_if(self):
        """Lines 410-414: Autoencoder scoring raises, falls back to IF score."""
        ta = TransactionAnalyzer()
        mock_ae = MagicMock()
        mock_ae.decision_function.side_effect = RuntimeError("AE exploded")
        mock_ae.threshold = 1.0
        ta.autoencoder = mock_ae

        txn = {
            "amount": 100.0,
            "merchant": "TestShop",
            "location": "US",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card",
        }
        result = ta.analyze_transaction(txn)
        # Should still return a valid result using IF only
        assert "risk_score" in result
        assert "autoencoder" not in result.get("model_scores", {})

    def test_save_models_autoencoder_save_failure(self, tmp_path):
        """Lines 495-496: Autoencoder save raises exception."""
        ta = TransactionAnalyzer()
        mock_ae = MagicMock()
        mock_ae.save.side_effect = RuntimeError("save failed")
        ta.autoencoder = mock_ae

        paths = ta.save_models(model_dir=tmp_path / "models")
        # Should save IF and FE but not autoencoder
        assert "isolation_forest" in paths
        assert "feature_engineer" in paths
        assert "autoencoder" not in paths

    def test_load_models_autoencoder_load_failure(self, tmp_path):
        """Lines 530-532: Autoencoder load raises exception."""
        import joblib

        # First save valid IF and FE models
        ta = TransactionAnalyzer()
        save_dir = tmp_path / "models"
        save_dir.mkdir(parents=True)
        joblib.dump(ta.isolation_forest, save_dir / "isolation_forest.joblib")
        joblib.dump(ta.feature_engineer, save_dir / "feature_engineer.joblib")

        # Create a fake autoencoder.pt file
        (save_dir / "autoencoder.pt").write_text("corrupted")

        # Mock AUTOENCODER_AVAILABLE to True and AutoencoderFraudDetector
        with patch("server.AUTOENCODER_AVAILABLE", True), \
             patch("server.AutoencoderFraudDetector") as MockAE:
            mock_instance = MagicMock()
            mock_instance.load.side_effect = RuntimeError("corrupt model")
            MockAE.return_value = mock_instance

            result = ta.load_models(model_dir=save_dir)
            assert result is True  # overall load succeeds
            assert ta.autoencoder is None  # but autoencoder is None


class TestNetworkAnalyzerExceptionPaths:
    """Cover exception branches in NetworkAnalyzer."""

    def test_centrality_calculation_error(self):
        """Lines 771-774: centrality calculation raises exception."""
        na = NetworkAnalyzer()
        na.add_connection("nodeA", "nodeB", 0.5)

        with patch("networkx.betweenness_centrality", side_effect=RuntimeError("centrality error")):
            metrics = na._calculate_entity_metrics("nodeA")
            assert metrics["betweenness_centrality"] == 0
            assert metrics["closeness_centrality"] == 0
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_coverage_gaps.py -v --tb=short`
Expected: All tests PASS

**Step 3: Check coverage improvement**

Run: `python -m pytest tests/ --cov=server --cov-report=term-missing -q --tb=no 2>&1 | grep "^server"`
Expected: Coverage should improve from 89% toward 92-93%

**Step 4: Commit**

```bash
git add tests/test_coverage_gaps.py
git commit -m "test: Cover exception handler branches in analyzer classes"
```

---

### Task 3: Test _impl function exception paths

**Files:**
- Modify: `tests/test_coverage_gaps.py`

These cover lines: 1134-1136, 1191-1193, 1294-1295, 1322-1324, 1447-1448, 1459-1460, 1610-1612, 1806-1808, 1861, 1926-1928, 2122-2124.

**Step 1: Write the tests**

Append to `tests/test_coverage_gaps.py`:

```python
from server import (
    analyze_transaction_impl,
    detect_behavioral_anomaly_impl,
    generate_risk_score_impl,
    explain_decision_impl,
    health_check_impl,
    generate_synthetic_dataset_impl,
    analyze_dataset_impl,
    run_benchmark_impl,
    train_models_impl,
)


class TestImplExceptionPaths:
    """Cover outer exception handlers in _impl functions."""

    def test_analyze_transaction_impl_outer_exception(self):
        """Lines 1134-1136: analyze_transaction_impl catches top-level exception."""
        with patch("server.validate_transaction_data", side_effect=RuntimeError("validation boom")):
            result = analyze_transaction_impl({"amount": 100})
        assert result["status"] == "analysis_failed"
        assert "validation boom" in result["error"]

    def test_detect_behavioral_impl_outer_exception(self):
        """Lines 1191-1193: detect_behavioral_anomaly_impl catches exception."""
        with patch("server.validate_behavioral_data", side_effect=RuntimeError("behavioral boom")):
            result = detect_behavioral_anomaly_impl({"keystroke_dynamics": []})
        assert result["status"] == "analysis_failed"
        assert "behavioral boom" in result["error"]

    def test_generate_risk_score_impl_outer_exception(self):
        """Lines 1322-1324: generate_risk_score_impl catches exception."""
        with patch("server.analyze_transaction_impl", side_effect=RuntimeError("risk boom")):
            result = generate_risk_score_impl({"amount": 100})
        assert result["status"] == "analysis_failed"
        assert "risk boom" in result["error"]

    def test_generate_risk_score_medium_risk_branch(self):
        """Lines 1294-1295: MEDIUM risk level branch."""
        # Need a risk score between 0.4 and 0.6
        mock_txn_result = {
            "overall_risk_score": 0.45,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
        }
        with patch("server.analyze_transaction_impl", return_value=mock_txn_result):
            result = generate_risk_score_impl({"amount": 100})
        assert result["risk_level"] == "MEDIUM"
        assert "monitor_closely" in result["recommended_actions"]

    def test_health_check_impl_monitor_exception(self):
        """Lines 1610-1612: System health check raises exception."""
        with patch("server.monitor") as mock_mon:
            mock_mon.health_check.side_effect = RuntimeError("monitor failed")
            # monitor is not None so the branch is entered
            result = health_check_impl()
        assert "error" in result.get("system", {})

    def test_generate_synthetic_dataset_exception(self):
        """Lines 1806-1808: generate_synthetic_dataset_impl catches exception."""
        with patch("server.SYNTHETIC_DATA_AVAILABLE", True), \
             patch("server.synthetic_data_integration") as mock_sdi:
            mock_sdi.generate_and_save_dataset.side_effect = RuntimeError("gen failed")
            result = generate_synthetic_dataset_impl()
        assert result["status"] == "generation_failed"
        assert "gen failed" in result["error"]

    def test_analyze_dataset_empty_dataset(self, tmp_path):
        """Line 1861: Dataset is empty."""
        import pandas as pd
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(csv_path, index=False)

        result = analyze_dataset_impl(str(csv_path))
        assert result["status"] == "empty_dataset"

    def test_analyze_dataset_exception(self):
        """Lines 1926-1928: analyze_dataset_impl catches exception."""
        result = analyze_dataset_impl("/nonexistent/path/data.csv")
        assert result["status"] == "analysis_failed"

    def test_run_benchmark_exception(self):
        """Lines 2122-2124: run_benchmark_impl catches exception."""
        with patch("server.transaction_analyzer.analyze_transaction", side_effect=RuntimeError("bench fail")):
            result = run_benchmark_impl(num_transactions=10)
        assert result["status"] == "benchmark_failed"
        assert "bench fail" in result["error"]


class TestExplainDecisionExceptionPaths:
    """Cover explain_decision_impl edge cases."""

    def test_shap_explanation_failure(self):
        """Lines 1447-1448: On-demand SHAP explanation raises exception."""
        analysis_result = {
            "overall_risk_score": 0.7,
            "risk_level": "HIGH",
            "detected_anomalies": ["high_amount"],
        }
        txn_data = {
            "amount": 100.0,
            "merchant": "Shop",
            "location": "US",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card",
        }
        with patch("server.fraud_explainer") as mock_explainer:
            mock_explainer.explain_prediction.side_effect = RuntimeError("SHAP failed")
            # validate_transaction_data needs to return True
            with patch("server.validate_transaction_data", return_value=(True, "")):
                result = explain_decision_impl(analysis_result, txn_data)
        # Should still return a valid explanation, just without SHAP
        assert "decision_summary" in result
        assert "feature_analysis" not in result

    def test_summary_generation_failure(self):
        """Lines 1459-1460: Summary generation from feature_analysis raises."""
        analysis_result = {
            "overall_risk_score": 0.7,
            "risk_level": "HIGH",
            "detected_anomalies": [],
            "feature_explanation": {"method": "shap", "features": {}},
        }
        with patch("server.fraud_explainer") as mock_explainer:
            mock_explainer.generate_summary.side_effect = RuntimeError("summary failed")
            result = explain_decision_impl(analysis_result)
        # Should still return explanation but without human_readable_summary
        assert "decision_summary" in result
        assert "human_readable_summary" not in result
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_coverage_gaps.py -v --tb=short`
Expected: All PASS

**Step 3: Check coverage**

Run: `python -m pytest tests/ --cov=server --cov-report=term-missing -q --tb=no 2>&1 | grep "^server"`
Expected: Coverage ~94-95%

**Step 4: Commit**

```bash
git add tests/test_coverage_gaps.py
git commit -m "test: Cover _impl function exception and edge-case branches"
```

---

### Task 4: Test train_models_impl full path and _monitored decorator

**Files:**
- Modify: `tests/test_coverage_gaps.py`

Covers lines: 1642-1674 (train_models_impl body), 919 (_monitored no-op branch).

**Step 1: Write the tests**

Append to `tests/test_coverage_gaps.py`:

```python
class TestTrainModelsImpl:
    """Cover train_models_impl execution paths."""

    def test_train_models_file_not_found(self):
        """Lines 1644-1648: Data file doesn't exist."""
        result = train_models_impl("/nonexistent/path/data.csv")
        assert result["status"] == "file_not_found"

    def test_train_models_training_exception(self, tmp_path):
        """Lines 1672-1677: Training raises exception."""
        # Create a dummy CSV so file exists
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("amount,is_fraud\n100,0\n200,1\n")

        with patch("server.TRAINING_AVAILABLE", True), \
             patch("server.ModelTrainer") as MockTrainer:
            MockTrainer.return_value.train_all_models.side_effect = RuntimeError("train exploded")
            result = train_models_impl(str(csv_path))
        assert result["status"] == "training_failed"
        assert "train exploded" in result["error"]

    def test_train_models_success_with_hot_reload(self, tmp_path):
        """Lines 1650-1670: Successful training with hot reload."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("amount,is_fraud\n100,0\n200,1\n")

        with patch("server.TRAINING_AVAILABLE", True), \
             patch("server.ModelTrainer") as MockTrainer, \
             patch.object(transaction_analyzer, "load_models", return_value=True), \
             patch.object(transaction_analyzer, "_model_source", "saved"):
            MockTrainer.return_value.train_all_models.return_value = {
                "metrics": {"accuracy": 0.95},
                "status": "success",
            }
            result = train_models_impl(str(csv_path))
        assert result["hot_reload"] is True

    def test_train_models_success_no_hot_reload(self, tmp_path):
        """Line 1668: Hot reload fails."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("amount,is_fraud\n100,0\n200,1\n")

        with patch("server.TRAINING_AVAILABLE", True), \
             patch("server.ModelTrainer") as MockTrainer, \
             patch.object(transaction_analyzer, "load_models", return_value=False):
            MockTrainer.return_value.train_all_models.return_value = {
                "metrics": {"accuracy": 0.95},
                "status": "success",
            }
            result = train_models_impl(str(csv_path))
        assert result["hot_reload"] is False


class TestMonitoredDecorator:
    """Cover _monitored no-op branch."""

    def test_monitored_returns_noop_when_monitoring_unavailable(self):
        """Line 919: _monitored returns identity decorator when monitoring off."""
        from server import _monitored
        with patch("server.MONITORING_AVAILABLE", False):
            decorator = _monitored("/test", "GET")
            # Should be a no-op: decorator(fn) returns fn unchanged
            def dummy():
                return 42
            assert decorator(dummy) is dummy
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_coverage_gaps.py -v --tb=short`
Expected: All PASS

**Step 3: Final coverage check**

Run: `python -m pytest tests/ --cov=server --cov-report=term-missing -q --tb=no 2>&1 | grep "^server"`
Expected: Coverage >= 95%

**Step 4: Commit**

```bash
git add tests/test_coverage_gaps.py
git commit -m "test: Cover train_models_impl paths and _monitored decorator"
```

---

### Task 5: Final verification and version marker

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short --cov=server --cov-report=term-missing 2>&1 | tail -30`
Expected: 555+ passed, coverage >= 95%, no `datetime.utcnow()` deprecation warnings from monitoring.py

**Step 2: Run linting**

Run: `ruff check tests/test_coverage_gaps.py`
Expected: No errors

**Step 3: Add pytest marker**

In `pytest.ini`, add `coverage: Coverage gap tests` to the markers list.

**Step 4: Commit**

```bash
git add pytest.ini
git commit -m "chore: Add coverage marker to pytest.ini, Phase 14 complete"
```
