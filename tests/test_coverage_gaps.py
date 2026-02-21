"""Tests targeting uncovered code paths in server.py to push coverage to 95%+."""

import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from server import (
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer,
    transaction_analyzer,
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


class TestBehavioralExceptionPaths:
    """Cover exception branches in BehavioralBiometrics."""

    def test_keystroke_analysis_exception_returns_error(self):
        """Line 242-244: Exception in analyze_keystroke_dynamics."""
        bb = BehavioralBiometrics()
        with patch.object(
            bb, "_extract_keystroke_features", side_effect=RuntimeError("boom")
        ):
            result = bb.analyze_keystroke_dynamics(
                [{"key": "a", "press_time": 100, "release_time": 150}] * 5
            )
        assert result["risk_score"] == 0.0
        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_feature_extraction_outer_exception_returns_none(self):
        """Lines 306-308: Outer exception in _extract_keystroke_features returns None."""
        bb = BehavioralBiometrics()
        # Mock np.mean to raise inside the try block after the lists are populated
        with patch("server.np.mean", side_effect=RuntimeError("numpy error")):
            data = [
                {"key": "a", "press_time": 100, "release_time": 150},
                {"key": "b", "press_time": 200, "release_time": 250},
            ]
            result = bb._extract_keystroke_features(data)
        assert result is None

    def test_empty_dwell_times_with_valid_flight_times(self):
        """Line 291: dwell_times is empty but flight_times has data."""
        bb = BehavioralBiometrics()
        # No entry has BOTH press_time and release_time -> no dwell times.
        # But consecutive entries have release_time (prev) and press_time (curr)
        # -> valid flight times.
        data = [
            {"key": "a", "release_time": 100},  # no press_time -> no dwell
            {
                "key": "b",
                "press_time": 150,
            },  # flight = 150 - 100 = 50; no release_time -> no dwell
            {
                "key": "c",
                "press_time": 250,
            },  # prev has no release_time -> no flight; no release_time -> no dwell
        ]
        result = bb._extract_keystroke_features(data)
        assert result is not None
        assert len(result) == 10
        # First 5 features (dwell) should be zeros since no valid dwell pairs
        assert result[:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
        # Flight times portion should have data (one valid flight time of 50)
        assert result[5] == 50.0  # mean of [50] = 50

    def test_empty_flight_times_with_dwell_times(self):
        """Line 302: flight_times is empty but dwell_times has data."""
        bb = BehavioralBiometrics()
        # Entries that have press/release (dwell data) but no valid flight
        # pairs (flight needs prev release_time + current press_time).
        data = [
            {"key": "a", "press_time": 100, "release_time": 150},
            {"key": "b", "release_time": 250},  # no press_time -> no flight calc
        ]
        result = bb._extract_keystroke_features(data)
        assert result is not None
        assert len(result) == 10
        # flight_times portion (indices 5-9) should be all zeros
        assert result[5:] == [0.0, 0.0, 0.0, 0.0, 0.0]


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
        assert "risk_score" in result
        assert "autoencoder" not in result.get("model_scores", {})

    def test_save_models_autoencoder_save_failure(self, tmp_path):
        """Lines 495-496: Autoencoder save raises exception."""
        ta = TransactionAnalyzer()
        mock_ae = MagicMock()
        mock_ae.save.side_effect = RuntimeError("save failed")
        ta.autoencoder = mock_ae

        paths = ta.save_models(model_dir=tmp_path / "models")
        assert "isolation_forest" in paths
        assert "feature_engineer" in paths
        assert "autoencoder" not in paths

    def test_load_models_autoencoder_load_failure(self, tmp_path):
        """Lines 530-532: Autoencoder load raises exception."""
        import joblib

        ta = TransactionAnalyzer()
        save_dir = tmp_path / "models"
        save_dir.mkdir(parents=True)
        joblib.dump(ta.isolation_forest, save_dir / "isolation_forest.joblib")
        joblib.dump(ta.feature_engineer, save_dir / "feature_engineer.joblib")

        # Create a fake autoencoder.pt file
        (save_dir / "autoencoder.pt").write_text("corrupted")

        with (
            patch("server.AUTOENCODER_AVAILABLE", True),
            patch("server.AutoencoderFraudDetector") as MockAE,
        ):
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
        # Use _update_graph to add nodes to the internal graph
        na._update_graph("nodeA", [{"entity_id": "nodeB", "strength": 0.5}])

        with patch(
            "networkx.betweenness_centrality",
            side_effect=RuntimeError("centrality error"),
        ):
            metrics = na._calculate_network_metrics("nodeA")
            assert metrics["betweenness_centrality"] == 0
            assert metrics["closeness_centrality"] == 0


class TestImplExceptionPaths:
    """Cover outer exception handlers in _impl functions."""

    def test_analyze_transaction_impl_outer_exception(self):
        """Lines 1134-1136: analyze_transaction_impl catches top-level exception."""
        with patch(
            "server.validate_transaction_data",
            side_effect=RuntimeError("validation boom"),
        ):
            result = analyze_transaction_impl({"amount": 100})
        assert result["status"] == "analysis_failed"
        assert "validation boom" in result["error"]

    def test_detect_behavioral_impl_outer_exception(self):
        """Lines 1191-1193: detect_behavioral_anomaly_impl catches exception."""
        with patch(
            "server.validate_behavioral_data",
            side_effect=RuntimeError("behavioral boom"),
        ):
            result = detect_behavioral_anomaly_impl({"keystroke_dynamics": []})
        assert result["status"] == "analysis_failed"
        assert "behavioral boom" in result["error"]

    def test_generate_risk_score_impl_outer_exception(self):
        """Lines 1322-1324: generate_risk_score_impl catches exception."""
        with patch(
            "server.validate_transaction_data",
            side_effect=RuntimeError("risk boom"),
        ):
            result = generate_risk_score_impl({"amount": 100})
        assert result["status"] == "analysis_failed"
        assert "risk boom" in result["error"]

    def test_generate_risk_score_medium_risk_branch(self):
        """Lines 1294-1295: MEDIUM risk level branch."""
        # Need a transaction that produces a medium-range score (0.4-0.6).
        # We can patch the analyzer to return a controlled score.
        mock_result = {
            "risk_score": 0.45,
            "is_anomaly": False,
            "risk_factors": [],
            "confidence": 0.85,
            "anomaly_score": -0.1,
            "model_scores": {"isolation_forest": 0.45, "ensemble": 0.45},
        }
        with patch.object(
            transaction_analyzer, "analyze_transaction", return_value=mock_result
        ):
            result = generate_risk_score_impl(
                {
                    "amount": 100.0,
                    "merchant": "Shop",
                    "location": "US",
                    "timestamp": datetime.now().isoformat(),
                    "payment_method": "credit_card",
                }
            )
        assert result["risk_level"] == "MEDIUM"
        assert "monitor_closely" in result["recommended_actions"]

    def test_health_check_impl_monitor_exception(self):
        """Lines 1610-1612: System health check raises exception."""
        with patch("server.monitor") as mock_mon:
            mock_mon.health_check.side_effect = RuntimeError("monitor failed")
            mock_mon.__bool__ = lambda self: True  # ensure truthiness
            # The health_check_impl checks "if monitor is not None"
            result = health_check_impl()
        assert "error" in result.get("system", {})

    def test_generate_synthetic_dataset_exception(self):
        """Lines 1806-1808: generate_synthetic_dataset_impl catches exception."""
        with (
            patch("server.SYNTHETIC_DATA_AVAILABLE", True),
            patch("server.synthetic_data_integration") as mock_sdi,
        ):
            mock_sdi.__bool__ = lambda self: True
            mock_sdi.generate_comprehensive_test_dataset.side_effect = RuntimeError(
                "gen failed"
            )
            result = generate_synthetic_dataset_impl()
        assert result["status"] == "generation_failed"
        assert "gen failed" in result["error"]

    def test_analyze_dataset_empty_dataset(self, tmp_path):
        """Line 1861: Dataset is empty."""
        import pandas as pd

        csv_path = tmp_path / "empty.csv"
        # Write a CSV with headers but no rows
        pd.DataFrame(columns=["amount", "merchant"]).to_csv(csv_path, index=False)

        result = analyze_dataset_impl(str(csv_path))
        assert result["status"] == "empty_dataset"

    def test_analyze_dataset_exception(self):
        """Lines 1926-1928: analyze_dataset_impl catches exception."""
        result = analyze_dataset_impl("/nonexistent/path/data.csv")
        assert result["status"] == "file_not_found"

    def test_analyze_dataset_general_exception(self, tmp_path):
        """Lines 1926-1928: analyze_dataset_impl catches a general exception."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("amount,merchant\n100,Shop\n")

        with patch(
            "server.transaction_analyzer.analyze_transaction",
            side_effect=RuntimeError("analysis error"),
        ):
            result = analyze_dataset_impl(str(csv_path))
        assert result["status"] == "analysis_failed"
        assert "analysis error" in result["error"]

    def test_run_benchmark_exception(self):
        """Lines 2122-2124: run_benchmark_impl catches exception."""
        with patch(
            "server.SyntheticDataIntegration",
            side_effect=RuntimeError("bench fail"),
        ):
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
            result = explain_decision_impl(analysis_result, txn_data)
        # Should still return a valid explanation, just without feature_analysis
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


class TestTrainModelsImpl:
    """Cover train_models_impl execution paths."""

    def test_train_models_file_not_found(self):
        """Lines 1644-1648: Data file doesn't exist."""
        with patch("server.TRAINING_AVAILABLE", True):
            result = train_models_impl("/nonexistent/path/data.csv")
        assert result["status"] == "file_not_found"

    def test_train_models_training_exception(self, tmp_path):
        """Lines 1672-1677: Training raises exception."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("amount,is_fraud\n100,0\n200,1\n")

        with (
            patch("server.TRAINING_AVAILABLE", True),
            patch("server.ModelTrainer") as MockTrainer,
        ):
            MockTrainer.return_value.train_all_models.side_effect = RuntimeError(
                "train exploded"
            )
            result = train_models_impl(str(csv_path))
        assert result["status"] == "training_failed"
        assert "train exploded" in result["error"]

    def test_train_models_success_with_hot_reload(self, tmp_path):
        """Lines 1650-1670: Successful training with hot reload."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("amount,is_fraud\n100,0\n200,1\n")

        with (
            patch("server.TRAINING_AVAILABLE", True),
            patch("server.ModelTrainer") as MockTrainer,
            patch.object(transaction_analyzer, "load_models", return_value=True),
        ):
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

        with (
            patch("server.TRAINING_AVAILABLE", True),
            patch("server.ModelTrainer") as MockTrainer,
            patch.object(transaction_analyzer, "load_models", return_value=False),
        ):
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
