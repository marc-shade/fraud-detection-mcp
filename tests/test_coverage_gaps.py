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
            {"key": "a", "release_time": 100},   # no press_time -> no dwell
            {"key": "b", "press_time": 150},      # flight = 150 - 100 = 50; no release_time -> no dwell
            {"key": "c", "press_time": 250},      # prev has no release_time -> no flight; no release_time -> no dwell
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
