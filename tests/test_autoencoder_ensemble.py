"""
Tests for autoencoder ensemble integration in TransactionAnalyzer.
"""

import pytest
import numpy as np


class TestAutoencoderAvailability:
    """Test autoencoder import and availability"""

    def test_autoencoder_available_flag(self):
        from server import AUTOENCODER_AVAILABLE
        assert isinstance(AUTOENCODER_AVAILABLE, bool)

    def test_autoencoder_import(self):
        from server import AUTOENCODER_AVAILABLE
        if AUTOENCODER_AVAILABLE:
            from models.autoencoder import AutoencoderFraudDetector
            assert AutoencoderFraudDetector is not None

    def test_autoencoder_initialized_on_analyzer(self):
        from server import transaction_analyzer, AUTOENCODER_AVAILABLE
        if AUTOENCODER_AVAILABLE:
            assert transaction_analyzer.autoencoder is not None
        else:
            assert transaction_analyzer.autoencoder is None


class TestEnsembleWeights:
    """Test ensemble weight configuration"""

    def test_default_weights(self):
        from server import transaction_analyzer
        weights = transaction_analyzer._ensemble_weights
        assert "isolation_forest" in weights
        assert "autoencoder" in weights
        assert weights["isolation_forest"] == 0.6
        assert weights["autoencoder"] == 0.4

    def test_weights_sum_to_one(self):
        from server import transaction_analyzer
        weights = transaction_analyzer._ensemble_weights
        total = weights["isolation_forest"] + weights["autoencoder"]
        assert abs(total - 1.0) < 1e-9


class TestEnsembleScoring:
    """Test ensemble scoring in analyze_transaction"""

    def test_model_scores_in_result(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 100})
        assert "model_scores" in result
        scores = result["model_scores"]
        assert "isolation_forest" in scores
        assert "ensemble" in scores

    def test_model_scores_autoencoder_present_when_available(self):
        from server import transaction_analyzer, AUTOENCODER_AVAILABLE
        result = transaction_analyzer.analyze_transaction({"amount": 100})
        scores = result["model_scores"]
        if AUTOENCODER_AVAILABLE and transaction_analyzer.autoencoder is not None:
            assert "autoencoder" in scores
        else:
            assert "autoencoder" not in scores

    def test_risk_score_within_bounds(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 100})
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_model_scores_within_bounds(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 500})
        for name, score in result["model_scores"].items():
            assert 0.0 <= score <= 1.0, f"{name} score {score} out of bounds"

    def test_high_risk_transaction_ensemble(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({
            "amount": 15000,
            "payment_method": "crypto",
            "location": "unknown",
        })
        assert result["risk_score"] > 0.0
        assert len(result["risk_factors"]) > 0

    def test_ensemble_still_returns_standard_keys(self):
        """Ensure backward compatibility -- all original keys still present."""
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 200})
        for key in ["risk_score", "is_anomaly", "risk_factors", "confidence",
                     "analysis_type", "anomaly_score"]:
            assert key in result, f"Missing key: {key}"

    def test_analysis_type_unchanged(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 100})
        assert result["analysis_type"] == "transaction_pattern"

    def test_confidence_unchanged(self):
        from server import transaction_analyzer
        result = transaction_analyzer.analyze_transaction({"amount": 100})
        assert result["confidence"] == 0.88


class TestAutoencoderPersistence:
    """Test autoencoder save/load with TransactionAnalyzer"""

    def test_save_includes_autoencoder(self, tmp_path):
        from server import transaction_analyzer, AUTOENCODER_AVAILABLE
        paths = transaction_analyzer.save_models(model_dir=tmp_path)
        assert "isolation_forest" in paths
        if AUTOENCODER_AVAILABLE and transaction_analyzer.autoencoder is not None:
            assert "autoencoder" in paths

    def test_load_restores_autoencoder(self, tmp_path):
        from server import TransactionAnalyzer, AUTOENCODER_AVAILABLE
        # Create and save an analyzer
        analyzer1 = TransactionAnalyzer()
        analyzer1.save_models(model_dir=tmp_path)

        # Create a new analyzer that loads from disk
        analyzer2 = TransactionAnalyzer(model_dir=tmp_path)
        assert analyzer2._model_source == "saved"
        if AUTOENCODER_AVAILABLE:
            assert analyzer2.autoencoder is not None


class TestModelStatusWithAutoencoder:
    """Test model status includes autoencoder info"""

    def test_model_status_has_autoencoder(self):
        from server import get_model_status_impl
        status = get_model_status_impl()
        assert "autoencoder" in status["models"]

    def test_model_status_has_ensemble_weights(self):
        from server import get_model_status_impl
        status = get_model_status_impl()
        assert "ensemble_weights" in status

    def test_health_check_has_autoencoder(self):
        from server import health_check_impl
        health = health_check_impl()
        assert "autoencoder" in health["models"]


class TestAutoencoderDirect:
    """Test AutoencoderFraudDetector directly"""

    def test_create_and_fit(self):
        from server import AUTOENCODER_AVAILABLE
        if not AUTOENCODER_AVAILABLE:
            pytest.skip("Autoencoder not available")
        from models.autoencoder import AutoencoderFraudDetector
        ae = AutoencoderFraudDetector(contamination=0.1, epochs=5, batch_size=16)
        X = np.random.randn(50, 10).astype(np.float32)
        ae.fit(X)
        assert ae.threshold is not None

    def test_predict_returns_binary(self):
        from server import AUTOENCODER_AVAILABLE
        if not AUTOENCODER_AVAILABLE:
            pytest.skip("Autoencoder not available")
        from models.autoencoder import AutoencoderFraudDetector
        ae = AutoencoderFraudDetector(contamination=0.1, epochs=5, batch_size=16)
        X = np.random.randn(50, 10).astype(np.float32)
        ae.fit(X)
        preds = ae.predict(X[:5])
        assert all(p in (0, 1) for p in preds)

    def test_decision_function_returns_floats(self):
        from server import AUTOENCODER_AVAILABLE
        if not AUTOENCODER_AVAILABLE:
            pytest.skip("Autoencoder not available")
        from models.autoencoder import AutoencoderFraudDetector
        ae = AutoencoderFraudDetector(contamination=0.1, epochs=5, batch_size=16)
        X = np.random.randn(50, 10).astype(np.float32)
        ae.fit(X)
        scores = ae.decision_function(X[:5])
        assert len(scores) == 5
        assert all(isinstance(float(s), float) for s in scores)

    def test_save_and_load_roundtrip(self, tmp_path):
        from server import AUTOENCODER_AVAILABLE
        if not AUTOENCODER_AVAILABLE:
            pytest.skip("Autoencoder not available")
        from models.autoencoder import AutoencoderFraudDetector
        ae1 = AutoencoderFraudDetector(contamination=0.1, epochs=5, batch_size=16)
        X = np.random.randn(50, 10).astype(np.float32)
        ae1.fit(X)
        scores1 = ae1.decision_function(X[:5])

        path = str(tmp_path / "ae_test.pt")
        ae1.save(path)

        ae2 = AutoencoderFraudDetector(contamination=0.1)
        ae2.load(path)
        scores2 = ae2.decision_function(X[:5])

        np.testing.assert_allclose(scores1, scores2, rtol=1e-4)
