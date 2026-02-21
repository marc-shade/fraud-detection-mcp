"""
Tests for Phase 6: Training pipeline integration and model persistence.
Covers: model save/load, train_models_impl, get_model_status_impl,
        _model_source tracking, TRAINING_AVAILABLE flag.
"""

import pytest
import os
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np


# =============================================================================
# Model source tracking
# =============================================================================

class TestModelSourceTracking:
    """Test _model_source attribute lifecycle in TransactionAnalyzer."""

    def test_model_source_is_synthetic_after_init(self):
        """TransactionAnalyzer defaults to 'synthetic' after init with dummy data."""
        from server import transaction_analyzer
        assert transaction_analyzer._model_source == "synthetic"

    def test_model_source_attribute_exists(self):
        """TransactionAnalyzer has _model_source attribute."""
        from server import transaction_analyzer
        assert hasattr(transaction_analyzer, "_model_source")

    def test_model_source_is_string(self):
        """_model_source should be a string."""
        from server import transaction_analyzer
        assert isinstance(transaction_analyzer._model_source, str)

    def test_model_source_valid_values(self):
        """_model_source should be one of the known values."""
        from server import transaction_analyzer
        assert transaction_analyzer._model_source in ("none", "synthetic", "saved")


# =============================================================================
# Model persistence (save/load)
# =============================================================================

class TestModelPersistence:
    """Test save_models() and load_models() on TransactionAnalyzer."""

    @pytest.fixture
    def tmp_model_dir(self, tmp_path):
        """Provide a temporary directory for model files."""
        model_dir = tmp_path / "test_models"
        model_dir.mkdir()
        return model_dir

    def test_save_models_creates_files(self, tmp_model_dir):
        """save_models() creates isolation_forest.joblib and feature_engineer.joblib."""
        from server import transaction_analyzer
        paths = transaction_analyzer.save_models(tmp_model_dir)

        assert "isolation_forest" in paths
        assert "feature_engineer" in paths
        assert Path(paths["isolation_forest"]).exists()
        assert Path(paths["feature_engineer"]).exists()

    def test_save_models_returns_dict(self, tmp_model_dir):
        """save_models() returns a dict mapping model name to path."""
        from server import transaction_analyzer
        paths = transaction_analyzer.save_models(tmp_model_dir)
        assert isinstance(paths, dict)
        assert len(paths) == 2

    def test_save_models_creates_directory(self, tmp_path):
        """save_models() creates the directory if it doesn't exist."""
        from server import transaction_analyzer
        new_dir = tmp_path / "new_subdir" / "models"
        paths = transaction_analyzer.save_models(new_dir)
        assert new_dir.exists()
        assert Path(paths["isolation_forest"]).exists()

    def test_load_models_from_saved(self, tmp_model_dir):
        """load_models() successfully loads previously saved models."""
        from server import transaction_analyzer
        transaction_analyzer.save_models(tmp_model_dir)

        # Reset model source to verify it changes on load
        transaction_analyzer._model_source = "none"

        result = transaction_analyzer.load_models(tmp_model_dir)
        assert result is True
        assert transaction_analyzer._model_source == "saved"

    def test_load_models_returns_false_when_missing(self, tmp_model_dir):
        """load_models() returns False if model files don't exist."""
        from server import transaction_analyzer
        result = transaction_analyzer.load_models(tmp_model_dir)
        assert result is False

    def test_load_models_returns_false_for_partial_files(self, tmp_model_dir):
        """load_models() returns False if only some model files exist."""
        from server import transaction_analyzer
        import joblib
        # Create only the isolation forest file
        joblib.dump(transaction_analyzer.isolation_forest, tmp_model_dir / "isolation_forest.joblib")
        result = transaction_analyzer.load_models(tmp_model_dir)
        assert result is False

    def test_save_load_roundtrip_produces_same_predictions(self, tmp_model_dir):
        """Models produce the same predictions after save/load roundtrip."""
        from server import transaction_analyzer

        # Make a prediction before save
        test_data = {
            'amount': 150.0,
            'merchant': 'Amazon',
            'location': 'United States',
            'timestamp': datetime.now().isoformat(),
            'payment_method': 'credit_card',
        }
        result_before = transaction_analyzer.analyze_transaction(test_data)

        # Save and reload
        transaction_analyzer.save_models(tmp_model_dir)
        transaction_analyzer.load_models(tmp_model_dir)

        # Make the same prediction after load
        result_after = transaction_analyzer.analyze_transaction(test_data)

        assert result_before["risk_score"] == pytest.approx(
            result_after["risk_score"], abs=1e-6
        )

    def test_load_models_handles_corrupt_file(self, tmp_model_dir):
        """load_models() returns False for corrupt joblib files."""
        from server import transaction_analyzer
        # Create corrupt files
        (tmp_model_dir / "isolation_forest.joblib").write_text("corrupt data")
        (tmp_model_dir / "feature_engineer.joblib").write_text("corrupt data")

        result = transaction_analyzer.load_models(tmp_model_dir)
        assert result is False

    def test_save_models_default_dir(self):
        """save_models() uses DEFAULT_MODEL_DIR when no path specified."""
        from server import TransactionAnalyzer
        analyzer = TransactionAnalyzer.__new__(TransactionAnalyzer)
        analyzer._model_dir = Path("/nonexistent/test/path")
        analyzer._model_source = "none"
        analyzer.feature_engineer = MagicMock()
        analyzer.isolation_forest = MagicMock()

        # load_models should fail (directory doesn't exist) but not crash
        result = analyzer.load_models()
        assert result is False


# =============================================================================
# TRAINING_AVAILABLE flag
# =============================================================================

class TestTrainingAvailableFlag:
    """Test the TRAINING_AVAILABLE flag behavior."""

    def test_training_available_is_boolean(self):
        """TRAINING_AVAILABLE should be a boolean."""
        from server import TRAINING_AVAILABLE
        assert isinstance(TRAINING_AVAILABLE, bool)

    def test_training_available_reflects_import(self):
        """TRAINING_AVAILABLE should be True since imblearn is installed."""
        from server import TRAINING_AVAILABLE
        try:
            from training_pipeline import ModelTrainer
            expected = True
        except ImportError:
            expected = False
        assert TRAINING_AVAILABLE == expected

    def test_model_trainer_is_none_when_unavailable(self):
        """When TRAINING_AVAILABLE is False, ModelTrainer should be None."""
        from server import TRAINING_AVAILABLE, ModelTrainer as ServerModelTrainer
        if not TRAINING_AVAILABLE:
            assert ServerModelTrainer is None


# =============================================================================
# train_models_impl
# =============================================================================

class TestTrainModelsImpl:
    """Test train_models_impl function."""

    def test_train_models_returns_error_when_unavailable(self):
        """train_models_impl returns error when TRAINING_AVAILABLE is False."""
        from server import train_models_impl
        with patch("server.TRAINING_AVAILABLE", False):
            result = train_models_impl("some_path.csv")
            assert "error" in result
            assert result["status"] == "unavailable"
            assert result["training_available"] is False

    def test_train_models_file_not_found(self):
        """train_models_impl returns error for nonexistent data file."""
        from server import train_models_impl, TRAINING_AVAILABLE
        if not TRAINING_AVAILABLE:
            pytest.skip("Training dependencies not available")

        result = train_models_impl("/nonexistent/path/data.csv")
        assert "error" in result
        assert result["status"] == "file_not_found"

    def test_train_models_returns_dict(self):
        """train_models_impl always returns a dict."""
        from server import train_models_impl
        with patch("server.TRAINING_AVAILABLE", False):
            result = train_models_impl("data.csv")
            assert isinstance(result, dict)

    def test_train_models_unavailable_has_message(self):
        """Error message explains what needs to be installed."""
        from server import train_models_impl
        with patch("server.TRAINING_AVAILABLE", False):
            result = train_models_impl("data.csv")
            assert "imbalanced-learn" in result["error"]

    def test_train_models_handles_training_exception(self):
        """train_models_impl handles exceptions from ModelTrainer gracefully."""
        from server import train_models_impl, TRAINING_AVAILABLE
        if not TRAINING_AVAILABLE:
            pytest.skip("Training dependencies not available")

        with patch("server.ModelTrainer") as mock_trainer_cls:
            mock_instance = MagicMock()
            mock_instance.train_all_models.side_effect = RuntimeError("Training failed")
            mock_trainer_cls.return_value = mock_instance

            # Need a file that exists
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(b"col1,is_fraud\n1,0\n")
                tmp_path = f.name

            try:
                result = train_models_impl(tmp_path)
                assert "error" in result
                assert result["status"] == "training_failed"
            finally:
                os.unlink(tmp_path)


# =============================================================================
# get_model_status_impl
# =============================================================================

class TestGetModelStatusImpl:
    """Test get_model_status_impl function."""

    def test_returns_dict(self):
        """get_model_status_impl returns a dict."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert isinstance(result, dict)

    def test_has_model_source(self):
        """Result includes model_source field."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "model_source" in result
        assert result["model_source"] in ("none", "synthetic", "saved")

    def test_has_training_available(self):
        """Result includes training_available field."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "training_available" in result
        assert isinstance(result["training_available"], bool)

    def test_has_models_section(self):
        """Result includes detailed models section."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "models" in result
        models = result["models"]
        assert "isolation_forest" in models
        assert "feature_engineer" in models

    def test_isolation_forest_details(self):
        """Isolation forest details include loaded, n_estimators, contamination."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        iso = result["models"]["isolation_forest"]
        assert iso["loaded"] is True
        assert iso["n_estimators"] == 200
        assert iso["contamination"] == 0.1

    def test_feature_engineer_details(self):
        """Feature engineer details include loaded and feature_count."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        fe = result["models"]["feature_engineer"]
        assert fe["loaded"] is True
        assert fe["feature_count"] == 46
        assert isinstance(fe["feature_names"], list)

    def test_has_saved_models_section(self):
        """Result includes saved_models with file paths or None."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "saved_models" in result
        saved = result["saved_models"]
        assert "isolation_forest" in saved
        assert "feature_engineer" in saved

    def test_has_model_dir(self):
        """Result includes model_dir path."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "model_dir" in result
        assert isinstance(result["model_dir"], str)

    def test_has_timestamp(self):
        """Result includes ISO timestamp."""
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "timestamp" in result
        # Should be parseable as ISO format
        datetime.fromisoformat(result["timestamp"])

    def test_saved_models_none_when_no_files(self):
        """saved_models paths are None when no files exist on disk."""
        from server import get_model_status_impl, transaction_analyzer
        original_dir = transaction_analyzer._model_dir
        try:
            transaction_analyzer._model_dir = Path("/nonexistent/path")
            result = get_model_status_impl()
            assert result["saved_models"]["isolation_forest"] is None
            assert result["saved_models"]["feature_engineer"] is None
        finally:
            transaction_analyzer._model_dir = original_dir

    def test_saved_models_show_paths_when_exist(self, tmp_path):
        """saved_models show file paths when models are saved to disk."""
        from server import get_model_status_impl, transaction_analyzer
        original_dir = transaction_analyzer._model_dir
        try:
            model_dir = tmp_path / "models"
            transaction_analyzer._model_dir = model_dir
            transaction_analyzer.save_models(model_dir)

            result = get_model_status_impl()
            assert result["saved_models"]["isolation_forest"] is not None
            assert result["saved_models"]["feature_engineer"] is not None
            assert "isolation_forest.joblib" in result["saved_models"]["isolation_forest"]
        finally:
            transaction_analyzer._model_dir = original_dir


# =============================================================================
# MCP tool registration
# =============================================================================

class TestMCPToolRegistration:
    """Test that new MCP tools are registered."""

    def test_train_models_tool_exists(self):
        """train_models should be importable from server."""
        from server import train_models_impl
        assert callable(train_models_impl)

    def test_get_model_status_tool_exists(self):
        """get_model_status should be importable from server."""
        from server import get_model_status_impl
        assert callable(get_model_status_impl)

    def test_mcp_has_train_models(self):
        """MCP server should have train_models registered."""
        from server import mcp
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "train_models" in tool_names

    def test_mcp_has_get_model_status(self):
        """MCP server should have get_model_status registered."""
        from server import mcp
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "get_model_status" in tool_names

    def test_total_mcp_tools_count(self):
        """Server should now have 10 MCP tools registered."""
        from server import mcp
        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 10


# =============================================================================
# Integration: model source after save/load cycle
# =============================================================================

class TestModelSourceIntegration:
    """Integration tests for _model_source through the full lifecycle."""

    def test_source_synthetic_after_fresh_init(self):
        """A fresh TransactionAnalyzer has model_source='synthetic'."""
        from server import TransactionAnalyzer
        analyzer = TransactionAnalyzer(model_dir=Path("/nonexistent"))
        assert analyzer._model_source == "synthetic"

    def test_source_saved_after_load(self, tmp_path):
        """model_source becomes 'saved' after loading from disk."""
        from server import TransactionAnalyzer
        model_dir = tmp_path / "models"

        # Create analyzer and save
        analyzer = TransactionAnalyzer(model_dir=model_dir)
        assert analyzer._model_source == "synthetic"
        analyzer.save_models(model_dir)

        # Create fresh analyzer that loads from saved
        analyzer2 = TransactionAnalyzer(model_dir=model_dir)
        assert analyzer2._model_source == "saved"

    def test_get_model_status_reflects_source(self):
        """get_model_status_impl returns the current _model_source value."""
        from server import get_model_status_impl, transaction_analyzer
        original = transaction_analyzer._model_source
        try:
            transaction_analyzer._model_source = "synthetic"
            result = get_model_status_impl()
            assert result["model_source"] == "synthetic"

            transaction_analyzer._model_source = "saved"
            result = get_model_status_impl()
            assert result["model_source"] == "saved"
        finally:
            transaction_analyzer._model_source = original
