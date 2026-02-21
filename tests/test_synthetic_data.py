"""
Tests for Phase 9: Synthetic data generation and dataset analysis integration.

Covers:
- SYNTHETIC_DATA_AVAILABLE flag and SyntheticDataIntegration initialization
- SyntheticDataIntegration class methods (fraud patterns, schemas, data gen)
- generate_synthetic_dataset_impl input validation and end-to-end flow
- analyze_dataset_impl with CSV/JSON, ground truth, and edge cases
- health_check_impl and get_model_status_impl synthetic_data sections
- Graceful degradation when synthetic data module unavailable
- Performance metrics calculation (_calculate_performance_metrics)
- MCP tool registration for new tools
"""

import os

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Availability and initialization
# ---------------------------------------------------------------------------

class TestSyntheticDataAvailability:
    """Test SYNTHETIC_DATA_AVAILABLE flag and integration initialization."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_synthetic_data_available_flag(self):
        from server import SYNTHETIC_DATA_AVAILABLE
        assert isinstance(SYNTHETIC_DATA_AVAILABLE, bool)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_synthetic_data_available_is_true(self):
        from server import SYNTHETIC_DATA_AVAILABLE
        assert SYNTHETIC_DATA_AVAILABLE is True

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_synthetic_data_integration_import(self):
        from integration import SyntheticDataIntegration
        assert SyntheticDataIntegration is not None

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_synthetic_data_integration_singleton(self):
        from server import synthetic_data_integration
        assert synthetic_data_integration is not None

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_synthetic_data_integration_has_output_dir(self):
        from server import synthetic_data_integration
        assert hasattr(synthetic_data_integration, "output_dir")
        assert synthetic_data_integration.output_dir.exists()


# ---------------------------------------------------------------------------
# SyntheticDataIntegration class unit tests
# ---------------------------------------------------------------------------

class TestSyntheticDataIntegrationClass:
    """Unit tests for the SyntheticDataIntegration class."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_init_creates_output_dir(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path / "out")}):
            sdi = SyntheticDataIntegration()
            assert sdi.output_dir.exists()

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraud_patterns_returns_dict(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        assert isinstance(patterns, dict)
        assert "transaction_fraud" in patterns
        assert "behavioral_fraud" in patterns
        assert "network_fraud" in patterns

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_fraud_patterns_transaction_types(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        txn_fraud = patterns["transaction_fraud"]
        expected_types = [
            "high_amount_patterns", "velocity_fraud",
            "geographic_anomaly", "temporal_anomaly",
            "payment_method_fraud",
        ]
        for fraud_type in expected_types:
            assert fraud_type in txn_fraud

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_create_dataset_schema_returns_dict(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        schema = sdi.create_dataset_schema()
        assert isinstance(schema, dict)
        assert "transaction_data" in schema
        assert "behavioral_data" in schema
        assert "network_data" in schema
        assert "labels" in schema

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_schema_required_fields(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        schema = sdi.create_dataset_schema()
        required = schema["transaction_data"]["required_fields"]
        assert "transaction_id" in required
        assert "amount" in required
        assert "timestamp" in required

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_legitimate_transaction(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        txn = sdi._generate_legitimate_transaction(0)
        assert txn["is_fraud"] is False
        assert txn["fraud_type"] == "none"
        assert "transaction_id" in txn
        assert "amount" in txn
        assert txn["amount"] > 0

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraudulent_transaction_high_amount(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        pattern = patterns["transaction_fraud"]["high_amount_patterns"]
        txn = sdi._generate_fraudulent_transaction(0, "high_amount_patterns", pattern)
        assert txn["is_fraud"] is True
        assert txn["fraud_type"] == "high_amount_patterns"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraudulent_transaction_velocity(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        pattern = patterns["transaction_fraud"]["velocity_fraud"]
        txn = sdi._generate_fraudulent_transaction(1, "velocity_fraud", pattern)
        assert txn["is_fraud"] is True
        assert txn["fraud_type"] == "velocity_fraud"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraudulent_transaction_geographic(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        pattern = patterns["transaction_fraud"]["geographic_anomaly"]
        txn = sdi._generate_fraudulent_transaction(2, "geographic_anomaly", pattern)
        assert txn["is_fraud"] is True
        assert txn["location"] in pattern["high_risk_locations"]

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraudulent_transaction_temporal(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        pattern = patterns["transaction_fraud"]["temporal_anomaly"]
        txn = sdi._generate_fraudulent_transaction(3, "temporal_anomaly", pattern)
        assert txn["is_fraud"] is True

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_fraudulent_transaction_payment_method(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        patterns = sdi.generate_fraud_patterns()
        pattern = patterns["transaction_fraud"]["payment_method_fraud"]
        txn = sdi._generate_fraudulent_transaction(4, "payment_method_fraud", pattern)
        assert txn["is_fraud"] is True
        assert txn["payment_method"] in pattern["high_risk_methods"]

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_normal_behavioral_data(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        data = sdi._generate_normal_behavioral_data("user_001")
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(d["is_anomaly"] is False for d in data)
        assert all(d["user_id"] == "user_001" for d in data)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_anomalous_behavioral_data(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        data = sdi._generate_anomalous_behavioral_data("user_002", "high_amount_patterns")
        assert isinstance(data, list)
        assert len(data) > 0
        assert all(d["is_anomaly"] is True for d in data)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_anomalous_behavioral_data_unmatched_type(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        data = sdi._generate_anomalous_behavioral_data("user_003", "geographic_anomaly")
        # geographic_anomaly doesn't match the if-condition, so returns empty
        assert isinstance(data, list)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_network_connections(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        transactions = [
            {"user_id": f"user_{i}", "is_fraud": i < 5, "transaction_id": f"txn_{i}"}
            for i in range(20)
        ]
        patterns = sdi.generate_fraud_patterns()
        network = sdi._generate_network_connections(transactions, patterns)
        assert isinstance(network, list)
        assert len(network) > 0

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validate_schema_compliance(self):
        from integration import SyntheticDataIntegration
        sdi = SyntheticDataIntegration()
        schema = sdi.create_dataset_schema()

        df = pd.DataFrame([{
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "amount": 100.0,
            "merchant": "Test",
            "merchant_category": "retail",
            "location": "US",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card",
            "is_fraud": False,
        }])

        compliance = sdi._validate_schema_compliance(df, schema)
        assert isinstance(compliance, dict)
        assert compliance["fraud_labels_present"] is True
        assert compliance["schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# Comprehensive dataset generation (end-to-end)
# ---------------------------------------------------------------------------

class TestComprehensiveDatasetGeneration:
    """Test generate_comprehensive_test_dataset end-to-end."""

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_small_csv_dataset(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=50,
                fraud_percentage=10.0,
                include_behavioral=False,
                include_network=False,
                output_format="csv",
            )
            assert result["integration_status"] == "success"
            assert result["ready_for_analysis"] is True
            assert result["generation_info"]["total_transactions"] == 50
            assert result["generation_info"]["fraudulent_transactions"] == 5
            assert result["dataset_paths"]["transactions"] is not None
            assert os.path.exists(result["dataset_paths"]["transactions"])

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_json_dataset(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=30,
                fraud_percentage=5.0,
                include_behavioral=False,
                include_network=False,
                output_format="json",
            )
            assert result["integration_status"] == "success"
            txn_path = result["dataset_paths"]["transactions"]
            assert txn_path.endswith(".json")
            assert os.path.exists(txn_path)

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_with_behavioral_data(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=30,
                fraud_percentage=10.0,
                include_behavioral=True,
                include_network=False,
                output_format="csv",
            )
            assert result["integration_status"] == "success"
            assert result["dataset_paths"]["behavioral"] is not None
            assert os.path.exists(result["dataset_paths"]["behavioral"])

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_with_network_data(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=30,
                fraud_percentage=10.0,
                include_behavioral=False,
                include_network=True,
                output_format="csv",
            )
            assert result["integration_status"] == "success"
            assert result["dataset_paths"]["network"] is not None

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_fraud_distribution_in_result(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=100,
                fraud_percentage=20.0,
                include_behavioral=False,
                include_network=False,
                output_format="csv",
            )
            assert "fraud_distribution" in result
            assert isinstance(result["fraud_distribution"], dict)

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_schema_compliance_in_result(self, tmp_path):
        from integration import SyntheticDataIntegration
        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            result = sdi.generate_comprehensive_test_dataset(
                num_transactions=30,
                fraud_percentage=10.0,
                include_behavioral=False,
                include_network=False,
                output_format="csv",
            )
            assert "schema_compliance" in result
            assert result["schema_compliance"]["fraud_labels_present"] is True


# ---------------------------------------------------------------------------
# generate_synthetic_dataset_impl tests
# ---------------------------------------------------------------------------

class TestGenerateSyntheticDatasetImpl:
    """Test generate_synthetic_dataset_impl input validation and flow."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_returns_result_with_success(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(
            num_transactions=20,
            fraud_percentage=10.0,
            include_behavioral=False,
            include_network=False,
            output_format="csv",
        )
        assert result.get("integration_status") == "success" or "error" not in result

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_num_transactions_zero(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(num_transactions=0)
        assert result["status"] == "validation_failed"
        assert "num_transactions" in result["error"]

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_num_transactions_negative(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(num_transactions=-5)
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_num_transactions_too_large(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(num_transactions=2_000_000)
        assert result["status"] == "validation_failed"
        assert "maximum" in result["error"]

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_fraud_percentage_negative(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(
            num_transactions=10, fraud_percentage=-1.0,
        )
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_fraud_percentage_over_100(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(
            num_transactions=10, fraud_percentage=101.0,
        )
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_validation_invalid_output_format(self):
        from server import generate_synthetic_dataset_impl
        result = generate_synthetic_dataset_impl(
            num_transactions=10, output_format="parquet",
        )
        assert result["status"] == "validation_failed"
        assert "output_format" in result["error"]

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_graceful_degradation(self):
        """When synthetic_data_integration is None, returns unavailable."""
        from server import generate_synthetic_dataset_impl
        with patch("server.synthetic_data_integration", None), \
             patch("server.SYNTHETIC_DATA_AVAILABLE", False):
            result = generate_synthetic_dataset_impl(num_transactions=10)
            assert result["status"] == "unavailable"
            assert result["synthetic_data_available"] is False


# ---------------------------------------------------------------------------
# analyze_dataset_impl tests
# ---------------------------------------------------------------------------

class TestAnalyzeDatasetImpl:
    """Test analyze_dataset_impl with various inputs."""

    def _create_csv_dataset(self, tmp_path, rows=20, include_fraud_labels=True):
        """Helper: create a small CSV dataset."""
        data = []
        for i in range(rows):
            txn = {
                "transaction_id": f"txn_{i:04d}",
                "user_id": f"user_{i % 5:03d}",
                "amount": round(50 + i * 10.0, 2),
                "merchant": "TestMerchant",
                "location": "United States",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "credit_card",
            }
            if include_fraud_labels:
                txn["is_fraud"] = i >= (rows - 2)  # last 2 are fraud
            data.append(txn)
        df = pd.DataFrame(data)
        path = tmp_path / "test_dataset.csv"
        df.to_csv(path, index=False)
        return str(path)

    def _create_json_dataset(self, tmp_path, rows=10):
        """Helper: create a small JSON dataset."""
        data = []
        for i in range(rows):
            data.append({
                "transaction_id": f"txn_{i:04d}",
                "user_id": f"user_{i % 3:03d}",
                "amount": round(100 + i * 5.0, 2),
                "merchant": "JsonMerchant",
                "location": "United States",
                "timestamp": datetime.now().isoformat(),
                "payment_method": "debit_card",
                "is_fraud": False,
            })
        path = tmp_path / "test_dataset.json"
        df = pd.DataFrame(data)
        df.to_json(path, orient="records", indent=2)
        return str(path)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_csv_dataset(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_csv_dataset(tmp_path)
        result = analyze_dataset_impl(path)
        assert result["analysis_status"] == "success"
        assert result["dataset_info"]["total_transactions"] == 20
        assert "risk_distribution" in result

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_json_dataset(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_json_dataset(tmp_path)
        result = analyze_dataset_impl(path)
        assert result["analysis_status"] == "success"
        assert result["dataset_info"]["total_transactions"] == 10

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_with_ground_truth(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_csv_dataset(tmp_path, include_fraud_labels=True)
        result = analyze_dataset_impl(path, fraud_threshold=0.3)
        assert result["analysis_status"] == "success"
        # Performance metrics should be present when is_fraud column exists
        assert result["performance_metrics"] is not None
        metrics = result["performance_metrics"]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "accuracy" in metrics

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_without_ground_truth(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_csv_dataset(tmp_path, include_fraud_labels=False)
        result = analyze_dataset_impl(path)
        assert result["analysis_status"] == "success"
        assert result["performance_metrics"] is None

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_empty_path(self):
        from server import analyze_dataset_impl
        result = analyze_dataset_impl("")
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_nonexistent_file(self):
        from server import analyze_dataset_impl
        result = analyze_dataset_impl("/nonexistent/path/data.csv")
        assert result["status"] == "file_not_found"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_unsupported_format(self, tmp_path):
        from server import analyze_dataset_impl
        path = tmp_path / "data.parquet"
        path.write_text("dummy")
        result = analyze_dataset_impl(str(path))
        assert result["status"] == "unsupported_format"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_invalid_threshold_low(self, tmp_path):
        from server import analyze_dataset_impl
        result = analyze_dataset_impl("test.csv", fraud_threshold=-0.1)
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_invalid_threshold_high(self, tmp_path):
        from server import analyze_dataset_impl
        result = analyze_dataset_impl("test.csv", fraud_threshold=1.5)
        assert result["status"] == "validation_failed"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_risk_distribution_keys(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_csv_dataset(tmp_path, rows=10, include_fraud_labels=False)
        result = analyze_dataset_impl(path)
        dist = result["risk_distribution"]
        assert set(dist.keys()) == {"low", "medium", "high", "critical"}

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_flagged_transactions_structure(self, tmp_path):
        from server import analyze_dataset_impl
        path = self._create_csv_dataset(tmp_path, rows=10, include_fraud_labels=False)
        result = analyze_dataset_impl(path, fraud_threshold=0.0)
        # With threshold 0, all should be flagged
        flagged = result["flagged_transactions"]
        assert isinstance(flagged, list)
        for item in flagged:
            assert "transaction_id" in item
            assert "risk_score" in item
            assert "risk_level" in item
            assert isinstance(item["risk_score"], float)


# ---------------------------------------------------------------------------
# _calculate_performance_metrics tests
# ---------------------------------------------------------------------------

class TestCalculatePerformanceMetrics:
    """Test the _calculate_performance_metrics helper."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_perfect_predictions(self):
        from server import _calculate_performance_metrics
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "is_fraud": [True, True, False, False],
        })
        flagged = [
            {"transaction_id": "t1"},
            {"transaction_id": "t2"},
        ]
        metrics = _calculate_performance_metrics(df, flagged)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 0

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_no_flagged_transactions(self):
        from server import _calculate_performance_metrics
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2"],
            "is_fraud": [True, False],
        })
        metrics = _calculate_performance_metrics(df, [])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["true_positives"] == 0
        assert metrics["false_negatives"] == 1

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_all_flagged_transactions(self):
        from server import _calculate_performance_metrics
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2", "t3"],
            "is_fraud": [True, False, False],
        })
        flagged = [
            {"transaction_id": "t1"},
            {"transaction_id": "t2"},
            {"transaction_id": "t3"},
        ]
        metrics = _calculate_performance_metrics(df, flagged)
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 2
        assert metrics["recall"] == 1.0

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_mixed_predictions(self):
        from server import _calculate_performance_metrics
        df = pd.DataFrame({
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "is_fraud": [True, False, True, False],
        })
        flagged = [
            {"transaction_id": "t1"},
            {"transaction_id": "t4"},
        ]
        metrics = _calculate_performance_metrics(df, flagged)
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5


# ---------------------------------------------------------------------------
# Health check and model status synthetic_data sections
# ---------------------------------------------------------------------------

class TestHealthCheckSyntheticData:
    """Test synthetic_data section in health_check_impl."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_health_check_has_synthetic_data_section(self):
        from server import health_check_impl
        result = health_check_impl()
        assert "synthetic_data" in result

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_health_check_synthetic_data_available(self):
        from server import health_check_impl
        result = health_check_impl()
        sd = result["synthetic_data"]
        assert "available" in sd
        assert isinstance(sd["available"], bool)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_health_check_synthetic_data_integration_loaded(self):
        from server import health_check_impl
        result = health_check_impl()
        sd = result["synthetic_data"]
        assert "integration_loaded" in sd
        assert isinstance(sd["integration_loaded"], bool)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_health_check_synthetic_data_output_dir(self):
        from server import health_check_impl
        result = health_check_impl()
        sd = result["synthetic_data"]
        assert "output_dir" in sd
        # If integration is loaded, output_dir should be a string
        if sd["integration_loaded"]:
            assert isinstance(sd["output_dir"], str)

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_health_check_synthetic_data_when_unavailable(self):
        from server import health_check_impl
        with patch("server.SYNTHETIC_DATA_AVAILABLE", False), \
             patch("server.synthetic_data_integration", None):
            result = health_check_impl()
            sd = result["synthetic_data"]
            assert sd["available"] is False
            assert sd["integration_loaded"] is False
            assert sd["output_dir"] is None


class TestModelStatusSyntheticData:
    """Test synthetic_data section in get_model_status_impl."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_model_status_has_synthetic_data_section(self):
        from server import get_model_status_impl
        result = get_model_status_impl()
        assert "synthetic_data" in result

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_model_status_synthetic_data_fields(self):
        from server import get_model_status_impl
        result = get_model_status_impl()
        sd = result["synthetic_data"]
        assert "available" in sd
        assert "integration_loaded" in sd
        assert "output_dir" in sd

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_model_status_synthetic_data_when_unavailable(self):
        from server import get_model_status_impl
        with patch("server.SYNTHETIC_DATA_AVAILABLE", False), \
             patch("server.synthetic_data_integration", None):
            result = get_model_status_impl()
            sd = result["synthetic_data"]
            assert sd["available"] is False
            assert sd["integration_loaded"] is False
            assert sd["output_dir"] is None


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

class TestSyntheticDataMCPRegistration:
    """Test that new MCP tools are registered."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_synthetic_dataset_tool_registered(self):
        from server import mcp
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "generate_synthetic_dataset" in tool_names

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_analyze_dataset_tool_registered(self):
        from server import mcp
        tool_names = list(mcp._tool_manager._tools.keys())
        assert "analyze_dataset" in tool_names

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_total_tools_count_is_12(self):
        from server import mcp
        tool_count = len(mcp._tool_manager._tools)
        assert tool_count == 12


# ---------------------------------------------------------------------------
# End-to-end: generate then analyze
# ---------------------------------------------------------------------------

class TestEndToEndSyntheticFlow:
    """End-to-end test: generate a dataset then analyze it."""

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_then_analyze(self, tmp_path):
        from integration import SyntheticDataIntegration
        from server import analyze_dataset_impl

        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            gen_result = sdi.generate_comprehensive_test_dataset(
                num_transactions=50,
                fraud_percentage=10.0,
                include_behavioral=False,
                include_network=False,
                output_format="csv",
            )
            assert gen_result["integration_status"] == "success"

            txn_path = gen_result["dataset_paths"]["transactions"]
            analysis = analyze_dataset_impl(txn_path, fraud_threshold=0.5)
            assert analysis["analysis_status"] == "success"
            assert analysis["dataset_info"]["total_transactions"] == 50
            assert analysis["performance_metrics"] is not None

    @pytest.mark.synthetic
    @pytest.mark.integration
    def test_generate_then_analyze_json(self, tmp_path):
        from integration import SyntheticDataIntegration
        from server import analyze_dataset_impl

        with patch.dict(os.environ, {"FRAUD_DETECTION_DATA_DIR": str(tmp_path)}):
            sdi = SyntheticDataIntegration()
            gen_result = sdi.generate_comprehensive_test_dataset(
                num_transactions=30,
                fraud_percentage=20.0,
                include_behavioral=False,
                include_network=False,
                output_format="json",
            )
            assert gen_result["integration_status"] == "success"

            txn_path = gen_result["dataset_paths"]["transactions"]
            analysis = analyze_dataset_impl(txn_path, fraud_threshold=0.5)
            assert analysis["analysis_status"] == "success"
            assert analysis["dataset_info"]["total_transactions"] == 30


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Test graceful degradation when integration module is unavailable."""

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_unavailable_returns_error(self):
        from server import generate_synthetic_dataset_impl
        with patch("server.SYNTHETIC_DATA_AVAILABLE", False), \
             patch("server.synthetic_data_integration", None):
            result = generate_synthetic_dataset_impl(num_transactions=10)
            assert "error" in result
            assert result["status"] == "unavailable"

    @pytest.mark.synthetic
    @pytest.mark.unit
    def test_generate_integration_none_returns_error(self):
        from server import generate_synthetic_dataset_impl
        with patch("server.synthetic_data_integration", None):
            result = generate_synthetic_dataset_impl(num_transactions=10)
            assert "error" in result
            assert result["status"] == "unavailable"
