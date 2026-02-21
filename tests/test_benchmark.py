"""Tests for Phase 10: Run Benchmark MCP Tool"""

import pytest


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestBenchmarkValidation:
    """Test input validation for run_benchmark_impl."""

    def test_too_few_transactions(self):
        import server
        result = server.run_benchmark_impl(num_transactions=5)
        assert "error" in result
        assert result["status"] == "validation_failed"

    def test_too_many_transactions(self):
        import server
        result = server.run_benchmark_impl(num_transactions=10000)
        assert "error" in result

    def test_negative_fraud_percentage(self):
        import server
        result = server.run_benchmark_impl(fraud_percentage=-5)
        assert "error" in result

    def test_over_100_fraud_percentage(self):
        import server
        result = server.run_benchmark_impl(fraud_percentage=150)
        assert "error" in result

    def test_valid_min_transactions(self):
        import server
        if not server.SYNTHETIC_DATA_AVAILABLE:
            pytest.skip("Synthetic data not available")
        result = server.run_benchmark_impl(num_transactions=10)
        assert result.get("status") == "success"


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------
class TestBenchmarkExecution:
    """Test benchmark execution and output structure."""

    @pytest.fixture
    def benchmark_result(self):
        import server
        if not server.SYNTHETIC_DATA_AVAILABLE:
            pytest.skip("Synthetic data not available")
        return server.run_benchmark_impl(
            num_transactions=30, fraud_percentage=20.0
        )

    def test_status_success(self, benchmark_result):
        assert benchmark_result["status"] == "success"

    def test_has_benchmark_config(self, benchmark_result):
        config = benchmark_result["benchmark_config"]
        assert config["num_transactions"] == 30
        assert config["fraud_percentage"] == 20.0
        assert config["actual_fraud_count"] == 6
        assert config["actual_legit_count"] == 24

    def test_has_throughput(self, benchmark_result):
        tp = benchmark_result["throughput"]
        assert "transactions_per_second" in tp
        assert tp["transactions_per_second"] > 0
        assert "total_time_ms" in tp
        assert tp["total_time_ms"] > 0

    def test_has_latency(self, benchmark_result):
        lat = benchmark_result["latency"]
        assert "avg_ms" in lat
        assert "min_ms" in lat
        assert "max_ms" in lat
        assert lat["avg_ms"] >= lat["min_ms"]
        assert lat["max_ms"] >= lat["avg_ms"]

    def test_has_latency_percentiles(self, benchmark_result):
        lat = benchmark_result["latency"]
        assert "p50_ms" in lat
        assert "p95_ms" in lat
        assert "p99_ms" in lat
        assert lat["p50_ms"] <= lat["p95_ms"]
        assert lat["p95_ms"] <= lat["p99_ms"]

    def test_without_percentiles(self):
        import server
        if not server.SYNTHETIC_DATA_AVAILABLE:
            pytest.skip("Synthetic data not available")
        result = server.run_benchmark_impl(
            num_transactions=10, include_latency_percentiles=False
        )
        assert result["status"] == "success"
        assert "p50_ms" not in result["latency"]
        assert "p95_ms" not in result["latency"]
        assert "avg_ms" in result["latency"]

    def test_has_accuracy(self, benchmark_result):
        acc = benchmark_result["accuracy"]
        assert "precision" in acc
        assert "recall" in acc
        assert "f1_score" in acc
        assert "accuracy" in acc
        assert 0 <= acc["precision"] <= 1
        assert 0 <= acc["recall"] <= 1
        assert 0 <= acc["f1_score"] <= 1

    def test_has_confusion_matrix(self, benchmark_result):
        acc = benchmark_result["accuracy"]
        assert "true_positives" in acc
        assert "false_positives" in acc
        assert "true_negatives" in acc
        assert "false_negatives" in acc
        total = acc["true_positives"] + acc["false_positives"] + \
                acc["true_negatives"] + acc["false_negatives"]
        assert total == 30

    def test_has_risk_distribution(self, benchmark_result):
        dist = benchmark_result["risk_distribution"]
        assert "LOW" in dist
        assert "MEDIUM" in dist
        assert "HIGH" in dist
        assert "CRITICAL" in dist
        total = sum(dist.values())
        assert total == 30

    def test_has_pipeline_config(self, benchmark_result):
        pipe = benchmark_result["pipeline"]
        assert "isolation_forest" in pipe
        assert "autoencoder" in pipe
        assert "explainer" in pipe
        assert "model_source" in pipe
        assert "ensemble_weights" in pipe

    def test_has_timestamp(self, benchmark_result):
        assert "timestamp" in benchmark_result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestBenchmarkEdgeCases:
    """Test edge cases."""

    def test_zero_fraud(self):
        import server
        if not server.SYNTHETIC_DATA_AVAILABLE:
            pytest.skip("Synthetic data not available")
        result = server.run_benchmark_impl(
            num_transactions=20, fraud_percentage=0.0
        )
        assert result["status"] == "success"
        assert result["benchmark_config"]["actual_fraud_count"] == 0

    def test_all_fraud(self):
        import server
        if not server.SYNTHETIC_DATA_AVAILABLE:
            pytest.skip("Synthetic data not available")
        result = server.run_benchmark_impl(
            num_transactions=20, fraud_percentage=100.0
        )
        assert result["status"] == "success"
        assert result["benchmark_config"]["actual_legit_count"] == 0

    def test_unavailable_when_no_integration(self):
        import server
        original = server.SYNTHETIC_DATA_AVAILABLE
        original_integration = server.synthetic_data_integration
        try:
            server.SYNTHETIC_DATA_AVAILABLE = False
            server.synthetic_data_integration = None
            result = server.run_benchmark_impl()
            assert "error" in result
            assert result["status"] == "unavailable"
        finally:
            server.SYNTHETIC_DATA_AVAILABLE = original
            server.synthetic_data_integration = original_integration


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------
class TestBenchmarkMCPRegistration:
    """Verify MCP tool is registered."""

    def test_run_benchmark_registered(self):
        import server
        tools = list(server.mcp._tool_manager._tools.keys())
        assert "run_benchmark" in tools

    def test_total_tool_count(self):
        import server
        tools = list(server.mcp._tool_manager._tools.keys())
        assert len(tools) >= 13
