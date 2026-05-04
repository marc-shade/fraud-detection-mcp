"""
Tests for monitoring integration.
"""

import pytest


class TestMonitoringManager:
    """Test MonitoringManager from monitoring.py"""

    def test_manager_initialization(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        assert mgr.app_name == "test-app"
        assert mgr.version == "0.0.1"

    def test_health_check_returns_status(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        health = mgr.health_check()
        assert health["status"] in ("healthy", "degraded", "unhealthy")
        assert "timestamp" in health
        assert "system" in health

    def test_health_check_system_metrics(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        health = mgr.health_check()
        system = health["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "disk_percent" in system

    def test_record_fraud_transaction(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        # Should not raise
        mgr.record_fraud_transaction(risk_level="high", status="blocked")

    def test_record_prediction(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        # Should not raise
        mgr.record_prediction(
            model_type="isolation_forest",
            feature_count=46,
            duration=0.05,
            risk_score=0.75,
            transaction_id="test-txn-001",
        )

    def test_record_cache_hit(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        mgr.record_cache_hit(cache_type="prediction")

    def test_get_metrics_returns_bytes(self):
        from monitoring import MonitoringManager

        mgr = MonitoringManager(app_name="test-app", version="0.0.1")
        metrics = mgr.get_metrics()
        assert isinstance(metrics, bytes)
        assert b"fraud_detection_app_info" in metrics

    def test_categorize_risk(self):
        from monitoring import MonitoringManager

        assert MonitoringManager._categorize_risk(0.9) == "high"
        assert MonitoringManager._categorize_risk(0.6) == "medium"
        assert MonitoringManager._categorize_risk(0.3) == "low"
        assert MonitoringManager._categorize_risk(0.1) == "minimal"


class TestTrackApiCallDecorator:
    """Test track_api_call decorator"""

    def test_decorator_preserves_return_value(self):
        from monitoring import track_api_call

        @track_api_call(endpoint="/test", method="TEST")
        def sample_func():
            return {"result": "ok"}

        assert sample_func() == {"result": "ok"}

    def test_decorator_preserves_function_name(self):
        from monitoring import track_api_call

        @track_api_call(endpoint="/test", method="TEST")
        def sample_func():
            return True

        assert sample_func.__name__ == "sample_func"

    def test_decorator_handles_exception(self):
        from monitoring import track_api_call

        @track_api_call(endpoint="/test", method="TEST")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()


class TestMonitoringIntegration:
    """Test monitoring integration in server.py"""

    def test_monitor_is_initialized(self):
        from server import monitor, MONITORING_AVAILABLE

        if MONITORING_AVAILABLE:
            assert monitor is not None
            assert monitor.app_name == "fraud-detection-mcp"
        else:
            assert monitor is None

    def test_monitoring_flag_set(self):
        from server import MONITORING_AVAILABLE

        assert isinstance(MONITORING_AVAILABLE, bool)


class TestMonitoringActuallyFiresOnMcpToolCall:
    """Pre-fix theater: @_monitored was OUTER and @mcp.tool() was INNER, so
    mcp.tool() registered the bare function; the monitoring wrapper sat
    above it and was never invoked when MCP dispatched a tool call.

    Demonstration: api_requests_total counter stayed at 0 even after
    successful tool invocations.

    Fix: swap decorator order — @mcp.tool() outer, @_monitored inner —
    so MCP registers the monitoring-wrapped callable. This test asserts
    the counter actually moves on a real mcp.call_tool().
    """

    async def _call_and_observe(self, tool_name: str, label_substr: str, payload):
        import asyncio
        from server import mcp, MONITORING_AVAILABLE
        if not MONITORING_AVAILABLE:
            import pytest
            pytest.skip("monitoring not available")
        from monitoring import api_requests_total

        def _count():
            return sum(
                s.value
                for s in list(api_requests_total.collect())[0].samples
                if label_substr in str(s.labels)
            )

        before = _count()
        await mcp.call_tool(tool_name, payload)
        after = _count()
        return before, after

    def test_analyze_transaction_increments_counter(self):
        import asyncio
        before, after = asyncio.run(self._call_and_observe(
            "analyze_transaction",
            "analyze_transaction",
            {"transaction_data": {
                "amount": 50.0, "merchant": "M", "location": "L",
                "timestamp": "2026-05-04T12:00:00Z",
                "payment_method": "credit_card",
            }},
        ))
        assert after > before, (
            f"api_requests_total did NOT move on a real MCP call (before={before}, "
            f"after={after}). The @_monitored decorator is not wired into the MCP "
            f"dispatch path — likely the @mcp.tool()/@_monitored decorator order "
            f"got swapped back."
        )


class TestCacheMissesAreRecorded:
    """Pre-2026-05-04 only model_cache_hits was exported to Prometheus —
    misses were tracked only in the in-process _inference_stats counter.
    This made the cache hit RATE uncomputable from external metrics.
    Now record_cache_miss exports model_cache_misses_total."""

    def test_cache_miss_increments_prometheus_counter(self):
        from server import MONITORING_AVAILABLE
        if not MONITORING_AVAILABLE:
            import pytest
            pytest.skip("monitoring not available")
        from monitoring import model_cache_misses, MonitoringManager

        # Use a fresh monitor to avoid cross-test contamination
        mgr = MonitoringManager(app_name="test", version="0.0.0")

        def _count():
            # ``Counter.collect()`` yields one Metric whose ``samples``
            # include both the ``<name>_total`` (actual count) and a
            # ``<name>_created`` (Unix timestamp) per labelset. Filter
            # on name suffix; pre-fix the test summed both and got a
            # 1.7B number from the timestamp.
            total = 0.0
            for metric in model_cache_misses.collect():
                for s in metric.samples:
                    if (
                        s.name.endswith("_total")
                        and s.labels.get("cache_type") == "prediction"
                    ):
                        total += s.value
            return total

        before = _count()
        mgr.record_cache_miss(cache_type="prediction")
        mgr.record_cache_miss(cache_type="prediction")
        after = _count()
        assert after == before + 2, f"Expected +2, got {after - before}"
