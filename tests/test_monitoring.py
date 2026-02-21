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
