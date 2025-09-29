"""
Fraud Detection Monitoring System

Comprehensive observability infrastructure with:
- Prometheus metrics (Counter, Histogram, Gauge)
- Structured logging with structlog
- Performance tracking and decorators
- Health checks and system metrics
- Error tracking and alerting
- Grafana dashboard configuration

Production-ready with proper metric naming conventions and labels.
"""

import time
import functools
import logging
import sys
import psutil
import platform
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
from pathlib import Path

import structlog
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ============================================================================
# Prometheus Metrics Configuration
# ============================================================================

# Custom registry for isolation
REGISTRY = CollectorRegistry()

# Fraud Detection Metrics
fraud_transactions_total = Counter(
    'fraud_transactions_total',
    'Total fraud transactions processed',
    ['risk_level', 'status'],
    registry=REGISTRY
)

fraud_prediction_duration_seconds = Histogram(
    'fraud_prediction_duration_seconds',
    'Time spent processing fraud predictions',
    ['model_type', 'feature_count'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY
)

fraud_risk_score_current = Gauge(
    'fraud_risk_score_current',
    'Current fraud risk score for transaction',
    ['transaction_id', 'risk_category'],
    registry=REGISTRY
)

fraud_model_accuracy = Gauge(
    'fraud_model_accuracy',
    'Model accuracy by type',
    ['model', 'metric_type'],
    registry=REGISTRY
)

# API Metrics
api_requests_total = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code'],
    registry=REGISTRY
)

api_errors_total = Counter(
    'fraud_api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type'],
    registry=REGISTRY
)

api_request_duration_seconds = Histogram(
    'fraud_api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY
)

# System Metrics
system_cpu_usage = Gauge(
    'fraud_system_cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

system_memory_usage = Gauge(
    'fraud_system_memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],
    registry=REGISTRY
)

system_disk_usage = Gauge(
    'fraud_system_disk_usage_percent',
    'Disk usage percentage',
    ['mount_point'],
    registry=REGISTRY
)

# Model Performance Metrics
model_predictions_total = Counter(
    'fraud_model_predictions_total',
    'Total predictions made',
    ['model', 'prediction_class'],
    registry=REGISTRY
)

model_features_processed = Counter(
    'fraud_model_features_processed_total',
    'Total features processed',
    ['feature_type'],
    registry=REGISTRY
)

model_cache_hits = Counter(
    'fraud_model_cache_hits_total',
    'Model cache hits',
    ['cache_type'],
    registry=REGISTRY
)

# Application Info
app_info = Info(
    'fraud_detection_app',
    'Fraud detection application information',
    registry=REGISTRY
)


# ============================================================================
# Structured Logging Configuration
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure structured logging with structlog."""

    # Processors for log formatting
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if sys.stderr.isatty():
        # Pretty colored output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer()
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Optional file logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


# ============================================================================
# Monitoring Manager
# ============================================================================

class MonitoringManager:
    """
    Central monitoring manager for fraud detection system.

    Handles metrics collection, health checks, and system monitoring.
    """

    def __init__(self, app_name: str = "fraud-detection", version: str = "1.0.0"):
        self.app_name = app_name
        self.version = version
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.start_time = time.time()

        # Initialize application info
        app_info.info({
            'app_name': app_name,
            'version': version,
            'python_version': platform.python_version(),
            'platform': platform.system(),
            'platform_release': platform.release(),
        })

        self.logger.info(
            "monitoring_initialized",
            app_name=app_name,
            version=version,
        )

    def record_fraud_transaction(
        self,
        risk_level: str,
        status: str = "processed"
    ) -> None:
        """Record a fraud transaction."""
        fraud_transactions_total.labels(
            risk_level=risk_level,
            status=status
        ).inc()

        self.logger.info(
            "fraud_transaction_recorded",
            risk_level=risk_level,
            status=status,
        )

    def record_prediction(
        self,
        model_type: str,
        feature_count: int,
        duration: float,
        risk_score: float,
        transaction_id: str,
        prediction_class: str = "fraud"
    ) -> None:
        """Record prediction metrics."""
        # Duration histogram
        fraud_prediction_duration_seconds.labels(
            model_type=model_type,
            feature_count=str(feature_count)
        ).observe(duration)

        # Risk score gauge
        risk_category = self._categorize_risk(risk_score)
        fraud_risk_score_current.labels(
            transaction_id=transaction_id,
            risk_category=risk_category
        ).set(risk_score)

        # Prediction counter
        model_predictions_total.labels(
            model=model_type,
            prediction_class=prediction_class
        ).inc()

        self.logger.info(
            "prediction_recorded",
            model_type=model_type,
            feature_count=feature_count,
            duration=duration,
            risk_score=risk_score,
            transaction_id=transaction_id,
        )

    def record_model_accuracy(
        self,
        model: str,
        accuracy: float,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None
    ) -> None:
        """Record model accuracy metrics."""
        fraud_model_accuracy.labels(
            model=model,
            metric_type="accuracy"
        ).set(accuracy)

        if precision is not None:
            fraud_model_accuracy.labels(
                model=model,
                metric_type="precision"
            ).set(precision)

        if recall is not None:
            fraud_model_accuracy.labels(
                model=model,
                metric_type="recall"
            ).set(recall)

        if f1_score is not None:
            fraud_model_accuracy.labels(
                model=model,
                metric_type="f1_score"
            ).set(f1_score)

        self.logger.info(
            "model_accuracy_recorded",
            model=model,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        error_type: Optional[str] = None
    ) -> None:
        """Record API request metrics."""
        api_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()

        api_request_duration_seconds.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)

        if error_type:
            api_errors_total.labels(
                endpoint=endpoint,
                error_type=error_type
            ).inc()

            self.logger.error(
                "api_request_error",
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                error_type=error_type,
                duration=duration,
            )
        else:
            self.logger.info(
                "api_request_completed",
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                duration=duration,
            )

    def record_feature_processing(self, feature_type: str, count: int = 1) -> None:
        """Record feature processing."""
        model_features_processed.labels(
            feature_type=feature_type
        ).inc(count)

    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        model_cache_hits.labels(
            cache_type=cache_type
        ).inc()

    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage.labels(type="used").set(memory.used)
        system_memory_usage.labels(type="available").set(memory.available)
        system_memory_usage.labels(type="total").set(memory.total)

        # Disk usage
        disk = psutil.disk_usage('/')
        system_disk_usage.labels(mount_point="/").set(disk.percent)

        self.logger.debug(
            "system_metrics_updated",
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status dictionary with system metrics
        """
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Application uptime
            uptime_seconds = time.time() - self.start_time

            # Determine overall health
            is_healthy = (
                cpu_percent < 90 and
                memory.percent < 90 and
                disk.percent < 90
            )

            health_status = {
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "app_name": self.app_name,
                "version": self.version,
                "uptime_seconds": uptime_seconds,
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                },
                "checks": {
                    "cpu": "ok" if cpu_percent < 90 else "warning",
                    "memory": "ok" if memory.percent < 90 else "warning",
                    "disk": "ok" if disk.percent < 90 else "warning",
                }
            }

            self.logger.info(
                "health_check_completed",
                status=health_status["status"],
                **health_status["system"]
            )

            return health_status

        except Exception as e:
            self.logger.error(
                "health_check_failed",
                error=str(e),
                exc_info=True
            )
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in exposition format."""
        return generate_latest(REGISTRY)

    @staticmethod
    def _categorize_risk(risk_score: float) -> str:
        """Categorize risk score into levels."""
        if risk_score >= 0.8:
            return "high"
        elif risk_score >= 0.5:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"


# ============================================================================
# Decorators for Automatic Tracking
# ============================================================================

def track_prediction(
    model_type: str,
    feature_count: Optional[int] = None
):
    """
    Decorator to track prediction performance.

    Usage:
        @track_prediction(model_type="random_forest", feature_count=10)
        def predict_fraud(transaction):
            return risk_score, prediction
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = structlog.get_logger(func.__name__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Extract risk score and transaction ID from result
                if isinstance(result, tuple) and len(result) >= 2:
                    risk_score = float(result[0])
                    prediction = result[1]
                    transaction_id = kwargs.get('transaction_id', 'unknown')

                    # Record metrics
                    fraud_prediction_duration_seconds.labels(
                        model_type=model_type,
                        feature_count=str(feature_count or 'unknown')
                    ).observe(duration)

                    logger.info(
                        "prediction_tracked",
                        model_type=model_type,
                        duration=duration,
                        risk_score=risk_score,
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "prediction_failed",
                    model_type=model_type,
                    duration=duration,
                    error=str(e),
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def track_api_call(endpoint: str, method: str = "GET"):
    """
    Decorator to track API calls.

    Usage:
        @track_api_call(endpoint="/predict", method="POST")
        def predict_endpoint(request):
            return response
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = structlog.get_logger(func.__name__)
            start_time = time.time()
            status_code = 200
            error_type = None

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                status_code = 500
                error_type = type(e).__name__
                logger.error(
                    "api_call_failed",
                    endpoint=endpoint,
                    method=method,
                    error=str(e),
                    exc_info=True
                )
                raise

            finally:
                duration = time.time() - start_time

                api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status_code=str(status_code)
                ).inc()

                api_request_duration_seconds.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)

                if error_type:
                    api_errors_total.labels(
                        endpoint=endpoint,
                        error_type=error_type
                    ).inc()

        return wrapper
    return decorator


# ============================================================================
# Setup Function
# ============================================================================

def setup_monitoring(
    app_name: str = "fraud-detection",
    version: str = "1.0.0",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> MonitoringManager:
    """
    Initialize monitoring system.

    Args:
        app_name: Application name
        version: Application version
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        MonitoringManager instance
    """
    # Setup structured logging
    setup_logging(log_level=log_level, log_file=log_file)

    # Create monitoring manager
    manager = MonitoringManager(app_name=app_name, version=version)

    # Initial system metrics update
    manager.update_system_metrics()

    return manager


# ============================================================================
# Grafana Dashboard Configuration
# ============================================================================

GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Fraud Detection Monitoring",
        "tags": ["fraud-detection", "ml", "security"],
        "timezone": "browser",
        "schemaVersion": 16,
        "version": 0,
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "Fraud Transactions by Risk Level",
                "type": "graph",
                "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "rate(fraud_transactions_total[5m])",
                        "legendFormat": "{{risk_level}} - {{status}}",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "short", "label": "Transactions/sec"},
                    {"format": "short"}
                ]
            },
            {
                "id": 2,
                "title": "Prediction Duration (p95)",
                "type": "graph",
                "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(fraud_prediction_duration_seconds_bucket[5m]))",
                        "legendFormat": "{{model_type}} p95",
                        "refId": "A"
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(fraud_prediction_duration_seconds_bucket[5m]))",
                        "legendFormat": "{{model_type}} p99",
                        "refId": "B"
                    }
                ],
                "yaxes": [
                    {"format": "s", "label": "Duration"},
                    {"format": "short"}
                ]
            },
            {
                "id": 3,
                "title": "Model Accuracy",
                "type": "graph",
                "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "fraud_model_accuracy",
                        "legendFormat": "{{model}} - {{metric_type}}",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "percentunit", "label": "Accuracy", "max": 1, "min": 0},
                    {"format": "short"}
                ]
            },
            {
                "id": 4,
                "title": "API Request Rate",
                "type": "graph",
                "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "rate(fraud_api_requests_total[5m])",
                        "legendFormat": "{{endpoint}} - {{method}} ({{status_code}})",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "reqps", "label": "Requests/sec"},
                    {"format": "short"}
                ]
            },
            {
                "id": 5,
                "title": "API Error Rate",
                "type": "graph",
                "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "rate(fraud_api_errors_total[5m])",
                        "legendFormat": "{{endpoint}} - {{error_type}}",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "short", "label": "Errors/sec"},
                    {"format": "short"}
                ],
                "alert": {
                    "conditions": [
                        {
                            "evaluator": {"params": [0.1], "type": "gt"},
                            "operator": {"type": "and"},
                            "query": {"params": ["A", "5m", "now"]},
                            "reducer": {"params": [], "type": "avg"},
                            "type": "query"
                        }
                    ],
                    "executionErrorState": "alerting",
                    "frequency": "60s",
                    "handler": 1,
                    "name": "High API Error Rate",
                    "noDataState": "no_data",
                    "notifications": []
                }
            },
            {
                "id": 6,
                "title": "System CPU Usage",
                "type": "graph",
                "gridPos": {"x": 12, "y": 16, "w": 6, "h": 8},
                "targets": [
                    {
                        "expr": "fraud_system_cpu_usage_percent",
                        "legendFormat": "CPU Usage",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "percent", "label": "Usage", "max": 100, "min": 0},
                    {"format": "short"}
                ],
                "thresholds": [
                    {"value": 80, "colorMode": "critical", "op": "gt", "fill": True, "line": True}
                ]
            },
            {
                "id": 7,
                "title": "System Memory Usage",
                "type": "graph",
                "gridPos": {"x": 18, "y": 16, "w": 6, "h": 8},
                "targets": [
                    {
                        "expr": "fraud_system_memory_usage_bytes{type='used'} / fraud_system_memory_usage_bytes{type='total'} * 100",
                        "legendFormat": "Memory Usage",
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "percent", "label": "Usage", "max": 100, "min": 0},
                    {"format": "short"}
                ],
                "thresholds": [
                    {"value": 80, "colorMode": "critical", "op": "gt", "fill": True, "line": True}
                ]
            },
            {
                "id": 8,
                "title": "Predictions by Model",
                "type": "piechart",
                "gridPos": {"x": 0, "y": 24, "w": 8, "h": 8},
                "targets": [
                    {
                        "expr": "sum by (model) (fraud_model_predictions_total)",
                        "legendFormat": "{{model}}",
                        "refId": "A"
                    }
                ]
            },
            {
                "id": 9,
                "title": "Current Risk Score Distribution",
                "type": "heatmap",
                "gridPos": {"x": 8, "y": 24, "w": 16, "h": 8},
                "targets": [
                    {
                        "expr": "fraud_risk_score_current",
                        "legendFormat": "{{risk_category}}",
                        "refId": "A"
                    }
                ],
                "dataFormat": "timeseries",
                "yAxis": {"format": "short", "decimals": 2}
            }
        ],
        "templating": {
            "list": [
                {
                    "name": "model",
                    "type": "query",
                    "query": "label_values(fraud_model_predictions_total, model)",
                    "multi": True,
                    "includeAll": True
                },
                {
                    "name": "endpoint",
                    "type": "query",
                    "query": "label_values(fraud_api_requests_total, endpoint)",
                    "multi": True,
                    "includeAll": True
                }
            ]
        },
        "annotations": {
            "list": [
                {
                    "name": "Deployments",
                    "datasource": "Prometheus",
                    "enable": True,
                    "expr": "fraud_detection_app_info",
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "tagKeys": "version"
                }
            ]
        }
    }
}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize monitoring
    monitor = setup_monitoring(
        app_name="fraud-detection",
        version="1.0.0",
        log_level="INFO"
    )

    # Example: Track fraud transaction
    monitor.record_fraud_transaction(risk_level="high", status="blocked")
    monitor.record_fraud_transaction(risk_level="low", status="approved")

    # Example: Track prediction
    import random
    transaction_id = f"txn_{random.randint(1000, 9999)}"
    monitor.record_prediction(
        model_type="random_forest",
        feature_count=15,
        duration=0.042,
        risk_score=0.87,
        transaction_id=transaction_id,
        prediction_class="fraud"
    )

    # Example: Track model accuracy
    monitor.record_model_accuracy(
        model="random_forest",
        accuracy=0.95,
        precision=0.92,
        recall=0.88,
        f1_score=0.90
    )

    # Example: Track API request
    monitor.record_api_request(
        endpoint="/api/predict",
        method="POST",
        status_code=200,
        duration=0.156
    )

    # Update system metrics
    monitor.update_system_metrics()

    # Health check
    health = monitor.health_check()
    print("\nHealth Check:", health)

    # Get metrics
    print("\nPrometheus Metrics:")
    print(monitor.get_metrics().decode('utf-8'))

    # Example decorator usage
    @track_prediction(model_type="gradient_boosting", feature_count=20)
    def predict_transaction(transaction_data, transaction_id=None):
        """Example prediction function."""
        import time
        time.sleep(0.01)  # Simulate processing
        risk_score = random.uniform(0, 1)
        prediction = "fraud" if risk_score > 0.5 else "legitimate"
        return risk_score, prediction

    # Test decorator
    result = predict_transaction(
        {"amount": 1000},
        transaction_id="txn_5678"
    )
    print(f"\nPrediction Result: {result}")