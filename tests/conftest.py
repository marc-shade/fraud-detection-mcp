"""
Pytest fixtures and configuration for fraud-detection-mcp tests
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, MagicMock


@pytest.fixture
def sample_keystroke_data():
    """Generate sample keystroke dynamics data"""
    data = []
    base_time = 1000.0

    for i in range(20):
        keystroke = {
            'key': chr(97 + i % 26),  # a-z
            'press_time': base_time + i * 150,
            'release_time': base_time + i * 150 + 80,  # 80ms dwell time
        }
        data.append(keystroke)
        base_time = keystroke['release_time']

    return data


@pytest.fixture
def anomalous_keystroke_data():
    """Generate anomalous keystroke dynamics data"""
    data = []
    base_time = 1000.0

    for i in range(20):
        # Irregular timing patterns
        dwell = 200 if i % 3 == 0 else 50  # Inconsistent dwell times
        flight = 500 if i % 5 == 0 else 100  # Irregular flight times

        keystroke = {
            'key': chr(97 + i % 26),
            'press_time': base_time,
            'release_time': base_time + dwell,
        }
        data.append(keystroke)
        base_time = keystroke['release_time'] + flight

    return data


@pytest.fixture
def sample_mouse_movements():
    """Generate sample mouse movement data"""
    movements = []
    x, y = 100, 100

    for i in range(100):
        x += np.random.randint(-10, 10)
        y += np.random.randint(-10, 10)
        movements.append({
            'x': x,
            'y': y,
            'timestamp': 1000 + i * 50,
            'event_type': 'move'
        })

    return movements


@pytest.fixture
def sample_transaction_data():
    """Generate sample transaction data"""
    return {
        'amount': 150.00,
        'merchant': 'Amazon',
        'location': 'United States',
        'timestamp': datetime.now().isoformat(),
        'payment_method': 'credit_card',
        'user_id': 'user123',
        'merchant_category': 'retail',
        'currency': 'USD'
    }


@pytest.fixture
def high_risk_transaction():
    """Generate high-risk transaction data"""
    return {
        'amount': 25000.00,
        'merchant': 'Unknown Merchant',
        'location': 'Nigeria',
        'timestamp': (datetime.now().replace(hour=3)).isoformat(),  # 3 AM
        'payment_method': 'crypto',
        'user_id': 'user456',
        'merchant_category': 'unknown',
        'currency': 'BTC'
    }


@pytest.fixture
def sample_behavioral_data(sample_keystroke_data, sample_mouse_movements):
    """Generate complete behavioral biometrics data"""
    return {
        'keystroke_dynamics': sample_keystroke_data,
        'mouse_movements': sample_mouse_movements,
        'user_id': 'user123',
        'session_id': 'session_abc',
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_network_data():
    """Generate sample network/graph data for fraud ring detection"""
    return {
        'entity_id': 'user123',
        'connections': [
            {'entity_id': 'user456', 'strength': 0.8, 'transaction_count': 5},
            {'entity_id': 'user789', 'strength': 0.6, 'transaction_count': 3},
            {'entity_id': 'merchant_abc', 'strength': 0.9, 'transaction_count': 10}
        ],
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def fraud_ring_network_data():
    """Generate network data indicative of fraud ring"""
    connections = []
    # Create highly connected network (fraud ring pattern)
    for i in range(60):
        connections.append({
            'entity_id': f'user{i}',
            'strength': 0.9,
            'transaction_count': 15
        })

    return {
        'entity_id': 'hub_user',
        'connections': connections,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def mock_isolation_forest():
    """Mock Isolation Forest model"""
    mock = MagicMock()
    mock.decision_function.return_value = np.array([0.3])
    mock.predict.return_value = np.array([1])  # Normal
    return mock


@pytest.fixture
def mock_isolation_forest_anomaly():
    """Mock Isolation Forest model detecting anomaly"""
    mock = MagicMock()
    mock.decision_function.return_value = np.array([-0.5])
    mock.predict.return_value = np.array([-1])  # Anomaly
    return mock


@pytest.fixture
def mock_one_class_svm():
    """Mock One-Class SVM model"""
    mock = MagicMock()
    mock.predict.return_value = np.array([1])  # Normal
    mock.decision_function.return_value = np.array([0.2])
    return mock


@pytest.fixture
def mock_xgboost():
    """Mock XGBoost classifier"""
    mock = MagicMock()
    mock.predict.return_value = np.array([0])  # Not fraud
    mock.predict_proba.return_value = np.array([[0.9, 0.1]])
    return mock


@pytest.fixture
def invalid_transaction_data():
    """Generate various invalid transaction data for validation testing"""
    return [
        # Negative amount
        {'amount': -100.00, 'merchant': 'Test'},
        # Excessive amount
        {'amount': 2_000_000_000, 'merchant': 'Test'},
        # Invalid timestamp
        {'amount': 100.00, 'timestamp': 'not-a-timestamp'},
        # Non-numeric amount
        {'amount': 'invalid', 'merchant': 'Test'},
        # Not a dictionary
        "not a dict",
        # Empty dict
        {}
    ]


@pytest.fixture
def invalid_behavioral_data():
    """Generate various invalid behavioral data for validation testing"""
    return [
        # Keystroke dynamics not a list
        {'keystroke_dynamics': 'invalid'},
        # Keystroke items not dicts
        {'keystroke_dynamics': ['invalid', 'items']},
        # Invalid timing values
        {'keystroke_dynamics': [{'dwell_time': -10}]},
        # Excessive timing values
        {'keystroke_dynamics': [{'dwell_time': 20000}]},
        # Mouse movements not a list
        {'mouse_movements': 'invalid'},
        # Not a dictionary
        "not a dict",
    ]


@pytest.fixture
def edge_case_keystroke_data():
    """Generate edge case keystroke data"""
    return {
        'empty': [],
        'single': [{'press_time': 100, 'release_time': 150}],
        'no_timing': [{'key': 'a'}, {'key': 'b'}],
        'partial_timing': [
            {'press_time': 100},
            {'release_time': 200}
        ]
    }


@pytest.fixture
def mock_network_graph():
    """Mock NetworkX graph for testing"""
    import networkx as nx

    G = nx.Graph()
    # Add sample nodes and edges
    G.add_node('user123')
    G.add_node('user456')
    G.add_node('user789')
    G.add_edge('user123', 'user456', weight=0.8)
    G.add_edge('user123', 'user789', weight=0.6)

    return G


@pytest.fixture
def performance_test_data():
    """Generate large dataset for performance testing"""
    return {
        'keystroke_data': [
            {
                'press_time': i * 100,
                'release_time': i * 100 + 80,
                'key': chr(97 + i % 26)
            }
            for i in range(1000)
        ],
        'mouse_movements': [
            {
                'x': i % 1920,
                'y': i % 1080,
                'timestamp': i * 10
            }
            for i in range(5000)
        ]
    }


@pytest.fixture
def datetime_helper():
    """Helper for generating various datetime scenarios"""
    class DateTimeHelper:
        @staticmethod
        def get_timestamp(hours_offset=0, days_offset=0):
            dt = datetime.now() + timedelta(hours=hours_offset, days=days_offset)
            return dt.isoformat()

        @staticmethod
        def get_unusual_hour_timestamp():
            """Get timestamp at unusual hour (2-5 AM)"""
            dt = datetime.now().replace(hour=3, minute=30)
            return dt.isoformat()

        @staticmethod
        def get_normal_hour_timestamp():
            """Get timestamp at normal hour (9 AM - 9 PM)"""
            dt = datetime.now().replace(hour=14, minute=30)
            return dt.isoformat()

    return DateTimeHelper()


@pytest.fixture
def comprehensive_analysis_result():
    """Sample comprehensive analysis result for testing explanation"""
    return {
        'overall_risk_score': 0.75,
        'risk_level': 'HIGH',
        'detected_anomalies': [
            'high_amount_transaction',
            'unusual_time_pattern',
            'keystroke_anomaly'
        ],
        'component_scores': {
            'transaction': 0.8,
            'behavioral': 0.7,
            'network': 0.6
        },
        'confidence': 0.85
    }


@pytest.fixture(autouse=True)
def reset_ml_models():
    """Reset ML models and caches before each test to ensure isolation"""
    from server import network_analyzer, prediction_cache, _inference_stats
    import networkx as nx
    from collections import deque
    yield
    # Reset network graph state after each test
    network_analyzer.transaction_graph = nx.Graph()
    network_analyzer._node_order = deque()
    # Reset prediction cache and inference stats
    prediction_cache.clear()
    _inference_stats["total_predictions"] = 0
    _inference_stats["cache_hits"] = 0
    _inference_stats["cache_misses"] = 0
    _inference_stats["total_time_ms"] = 0.0
    _inference_stats["batch_predictions"] = 0


@pytest.fixture
def mock_logger(monkeypatch):
    """Mock logger to capture log messages"""
    mock_log = Mock()
    import logging

    # Patch the logger
    monkeypatch.setattr('server.logger', mock_log)

    return mock_log
