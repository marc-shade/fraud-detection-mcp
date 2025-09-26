#!/usr/bin/env python3
"""
Test suite for Advanced Fraud Detection MCP
Comprehensive testing of all fraud detection capabilities
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the fraud detection components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import (
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer
)

class TestBehavioralBiometrics:
    """Test behavioral biometrics fraud detection"""

    def setup_method(self):
        """Setup test environment"""
        self.behavioral_analyzer = BehavioralBiometrics()

    def test_keystroke_dynamics_normal(self):
        """Test normal keystroke patterns"""
        # Normal typing pattern
        keystroke_data = [
            {"key": "p", "press_time": 1000, "release_time": 1050},
            {"key": "a", "press_time": 1120, "release_time": 1160},
            {"key": "s", "press_time": 1200, "release_time": 1240},
            {"key": "s", "press_time": 1280, "release_time": 1320},
            {"key": "w", "press_time": 1360, "release_time": 1400},
            {"key": "o", "press_time": 1480, "release_time": 1520},
            {"key": "r", "press_time": 1560, "release_time": 1600},
            {"key": "d", "press_time": 1680, "release_time": 1720},
        ]

        result = self.behavioral_analyzer.analyze_keystroke_dynamics(keystroke_data)

        assert result["confidence"] > 0.8
        assert "risk_score" in result
        assert result["analysis_type"] == "keystroke_dynamics"

    def test_keystroke_dynamics_suspicious(self):
        """Test suspicious keystroke patterns"""
        # Suspicious pattern - very slow and irregular
        keystroke_data = [
            {"key": "p", "press_time": 1000, "release_time": 1200},  # Very long press
            {"key": "a", "press_time": 2000, "release_time": 2100},  # Long gap
            {"key": "s", "press_time": 3500, "release_time": 3600},  # Very long gap
        ]

        result = self.behavioral_analyzer.analyze_keystroke_dynamics(keystroke_data)

        assert "risk_score" in result
        assert result["features_analyzed"] == 10  # 5 dwell + 5 flight features

    def test_empty_keystroke_data(self):
        """Test handling of empty keystroke data"""
        result = self.behavioral_analyzer.analyze_keystroke_dynamics([])

        assert result["risk_score"] == 0.0
        assert result["status"] == "no_data"

    def test_insufficient_keystroke_data(self):
        """Test handling of insufficient keystroke data"""
        keystroke_data = [
            {"key": "p", "press_time": 1000, "release_time": 1050}
        ]

        result = self.behavioral_analyzer.analyze_keystroke_dynamics(keystroke_data)

        assert result["risk_score"] == 0.0
        assert result["status"] == "invalid_data"

class TestTransactionAnalyzer:
    """Test transaction pattern analysis"""

    def setup_method(self):
        """Setup test environment"""
        self.transaction_analyzer = TransactionAnalyzer()

    def test_normal_transaction(self):
        """Test normal transaction analysis"""
        transaction_data = {
            "amount": 50.00,
            "merchant": "Coffee Shop",
            "location": "New York, NY",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card"
        }

        result = self.transaction_analyzer.analyze_transaction(transaction_data)

        assert "risk_score" in result
        assert result["analysis_type"] == "transaction_pattern"
        assert result["confidence"] > 0.8

    def test_high_amount_transaction(self):
        """Test high amount transaction flagging"""
        transaction_data = {
            "amount": 25000.00,
            "merchant": "Electronics Store",
            "location": "Las Vegas, NV",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card"
        }

        result = self.transaction_analyzer.analyze_transaction(transaction_data)

        assert "high_amount_transaction" in result.get("risk_factors", [])
        assert result["risk_score"] > 0.3

    def test_unusual_time_transaction(self):
        """Test unusual time pattern detection"""
        # 3 AM transaction
        unusual_time = datetime.now().replace(hour=3, minute=30)
        transaction_data = {
            "amount": 1000.00,
            "merchant": "Gas Station",
            "location": "Miami, FL",
            "timestamp": unusual_time.isoformat(),
            "payment_method": "debit_card"
        }

        result = self.transaction_analyzer.analyze_transaction(transaction_data)

        assert "unusual_time_pattern" in result.get("risk_factors", [])

    def test_crypto_payment_risk(self):
        """Test cryptocurrency payment risk detection"""
        transaction_data = {
            "amount": 5000.00,
            "merchant": "Crypto Exchange",
            "location": "Online",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "crypto"
        }

        result = self.transaction_analyzer.analyze_transaction(transaction_data)

        assert "high_risk_payment_method" in result.get("risk_factors", [])

    def test_feature_extraction(self):
        """Test transaction feature extraction"""
        transaction_data = {
            "amount": 1000.00,
            "merchant": "Test Merchant",
            "location": "Test Location",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card"
        }

        features = self.transaction_analyzer._extract_transaction_features(transaction_data)

        assert len(features) == 8  # Expected number of features
        assert features[0] == 1000.00  # Amount
        assert features[1] > 0  # Log-transformed amount

class TestNetworkAnalyzer:
    """Test network-based fraud detection"""

    def setup_method(self):
        """Setup test environment"""
        self.network_analyzer = NetworkAnalyzer()

    def test_single_entity_analysis(self):
        """Test analysis of single entity with connections"""
        entity_data = {
            "entity_id": "user_001",
            "connections": [
                {"entity_id": "user_002", "strength": 0.8, "transaction_count": 5},
                {"entity_id": "merchant_001", "strength": 0.3, "transaction_count": 1}
            ]
        }

        result = self.network_analyzer.analyze_network_risk(entity_data)

        assert "risk_score" in result
        assert result["analysis_type"] == "network_analysis"
        assert "network_metrics" in result

    def test_high_connectivity_detection(self):
        """Test detection of unusually high connectivity"""
        # Create entity with many connections
        connections = [
            {"entity_id": f"user_{i:03d}", "strength": 0.5, "transaction_count": 1}
            for i in range(60)  # 60 connections (above threshold of 50)
        ]

        entity_data = {
            "entity_id": "hub_user",
            "connections": connections
        }

        result = self.network_analyzer.analyze_network_risk(entity_data)

        assert "unusually_high_connectivity" in result.get("risk_patterns", [])
        assert result["risk_score"] > 0.3

    def test_no_entity_id(self):
        """Test handling of missing entity ID"""
        entity_data = {
            "connections": []
        }

        result = self.network_analyzer.analyze_network_risk(entity_data)

        assert result["status"] == "no_entity_id"
        assert result["risk_score"] == 0.0

    def test_network_metrics_calculation(self):
        """Test network metrics calculation"""
        # Build a small network
        entity_data = {
            "entity_id": "center_node",
            "connections": [
                {"entity_id": "node_1", "strength": 1.0, "transaction_count": 3},
                {"entity_id": "node_2", "strength": 0.8, "transaction_count": 2},
                {"entity_id": "node_3", "strength": 0.6, "transaction_count": 1}
            ]
        }

        result = self.network_analyzer.analyze_network_risk(entity_data)
        metrics = result["network_metrics"]

        assert "degree" in metrics
        assert "clustering_coefficient" in metrics
        assert "betweenness_centrality" in metrics
        assert "closeness_centrality" in metrics

class TestIntegration:
    """Integration tests for the complete fraud detection system"""

    def test_comprehensive_fraud_scenario(self):
        """Test a comprehensive fraud detection scenario"""
        # High-risk transaction
        transaction_data = {
            "amount": 15000.00,
            "merchant": "Unknown Electronics",
            "location": "Unknown Location",
            "timestamp": datetime.now().replace(hour=2).isoformat(),  # 2 AM
            "payment_method": "crypto"
        }

        # Suspicious behavioral patterns
        behavioral_data = {
            "keystroke_dynamics": [
                {"key": "1", "press_time": 1000, "release_time": 1300},  # Very slow
                {"key": "2", "press_time": 2000, "release_time": 2100},  # Long gap
                {"key": "3", "press_time": 3000, "release_time": 3100},  # Long gap
            ]
        }

        # Suspicious network patterns
        network_data = {
            "entity_id": "suspicious_user",
            "connections": [
                {"entity_id": f"bot_{i}", "strength": 0.9, "transaction_count": 1}
                for i in range(55)  # High connectivity
            ]
        }

        # Analyze each component
        transaction_analyzer = TransactionAnalyzer()
        behavioral_analyzer = BehavioralBiometrics()
        network_analyzer = NetworkAnalyzer()

        transaction_result = transaction_analyzer.analyze_transaction(transaction_data)
        behavioral_result = behavioral_analyzer.analyze_keystroke_dynamics(
            behavioral_data["keystroke_dynamics"]
        )
        network_result = network_analyzer.analyze_network_risk(network_data)

        # All should indicate high risk
        assert transaction_result["risk_score"] > 0.5
        assert len(transaction_result["risk_factors"]) >= 2
        assert behavioral_result["risk_score"] > 0.0
        assert network_result["risk_score"] > 0.3

    def test_low_risk_scenario(self):
        """Test a low-risk transaction scenario"""
        # Normal transaction
        transaction_data = {
            "amount": 25.00,
            "merchant": "Coffee Shop",
            "location": "New York, NY",
            "timestamp": datetime.now().replace(hour=10).isoformat(),  # 10 AM
            "payment_method": "credit_card"
        }

        # Normal behavioral patterns
        behavioral_data = {
            "keystroke_dynamics": [
                {"key": "p", "press_time": 1000, "release_time": 1050},
                {"key": "a", "press_time": 1120, "release_time": 1160},
                {"key": "s", "press_time": 1200, "release_time": 1240},
                {"key": "s", "press_time": 1280, "release_time": 1320},
                {"key": "w", "press_time": 1360, "release_time": 1400},
            ]
        }

        # Normal network patterns
        network_data = {
            "entity_id": "normal_user",
            "connections": [
                {"entity_id": "normal_merchant", "strength": 0.3, "transaction_count": 5}
            ]
        }

        # Analyze each component
        transaction_analyzer = TransactionAnalyzer()
        behavioral_analyzer = BehavioralBiometrics()
        network_analyzer = NetworkAnalyzer()

        transaction_result = transaction_analyzer.analyze_transaction(transaction_data)
        behavioral_result = behavioral_analyzer.analyze_keystroke_dynamics(
            behavioral_data["keystroke_dynamics"]
        )
        network_result = network_analyzer.analyze_network_risk(network_data)

        # All should indicate low risk
        assert transaction_result["risk_score"] < 0.4
        assert len(transaction_result.get("risk_factors", [])) == 0
        assert not behavioral_result.get("is_anomaly", False)
        assert network_result["risk_score"] < 0.3

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_malformed_transaction_data(self):
        """Test handling of malformed transaction data"""
        transaction_analyzer = TransactionAnalyzer()

        # Missing required fields
        malformed_data = {"invalid": "data"}

        result = transaction_analyzer.analyze_transaction(malformed_data)

        assert "risk_score" in result
        # Should handle gracefully without crashing

    def test_invalid_keystroke_data(self):
        """Test handling of invalid keystroke data"""
        behavioral_analyzer = BehavioralBiometrics()

        # Invalid keystroke format
        invalid_data = [
            {"invalid": "format"},
            {"also": "invalid"}
        ]

        result = behavioral_analyzer.analyze_keystroke_dynamics(invalid_data)

        assert result["risk_score"] == 0.0
        assert result["status"] == "invalid_data"

    def test_network_analysis_errors(self):
        """Test network analysis error handling"""
        network_analyzer = NetworkAnalyzer()

        # Empty connections with valid entity ID
        edge_case_data = {
            "entity_id": "isolated_user",
            "connections": []
        }

        result = network_analyzer.analyze_network_risk(edge_case_data)

        # Should handle gracefully
        assert "risk_score" in result
        assert result["analysis_type"] == "network_analysis"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])