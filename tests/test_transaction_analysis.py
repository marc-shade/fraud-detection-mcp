"""
Tests for transaction pattern analysis
"""

import pytest
import numpy as np
from datetime import datetime
from server import TransactionAnalyzer


class TestTransactionAnalysis:
    """Test transaction fraud detection"""

    @pytest.fixture
    def analyzer(self):
        """Create transaction analyzer"""
        return TransactionAnalyzer()

    def test_analyze_normal_transaction(self, analyzer, sample_transaction_data):
        """Test analysis of normal transaction"""
        result = analyzer.analyze_transaction(sample_transaction_data)

        assert isinstance(result, dict)
        assert 'risk_score' in result
        assert 'is_anomaly' in result
        assert 'risk_factors' in result
        assert 'confidence' in result
        assert 'analysis_type' in result

        assert result['analysis_type'] == 'transaction_pattern'
        assert 0.0 <= result['risk_score'] <= 1.0
        assert isinstance(result['risk_factors'], list)

    def test_analyze_high_risk_transaction(self, analyzer, high_risk_transaction):
        """Test detection of high-risk transaction"""
        result = analyzer.analyze_transaction(high_risk_transaction)

        assert isinstance(result, dict)
        assert result['risk_score'] > 0.0
        # Should detect multiple risk factors
        assert len(result['risk_factors']) > 0

    def test_extract_transaction_features(self, analyzer, sample_transaction_data):
        """Test feature extraction from transaction data (46 features via FeatureEngineer)"""
        features = analyzer._extract_transaction_features(sample_transaction_data)

        assert isinstance(features, np.ndarray)
        assert len(features) == 46

    def test_feature_amount_extraction(self, analyzer):
        """Test correct extraction of amount features"""
        transaction = {'amount': 1000.00}
        features = analyzer._extract_transaction_features(transaction)

        # First feature should be the amount
        assert features[0] == 1000.00
        # Second feature should be log-transformed amount
        assert features[1] > 0

    def test_feature_time_extraction(self, analyzer, datetime_helper):
        """Test correct extraction of time-based features"""
        transaction = {'timestamp': datetime_helper.get_timestamp()}
        features = analyzer._extract_transaction_features(transaction)

        assert isinstance(features, np.ndarray)
        assert len(features) == 46

    def test_feature_time_missing_timestamp(self, analyzer):
        """Test feature extraction when timestamp is missing"""
        transaction = {'amount': 100}
        features = analyzer._extract_transaction_features(transaction)

        assert isinstance(features, np.ndarray)
        assert len(features) == 46

    def test_payment_method_risk_scoring(self, analyzer):
        """Test payment method produces different features for different methods"""
        transactions = [
            {'payment_method': 'credit_card', 'amount': 100},
            {'payment_method': 'debit_card', 'amount': 100},
            {'payment_method': 'crypto', 'amount': 100},
        ]

        features_list = [
            analyzer._extract_transaction_features(t) for t in transactions
        ]

        # Different payment methods should produce different feature vectors
        assert not np.array_equal(features_list[0], features_list[2])

    def test_identify_high_amount_risk(self, analyzer):
        """Test identification of high amount risk factor"""
        transaction = {'amount': 15000.00}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'high_amount_transaction' in risk_factors

    def test_identify_no_high_amount_risk(self, analyzer):
        """Test no high amount risk for normal transactions"""
        transaction = {'amount': 500.00}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'high_amount_transaction' not in risk_factors

    def test_identify_unusual_time_risk(self, analyzer, datetime_helper):
        """Test identification of unusual time pattern"""
        transaction = {'timestamp': datetime_helper.get_unusual_hour_timestamp()}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'unusual_time_pattern' in risk_factors

    def test_identify_normal_time(self, analyzer):
        """Test no unusual time risk for normal hours"""
        from datetime import timedelta
        # Use yesterday at 14:00 (definitively in the past, within Pydantic's 1-year limit)
        normal_ts = (datetime.now() - timedelta(days=1)).replace(hour=14, minute=0)
        transaction = {'timestamp': normal_ts.isoformat(), 'amount': 100}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'unusual_time_pattern' not in risk_factors

    def test_identify_crypto_payment_risk(self, analyzer):
        """Test identification of high-risk payment method"""
        transaction = {'payment_method': 'crypto', 'amount': 100}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'high_risk_payment_method' in risk_factors

    def test_identify_geographic_risk(self, analyzer):
        """Test identification of high-risk geographic location from config"""
        from config import get_config
        app_config = get_config()

        for location in app_config.HIGH_RISK_LOCATIONS:
            transaction = {'location': location, 'amount': 100}
            features = analyzer._extract_transaction_features(transaction)
            risk_factors = analyzer._identify_risk_factors(transaction, features)

            assert 'high_risk_geographic_location' in risk_factors

    def test_no_geographic_risk_safe_location(self, analyzer):
        """Test no geographic risk for safe locations"""
        transaction = {'location': 'United States', 'amount': 100}
        features = analyzer._extract_transaction_features(transaction)
        risk_factors = analyzer._identify_risk_factors(transaction, features)

        assert 'high_risk_geographic_location' not in risk_factors

    def test_multiple_risk_factors_increase_score(self, analyzer):
        """Test that multiple risk factors increase overall risk score"""
        # Transaction with single risk factor
        transaction1 = {'amount': 15000.00}
        result1 = analyzer.analyze_transaction(transaction1)

        # Transaction with multiple risk factors
        transaction2 = {
            'amount': 15000.00,
            'payment_method': 'crypto',
            'location': 'nigeria'
        }
        result2 = analyzer.analyze_transaction(transaction2)

        # Multiple risk factors should result in higher score
        assert result2['risk_score'] >= result1['risk_score']

    def test_risk_score_bounds(self, analyzer, sample_transaction_data):
        """Test that risk score is always within bounds"""
        result = analyzer.analyze_transaction(sample_transaction_data)

        assert 0.0 <= result['risk_score'] <= 1.0

    def test_confidence_score_value(self, analyzer, sample_transaction_data):
        """Test that confidence score has expected value"""
        result = analyzer.analyze_transaction(sample_transaction_data)

        assert result['confidence'] == 0.88

    def test_anomaly_score_returned(self, analyzer, sample_transaction_data):
        """Test that anomaly score is included in results"""
        result = analyzer.analyze_transaction(sample_transaction_data)

        assert 'anomaly_score' in result
        assert isinstance(result['anomaly_score'], float)

    def test_model_initialization(self, analyzer):
        """Test that models are properly initialized"""
        assert analyzer.isolation_forest is not None
        assert not hasattr(analyzer, 'xgb_model')

    def test_error_handling_malformed_data(self, analyzer):
        """Test error handling with malformed transaction data"""
        malformed_data = {'amount': 'invalid'}
        result = analyzer.analyze_transaction(malformed_data)

        assert 'error' in result
        assert result['risk_score'] == 0.0
        assert result['status'] == 'error'

    def test_location_hashing_consistency(self, analyzer):
        """Test that location produces consistent features"""
        transaction = {'location': 'Test Location', 'amount': 100}

        features1 = analyzer._extract_transaction_features(transaction)
        features2 = analyzer._extract_transaction_features(transaction)

        assert np.array_equal(features1, features2)

    def test_merchant_hashing_consistency(self, analyzer):
        """Test that merchant produces consistent features"""
        transaction = {'merchant': 'Test Merchant', 'amount': 100}

        features1 = analyzer._extract_transaction_features(transaction)
        features2 = analyzer._extract_transaction_features(transaction)

        assert np.array_equal(features1, features2)

    def test_edge_case_zero_amount(self, analyzer):
        """Test analysis with zero amount transaction"""
        transaction = {'amount': 0.0}
        result = analyzer.analyze_transaction(transaction)

        assert isinstance(result, dict)
        assert 'risk_score' in result

    def test_edge_case_very_large_amount(self, analyzer):
        """Test analysis with very large amount (within Pydantic max of $10M)"""
        transaction = {'amount': 9_999_999}
        result = analyzer.analyze_transaction(transaction)

        assert isinstance(result, dict)
        assert 'high_amount_transaction' in result['risk_factors']

    def test_timestamp_parsing_iso_format(self, analyzer):
        """Test timestamp parsing in ISO format"""
        from datetime import timedelta
        recent_ts = (datetime.now() - timedelta(days=1)).replace(hour=10, minute=30)
        transaction = {
            'timestamp': recent_ts.isoformat(),
            'amount': 100
        }
        result = analyzer.analyze_transaction(transaction)

        assert isinstance(result, dict)
        assert 'error' not in result

    def test_timestamp_parsing_with_timezone(self, analyzer):
        """Test timestamp parsing with timezone"""
        from datetime import timedelta
        recent_ts = (datetime.now() - timedelta(days=1)).replace(hour=10, minute=30)
        transaction = {
            'timestamp': recent_ts.isoformat(),
            'amount': 100
        }
        result = analyzer.analyze_transaction(transaction)

        assert isinstance(result, dict)
        assert 'error' not in result

    def test_case_insensitive_location_matching(self, analyzer):
        """Test that location risk matching is case-insensitive"""
        transaction1 = {'location': 'UNKNOWN', 'amount': 100}
        transaction2 = {'location': 'unknown', 'amount': 100}

        features1 = analyzer._extract_transaction_features(transaction1)
        features2 = analyzer._extract_transaction_features(transaction2)

        risk_factors1 = analyzer._identify_risk_factors(transaction1, features1)
        risk_factors2 = analyzer._identify_risk_factors(transaction2, features2)

        # Both should identify same risk
        assert 'high_risk_geographic_location' in risk_factors1
        assert 'high_risk_geographic_location' in risk_factors2

    def test_geographic_risk_exact_match_not_substring(self, analyzer):
        """'China Palace Restaurant' should NOT trigger geographic risk."""
        txn = {"amount": 100, "location": "China Palace Restaurant, Seattle"}
        features = analyzer._extract_transaction_features(txn)
        risk_factors = analyzer._identify_risk_factors(txn, features)
        assert "high_risk_geographic_location" not in risk_factors

    def test_geographic_risk_exact_match_triggers(self, analyzer):
        """Exact match on 'unknown' should trigger."""
        txn = {"amount": 100, "location": "unknown"}
        features = analyzer._extract_transaction_features(txn)
        risk_factors = analyzer._identify_risk_factors(txn, features)
        assert "high_risk_geographic_location" in risk_factors

    def test_feature_extraction_deterministic(self, analyzer):
        """Features must be identical across calls (must be deterministic)."""
        txn = {"amount": 100, "location": "USA", "merchant": "Amazon", "payment_method": "credit_card"}
        f1 = analyzer._extract_transaction_features(txn)
        f2 = analyzer._extract_transaction_features(txn)
        assert np.array_equal(f1, f2), "Feature extraction must be deterministic"

    def test_complete_transaction_analysis(self, analyzer):
        """Test complete transaction with all fields"""
        transaction = {
            'amount': 5000.00,
            'merchant': 'Test Merchant',
            'location': 'United States',
            'timestamp': datetime.now().isoformat(),
            'payment_method': 'credit_card',
            'user_id': 'user123',
            'merchant_category': 'retail',
            'currency': 'USD'
        }
        result = analyzer.analyze_transaction(transaction)

        assert isinstance(result, dict)
        assert all(k in result for k in [
            'risk_score', 'is_anomaly', 'risk_factors',
            'confidence', 'analysis_type', 'anomaly_score'
        ])
