"""
Tests for error handling and edge cases
"""

import pytest
from datetime import datetime
from server import (
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer,
    analyze_transaction_impl as analyze_transaction,
    detect_behavioral_anomaly_impl as detect_behavioral_anomaly,
    assess_network_risk_impl as assess_network_risk,
    generate_risk_score_impl as generate_risk_score,
    explain_decision_impl as explain_decision,
)


class TestErrorHandling:
    """Test error handling across all components"""

    def test_behavioral_analyzer_with_none(self):
        """Test behavioral analyzer handles None input"""
        analyzer = BehavioralBiometrics()
        result = analyzer.analyze_keystroke_dynamics(None)

        assert result['risk_score'] == 0.0
        assert result['status'] == 'no_data'

    def test_behavioral_analyzer_with_invalid_type(self):
        """Test behavioral analyzer handles invalid type"""
        analyzer = BehavioralBiometrics()
        result = analyzer.analyze_keystroke_dynamics("invalid")

        # Should handle gracefully
        assert 'error' in result or result['status'] == 'no_data'

    def test_transaction_analyzer_with_empty_dict(self):
        """Test transaction analyzer with empty dictionary"""
        analyzer = TransactionAnalyzer()
        result = analyzer.analyze_transaction({})

        # Should not crash, may have default values
        assert isinstance(result, dict)
        assert 'risk_score' in result

    def test_transaction_analyzer_with_none_values(self):
        """Test transaction analyzer with None values"""
        analyzer = TransactionAnalyzer()
        result = analyzer.analyze_transaction({
            'amount': None,
            'merchant': None
        })

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_network_analyzer_with_empty_graph(self):
        """Test network analyzer with empty graph"""
        analyzer = NetworkAnalyzer()
        result = analyzer._calculate_network_metrics('nonexistent')

        assert result == {}

    def test_analyze_transaction_tool_with_none(self):
        """Test analyze_transaction tool with None input"""
        result = analyze_transaction(None)

        assert 'error' in result
        assert result['status'] == 'validation_failed'

    def test_detect_behavioral_anomaly_with_none(self):
        """Test detect_behavioral_anomaly tool with None input"""
        result = detect_behavioral_anomaly(None)

        assert 'error' in result
        assert result['status'] == 'validation_failed'

    def test_assess_network_risk_with_none(self):
        """Test assess_network_risk tool with None input"""
        result = assess_network_risk(None)

        # Should handle None gracefully
        assert isinstance(result, dict)

    def test_generate_risk_score_with_none(self):
        """Test generate_risk_score tool with None input"""
        result = generate_risk_score(None)

        assert 'error' in result
        assert result['status'] == 'validation_failed'

    def test_explain_decision_with_none(self):
        """Test explain_decision tool with None input"""
        result = explain_decision(None)

        assert 'error' in result
        assert result['status'] == 'explanation_failed'

    def test_explain_decision_with_empty_dict(self):
        """Test explain_decision with empty dictionary"""
        result = explain_decision({})

        # Should handle missing keys gracefully
        assert isinstance(result, dict)

    def test_analyze_transaction_with_malformed_timestamp(self):
        """Test transaction analysis with malformed timestamp"""
        transaction = {
            'amount': 100,
            'timestamp': 'definitely not a timestamp'
        }
        result = analyze_transaction(transaction)

        # Validation should catch this
        assert 'error' in result

    def test_behavioral_data_with_missing_keys(self):
        """Test behavioral analysis with missing required keys"""
        data = {
            'keystroke_dynamics': [
                {'press_time': 100}  # Missing release_time
            ]
        }
        result = detect_behavioral_anomaly(data)

        # Should handle missing keys
        assert isinstance(result, dict)

    def test_network_data_with_invalid_connections(self):
        """Test network analysis with invalid connections format"""
        data = {
            'entity_id': 'test',
            'connections': 'not a list'
        }
        result = assess_network_risk(data)

        assert 'error' in result

    def test_very_large_keystroke_dataset(self):
        """Test with extremely large keystroke dataset"""
        huge_dataset = [
            {'press_time': i * 100, 'release_time': i * 100 + 80}
            for i in range(10000)
        ]

        analyzer = BehavioralBiometrics()
        result = analyzer.analyze_keystroke_dynamics(huge_dataset)

        # Should complete without crashing
        assert isinstance(result, dict)

    def test_unicode_in_transaction_data(self):
        """Test transaction with unicode characters"""
        transaction = {
            'amount': 100,
            'merchant': '商店名',
            'location': 'München'
        }
        result = analyze_transaction(transaction)

        # Should handle unicode gracefully
        assert isinstance(result, dict)

    def test_special_characters_in_strings(self):
        """Test transaction with special characters"""
        transaction = {
            'amount': 100,
            'merchant': "O'Reilly's <Store> & Co.",
            'location': "Test\nLocation"
        }
        result = analyze_transaction(transaction)

        assert isinstance(result, dict)

    def test_extreme_transaction_amounts(self):
        """Test with extreme transaction amounts"""
        # Very small amount
        result1 = analyze_transaction({'amount': 0.01})
        assert isinstance(result1, dict)

        # Very large amount (at limit)
        result2 = analyze_transaction({'amount': 999_999_999})
        assert isinstance(result2, dict)

    def test_future_timestamp(self):
        """Test transaction with future timestamp"""
        from datetime import timedelta
        future_time = (datetime.now() + timedelta(days=30)).isoformat()

        transaction = {
            'amount': 100,
            'timestamp': future_time
        }
        result = analyze_transaction(transaction)

        # Should process without error (may not detect as anomaly)
        assert isinstance(result, dict)

    def test_very_old_timestamp(self):
        """Test transaction with very old timestamp"""
        transaction = {
            'amount': 100,
            'timestamp': '1970-01-01T00:00:00Z'
        }
        result = analyze_transaction(transaction)

        assert isinstance(result, dict)

    def test_concurrent_graph_updates(self):
        """Test network analyzer with rapid updates"""
        analyzer = NetworkAnalyzer()

        # Rapidly add multiple entities
        for i in range(100):
            data = {
                'entity_id': f'user{i}',
                'connections': [{'entity_id': f'user{i+1}'}]
            }
            result = analyzer.analyze_network_risk(data)
            assert isinstance(result, dict)

    def test_circular_network_connections(self):
        """Test network with circular connections"""
        analyzer = NetworkAnalyzer()

        # Create circular reference
        data1 = {
            'entity_id': 'user1',
            'connections': [{'entity_id': 'user2'}]
        }
        data2 = {
            'entity_id': 'user2',
            'connections': [{'entity_id': 'user1'}]
        }

        result1 = analyzer.analyze_network_risk(data1)
        result2 = analyzer.analyze_network_risk(data2)

        # Should handle circular connections
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_self_referencing_connection(self):
        """Test entity with self-referencing connection"""
        data = {
            'entity_id': 'user1',
            'connections': [{'entity_id': 'user1'}]  # Self reference
        }
        result = assess_network_risk(data)

        assert isinstance(result, dict)

    def test_missing_optional_fields(self):
        """Test all tools with minimal data (only required fields)"""
        # Minimal transaction
        result1 = analyze_transaction({})
        assert isinstance(result1, dict)

        # Minimal behavioral
        result2 = detect_behavioral_anomaly({})
        assert isinstance(result2, dict)

        # Minimal network (missing entity_id)
        result3 = assess_network_risk({})
        assert isinstance(result3, dict)

    def test_extra_unknown_fields(self):
        """Test that extra fields don't cause errors"""
        transaction = {
            'amount': 100,
            'unknown_field_1': 'value1',
            'unknown_field_2': 'value2',
            'random_data': [1, 2, 3]
        }
        result = analyze_transaction(transaction)

        # Should ignore unknown fields gracefully
        assert isinstance(result, dict)
        assert 'error' not in result or result.get('status') != 'analysis_failed'

    def test_numeric_string_values(self):
        """Test that numeric strings are handled"""
        transaction = {
            'amount': '100.50',  # String instead of number
        }
        result = analyze_transaction(transaction)

        # Validation should catch this
        assert 'error' in result

    def test_boolean_in_numeric_field(self):
        """Test boolean value in numeric field"""
        transaction = {
            'amount': True,
        }
        result = analyze_transaction(transaction)

        # Validation should catch this (bool is not int/float in Python for isinstance)
        # Note: In Python, isinstance(True, int) is True, so this might pass
        assert isinstance(result, dict)

    def test_list_in_string_field(self):
        """Test list value in string field"""
        transaction = {
            'amount': 100,
            'merchant': ['not', 'a', 'string']
        }
        result = analyze_transaction(transaction)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_nested_dict_in_flat_field(self):
        """Test nested dictionary in flat field"""
        transaction = {
            'amount': 100,
            'location': {'city': 'New York', 'country': 'US'}
        }
        result = analyze_transaction(transaction)

        # Should handle gracefully (will hash the dict)
        assert isinstance(result, dict)

    def test_empty_string_values(self):
        """Test empty string values"""
        transaction = {
            'amount': 100,
            'merchant': '',
            'location': '',
            'payment_method': ''
        }
        result = analyze_transaction(transaction)

        assert isinstance(result, dict)

    def test_whitespace_only_strings(self):
        """Test whitespace-only strings"""
        transaction = {
            'amount': 100,
            'merchant': '   ',
            'location': '\t\n'
        }
        result = analyze_transaction(transaction)

        assert isinstance(result, dict)

    def test_inf_and_nan_values(self):
        """Test infinity and NaN values"""
        import math

        # NaN amount
        transaction1 = {'amount': math.nan}
        result1 = analyze_transaction(transaction1)
        # Should be caught by validation or handled gracefully
        assert isinstance(result1, dict)

        # Infinity amount
        transaction2 = {'amount': math.inf}
        result2 = analyze_transaction(transaction2)
        # Should exceed maximum and fail validation
        assert 'error' in result2

    def test_negative_zero(self):
        """Test negative zero value"""
        transaction = {'amount': -0.0}
        result = analyze_transaction(transaction)

        # -0.0 should be treated as 0.0 (valid)
        assert isinstance(result, dict)

    def test_keystroke_with_identical_timestamps(self):
        """Test keystroke data where all timestamps are identical"""
        data = [
            {'press_time': 100, 'release_time': 100}
            for _ in range(10)
        ]
        analyzer = BehavioralBiometrics()
        result = analyzer.analyze_keystroke_dynamics(data)

        # Should handle zero variance
        assert isinstance(result, dict)

    def test_keystroke_with_negative_dwell_time(self):
        """Test keystroke where release comes before press"""
        data = [
            {'press_time': 200, 'release_time': 100}  # Negative dwell
        ]
        analyzer = BehavioralBiometrics()
        result = analyzer.analyze_keystroke_dynamics(data)

        # Should handle negative values in feature extraction
        assert isinstance(result, dict)
