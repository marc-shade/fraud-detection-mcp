"""
Tests for input validation functions
"""

import math
import pytest
from server import validate_transaction_data, validate_behavioral_data


class TestTransactionValidation:
    """Test transaction data validation"""

    def test_valid_transaction_basic(self, sample_transaction_data):
        """Test validation with valid transaction data"""
        valid, msg = validate_transaction_data(sample_transaction_data)
        assert valid is True
        assert msg == "valid"

    def test_valid_transaction_minimal(self):
        """Test validation with minimal valid data"""
        data = {'merchant': 'Test'}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_invalid_not_dict(self):
        """Test validation fails when not a dictionary"""
        valid, msg = validate_transaction_data("not a dict")
        assert valid is False
        assert "must be a dictionary" in msg

    def test_invalid_negative_amount(self):
        """Test validation fails with negative amount"""
        data = {'amount': -100.00}
        valid, msg = validate_transaction_data(data)
        assert valid is False
        assert "cannot be negative" in msg

    def test_invalid_excessive_amount(self):
        """Test validation fails with excessive amount"""
        data = {'amount': 2_000_000_000}
        valid, msg = validate_transaction_data(data)
        assert valid is False
        assert "exceeds maximum" in msg

    def test_invalid_non_numeric_amount(self):
        """Test validation fails with non-numeric amount"""
        data = {'amount': 'invalid'}
        valid, msg = validate_transaction_data(data)
        assert valid is False
        assert "must be numeric" in msg

    def test_valid_amount_zero(self):
        """Test validation passes with zero amount"""
        data = {'amount': 0.0}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_valid_amount_at_limit(self):
        """Test validation passes at maximum allowed amount"""
        data = {'amount': 1_000_000_000}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_invalid_timestamp_format(self):
        """Test validation fails with invalid timestamp"""
        data = {'timestamp': 'not-a-timestamp'}
        valid, msg = validate_transaction_data(data)
        assert valid is False
        assert "invalid timestamp" in msg

    def test_valid_timestamp_iso_format(self):
        """Test validation passes with ISO format timestamp"""
        data = {'timestamp': '2024-01-01T12:00:00'}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_valid_timestamp_with_z(self):
        """Test validation passes with Z timezone indicator"""
        data = {'timestamp': '2024-01-01T12:00:00Z'}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_valid_amount_integer(self):
        """Test validation passes with integer amount"""
        data = {'amount': 100}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_valid_amount_float(self):
        """Test validation passes with float amount"""
        data = {'amount': 99.99}
        valid, msg = validate_transaction_data(data)
        assert valid is True

    def test_boolean_amount_rejected(self):
        """Test validation fails with boolean amount"""
        valid, msg = validate_transaction_data({"amount": True})
        assert not valid
        assert "boolean" in msg.lower() or "numeric" in msg.lower()

    def test_nan_amount_rejected(self):
        """Test validation fails with NaN amount"""
        valid, msg = validate_transaction_data({"amount": float('nan')})
        assert not valid

    def test_inf_amount_rejected(self):
        """Test validation fails with infinity amount"""
        valid, msg = validate_transaction_data({"amount": float('inf')})
        assert not valid

    def test_negative_inf_amount_rejected(self):
        """Test validation fails with negative infinity amount"""
        valid, msg = validate_transaction_data({"amount": float('-inf')})
        assert not valid


class TestBehavioralValidation:
    """Test behavioral data validation"""

    def test_valid_behavioral_data(self, sample_behavioral_data):
        """Test validation with valid behavioral data"""
        valid, msg = validate_behavioral_data(sample_behavioral_data)
        assert valid is True
        assert msg == "valid"

    def test_invalid_not_dict(self):
        """Test validation fails when not a dictionary"""
        valid, msg = validate_behavioral_data("not a dict")
        assert valid is False
        assert "must be a dictionary" in msg

    def test_invalid_keystroke_not_list(self):
        """Test validation fails when keystroke_dynamics is not a list"""
        data = {'keystroke_dynamics': 'invalid'}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "must be a list" in msg

    def test_invalid_keystroke_items_not_dicts(self):
        """Test validation fails when keystroke items are not dictionaries"""
        data = {'keystroke_dynamics': ['invalid', 'items']}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "must be dictionaries" in msg

    def test_invalid_dwell_time_negative(self):
        """Test validation fails with negative dwell_time"""
        data = {'keystroke_dynamics': [{'dwell_time': -10}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "invalid dwell_time" in msg

    def test_invalid_dwell_time_excessive(self):
        """Test validation fails with excessive dwell_time"""
        data = {'keystroke_dynamics': [{'dwell_time': 20000}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "invalid dwell_time" in msg

    def test_invalid_flight_time_negative(self):
        """Test validation fails with negative flight_time"""
        data = {'keystroke_dynamics': [{'flight_time': -5}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "invalid flight_time" in msg

    def test_invalid_flight_time_excessive(self):
        """Test validation fails with excessive flight_time"""
        data = {'keystroke_dynamics': [{'flight_time': 15000}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "invalid flight_time" in msg

    def test_valid_dwell_time_at_boundaries(self):
        """Test validation passes with dwell_time at boundaries"""
        data = {'keystroke_dynamics': [{'dwell_time': 0}, {'dwell_time': 10000}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is True

    def test_invalid_mouse_not_list(self):
        """Test validation fails when mouse_movements is not a list"""
        data = {'mouse_movements': 'invalid'}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "must be a list" in msg

    def test_invalid_mouse_items_not_dicts(self):
        """Test validation fails when mouse items are not dictionaries"""
        data = {'mouse_movements': ['invalid']}
        valid, msg = validate_behavioral_data(data)
        assert valid is False
        assert "must be dictionaries" in msg

    def test_valid_empty_lists(self):
        """Test validation passes with empty lists"""
        data = {
            'keystroke_dynamics': [],
            'mouse_movements': []
        }
        valid, msg = validate_behavioral_data(data)
        assert valid is True

    def test_valid_partial_data(self):
        """Test validation passes with only keystroke data"""
        data = {'keystroke_dynamics': [{'dwell_time': 100}]}
        valid, msg = validate_behavioral_data(data)
        assert valid is True

    def test_valid_large_keystroke_dataset(self, performance_test_data):
        """Test validation handles large keystroke datasets efficiently"""
        data = {'keystroke_dynamics': performance_test_data['keystroke_data']}
        valid, msg = validate_behavioral_data(data)
        assert valid is True

    def test_valid_large_mouse_dataset(self, performance_test_data):
        """Test validation handles large mouse movement datasets efficiently"""
        data = {'mouse_movements': performance_test_data['mouse_movements']}
        valid, msg = validate_behavioral_data(data)
        assert valid is True

    def test_validation_checks_first_100_keystrokes(self):
        """Test that validation only checks first 100 keystroke items for efficiency"""
        # Create 200 items, first 100 valid, last 100 invalid
        keystrokes = [{'dwell_time': 100}] * 100
        keystrokes.extend([{'dwell_time': -10}] * 100)  # Invalid but beyond check limit

        data = {'keystroke_dynamics': keystrokes}
        valid, msg = validate_behavioral_data(data)
        # Should pass because only first 100 are checked
        assert valid is True

    def test_validation_checks_first_1000_mouse_movements(self):
        """Test that validation only checks first 1000 mouse movements for efficiency"""
        # Create 2000 items, first 1000 valid, last 1000 invalid
        movements = [{'x': 100, 'y': 200}] * 1000
        movements.extend(['invalid'] * 1000)  # Invalid but beyond check limit

        data = {'mouse_movements': movements}
        valid, msg = validate_behavioral_data(data)
        # Should pass because only first 1000 are checked
        assert valid is True


class TestDictToTransactionData:
    """Test dict-to-TransactionData adapter"""

    def test_full_dict_conversion(self, sample_transaction_data):
        from server import _dict_to_transaction_data
        txn = _dict_to_transaction_data(sample_transaction_data)
        assert txn.amount == 150.00
        assert txn.merchant == 'Amazon'
        assert txn.payment_method == 'credit_card'

    def test_minimal_dict_conversion(self):
        from server import _dict_to_transaction_data
        txn = _dict_to_transaction_data({'amount': 100.0})
        assert txn.amount == 100.0
        assert txn.merchant == 'unknown'
        assert txn.user_id == 'anonymous'
        assert txn.transaction_id.startswith('txn-')

    def test_unknown_payment_method(self):
        from server import _dict_to_transaction_data
        txn = _dict_to_transaction_data({'amount': 50.0, 'payment_method': 'unknown'})
        assert txn.payment_method == 'other'

    def test_timestamp_string_parsing(self):
        from server import _dict_to_transaction_data
        from datetime import datetime
        ts = datetime.now().isoformat()
        txn = _dict_to_transaction_data({'amount': 50.0, 'timestamp': ts})
        assert isinstance(txn.timestamp, datetime)
