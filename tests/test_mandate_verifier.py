"""Tests for MandateVerifier class."""

import pytest
from datetime import datetime


class TestMandateVerifier:
    """Test MandateVerifier constraint checking."""

    def test_fully_compliant_transaction(self):
        """Transaction within all mandate constraints passes."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {
            "max_amount": 500.0,
            "allowed_merchants": ["Amazon", "Office Depot"],
        }
        transaction = {
            "amount": 100.0,
            "merchant": "Amazon",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True
        assert result["violations"] == []
        assert result["drift_score"] == 0.0

    def test_amount_exceeds_max(self):
        """Transaction over max_amount is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"max_amount": 100.0}
        transaction = {"amount": 200.0, "timestamp": datetime.now().isoformat()}
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("amount_exceeded" in v for v in result["violations"])

    def test_blocked_merchant(self):
        """Transaction with blocked merchant is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"blocked_merchants": ["Casino", "Gambling"]}
        transaction = {
            "amount": 50.0,
            "merchant": "Casino",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("blocked_merchant" in v for v in result["violations"])

    def test_merchant_not_in_allowed_list(self):
        """Transaction with merchant not in allowed list is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"allowed_merchants": ["Amazon", "Office Depot"]}
        transaction = {
            "amount": 50.0,
            "merchant": "Unauthorized Store",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("merchant_not_allowed" in v for v in result["violations"])

    def test_outside_time_window(self):
        """Transaction outside operating hours is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"time_window": {"start": "09:00", "end": "17:00"}}
        # Create a transaction at 3 AM
        transaction = {
            "amount": 50.0,
            "timestamp": "2026-02-21T03:00:00",
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("outside_time_window" in v for v in result["violations"])

    def test_within_time_window(self):
        """Transaction within operating hours is compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"time_window": {"start": "09:00", "end": "17:00"}}
        transaction = {
            "amount": 50.0,
            "timestamp": "2026-02-21T12:00:00",
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True

    def test_location_not_allowed(self):
        """Transaction from disallowed location is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"allowed_locations": ["United States", "Canada"]}
        transaction = {
            "amount": 50.0,
            "location": "Russia",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("location_not_allowed" in v for v in result["violations"])

    def test_daily_limit_exceeded(self):
        """Transaction pushing over daily_limit is non-compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"daily_limit": 500.0}
        transaction = {
            "amount": 600.0,
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        assert any("daily_limit_exceeded" in v for v in result["violations"])

    def test_drift_score_multiple_violations(self):
        """Drift score reflects ratio of violations to checks."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {
            "max_amount": 100.0,
            "allowed_merchants": ["Amazon"],
            "allowed_locations": ["United States"],
        }
        transaction = {
            "amount": 200.0,
            "merchant": "Bad Store",
            "location": "Russia",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is False
        # 3 violations out of 3 constraints = drift_score 1.0
        assert result["drift_score"] == pytest.approx(1.0, abs=0.01)

    def test_empty_mandate_always_compliant(self):
        """Empty mandate means no constraints, always compliant."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {}
        transaction = {
            "amount": 99999.0,
            "merchant": "Anything",
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert result["compliant"] is True
        assert result["drift_score"] == 0.0

    def test_mandate_utilization(self):
        """Mandate utilization shows how close to limits."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"max_amount": 100.0, "daily_limit": 1000.0}
        transaction = {
            "amount": 80.0,
            "timestamp": datetime.now().isoformat(),
        }
        result = verifier.verify(transaction, mandate)
        assert "mandate_utilization" in result
        # 80% of max_amount
        assert result["mandate_utilization"]["amount_pct"] == pytest.approx(
            0.8, abs=0.01
        )

    def test_result_has_required_fields(self):
        """Verify result contains all required fields."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        result = verifier.verify(
            {"amount": 10.0, "timestamp": datetime.now().isoformat()}, {}
        )
        assert "compliant" in result
        assert "violations" in result
        assert "drift_score" in result
        assert "mandate_utilization" in result
        assert "checks_performed" in result
