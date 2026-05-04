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

    def test_daily_limit_cumulative_with_history(self):
        """daily_limit MUST aggregate across the last 24h, not just check
        the single transaction. Pre-fix this was theater — daily_limit was
        functionally identical to max_amount because no history was
        consulted.

        Scenario: daily_limit=$1000, current txn $200, prior 24h $850.
        Expected: violation (200+850 = 1050 > 1000) even though no single
        transaction came close to the limit.
        """
        import time as _time
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"daily_limit": 1000.0}
        transaction = {"amount": 200.0, "timestamp": datetime.now().isoformat()}

        now_mono = _time.monotonic()
        history = [
            {"amount": 300.0, "recorded_at": now_mono - 600.0},   # 10 min ago
            {"amount": 250.0, "recorded_at": now_mono - 3600.0},  #  1h ago
            {"amount": 300.0, "recorded_at": now_mono - 7200.0},  #  2h ago
            # This one is >24h old and must NOT be counted:
            {"amount": 5000.0, "recorded_at": now_mono - 90000.0},
        ]
        result = verifier.verify(transaction, mandate, history=history)
        assert result["compliant"] is False
        assert any("daily_limit_exceeded" in v for v in result["violations"])
        # Utilization should reflect the cumulative projected total
        u = result["mandate_utilization"]
        assert u["daily_total_prior"] == 850.0
        assert u["daily_total_projected"] == 1050.0

    def test_daily_limit_under_with_history_passes(self):
        """Same setup but the 24h cumulative is below the limit — pass."""
        import time as _time
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"daily_limit": 1000.0}
        transaction = {"amount": 200.0, "timestamp": datetime.now().isoformat()}

        now_mono = _time.monotonic()
        history = [
            {"amount": 300.0, "recorded_at": now_mono - 600.0},
            {"amount": 250.0, "recorded_at": now_mono - 3600.0},
        ]
        result = verifier.verify(transaction, mandate, history=history)
        assert result["compliant"] is True
        assert result["mandate_utilization"]["daily_total_prior"] == 550.0
        assert result["mandate_utilization"]["daily_total_projected"] == 750.0

    def test_daily_limit_no_history_warns(self):
        """Without history, daily_limit degrades to single-txn check and
        emits ``daily_limit_no_history`` warning so the operator knows."""
        from server import MandateVerifier

        verifier = MandateVerifier()
        mandate = {"daily_limit": 1000.0}
        transaction = {"amount": 200.0, "timestamp": datetime.now().isoformat()}

        result = verifier.verify(transaction, mandate)  # no history arg
        assert "daily_limit_no_history" in result.get("warnings", [])

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
