"""Tests for Phase 12: User Transaction History & Velocity Features"""

import threading


# ---------------------------------------------------------------------------
# UserTransactionHistory unit tests
# ---------------------------------------------------------------------------
class TestUserTransactionHistory:
    """Test UserTransactionHistory class."""

    def test_record_and_retrieve(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record(
            "user1",
            {"amount": 100, "merchant": "Amazon", "location": "US", "timestamp": ""},
        )
        history = h.get_history("user1")
        assert len(history) == 1
        assert history[0]["amount"] == 100.0

    def test_empty_history(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        assert h.get_history("nonexistent") == []

    def test_max_history_per_user(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory(max_history=5)
        for i in range(10):
            h.record(
                "user1",
                {"amount": i, "merchant": "M", "location": "US", "timestamp": ""},
            )
        history = h.get_history("user1")
        assert len(history) == 5
        # Oldest entries evicted; most recent kept
        assert history[0]["amount"] == 5.0

    def test_max_users_eviction(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory(max_users=3)
        h.record(
            "user1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""}
        )
        h.record(
            "user2", {"amount": 20, "merchant": "M", "location": "US", "timestamp": ""}
        )
        h.record(
            "user3", {"amount": 30, "merchant": "M", "location": "US", "timestamp": ""}
        )
        # Adding a 4th user evicts the oldest (user1)
        h.record(
            "user4", {"amount": 40, "merchant": "M", "location": "US", "timestamp": ""}
        )
        assert h.get_history("user1") == []
        assert len(h.get_history("user4")) == 1

    def test_get_stats(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory(max_history=50, max_users=1000)
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        stats = h.get_stats()
        assert stats["tracked_users"] == 2
        assert stats["total_entries"] == 2
        assert stats["max_users"] == 1000

    def test_reset_specific_user(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        h.reset("a")
        assert h.get_history("a") == []
        assert len(h.get_history("b")) == 1

    def test_reset_all(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record("a", {"amount": 1, "merchant": "M", "location": "US", "timestamp": ""})
        h.record("b", {"amount": 2, "merchant": "M", "location": "US", "timestamp": ""})
        h.reset()
        assert h.get_stats()["tracked_users"] == 0

    def test_thread_safety(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory(max_history=10000)
        errors = []

        def add_records(uid):
            try:
                for i in range(100):
                    h.record(
                        uid,
                        {
                            "amount": i,
                            "merchant": "M",
                            "location": "US",
                            "timestamp": "",
                        },
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_records, args=(f"user{i}",)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert h.get_stats()["tracked_users"] == 5


# ---------------------------------------------------------------------------
# Velocity checks
# ---------------------------------------------------------------------------
class TestVelocityChecks:
    """Test velocity analysis methods."""

    def test_velocity_under_limit(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        for i in range(3):
            h.record(
                "u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""}
            )
        result = h.check_velocity("u1")
        assert result["transaction_count"] == 3
        assert not result["is_suspicious"]

    def test_velocity_over_limit(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        for i in range(12):
            h.record(
                "u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""}
            )
        result = h.check_velocity("u1")
        assert result["transaction_count"] == 12
        assert result["is_suspicious"]

    def test_velocity_empty_user(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        result = h.check_velocity("nobody")
        assert result["transaction_count"] == 0
        assert not result["is_suspicious"]

    def test_amount_deviation_normal(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        for i in range(10):
            h.record(
                "u1",
                {"amount": 50 + i, "merchant": "M", "location": "US", "timestamp": ""},
            )
        result = h.check_amount_deviation("u1", 55.0)
        assert not result["is_suspicious"]
        assert not result["insufficient_history"]
        assert abs(result["z_score"]) < 3.0

    def test_amount_deviation_suspicious(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        for i in range(10):
            h.record(
                "u1",
                {
                    "amount": 20 + i * 0.1,
                    "merchant": "M",
                    "location": "US",
                    "timestamp": "",
                },
            )
        result = h.check_amount_deviation("u1", 50000.0)
        assert result["is_suspicious"]
        assert result["z_score"] > 3.0

    def test_amount_deviation_insufficient_history(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record(
            "u1", {"amount": 100, "merchant": "M", "location": "US", "timestamp": ""}
        )
        result = h.check_amount_deviation("u1", 500.0)
        assert result["insufficient_history"]
        assert not result["is_suspicious"]

    def test_geographic_velocity_same_location(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record(
            "u1",
            {
                "amount": 10,
                "merchant": "M",
                "location": "United States",
                "timestamp": "",
            },
        )
        h.record(
            "u1",
            {
                "amount": 20,
                "merchant": "M",
                "location": "United States",
                "timestamp": "",
            },
        )
        result = h.check_geographic_velocity("u1")
        assert result["location_changes"] == 0
        assert not result["is_suspicious"]

    def test_geographic_velocity_different_location_fast(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record(
            "u1",
            {
                "amount": 10,
                "merchant": "M",
                "location": "United States",
                "timestamp": "",
            },
        )
        # Recorded immediately after -- within 300s window
        h.record(
            "u1", {"amount": 20, "merchant": "M", "location": "Japan", "timestamp": ""}
        )
        result = h.check_geographic_velocity("u1")
        assert result["location_changes"] == 1
        assert result["is_suspicious"]

    def test_geographic_velocity_insufficient_history(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        h.record(
            "u1", {"amount": 10, "merchant": "M", "location": "US", "timestamp": ""}
        )
        result = h.check_geographic_velocity("u1")
        assert result["insufficient_history"]

    def test_merchant_diversity_low(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        for i in range(3):
            h.record(
                "u1",
                {"amount": 10, "merchant": "Amazon", "location": "US", "timestamp": ""},
            )
        result = h.check_merchant_diversity("u1")
        assert result["unique_merchants"] == 1
        assert not result["is_suspicious"]

    def test_merchant_diversity_suspicious(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        merchants = [
            "Amazon",
            "Walmart",
            "Target",
            "BestBuy",
            "Costco",
            "Apple",
            "Google",
        ]
        for m in merchants:
            h.record(
                "u1", {"amount": 10, "merchant": m, "location": "US", "timestamp": ""}
            )
        result = h.check_merchant_diversity("u1")
        assert result["unique_merchants"] == 7
        assert result["is_suspicious"]


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------
class TestHistoryServerIntegration:
    """Test history integration in server.py."""

    def test_user_history_exists(self):
        import server

        assert hasattr(server, "user_history")
        assert server.user_history is not None

    def test_user_history_class_available(self):
        from server import UserTransactionHistory

        h = UserTransactionHistory()
        assert h.get_stats()["tracked_users"] == 0

    def test_health_check_has_user_history(self):
        from server import health_check_impl

        result = health_check_impl()
        assert "user_history" in result
        assert "tracked_users" in result["user_history"]
        assert "max_users" in result["user_history"]

    def test_analyze_transaction_returns_velocity(self):
        import server

        txn = {
            "transaction_id": "vel-test-001",
            "user_id": "velocity-user",
            "amount": 100.0,
            "merchant": "TestStore",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn)
        assert "velocity_analysis" in result
        va = result["velocity_analysis"]
        assert "velocity" in va
        assert "amount_deviation" in va
        assert "geographic" in va
        assert "merchant_diversity" in va

    def test_velocity_accumulates_across_calls(self):
        import server

        # Reset history for clean test
        server.user_history.reset("velocity-accum-user")
        for i in range(3):
            txn = {
                "transaction_id": f"vel-accum-{i}",
                "user_id": "velocity-accum-user",
                "amount": 50.0 + i,
                "merchant": "Store",
                "location": "United States",
                "timestamp": f"2026-01-15T14:3{i}:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        va = result["velocity_analysis"]
        assert va["velocity"]["transaction_count"] >= 3

    def test_high_velocity_adds_risk_factor(self):
        import server

        server.user_history.reset("burst-user")
        # Flood 12 transactions to trigger velocity flag (unique amounts bypass cache)
        for i in range(12):
            txn = {
                "transaction_id": f"burst-{i}",
                "user_id": "burst-user",
                "amount": 25.0 + i * 0.01,
                "merchant": "Store",
                "location": "United States",
                "timestamp": f"2026-01-15T14:{i:02d}:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        # After 12 rapid transactions, velocity should be flagged
        anomalies = result.get("detected_anomalies", [])
        assert "high_transaction_velocity" in anomalies

    def test_amount_deviation_adds_risk_factor(self):
        import server

        server.user_history.reset("deviation-user")
        # Build history with low amounts (varied to avoid zero std, unique for cache)
        for i in range(10):
            txn = {
                "transaction_id": f"dev-small-{i}",
                "user_id": "deviation-user",
                "amount": 20.0 + i * 0.5,
                "merchant": "Store",
                "location": "United States",
                "timestamp": f"2026-01-15T14:{20 + i}:00Z",
                "payment_method": "credit_card",
            }
            server.analyze_transaction_impl(txn)
        # Now send a very large transaction (extreme outlier survives being in history)
        big_txn = {
            "transaction_id": "dev-big",
            "user_id": "deviation-user",
            "amount": 999999.0,
            "merchant": "Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:45:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(big_txn)
        anomalies = result.get("detected_anomalies", [])
        assert "unusual_amount_deviation" in anomalies

    def test_impossible_travel_adds_risk_factor(self):
        import server

        server.user_history.reset("travel-user")
        txn1 = {
            "transaction_id": "travel-1",
            "user_id": "travel-user",
            "amount": 50.0,
            "merchant": "Store",
            "location": "United States",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        server.analyze_transaction_impl(txn1)
        # Immediately from a different country
        txn2 = {
            "transaction_id": "travel-2",
            "user_id": "travel-user",
            "amount": 50.0,
            "merchant": "Store",
            "location": "Japan",
            "timestamp": "2026-01-15T14:30:00Z",
            "payment_method": "credit_card",
        }
        result = server.analyze_transaction_impl(txn2)
        anomalies = result.get("detected_anomalies", [])
        assert "impossible_travel_detected" in anomalies

    def test_merchant_diversity_adds_risk_factor(self):
        import server

        server.user_history.reset("merchant-user")
        merchants = ["Amazon", "Walmart", "Target", "BestBuy", "Costco", "Apple"]
        for i, m in enumerate(merchants):
            txn = {
                "transaction_id": f"merch-{i}",
                "user_id": "merchant-user",
                "amount": 10.0,
                "merchant": m,
                "location": "United States",
                "timestamp": "2026-01-15T14:30:00Z",
                "payment_method": "credit_card",
            }
            result = server.analyze_transaction_impl(txn)
        anomalies = result.get("detected_anomalies", [])
        assert "high_merchant_diversity" in anomalies
