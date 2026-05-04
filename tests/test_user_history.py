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


class TestImpossibleTravelDetection:
    """Pre-2026-05-04 check_geographic_velocity was theater:
      - Compared raw location strings — 'NYC' vs 'NYC, USA' was flagged
      - Hardcoded suspicion window of <5min — NYC→Tokyo in 8min not flagged
      - No distance computation — the entire 'impossible travel' premise

    Now uses haversine distance + velocity threshold (1000 km/h) when
    coordinates are present, with canonicalised string fallback.
    """

    def setup_method(self):
        from server import user_history
        user_history.reset()

    def test_nyc_to_tokyo_in_8min_is_flagged(self):
        """Pre-fix this was NOT flagged because the suspicious-window
        threshold was hardcoded at <5min."""
        from server import user_history
        import time as _time

        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "NYC",
                                   "location_lat": 40.7128, "location_lon": -74.0060,
                                   "timestamp": "2026-05-04T12:00:00Z"})
        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "Tokyo",
                                   "location_lat": 35.6762, "location_lon": 139.6503,
                                   "timestamp": "2026-05-04T12:08:00Z"})
        # Force the recorded_at gap to be exactly 8 min
        hist = user_history._history["u"]
        now = _time.monotonic()
        hist[0]["recorded_at"] = now - 480
        hist[1]["recorded_at"] = now

        result = user_history.check_geographic_velocity("u")
        assert result["method"] == "haversine"
        assert result["distance_km"] > 10000  # NYC↔Tokyo ~10,800 km
        assert result["velocity_kmh"] > 70000  # 10800/(8/60) ≈ 81000
        assert result["is_suspicious"] is True

    def test_nyc_to_sf_in_4h_is_flagged(self):
        """4h NYC→SF requires ~1030 km/h — above commercial cruise."""
        from server import user_history
        import time as _time

        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "NYC",
                                   "location_lat": 40.7128, "location_lon": -74.0060,
                                   "timestamp": "2026-05-04T12:00:00Z"})
        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "SF",
                                   "location_lat": 37.7749, "location_lon": -122.4194,
                                   "timestamp": "2026-05-04T16:00:00Z"})
        hist = user_history._history["u"]
        now = _time.monotonic()
        hist[0]["recorded_at"] = now - 14400  # 4h ago
        hist[1]["recorded_at"] = now

        result = user_history.check_geographic_velocity("u")
        assert result["is_suspicious"] is True

    def test_nyc_to_sf_in_24h_is_not_flagged(self):
        """24h NYC→SF is ~172 km/h — totally normal travel."""
        from server import user_history
        import time as _time

        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "NYC",
                                   "location_lat": 40.7128, "location_lon": -74.0060,
                                   "timestamp": "2026-05-04T12:00:00Z"})
        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "SF",
                                   "location_lat": 37.7749, "location_lon": -122.4194,
                                   "timestamp": "2026-05-05T12:00:00Z"})
        hist = user_history._history["u"]
        now = _time.monotonic()
        hist[0]["recorded_at"] = now - 86400  # 24h ago
        hist[1]["recorded_at"] = now

        result = user_history.check_geographic_velocity("u")
        assert result["is_suspicious"] is False
        assert result["velocity_kmh"] < 200

    def test_string_fallback_canonicalises_city_names(self):
        """Pre-fix 'New York, NY' vs 'New York' was flagged as different
        cities — false positive on the SAME city."""
        from server import user_history

        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "New York, NY",
                                   "timestamp": "2026-05-04T12:00:00Z"})
        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "New York",
                                   "timestamp": "2026-05-04T12:00:01Z"})
        result = user_history.check_geographic_velocity("u")
        assert result["method"] == "string_fallback"
        assert result["is_suspicious"] is False, (
            "'New York, NY' and 'New York' must be treated as the same city"
        )

    def test_string_fallback_still_flags_different_cities(self):
        """No coordinates → string fallback should still flag NYC vs LA
        within 5 min as suspicious (the only signal we have)."""
        from server import user_history

        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "New York",
                                   "timestamp": "2026-05-04T12:00:00Z"})
        user_history.record("u", {"amount": 100, "merchant": "M",
                                   "location": "Los Angeles",
                                   "timestamp": "2026-05-04T12:00:30Z"})
        result = user_history.check_geographic_velocity("u")
        assert result["method"] == "string_fallback"
        assert result["is_suspicious"] is True

    def test_haversine_self_distance_is_zero(self):
        """Sanity — haversine(p, p) == 0."""
        from server import UserTransactionHistory
        d = UserTransactionHistory._haversine_km(40.7128, -74.0060, 40.7128, -74.0060)
        assert d == 0.0

    def test_haversine_known_distance(self):
        """NYC↔SF haversine should match published value (~4130 km)."""
        from server import UserTransactionHistory
        d = UserTransactionHistory._haversine_km(40.7128, -74.0060, 37.7749, -122.4194)
        assert 4100 <= d <= 4150
