"""
Tests for prediction caching, batch prediction, and inference statistics.
Phase 4: LRUCache integration, analyze_batch, get_inference_stats.
"""


# =============================================================================
# LRUCache direct tests
# =============================================================================


class TestLRUCacheDirect:
    """Test the LRUCache class imported from async_inference"""

    def test_cache_put_and_get(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=5)
        cache.put("key1", {"score": 0.5})
        result = cache.get("key1")
        assert result == {"score": 0.5}

    def test_cache_miss_returns_none(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=5)
        assert cache.get("nonexistent") is None

    def test_cache_eviction_on_capacity(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_cache_lru_order_on_access(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" to make it most recently used
        cache.get("a")
        cache.put("d", 4)  # Should evict "b" (least recently used)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_cache_update_existing_key(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=5)
        cache.put("key1", "old_value")
        cache.put("key1", "new_value")
        assert cache.get("key1") == "new_value"
        assert cache.size() == 1

    def test_cache_size(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=10)
        assert cache.size() == 0
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 2

    def test_cache_clear(self):
        from async_inference import LRUCache

        cache = LRUCache(capacity=10)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None


# =============================================================================
# Prediction cache integration tests
# =============================================================================


class TestPredictionCache:
    """Test prediction caching in analyze_transaction_impl"""

    def test_first_call_is_cache_miss(self, sample_transaction_data):
        from server import analyze_transaction_impl

        result = analyze_transaction_impl(sample_transaction_data)
        assert result.get("cache_hit") is False

    def test_second_call_is_cache_hit(self, sample_transaction_data):
        from server import analyze_transaction_impl

        # First call populates cache
        result1 = analyze_transaction_impl(sample_transaction_data)
        assert result1.get("cache_hit") is False
        # Second call should hit cache
        result2 = analyze_transaction_impl(sample_transaction_data)
        assert result2.get("cache_hit") is True

    def test_cache_returns_same_result(self, sample_transaction_data):
        from server import analyze_transaction_impl

        result1 = analyze_transaction_impl(sample_transaction_data)
        result2 = analyze_transaction_impl(sample_transaction_data)
        assert result1["overall_risk_score"] == result2["overall_risk_score"]
        assert result1["risk_level"] == result2["risk_level"]

    def test_different_transactions_no_cache_hit(
        self, sample_transaction_data, high_risk_transaction
    ):
        from server import analyze_transaction_impl

        result1 = analyze_transaction_impl(sample_transaction_data)
        result2 = analyze_transaction_impl(high_risk_transaction)
        assert result1.get("cache_hit") is False
        assert result2.get("cache_hit") is False

    def test_cache_disabled(self, sample_transaction_data):
        from server import analyze_transaction_impl

        # First call with cache
        analyze_transaction_impl(sample_transaction_data, use_cache=True)
        # Second call with cache disabled
        result = analyze_transaction_impl(sample_transaction_data, use_cache=False)
        # When cache is disabled, cache_hit should be False
        assert result.get("cache_hit") is False

    def test_cache_skipped_for_behavioral(
        self, sample_transaction_data, sample_behavioral_data
    ):
        from server import analyze_transaction_impl

        # First call with behavioral (should not cache)
        result1 = analyze_transaction_impl(
            sample_transaction_data,
            include_behavioral=True,
            behavioral_data=sample_behavioral_data,
        )
        # cache_hit should be False since behavioral analysis bypasses cache
        assert result1.get("cache_hit") is False

    def test_stats_update_on_cache_miss(self, sample_transaction_data):
        from server import analyze_transaction_impl, _inference_stats

        analyze_transaction_impl(sample_transaction_data)
        assert _inference_stats["total_predictions"] >= 1
        assert _inference_stats["cache_misses"] >= 1

    def test_stats_update_on_cache_hit(self, sample_transaction_data):
        from server import analyze_transaction_impl, _inference_stats

        analyze_transaction_impl(sample_transaction_data)
        analyze_transaction_impl(sample_transaction_data)
        assert _inference_stats["cache_hits"] >= 1
        assert _inference_stats["total_predictions"] >= 2


# =============================================================================
# Cache key generation tests
# =============================================================================


class TestCacheKeyGeneration:
    """Test _generate_cache_key determinism and uniqueness"""

    def test_same_data_same_key(self, sample_transaction_data):
        from server import _generate_cache_key

        key1 = _generate_cache_key(sample_transaction_data)
        key2 = _generate_cache_key(sample_transaction_data)
        assert key1 == key2

    def test_different_amount_different_key(self, sample_transaction_data):
        from server import _generate_cache_key

        key1 = _generate_cache_key(sample_transaction_data)
        modified = dict(sample_transaction_data)
        modified["amount"] = 999.99
        key2 = _generate_cache_key(modified)
        assert key1 != key2

    def test_different_merchant_different_key(self, sample_transaction_data):
        from server import _generate_cache_key

        key1 = _generate_cache_key(sample_transaction_data)
        modified = dict(sample_transaction_data)
        modified["merchant"] = "Walmart"
        key2 = _generate_cache_key(modified)
        assert key1 != key2

    def test_key_is_hex_hash(self, sample_transaction_data):
        from server import _generate_cache_key

        key = _generate_cache_key(sample_transaction_data)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest
        int(key, 16)  # Should not raise - valid hex

    def test_empty_dict_produces_key(self):
        from server import _generate_cache_key

        key = _generate_cache_key({})
        assert isinstance(key, str)
        assert len(key) == 64


# =============================================================================
# Batch prediction tests
# =============================================================================


class TestBatchPrediction:
    """Test analyze_batch_impl"""

    def test_batch_single_transaction(self, sample_transaction_data):
        from server import analyze_batch_impl

        result = analyze_batch_impl([sample_transaction_data])
        assert result["batch_size"] == 1
        assert len(result["results"]) == 1
        assert "summary" in result
        assert result["summary"]["total_analyzed"] == 1

    def test_batch_multiple_transactions(
        self, sample_transaction_data, high_risk_transaction
    ):
        from server import analyze_batch_impl

        result = analyze_batch_impl([sample_transaction_data, high_risk_transaction])
        assert result["batch_size"] == 2
        assert len(result["results"]) == 2
        assert result["summary"]["total_analyzed"] == 2

    def test_batch_risk_distribution(self, sample_transaction_data):
        from server import analyze_batch_impl

        # Create a batch of identical transactions
        batch = [sample_transaction_data] * 5
        result = analyze_batch_impl(batch)
        summary = result["summary"]
        total_in_distribution = sum(summary["risk_distribution"].values())
        assert total_in_distribution == 5

    def test_batch_summary_statistics(self, sample_transaction_data):
        from server import analyze_batch_impl

        result = analyze_batch_impl([sample_transaction_data])
        summary = result["summary"]
        assert "average_risk_score" in summary
        assert "max_risk_score" in summary
        assert "min_risk_score" in summary
        assert "processing_time_ms" in summary
        assert summary["average_risk_score"] >= 0.0
        assert summary["average_risk_score"] <= 1.0

    def test_batch_cache_hits(self, sample_transaction_data):
        from server import analyze_batch_impl

        # First batch populates cache
        analyze_batch_impl([sample_transaction_data])
        # Second batch should have cache hits
        result = analyze_batch_impl([sample_transaction_data])
        assert result["summary"]["cache_hits"] >= 1

    def test_batch_empty_list_error(self):
        from server import analyze_batch_impl

        result = analyze_batch_impl([])
        assert "error" in result
        assert result["status"] == "validation_failed"

    def test_batch_not_a_list_error(self):
        from server import analyze_batch_impl

        result = analyze_batch_impl("not_a_list")
        assert "error" in result
        assert result["status"] == "validation_failed"

    def test_batch_too_large_error(self):
        from server import analyze_batch_impl

        # Create a list exceeding 1000
        result = analyze_batch_impl([{"amount": 1}] * 1001)
        assert "error" in result
        assert "maximum" in result["error"]

    def test_batch_has_timestamp(self, sample_transaction_data):
        from server import analyze_batch_impl

        result = analyze_batch_impl([sample_transaction_data])
        assert "analysis_timestamp" in result

    def test_batch_updates_batch_stats(self, sample_transaction_data):
        from server import analyze_batch_impl, _inference_stats

        analyze_batch_impl([sample_transaction_data])
        assert _inference_stats["batch_predictions"] >= 1


# =============================================================================
# Inference stats tests
# =============================================================================


class TestInferenceStats:
    """Test get_inference_stats_impl"""

    def test_initial_stats_are_zero(self):
        from server import get_inference_stats_impl

        stats = get_inference_stats_impl()
        assert stats["total_predictions"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_hit_rate"] == 0.0

    def test_stats_after_predictions(self, sample_transaction_data):
        from server import analyze_transaction_impl, get_inference_stats_impl

        analyze_transaction_impl(sample_transaction_data)
        analyze_transaction_impl(sample_transaction_data)  # cache hit
        stats = get_inference_stats_impl()
        assert stats["total_predictions"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate"] == 0.5

    def test_stats_cache_size(self, sample_transaction_data, high_risk_transaction):
        from server import analyze_transaction_impl, get_inference_stats_impl

        analyze_transaction_impl(sample_transaction_data)
        analyze_transaction_impl(high_risk_transaction)
        stats = get_inference_stats_impl()
        assert stats["cache_size"] == 2

    def test_stats_has_capacity(self):
        from server import get_inference_stats_impl

        stats = get_inference_stats_impl()
        assert stats["cache_capacity"] == 1000

    def test_stats_average_time(self, sample_transaction_data):
        from server import analyze_transaction_impl, get_inference_stats_impl

        analyze_transaction_impl(sample_transaction_data)
        stats = get_inference_stats_impl()
        assert stats["average_prediction_time_ms"] >= 0.0
        assert stats["total_time_ms"] >= 0.0

    def test_stats_batch_count(self, sample_transaction_data):
        from server import analyze_batch_impl, get_inference_stats_impl

        analyze_batch_impl([sample_transaction_data])
        stats = get_inference_stats_impl()
        assert stats["batch_predictions"] == 1

    def test_stats_all_fields_present(self):
        from server import get_inference_stats_impl

        stats = get_inference_stats_impl()
        expected_fields = [
            "total_predictions",
            "cache_hits",
            "cache_misses",
            "cache_hit_rate",
            "cache_size",
            "cache_capacity",
            "average_prediction_time_ms",
            "total_time_ms",
            "batch_predictions",
        ]
        for field in expected_fields:
            assert field in stats, f"Missing field: {field}"


class TestCacheDoesNotBypassVelocity:
    """Pre-2026-05-04 the prediction_cache served stateful results that
    silently bypassed velocity tracking: identical-content transactions
    after the first only recorded ONE history entry, so a fraudster
    spamming identical transactions would never trip the velocity gate.

    Fix: history.record() runs BEFORE the cache check; the cache only
    stores the ML inference output, not the velocity-augmented final
    result. Velocity flags are recomputed every call from current state.
    """

    def setup_method(self):
        from server import prediction_cache
        import server
        prediction_cache.clear()
        server.user_history._history.clear()

    def test_identical_transactions_increment_velocity_counter(self):
        from server import analyze_transaction_impl, user_history

        uid = "velocity-cache-test"
        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "timestamp": "2026-05-04T12:00:00Z",
            "payment_method": "credit_card",
            "user_id": uid,
        }

        for _ in range(12):
            analyze_transaction_impl(txn)

        # All 12 should be in history (pre-fix: only 1)
        assert len(user_history.get_history(uid)) == 12

        # Velocity check should report 12, not 1
        v = user_history.check_velocity(uid)
        assert v["transaction_count"] == 12
        assert v["is_suspicious"] is True

    def test_velocity_flag_appears_after_threshold(self):
        from server import analyze_transaction_impl

        uid = "velocity-flag-test"
        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "timestamp": "2026-05-04T12:00:00Z",
            "payment_method": "credit_card",
            "user_id": uid,
        }

        # Threshold for is_suspicious is 10 (UserTransactionHistory.check_velocity)
        for _ in range(9):
            r = analyze_transaction_impl(txn)
        assert "high_transaction_velocity" not in r.get("detected_anomalies", [])

        for _ in range(3):
            r = analyze_transaction_impl(txn)
        assert "high_transaction_velocity" in r.get("detected_anomalies", []), (
            "velocity flag did not appear after 12 identical transactions — "
            "the prediction_cache is masking velocity tracking again"
        )

    def test_cache_hit_still_serves_fresh_velocity_info(self):
        from server import analyze_transaction_impl

        uid = "velocity-fresh-test"
        txn = {
            "amount": 50.0, "merchant": "M", "location": "L",
            "timestamp": "2026-05-04T12:00:00Z",
            "payment_method": "credit_card",
            "user_id": uid,
        }

        # Prime the cache
        analyze_transaction_impl(txn)
        # Pile on more so velocity threshold is crossed
        for _ in range(15):
            r = analyze_transaction_impl(txn)
        assert r["cache_hit"] is True  # subsequent calls hit cache
        v = r["velocity_analysis"]["velocity"]
        assert v["transaction_count"] >= 10
        assert v["is_suspicious"] is True


class TestLRUCacheTTL:
    """LRUCache acquired a TTL at 2026-05-04 to bound stale-state risk.
    Verify it actually enforces it."""

    def test_ttl_evicts_expired_entry_on_get(self):
        from async_inference import LRUCache
        import time as _time

        c = LRUCache(capacity=10, ttl_seconds=0.05)
        c.put("k1", "v1")
        # Immediately readable
        assert c.get("k1") == "v1"
        # After TTL
        _time.sleep(0.10)
        assert c.get("k1") is None
        # Eviction freed the slot
        assert c.size() == 0

    def test_ttl_none_preserves_legacy_behaviour(self):
        from async_inference import LRUCache
        import time as _time

        c = LRUCache(capacity=10, ttl_seconds=None)
        c.put("k1", "v1")
        _time.sleep(0.05)
        assert c.get("k1") == "v1"  # never expires
