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
