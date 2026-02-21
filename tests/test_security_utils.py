"""
Tests for security_utils module (InputSanitizer + InMemoryRateLimiter)
and their integration into the server pipeline.
"""

import time

import pytest

from security_utils import InMemoryRateLimiter, InputSanitizer


# =========================================================================
# InputSanitizer
# =========================================================================

class TestStripHtmlTags:
    """InputSanitizer.strip_html_tags"""

    @pytest.mark.unit
    @pytest.mark.security
    def test_removes_simple_tags(self):
        assert InputSanitizer.strip_html_tags("<b>bold</b>") == "bold"

    @pytest.mark.unit
    @pytest.mark.security
    def test_removes_script_tags(self):
        assert InputSanitizer.strip_html_tags('<script>alert("x")</script>') == 'alert("x")'

    @pytest.mark.unit
    @pytest.mark.security
    def test_preserves_plain_text(self):
        assert InputSanitizer.strip_html_tags("hello world") == "hello world"

    @pytest.mark.unit
    @pytest.mark.security
    def test_handles_empty_string(self):
        assert InputSanitizer.strip_html_tags("") == ""

    @pytest.mark.unit
    @pytest.mark.security
    def test_nested_tags(self):
        assert InputSanitizer.strip_html_tags("<div><p>text</p></div>") == "text"


class TestSanitizeString:
    """InputSanitizer.sanitize_string"""

    @pytest.mark.unit
    @pytest.mark.security
    def test_strips_control_characters(self):
        result = InputSanitizer.sanitize_string("hello\x00world\x07!")
        assert result == "helloworld!"

    @pytest.mark.unit
    @pytest.mark.security
    def test_preserves_tab_newline_cr(self):
        result = InputSanitizer.sanitize_string("line1\nline2\ttab\rcarriage")
        assert "\n" in result
        assert "\t" in result
        assert "\r" in result

    @pytest.mark.unit
    @pytest.mark.security
    def test_truncates_long_string(self):
        long = "A" * 20_000
        result = InputSanitizer.sanitize_string(long, max_length=100)
        assert len(result) == 100

    @pytest.mark.unit
    @pytest.mark.security
    def test_strips_html_and_controls(self):
        result = InputSanitizer.sanitize_string("<b>test\x00</b>")
        assert result == "test"


class TestSanitizeDict:
    """InputSanitizer.sanitize_dict"""

    @pytest.mark.unit
    @pytest.mark.security
    def test_sanitizes_string_values(self):
        data = {"name": "<b>Alice</b>", "amount": 100}
        result = InputSanitizer.sanitize_dict(data)
        assert result["name"] == "Alice"
        assert result["amount"] == 100

    @pytest.mark.unit
    @pytest.mark.security
    def test_sanitizes_nested_dicts(self):
        data = {"outer": {"inner": "<script>x</script>"}}
        result = InputSanitizer.sanitize_dict(data)
        assert result["outer"]["inner"] == "x"

    @pytest.mark.unit
    @pytest.mark.security
    def test_sanitizes_lists(self):
        data = {"items": ["<b>a</b>", "<i>b</i>", 42]}
        result = InputSanitizer.sanitize_dict(data)
        assert result["items"] == ["a", "b", 42]

    @pytest.mark.unit
    @pytest.mark.security
    def test_max_depth_prevents_stack_overflow(self):
        # Build deeply nested dict
        data: dict = {"val": "ok"}
        for _ in range(20):
            data = {"nested": data}
        result = InputSanitizer.sanitize_dict(data, max_depth=5)
        assert isinstance(result, dict)

    @pytest.mark.unit
    @pytest.mark.security
    def test_passes_through_none_bool_float(self):
        data = {"a": None, "b": True, "c": 3.14}
        result = InputSanitizer.sanitize_dict(data)
        assert result == data

    @pytest.mark.unit
    @pytest.mark.security
    def test_sanitizes_keys(self):
        data = {"<b>key</b>": "value"}
        result = InputSanitizer.sanitize_dict(data)
        assert "key" in result
        assert "<b>key</b>" not in result


# =========================================================================
# InMemoryRateLimiter
# =========================================================================

class TestRateLimiter:
    """InMemoryRateLimiter core behaviour"""

    @pytest.mark.unit
    @pytest.mark.security
    def test_allows_under_limit(self):
        rl = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        result = rl.check_rate_limit("user1")
        assert result["allowed"] is True
        assert result["remaining"] == 4

    @pytest.mark.unit
    @pytest.mark.security
    def test_blocks_over_limit(self):
        rl = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check_rate_limit("user1")
        result = rl.check_rate_limit("user1")
        assert result["allowed"] is False
        assert result["remaining"] == 0
        assert result["retry_after"] is not None

    @pytest.mark.unit
    @pytest.mark.security
    def test_separate_keys(self):
        rl = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        rl.check_rate_limit("user1")
        rl.check_rate_limit("user1")
        result = rl.check_rate_limit("user2")
        assert result["allowed"] is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_window_expiry(self):
        rl = InMemoryRateLimiter(max_requests=1, window_seconds=0.1)
        rl.check_rate_limit("user1")
        time.sleep(0.15)
        result = rl.check_rate_limit("user1")
        assert result["allowed"] is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_reset_single_key(self):
        rl = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        rl.check_rate_limit("user1")
        rl.reset("user1")
        result = rl.check_rate_limit("user1")
        assert result["allowed"] is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_reset_all(self):
        rl = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        rl.check_rate_limit("user1")
        rl.check_rate_limit("user2")
        rl.reset()
        r1 = rl.check_rate_limit("user1")
        r2 = rl.check_rate_limit("user2")
        assert r1["allowed"] is True
        assert r2["allowed"] is True

    @pytest.mark.unit
    @pytest.mark.security
    def test_get_status(self):
        rl = InMemoryRateLimiter(max_requests=10, window_seconds=60)
        rl.check_rate_limit("a")
        rl.check_rate_limit("b")
        status = rl.get_status()
        assert status["max_requests"] == 10
        assert status["window_seconds"] == 60
        assert status["active_keys"] == 2
        assert status["total_active_requests"] == 2


# =========================================================================
# Server integration
# =========================================================================

class TestServerIntegration:
    """Verify security_utils is wired into the server."""

    @pytest.mark.integration
    @pytest.mark.security
    def test_security_utils_available_flag(self):
        import server
        assert server.SECURITY_UTILS_AVAILABLE is True

    @pytest.mark.integration
    @pytest.mark.security
    def test_sanitizer_loaded(self):
        import server
        assert server.sanitizer is not None

    @pytest.mark.integration
    @pytest.mark.security
    def test_rate_limiter_loaded(self):
        import server
        assert server.rate_limiter is not None

    @pytest.mark.integration
    @pytest.mark.security
    def test_health_check_includes_security(self):
        from server import health_check_impl
        result = health_check_impl()
        assert "security_utils" in result
        sec = result["security_utils"]
        assert sec["available"] is True
        assert sec["sanitizer_loaded"] is True
        assert sec["rate_limiter"] is not None
        assert "max_requests" in sec["rate_limiter"]

    @pytest.mark.integration
    @pytest.mark.security
    def test_transaction_analysis_with_html_input(self):
        """Sanitisation should strip HTML but analysis should still succeed."""
        from server import analyze_transaction_impl
        result = analyze_transaction_impl({
            "amount": 150.0,
            "merchant": "<script>alert('xss')</script>Amazon",
            "location": "US",
        })
        # Should not error -- sanitiser strips the tags before validation
        assert "error" not in result or result.get("status") != "validation_failed"

    @pytest.mark.integration
    @pytest.mark.security
    def test_behavioral_analysis_with_html_input(self):
        """Sanitisation should strip HTML from behavioral data."""
        from server import detect_behavioral_anomaly_impl
        result = detect_behavioral_anomaly_impl({
            "user_id": "<img src=x onerror=alert(1)>user123",
            "keystroke_dynamics": [],
        })
        # Should run without error (empty keystroke list is valid)
        assert "error" not in result

    @pytest.mark.integration
    @pytest.mark.security
    def test_rate_limit_enforced_on_transaction(self):
        """Exhaust rate limit and verify the impl returns rate_limited status."""
        from server import analyze_transaction_impl, rate_limiter
        # Use a unique user_id to avoid collision with other tests
        uid = "rate_limit_test_user_unique_12345"
        # Reset to ensure clean state
        if rate_limiter is not None:
            rate_limiter.reset(uid)
            # Temporarily lower limit for the test
            original_max = rate_limiter.max_requests
            rate_limiter.max_requests = 2
            try:
                for _ in range(2):
                    analyze_transaction_impl({"amount": 10, "user_id": uid})
                result = analyze_transaction_impl({"amount": 10, "user_id": uid})
                assert result["status"] == "rate_limited"
                assert "retry_after" in result
            finally:
                rate_limiter.max_requests = original_max
                rate_limiter.reset(uid)
