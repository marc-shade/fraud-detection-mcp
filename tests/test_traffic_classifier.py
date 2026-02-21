"""Tests for TrafficClassifier and classify_traffic_source MCP tool"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
from server import TrafficClassifier


class TestTrafficClassifier:
    """Tests for the TrafficClassifier class"""

    def setup_method(self):
        self.classifier = TrafficClassifier()

    @pytest.mark.unit
    def test_classify_explicit_agent_flag(self):
        """When is_agent=True is set, classify as agent"""
        result = self.classifier.classify({"is_agent": True})
        assert result["source"] == "agent"
        assert result["confidence"] >= 0.8
        assert "explicit_flag" in result["signals"]

    @pytest.mark.unit
    def test_classify_explicit_human_flag(self):
        """When is_agent=False is set, classify as human"""
        result = self.classifier.classify({"is_agent": False})
        assert result["source"] == "human"
        assert result["confidence"] >= 0.8
        assert "explicit_flag" in result["signals"]

    @pytest.mark.unit
    def test_classify_stripe_acp_user_agent(self):
        """Stripe ACP user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "Stripe-ACP/1.0 agent-id:abc123"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "stripe_acp"
        assert "user_agent_match" in result["signals"]

    @pytest.mark.unit
    def test_classify_visa_tap_user_agent(self):
        """Visa TAP user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "Visa-TAP/2.0 commerce-agent"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "visa_tap"

    @pytest.mark.unit
    def test_classify_openai_operator_user_agent(self):
        """OpenAI Operator user agent detected as agent"""
        result = self.classifier.classify({
            "user_agent": "OpenAI-Operator/1.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "openai"

    @pytest.mark.unit
    def test_classify_browser_user_agent(self):
        """Standard browser user agent classified as human"""
        result = self.classifier.classify({
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0"
        })
        assert result["source"] == "human"
        assert "user_agent_match" in result["signals"]

    @pytest.mark.unit
    def test_classify_agent_identifier_present(self):
        """Presence of agent_identifier signals agent"""
        result = self.classifier.classify({
            "agent_identifier": "mastercard-agent-pay:agent-456"
        })
        assert result["source"] == "agent"
        assert "agent_identifier_present" in result["signals"]

    @pytest.mark.unit
    def test_classify_unknown_no_signals(self):
        """No signals results in unknown classification"""
        result = self.classifier.classify({
            "amount": 100.0,
            "merchant": "Store"
        })
        assert result["source"] == "unknown"
        assert result["confidence"] < 0.5

    @pytest.mark.unit
    def test_classify_returns_all_required_fields(self):
        """Result contains all required fields"""
        result = self.classifier.classify({"is_agent": True})
        assert "source" in result
        assert "confidence" in result
        assert "agent_type" in result
        assert "signals" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.unit
    def test_classify_empty_input(self):
        """Empty dict returns unknown"""
        result = self.classifier.classify({})
        assert result["source"] == "unknown"

    @pytest.mark.unit
    def test_classify_multiple_signals_boost_confidence(self):
        """Multiple agent signals increase confidence"""
        result = self.classifier.classify({
            "is_agent": True,
            "agent_identifier": "stripe-acp:agent-789",
            "user_agent": "Stripe-ACP/1.0"
        })
        assert result["source"] == "agent"
        assert result["confidence"] >= 0.95

    @pytest.mark.unit
    def test_classify_coinbase_x402(self):
        """Coinbase x402 protocol detected"""
        result = self.classifier.classify({
            "user_agent": "x402-client/1.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "x402"

    @pytest.mark.unit
    def test_classify_google_ap2(self):
        """Google AP2 protocol detected"""
        result = self.classifier.classify({
            "user_agent": "Google-AP2/1.0 agent"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "google_ap2"

    @pytest.mark.unit
    def test_classify_paypal_agent(self):
        """PayPal Agent Ready detected"""
        result = self.classifier.classify({
            "user_agent": "PayPal-Agent/2.0"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "paypal"

    @pytest.mark.unit
    def test_classify_anthropic_agent(self):
        """Anthropic Claude agent detected"""
        result = self.classifier.classify({
            "user_agent": "Anthropic-Agent/1.0 claude"
        })
        assert result["source"] == "agent"
        assert result["agent_type"] == "anthropic"


class TestClassifyTrafficSourceImpl:
    """Tests for the classify_traffic_source_impl function"""

    @pytest.mark.unit
    def test_impl_returns_valid_result(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({
            "user_agent": "Stripe-ACP/1.0",
            "amount": 100.0
        })
        assert result["source"] == "agent"
        assert "classification_timestamp" in result

    @pytest.mark.unit
    def test_impl_with_transaction_data_and_metadata(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl(
            {"amount": 100.0, "merchant": "Store"},
            {"user_agent": "Mozilla/5.0 Chrome/120", "is_agent": False}
        )
        assert result["source"] == "human"

    @pytest.mark.unit
    def test_impl_invalid_input(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl("not a dict")
        assert "error" in result

    @pytest.mark.unit
    def test_impl_metadata_in_transaction_data(self):
        """Agent fields in transaction_data itself should be detected"""
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({
            "amount": 100.0,
            "is_agent": True,
            "agent_identifier": "test-agent-1"
        })
        assert result["source"] == "agent"

    @pytest.mark.unit
    def test_impl_empty_input(self):
        from server import classify_traffic_source_impl
        result = classify_traffic_source_impl({})
        assert result["source"] == "unknown"
