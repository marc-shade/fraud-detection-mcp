"""
Tests for Phase 8: Explainability integration into active server.
Covers: EXPLAINABILITY_AVAILABLE flag, explain_decision_impl with
        transaction_data, feature explanation in analyze_transaction_impl,
        generate_summary integration, fallback mode, health_check and
        model_status explainability fields.
"""

import pytest
import numpy as np
from unittest.mock import patch


# =============================================================================
# EXPLAINABILITY_AVAILABLE flag
# =============================================================================


class TestExplainabilityAvailableFlag:
    """Test the EXPLAINABILITY_AVAILABLE graceful degradation flag."""

    def test_explainability_available_is_bool(self):
        """EXPLAINABILITY_AVAILABLE is a boolean."""
        from server import EXPLAINABILITY_AVAILABLE

        assert isinstance(EXPLAINABILITY_AVAILABLE, bool)

    def test_explainability_available_is_true(self):
        """EXPLAINABILITY_AVAILABLE is True when explainability module loads."""
        from server import EXPLAINABILITY_AVAILABLE

        assert EXPLAINABILITY_AVAILABLE is True

    def test_shap_available_is_bool(self):
        """SHAP_AVAILABLE is a boolean."""
        from server import SHAP_AVAILABLE

        assert isinstance(SHAP_AVAILABLE, bool)

    def test_fraud_explainer_is_initialized(self):
        """fraud_explainer is initialized when EXPLAINABILITY_AVAILABLE is True."""
        from server import fraud_explainer, EXPLAINABILITY_AVAILABLE

        if EXPLAINABILITY_AVAILABLE:
            assert fraud_explainer is not None

    def test_fraud_explainer_has_model(self):
        """fraud_explainer references the transaction analyzer's isolation forest."""
        from server import fraud_explainer, transaction_analyzer

        if fraud_explainer is not None:
            assert fraud_explainer.model is transaction_analyzer.isolation_forest

    def test_fraud_explainer_has_feature_names(self):
        """fraud_explainer has the correct feature names from FeatureEngineer."""
        from server import fraud_explainer, transaction_analyzer

        if fraud_explainer is not None:
            assert fraud_explainer.feature_names == (
                transaction_analyzer.feature_engineer.feature_names
            )

    def test_fraud_explainer_fallback_mode_is_bool(self):
        """fraud_explainer.fallback_mode is a boolean."""
        from server import fraud_explainer

        if fraud_explainer is not None:
            assert isinstance(fraud_explainer.fallback_mode, bool)


# =============================================================================
# FraudExplainer unit tests
# =============================================================================


class TestFraudExplainerUnit:
    """Unit tests for the FraudExplainer class."""

    def test_explain_prediction_returns_dict(self):
        """explain_prediction returns a dictionary."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert isinstance(result, dict)

    def test_explain_prediction_has_method(self):
        """explain_prediction result contains 'method' field."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert "method" in result
        assert result["method"] in ("SHAP", "Feature Importance", "Basic Analysis")

    def test_explain_prediction_has_risk_factors(self):
        """explain_prediction result contains risk_factors list."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert "risk_factors" in result
        assert isinstance(result["risk_factors"], list)

    def test_explain_prediction_has_protective_factors(self):
        """explain_prediction result contains protective_factors list."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert "protective_factors" in result
        assert isinstance(result["protective_factors"], list)

    def test_explain_prediction_has_top_features(self):
        """explain_prediction result contains top_features list."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert "top_features" in result
        assert isinstance(result["top_features"], list)

    def test_explain_prediction_top_n_limit(self):
        """explain_prediction respects top_n parameter."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(46)
        result = fraud_explainer.explain_prediction(features, 0.5, top_n=3)
        assert len(result.get("top_features", [])) <= 3

    def test_explain_prediction_2d_features(self):
        """explain_prediction handles 2D feature array."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features = np.random.randn(1, 46)
        result = fraud_explainer.explain_prediction(features, 0.5)
        assert isinstance(result, dict)

    def test_generate_summary_returns_string(self):
        """generate_summary returns a string."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        explanation = {
            "method": "Feature Importance",
            "prediction": 0.7,
            "risk_factors": [{"feature": "amount", "description": "High amount"}],
            "protective_factors": [],
        }
        summary = fraud_explainer.generate_summary(explanation)
        assert isinstance(summary, str)
        assert "Risk Score" in summary

    def test_generate_summary_includes_risk_factors(self):
        """generate_summary includes risk factor descriptions."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        explanation = {
            "method": "Feature Importance",
            "prediction": 0.7,
            "risk_factors": [
                {"feature": "amount", "description": "Transaction amount: $500.00"}
            ],
            "protective_factors": [],
        }
        summary = fraud_explainer.generate_summary(explanation)
        assert "amount" in summary

    def test_generate_summary_includes_protective_factors(self):
        """generate_summary includes protective factor descriptions."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        explanation = {
            "method": "Feature Importance",
            "prediction": 0.2,
            "risk_factors": [],
            "protective_factors": [
                {"feature": "payment_method", "description": "Traditional payment"}
            ],
        }
        summary = fraud_explainer.generate_summary(explanation)
        assert "Protective Factors" in summary

    def test_batch_explain_returns_list(self):
        """batch_explain returns a list of explanations."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features_batch = np.random.randn(3, 46)
        predictions = np.array([0.2, 0.5, 0.8])
        results = fraud_explainer.batch_explain(features_batch, predictions)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_explain_each_has_method(self):
        """Each batch_explain result has a 'method' field."""
        from server import fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        features_batch = np.random.randn(2, 46)
        predictions = np.array([0.3, 0.7])
        results = fraud_explainer.batch_explain(features_batch, predictions)
        for result in results:
            assert "method" in result


# =============================================================================
# explain_decision_impl with transaction_data
# =============================================================================


class TestExplainDecisionWithTransactionData:
    """Test explain_decision_impl with the new transaction_data parameter."""

    def test_explain_without_transaction_data(self):
        """explain_decision_impl works without transaction_data (backward compat)."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.5,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
        }
        result = explain_decision_impl(analysis_result)
        assert isinstance(result, dict)
        assert "decision_summary" in result
        assert "explainability_method" in result
        assert result["explainability_method"] == "rule_based"

    def test_explain_with_transaction_data(self, sample_transaction_data):
        """explain_decision_impl generates feature analysis with transaction_data."""
        from server import explain_decision_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        analysis_result = {
            "overall_risk_score": 0.5,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
        }
        result = explain_decision_impl(
            analysis_result, transaction_data=sample_transaction_data
        )
        assert "feature_analysis" in result
        assert "explainability_method" in result
        assert result["explainability_method"] in (
            "SHAP",
            "Feature Importance",
            "Basic Analysis",
        )

    def test_explain_with_transaction_data_has_summary(self, sample_transaction_data):
        """explain_decision_impl generates human_readable_summary with transaction_data."""
        from server import explain_decision_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        analysis_result = {
            "overall_risk_score": 0.6,
            "risk_level": "HIGH",
            "detected_anomalies": ["high_amount_transaction"],
        }
        result = explain_decision_impl(
            analysis_result, transaction_data=sample_transaction_data
        )
        assert "human_readable_summary" in result
        assert isinstance(result["human_readable_summary"], str)

    def test_explain_with_preexisting_feature_explanation(
        self, sample_transaction_data
    ):
        """When analysis_result already has feature_explanation, it is used directly."""
        from server import explain_decision_impl

        preexisting = {
            "method": "SHAP",
            "prediction": 0.6,
            "risk_factors": [],
            "protective_factors": [],
            "top_features": [],
        }
        analysis_result = {
            "overall_risk_score": 0.6,
            "risk_level": "HIGH",
            "detected_anomalies": [],
            "feature_explanation": preexisting,
        }
        result = explain_decision_impl(
            analysis_result, transaction_data=sample_transaction_data
        )
        # Should use the pre-existing one, not generate a new one
        assert result["feature_analysis"] == preexisting
        assert result["explainability_method"] == "SHAP"

    def test_explain_with_invalid_transaction_data(self):
        """Invalid transaction_data does not crash, falls back to rule-based."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.3,
            "risk_level": "LOW",
            "detected_anomalies": [],
        }
        # Invalid amount
        invalid_txn = {"amount": -100}
        result = explain_decision_impl(analysis_result, transaction_data=invalid_txn)
        assert isinstance(result, dict)
        assert "decision_summary" in result
        # Should NOT have feature_analysis since validation fails
        assert "feature_analysis" not in result

    def test_explain_with_none_transaction_data(self):
        """Passing None transaction_data is same as not passing it."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.3,
            "risk_level": "LOW",
            "detected_anomalies": [],
        }
        result = explain_decision_impl(analysis_result, transaction_data=None)
        assert result["explainability_method"] == "rule_based"

    def test_explain_decision_key_factors(self):
        """explain_decision_impl populates key_factors from detected_anomalies."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.75,
            "risk_level": "HIGH",
            "detected_anomalies": [
                "high_amount_transaction",
                "unusual_time_pattern",
            ],
        }
        result = explain_decision_impl(analysis_result)
        assert len(result["key_factors"]) == 2
        factors = [f["factor"] for f in result["key_factors"]]
        assert "high_amount_transaction" in factors
        assert "unusual_time_pattern" in factors

    def test_explain_decision_algorithm_contributions(self):
        """explain_decision_impl calculates algorithm contributions from component_scores."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.65,
            "risk_level": "HIGH",
            "detected_anomalies": [],
            "component_scores": {
                "transaction": 0.8,
                "behavioral": 0.5,
                "network": 0.3,
            },
        }
        result = explain_decision_impl(analysis_result)
        contributions = result["algorithm_contributions"]
        assert "transaction" in contributions
        assert "behavioral" in contributions
        assert "network" in contributions
        assert contributions["transaction"]["weight"] == 0.5
        assert contributions["behavioral"]["weight"] == 0.3
        assert contributions["network"]["weight"] == 0.2

    def test_explain_decision_confidence_breakdown(self):
        """explain_decision_impl provides confidence breakdown."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.9,
            "risk_level": "CRITICAL",
            "detected_anomalies": ["high_amount_transaction"],
        }
        result = explain_decision_impl(analysis_result)
        breakdown = result["confidence_breakdown"]
        assert breakdown["model_confidence"] == "High"
        assert breakdown["data_quality"] == "Good"
        assert breakdown["recommendation_strength"] == "Strong"

    def test_explain_decision_alternative_scenarios(self):
        """explain_decision_impl generates alternative scenarios for high risk."""
        from server import explain_decision_impl

        analysis_result = {
            "overall_risk_score": 0.75,
            "risk_level": "HIGH",
            "detected_anomalies": ["high_amount_transaction"],
        }
        result = explain_decision_impl(analysis_result)
        assert len(result["alternative_scenarios"]) >= 2


# =============================================================================
# Feature explanation integration in analyze_transaction_impl
# =============================================================================


class TestFeatureExplanationInTransaction:
    """Test that analyze_transaction_impl includes feature explanations."""

    def test_analyze_transaction_has_feature_explanation(self, sample_transaction_data):
        """analyze_transaction_impl includes feature_explanation when explainer is loaded."""
        from server import analyze_transaction_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        result = analyze_transaction_impl(sample_transaction_data)
        assert "feature_explanation" in result
        assert isinstance(result["feature_explanation"], dict)

    def test_feature_explanation_has_method(self, sample_transaction_data):
        """Feature explanation includes the method used."""
        from server import analyze_transaction_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        result = analyze_transaction_impl(sample_transaction_data)
        fe = result.get("feature_explanation", {})
        assert "method" in fe

    def test_feature_explanation_has_top_features(self, sample_transaction_data):
        """Feature explanation includes top_features list."""
        from server import analyze_transaction_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        result = analyze_transaction_impl(sample_transaction_data)
        fe = result.get("feature_explanation", {})
        assert "top_features" in fe
        assert isinstance(fe["top_features"], list)

    def test_feature_explanation_not_present_when_explainer_none(
        self, sample_transaction_data
    ):
        """When fraud_explainer is None, feature_explanation is absent."""
        from server import analyze_transaction_impl

        with patch("server.fraud_explainer", None):
            result = analyze_transaction_impl(sample_transaction_data)
            assert "feature_explanation" not in result

    def test_high_risk_transaction_explanation(self, high_risk_transaction):
        """High risk transaction includes feature_explanation with risk_factors."""
        from server import analyze_transaction_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")
        result = analyze_transaction_impl(high_risk_transaction)
        if "feature_explanation" in result:
            fe = result["feature_explanation"]
            assert "risk_factors" in fe


# =============================================================================
# Health check and model status explainability fields
# =============================================================================


class TestHealthCheckExplainability:
    """Test explainability fields in health_check_impl."""

    def test_health_check_has_explainability_section(self):
        """health_check_impl includes 'explainability' section."""
        from server import health_check_impl

        result = health_check_impl()
        assert "explainability" in result

    def test_health_check_explainability_available(self):
        """health_check explainability section has 'available' field."""
        from server import health_check_impl

        result = health_check_impl()
        assert "available" in result["explainability"]
        assert isinstance(result["explainability"]["available"], bool)

    def test_health_check_shap_available(self):
        """health_check explainability section has 'shap_available' field."""
        from server import health_check_impl

        result = health_check_impl()
        assert "shap_available" in result["explainability"]
        assert isinstance(result["explainability"]["shap_available"], bool)

    def test_health_check_explainer_loaded(self):
        """health_check explainability section has 'explainer_loaded' field."""
        from server import health_check_impl

        result = health_check_impl()
        assert "explainer_loaded" in result["explainability"]
        assert isinstance(result["explainability"]["explainer_loaded"], bool)

    def test_health_check_fallback_mode(self):
        """health_check explainability section has 'fallback_mode' field."""
        from server import health_check_impl

        result = health_check_impl()
        assert "fallback_mode" in result["explainability"]

    def test_health_check_explainer_in_models(self):
        """health_check models section still has 'explainer' field."""
        from server import health_check_impl

        result = health_check_impl()
        assert "explainer" in result["models"]


class TestModelStatusExplainability:
    """Test explainability fields in get_model_status_impl."""

    def test_model_status_has_explainer(self):
        """get_model_status_impl includes 'explainer' in models section."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        assert "explainer" in result["models"]

    def test_model_status_explainer_loaded(self):
        """get_model_status models.explainer has 'loaded' field."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        explainer_status = result["models"]["explainer"]
        assert "loaded" in explainer_status
        assert isinstance(explainer_status["loaded"], bool)

    def test_model_status_explainer_available(self):
        """get_model_status models.explainer has 'available' field."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        explainer_status = result["models"]["explainer"]
        assert "available" in explainer_status
        assert isinstance(explainer_status["available"], bool)

    def test_model_status_explainer_shap_available(self):
        """get_model_status models.explainer has 'shap_available' field."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        explainer_status = result["models"]["explainer"]
        assert "shap_available" in explainer_status
        assert isinstance(explainer_status["shap_available"], bool)

    def test_model_status_explainer_fallback_mode(self):
        """get_model_status models.explainer has 'fallback_mode' field."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        explainer_status = result["models"]["explainer"]
        assert "fallback_mode" in explainer_status

    def test_model_status_explainer_method(self):
        """get_model_status models.explainer has 'method' field."""
        from server import get_model_status_impl

        result = get_model_status_impl()
        explainer_status = result["models"]["explainer"]
        assert "method" in explainer_status
        assert explainer_status["method"] in (
            "SHAP",
            "Feature Importance",
            "unavailable",
        )


# =============================================================================
# Graceful degradation when explainability unavailable
# =============================================================================


class TestExplainabilityGracefulDegradation:
    """Test behavior when explainability module is unavailable."""

    def test_explain_decision_works_without_explainer(self):
        """explain_decision_impl works when fraud_explainer is None."""
        from server import explain_decision_impl

        with patch("server.fraud_explainer", None):
            analysis_result = {
                "overall_risk_score": 0.5,
                "risk_level": "MEDIUM",
                "detected_anomalies": [],
            }
            result = explain_decision_impl(analysis_result)
            assert "decision_summary" in result
            assert result["explainability_method"] == "rule_based"

    def test_explain_decision_skips_feature_analysis_without_explainer(
        self, sample_transaction_data
    ):
        """No feature_analysis generated when fraud_explainer is None."""
        from server import explain_decision_impl

        with patch("server.fraud_explainer", None):
            analysis_result = {
                "overall_risk_score": 0.5,
                "risk_level": "MEDIUM",
                "detected_anomalies": [],
            }
            result = explain_decision_impl(
                analysis_result, transaction_data=sample_transaction_data
            )
            assert "feature_analysis" not in result

    def test_analyze_transaction_works_without_explainer(self, sample_transaction_data):
        """analyze_transaction_impl works when fraud_explainer is None."""
        from server import analyze_transaction_impl

        with patch("server.fraud_explainer", None):
            result = analyze_transaction_impl(sample_transaction_data)
            assert "overall_risk_score" in result
            assert "risk_level" in result
            # feature_explanation should be absent
            assert "feature_explanation" not in result

    def test_health_check_when_explainer_none(self):
        """health_check_impl shows explainer as not loaded when None."""
        from server import health_check_impl

        with patch("server.fraud_explainer", None):
            result = health_check_impl()
            assert result["models"]["explainer"] is False
            assert result["explainability"]["explainer_loaded"] is False
            assert result["explainability"]["fallback_mode"] is None

    def test_model_status_when_explainer_none(self):
        """get_model_status_impl shows explainer as unavailable when None."""
        from server import get_model_status_impl

        with patch("server.fraud_explainer", None):
            result = get_model_status_impl()
            explainer_status = result["models"]["explainer"]
            assert explainer_status["loaded"] is False
            assert explainer_status["method"] == "unavailable"


# =============================================================================
# End-to-end explainability flow
# =============================================================================


class TestExplainabilityEndToEnd:
    """End-to-end tests for the full explainability pipeline."""

    @pytest.mark.integration
    def test_analyze_then_explain_flow(self, sample_transaction_data):
        """Full flow: analyze transaction, then explain the result."""
        from server import (
            analyze_transaction_impl,
            explain_decision_impl,
            fraud_explainer,
        )

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")

        # Step 1: Analyze transaction
        analysis = analyze_transaction_impl(sample_transaction_data)
        assert "overall_risk_score" in analysis

        # Step 2: Explain the result with original transaction data
        explanation = explain_decision_impl(
            analysis, transaction_data=sample_transaction_data
        )
        assert "decision_summary" in explanation
        assert "feature_analysis" in explanation
        assert "human_readable_summary" in explanation
        assert explanation["explainability_method"] != "rule_based"

    @pytest.mark.integration
    def test_analyze_then_explain_without_transaction_data(
        self, sample_transaction_data
    ):
        """Explain result without transaction_data uses pre-existing feature_explanation."""
        from server import (
            analyze_transaction_impl,
            explain_decision_impl,
            fraud_explainer,
        )

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")

        analysis = analyze_transaction_impl(sample_transaction_data)
        # When we don't pass transaction_data but analysis has feature_explanation,
        # it should still include the pre-existing explanation
        explanation = explain_decision_impl(analysis)
        if "feature_explanation" in analysis:
            assert "feature_analysis" in explanation

    @pytest.mark.integration
    def test_high_risk_explain_flow(self, high_risk_transaction):
        """Full flow with high-risk transaction produces detailed explanation."""
        from server import (
            analyze_transaction_impl,
            explain_decision_impl,
            fraud_explainer,
        )

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")

        analysis = analyze_transaction_impl(high_risk_transaction)
        explanation = explain_decision_impl(
            analysis, transaction_data=high_risk_transaction
        )
        assert "decision_summary" in explanation
        # High risk should mention risk factors
        if analysis.get("detected_anomalies"):
            assert len(explanation["key_factors"]) > 0

    @pytest.mark.integration
    def test_batch_analyze_then_explain(
        self, sample_transaction_data, high_risk_transaction
    ):
        """Analyze a batch, then explain individual results."""
        from server import analyze_batch_impl, explain_decision_impl, fraud_explainer

        if fraud_explainer is None:
            pytest.skip("FraudExplainer not available")

        batch_result = analyze_batch_impl(
            [sample_transaction_data, high_risk_transaction]
        )
        assert batch_result["batch_size"] == 2

        # Explain each result
        for individual in batch_result["results"]:
            explanation = explain_decision_impl(individual)
            assert "decision_summary" in explanation
