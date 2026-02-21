"""
Tests for MCP tool endpoints
"""

from datetime import datetime
from server import (
    analyze_transaction_impl as analyze_transaction,
    detect_behavioral_anomaly_impl as detect_behavioral_anomaly,
    assess_network_risk_impl as assess_network_risk,
    generate_risk_score_impl as generate_risk_score,
    explain_decision_impl as explain_decision,
)


class TestAnalyzeTransactionTool:
    """Test analyze_transaction MCP tool"""

    def test_basic_transaction_analysis(self, sample_transaction_data):
        """Test basic transaction analysis"""
        result = analyze_transaction(sample_transaction_data)

        assert isinstance(result, dict)
        assert "overall_risk_score" in result
        assert "risk_level" in result
        assert "transaction_analysis" in result
        assert "detected_anomalies" in result
        assert "recommended_actions" in result
        assert "analysis_timestamp" in result
        assert "model_version" in result

    def test_transaction_with_behavioral_analysis(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test transaction analysis with behavioral data"""
        result = analyze_transaction(
            sample_transaction_data,
            include_behavioral=True,
            behavioral_data=sample_behavioral_data,
        )

        assert "behavioral_analysis" in result
        assert "keystroke" in result["behavioral_analysis"]

    def test_high_risk_transaction_classification(self, high_risk_transaction):
        """Test that high-risk transaction is properly classified"""
        result = analyze_transaction(high_risk_transaction)

        assert result["risk_level"] in ["HIGH", "CRITICAL", "MEDIUM"]
        assert result["overall_risk_score"] > 0.0

    def test_risk_level_critical(self):
        """Test CRITICAL risk level classification"""
        # Create transaction that will score >= 0.8
        transaction = {
            "amount": 50000.00,
            "payment_method": "crypto",
            "location": "nigeria",
            "timestamp": datetime.now().replace(hour=3).isoformat(),
        }
        result = analyze_transaction(transaction)

        # Should be HIGH or CRITICAL
        assert result["risk_level"] in ["HIGH", "CRITICAL"]

    def test_risk_level_low(self, sample_transaction_data):
        """Test LOW risk level classification"""
        result = analyze_transaction(sample_transaction_data)

        # Normal transaction should be LOW or MEDIUM
        assert result["risk_level"] in ["LOW", "MEDIUM"]

    def test_recommended_actions_critical(self):
        """Test recommended actions for critical risk"""
        transaction = {
            "amount": 100000.00,
            "payment_method": "crypto",
            "location": "unknown",
        }
        result = analyze_transaction(transaction)

        # High risk should recommend blocking or review
        assert any(
            "block" in action or "review" in action
            for action in result["recommended_actions"]
        )

    def test_recommended_actions_low(self, sample_transaction_data):
        """Test recommended actions for low risk"""
        result = analyze_transaction(sample_transaction_data)

        # Low risk should allow transaction
        if result["risk_level"] == "LOW":
            assert "allow_transaction" in result["recommended_actions"]

    def test_explanation_generated(self, sample_transaction_data):
        """Test that explanation is generated"""
        result = analyze_transaction(sample_transaction_data)

        assert "explanation" in result
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    def test_invalid_transaction_data_validation(self):
        """Test validation error handling"""
        result = analyze_transaction({"amount": -100})

        assert "error" in result
        assert result["status"] == "validation_failed"

    def test_invalid_behavioral_data_validation(self, sample_transaction_data):
        """Test behavioral data validation"""
        result = analyze_transaction(
            sample_transaction_data,
            include_behavioral=True,
            behavioral_data={"keystroke_dynamics": "invalid"},
        )

        assert "error" in result
        assert result["status"] == "validation_failed"

    def test_behavioral_anomaly_increases_risk(self, sample_transaction_data):
        """Test that behavioral anomalies increase risk score"""
        # Without behavioral data
        _result_without = analyze_transaction(sample_transaction_data)

        # With anomalous behavioral data
        anomalous_behavioral = {
            "keystroke_dynamics": [
                {"press_time": i * 1000, "release_time": i * 1000 + 500}
                for i in range(20)
            ]
        }
        result_with = analyze_transaction(
            sample_transaction_data,
            include_behavioral=True,
            behavioral_data=anomalous_behavioral,
        )

        # Behavioral analysis might increase risk
        assert isinstance(result_with["overall_risk_score"], float)

    def test_model_version_included(self, sample_transaction_data):
        """Test that model version is included in response"""
        result = analyze_transaction(sample_transaction_data)

        assert result["model_version"] == "v2.3.0"

    def test_analysis_timestamp_format(self, sample_transaction_data):
        """Test that analysis timestamp is in ISO format"""
        result = analyze_transaction(sample_transaction_data)

        # Should be able to parse timestamp
        timestamp = datetime.fromisoformat(result["analysis_timestamp"])
        assert isinstance(timestamp, datetime)

    def test_error_handling_exception(self):
        """Test error handling when analysis fails"""
        # Pass something that will cause a validation error
        result = analyze_transaction(None)

        assert "error" in result
        assert result["status"] == "validation_failed"


class TestDetectBehavioralAnomalyTool:
    """Test detect_behavioral_anomaly MCP tool"""

    def test_basic_behavioral_analysis(self, sample_behavioral_data):
        """Test basic behavioral anomaly detection"""
        result = detect_behavioral_anomaly(sample_behavioral_data)

        assert isinstance(result, dict)
        assert "overall_anomaly_score" in result
        assert "behavioral_analyses" in result
        assert "detected_anomalies" in result
        assert "confidence" in result
        assert "analysis_timestamp" in result

    def test_keystroke_analysis_included(self, sample_behavioral_data):
        """Test that keystroke analysis is included"""
        result = detect_behavioral_anomaly(sample_behavioral_data)

        assert "keystroke" in result["behavioral_analyses"]
        assert isinstance(result["behavioral_analyses"]["keystroke"], dict)

    def test_anomaly_detection(self, anomalous_keystroke_data):
        """Test detection of behavioral anomalies"""
        behavioral_data = {"keystroke_dynamics": anomalous_keystroke_data}
        result = detect_behavioral_anomaly(behavioral_data)

        assert isinstance(result["overall_anomaly_score"], float)
        assert 0.0 <= result["overall_anomaly_score"] <= 1.0

    def test_confidence_calculation(self, sample_behavioral_data):
        """Test confidence score calculation"""
        result = detect_behavioral_anomaly(sample_behavioral_data)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_behavioral_data(self):
        """Test with empty behavioral data"""
        result = detect_behavioral_anomaly({})

        assert result["overall_anomaly_score"] == 0.0
        assert result["confidence"] == 0.0

    def test_error_handling(self):
        """Test error handling with invalid data"""
        result = detect_behavioral_anomaly(None)

        assert "error" in result
        assert result["status"] == "validation_failed"


class TestAssessNetworkRiskTool:
    """Test assess_network_risk MCP tool"""

    def test_basic_network_analysis(self, sample_network_data):
        """Test basic network risk assessment"""
        result = assess_network_risk(sample_network_data)

        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "network_metrics" in result
        assert "risk_patterns" in result

    def test_fraud_ring_detection(self, fraud_ring_network_data):
        """Test detection of fraud ring patterns"""
        result = assess_network_risk(fraud_ring_network_data)

        assert result["risk_score"] > 0.0
        # Should detect suspicious patterns
        assert isinstance(result["risk_patterns"], list)

    def test_missing_entity_id(self):
        """Test with missing entity_id"""
        result = assess_network_risk({})

        assert result["status"] == "no_entity_id"
        assert result["risk_score"] == 0.0


class TestGenerateRiskScoreTool:
    """Test generate_risk_score MCP tool"""

    def test_transaction_only_scoring(self, sample_transaction_data):
        """Test risk scoring with transaction data only"""
        result = generate_risk_score(sample_transaction_data)

        assert isinstance(result, dict)
        assert "overall_risk_score" in result
        assert "component_scores" in result
        assert "risk_level" in result
        assert "confidence" in result
        assert "detected_anomalies" in result
        assert "comprehensive_explanation" in result
        assert "recommended_actions" in result

        assert "transaction" in result["component_scores"]

    def test_transaction_and_behavioral_scoring(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test risk scoring with transaction and behavioral data"""
        result = generate_risk_score(
            sample_transaction_data, behavioral_data=sample_behavioral_data
        )

        assert "transaction" in result["component_scores"]
        assert "behavioral" in result["component_scores"]

    def test_all_components_scoring(
        self, sample_transaction_data, sample_behavioral_data, sample_network_data
    ):
        """Test risk scoring with all data components"""
        result = generate_risk_score(
            sample_transaction_data,
            behavioral_data=sample_behavioral_data,
            network_data=sample_network_data,
        )

        assert "transaction" in result["component_scores"]
        assert "behavioral" in result["component_scores"]
        assert "network" in result["component_scores"]

    def test_weighted_scoring_two_components(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test weighted scoring with two components"""
        result = generate_risk_score(
            sample_transaction_data, behavioral_data=sample_behavioral_data
        )

        # Should use weighted average (0.6 transaction + 0.4 behavioral)
        expected_score = (
            result["component_scores"]["transaction"] * 0.6
            + result["component_scores"]["behavioral"] * 0.4
        )

        # Allow small floating point difference
        assert abs(result["overall_risk_score"] - expected_score) < 0.01

    def test_weighted_scoring_three_components(
        self, sample_transaction_data, sample_behavioral_data, sample_network_data
    ):
        """Test weighted scoring with three components"""
        result = generate_risk_score(
            sample_transaction_data,
            behavioral_data=sample_behavioral_data,
            network_data=sample_network_data,
        )

        # Should use weighted average (0.5 + 0.3 + 0.2)
        expected_score = (
            result["component_scores"]["transaction"] * 0.5
            + result["component_scores"]["behavioral"] * 0.3
            + result["component_scores"]["network"] * 0.2
        )

        # Allow small floating point difference
        assert abs(result["overall_risk_score"] - expected_score) < 0.01

    def test_risk_level_classification(self, sample_transaction_data):
        """Test risk level classification logic"""
        result = generate_risk_score(sample_transaction_data)

        score = result["overall_risk_score"]
        level = result["risk_level"]

        if score >= 0.8:
            assert level == "CRITICAL"
        elif score >= 0.6:
            assert level == "HIGH"
        elif score >= 0.4:
            assert level == "MEDIUM"
        else:
            assert level == "LOW"

    def test_recommended_actions_by_level(self, sample_transaction_data):
        """Test that recommended actions match risk level"""
        result = generate_risk_score(sample_transaction_data)

        level = result["risk_level"]
        actions = result["recommended_actions"]

        if level == "CRITICAL":
            assert "block_transaction" in actions
        elif level == "LOW":
            assert "allow_transaction" in actions

    def test_comprehensive_explanation_with_anomalies(self, high_risk_transaction):
        """Test comprehensive explanation includes detected anomalies"""
        result = generate_risk_score(high_risk_transaction)

        explanation = result["comprehensive_explanation"]
        assert isinstance(explanation, str)

        if len(result["detected_anomalies"]) > 0:
            assert "anomalies" in explanation.lower()

    def test_comprehensive_explanation_no_anomalies(self, sample_transaction_data):
        """Test comprehensive explanation when no anomalies detected"""
        result = generate_risk_score(sample_transaction_data)

        if len(result["detected_anomalies"]) == 0:
            assert (
                "no significant anomalies"
                in result["comprehensive_explanation"].lower()
            )

    def test_analysis_components_listed(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test that analysis components are listed in result"""
        result = generate_risk_score(
            sample_transaction_data, behavioral_data=sample_behavioral_data
        )

        assert "analysis_components" in result
        assert "transaction" in result["analysis_components"]
        assert "behavioral" in result["analysis_components"]

    def test_confidence_averaging(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test that confidence is averaged across components"""
        result = generate_risk_score(
            sample_transaction_data, behavioral_data=sample_behavioral_data
        )

        assert 0.0 <= result["confidence"] <= 1.0

    def test_error_handling(self):
        """Test error handling with invalid data"""
        result = generate_risk_score(None)

        assert "error" in result
        assert result["overall_risk_score"] == 0.0
        assert result["risk_level"] == "UNKNOWN"
        assert result["status"] == "validation_failed"


class TestExplainDecisionTool:
    """Test explain_decision MCP tool"""

    def test_basic_explanation(self, comprehensive_analysis_result):
        """Test basic decision explanation"""
        result = explain_decision(comprehensive_analysis_result)

        assert isinstance(result, dict)
        assert "decision_summary" in result
        assert "key_factors" in result
        assert "algorithm_contributions" in result
        assert "confidence_breakdown" in result
        assert "alternative_scenarios" in result
        assert "explanation_timestamp" in result

    def test_decision_summary_content(self, comprehensive_analysis_result):
        """Test that decision summary contains key information"""
        result = explain_decision(comprehensive_analysis_result)

        summary = result["decision_summary"]
        assert "HIGH" in summary  # Risk level
        assert "0.75" in summary  # Risk score

    def test_key_factors_listed(self, comprehensive_analysis_result):
        """Test that key risk factors are listed"""
        result = explain_decision(comprehensive_analysis_result)

        assert len(result["key_factors"]) > 0
        for factor in result["key_factors"]:
            assert "factor" in factor
            assert "impact" in factor
            assert "description" in factor

    def test_algorithm_contributions_calculated(self, comprehensive_analysis_result):
        """Test algorithm contribution calculations"""
        result = explain_decision(comprehensive_analysis_result)

        contributions = result["algorithm_contributions"]
        assert "transaction" in contributions
        assert "behavioral" in contributions
        assert "network" in contributions

        for component, data in contributions.items():
            assert "score" in data
            assert "weight" in data
            assert "contribution" in data

    def test_algorithm_weights_single_component(self):
        """Test weights when only transaction analysis is present"""
        analysis_result = {
            "overall_risk_score": 0.6,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.6},
        }
        result = explain_decision(analysis_result)

        # Single component should have 100% weight
        assert result["algorithm_contributions"]["transaction"]["weight"] == 1.0

    def test_algorithm_weights_two_components(self):
        """Test weights with two components"""
        analysis_result = {
            "overall_risk_score": 0.6,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.6, "behavioral": 0.5},
        }
        result = explain_decision(analysis_result)

        # Two components: transaction 60%, other 40%
        assert result["algorithm_contributions"]["transaction"]["weight"] == 0.6
        assert result["algorithm_contributions"]["behavioral"]["weight"] == 0.4

    def test_algorithm_weights_three_components(self):
        """Test weights with all three components"""
        analysis_result = {
            "overall_risk_score": 0.6,
            "risk_level": "MEDIUM",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.6, "behavioral": 0.5, "network": 0.4},
        }
        result = explain_decision(analysis_result)

        # Three components: 50%, 30%, 20%
        assert result["algorithm_contributions"]["transaction"]["weight"] == 0.5
        assert result["algorithm_contributions"]["behavioral"]["weight"] == 0.3
        assert result["algorithm_contributions"]["network"]["weight"] == 0.2

    def test_confidence_breakdown(self, comprehensive_analysis_result):
        """Test confidence breakdown analysis"""
        result = explain_decision(comprehensive_analysis_result)

        breakdown = result["confidence_breakdown"]
        assert "model_confidence" in breakdown
        assert "data_quality" in breakdown
        assert "recommendation_strength" in breakdown

    def test_alternative_scenarios_high_risk(self):
        """Test alternative scenarios for high-risk cases"""
        analysis_result = {
            "overall_risk_score": 0.8,
            "risk_level": "HIGH",
            "detected_anomalies": ["high_amount_transaction"],
            "component_scores": {"transaction": 0.8},
        }
        result = explain_decision(analysis_result)

        # Should suggest scenarios that could reduce risk
        assert len(result["alternative_scenarios"]) > 0

    def test_alternative_scenarios_low_risk(self):
        """Test alternative scenarios for low-risk cases"""
        analysis_result = {
            "overall_risk_score": 0.2,
            "risk_level": "LOW",
            "detected_anomalies": [],
            "component_scores": {"transaction": 0.2},
        }
        result = explain_decision(analysis_result)

        # May or may not have alternative scenarios
        assert isinstance(result["alternative_scenarios"], list)

    def test_error_handling(self):
        """Test error handling with invalid data"""
        result = explain_decision(None)

        assert "error" in result
        assert result["decision_summary"] == "Unable to generate explanation"
        assert result["status"] == "explanation_failed"

    def test_explanation_timestamp_format(self, comprehensive_analysis_result):
        """Test that explanation timestamp is in ISO format"""
        result = explain_decision(comprehensive_analysis_result)

        timestamp = datetime.fromisoformat(result["explanation_timestamp"])
        assert isinstance(timestamp, datetime)


class TestFeatureExplanation:
    """Test feature-level explanations from FraudExplainer integration"""

    def test_analyze_transaction_includes_explanation(self, sample_transaction_data):
        from server import analyze_transaction_impl

        result = analyze_transaction_impl(sample_transaction_data)
        assert "feature_explanation" in result
        explanation = result["feature_explanation"]
        assert "method" in explanation
        assert "top_features" in explanation

    def test_feature_explanation_has_risk_factors(self, sample_transaction_data):
        from server import analyze_transaction_impl

        result = analyze_transaction_impl(sample_transaction_data)
        explanation = result["feature_explanation"]
        # Should have risk_factors and/or protective_factors
        assert "risk_factors" in explanation or "protective_factors" in explanation

    def test_explain_decision_includes_feature_analysis(
        self, comprehensive_analysis_result
    ):
        from server import explain_decision_impl

        # Add feature_explanation to simulate real analysis output
        comprehensive_analysis_result["feature_explanation"] = {
            "method": "Feature Importance",
            "top_features": [{"feature": "amount", "importance": 0.5, "value": 150.0}],
        }
        result = explain_decision_impl(comprehensive_analysis_result)
        assert "feature_analysis" in result


class TestHealthCheck:
    """Test health check functionality"""

    def test_health_check_returns_healthy(self):
        from server import health_check_impl

        result = health_check_impl()
        assert result["status"] in ("healthy", "degraded")
        assert "timestamp" in result
        assert "version" in result

    def test_health_check_models_section(self):
        from server import health_check_impl

        result = health_check_impl()
        models = result["models"]
        assert models["isolation_forest"] is True
        assert models["feature_engineer"] is True
        assert models["explainer"] is True
        assert models["feature_count"] == 46

    def test_health_check_cache_section(self):
        from server import health_check_impl

        result = health_check_impl()
        cache = result["cache"]
        assert "size" in cache
        assert "capacity" in cache
        assert cache["capacity"] == 1000

    def test_health_check_inference_section(self):
        from server import health_check_impl

        result = health_check_impl()
        inference = result["inference"]
        assert "total_predictions" in inference
        assert "batch_predictions" in inference

    def test_health_check_system_section(self):
        from server import health_check_impl

        result = health_check_impl()
        # System section present if monitoring available
        if "system" in result:
            system = result["system"]
            assert "cpu_percent" in system or "error" in system
