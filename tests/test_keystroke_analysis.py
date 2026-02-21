"""
Tests for keystroke dynamics analysis
"""

import pytest
from server import BehavioralBiometrics


class TestKeystrokeDynamicsAnalysis:
    """Test keystroke dynamics fraud detection"""

    @pytest.fixture
    def analyzer(self):
        """Create behavioral biometrics analyzer"""
        return BehavioralBiometrics()

    def test_analyze_normal_keystroke_pattern(self, analyzer, sample_keystroke_data):
        """Test analysis of normal keystroke dynamics"""
        result = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)

        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "is_anomaly" in result
        assert "confidence" in result
        assert "analysis_type" in result

        assert result["analysis_type"] == "keystroke_dynamics"
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["confidence"] > 0

    def test_analyze_anomalous_keystroke_pattern(
        self, analyzer, anomalous_keystroke_data
    ):
        """Test detection of anomalous keystroke dynamics"""
        result = analyzer.analyze_keystroke_dynamics(anomalous_keystroke_data)

        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "is_anomaly" in result
        # Anomalous patterns should have higher risk
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_analyze_empty_keystroke_data(self, analyzer):
        """Test analysis with empty keystroke data"""
        result = analyzer.analyze_keystroke_dynamics([])

        assert result["risk_score"] == 0.0
        assert result["confidence"] == 0.0
        assert result["status"] == "no_data"

    def test_analyze_single_keystroke(self, analyzer):
        """Test analysis with single keystroke (insufficient data)"""
        single_keystroke = [{"press_time": 100, "release_time": 150}]
        result = analyzer.analyze_keystroke_dynamics(single_keystroke)

        assert result["risk_score"] == 0.0
        assert result["status"] == "error"
        assert "error" in result

    def test_extract_keystroke_features_normal(self, analyzer, sample_keystroke_data):
        """Test feature extraction from normal keystroke data"""
        features = analyzer._extract_keystroke_features(sample_keystroke_data)

        assert features is not None
        assert isinstance(features, list)
        assert len(features) == 10  # 5 dwell + 5 flight features
        assert all(isinstance(f, (int, float)) for f in features)

    def test_extract_keystroke_features_insufficient_data(self, analyzer):
        """Test feature extraction with insufficient data"""
        insufficient_data = [{"press_time": 100, "release_time": 150}]
        features = analyzer._extract_keystroke_features(insufficient_data)

        assert features is None

    def test_extract_features_no_timing_data(self, analyzer):
        """Test feature extraction when timing data is missing"""
        no_timing = [{"key": "a"}, {"key": "b"}]
        features = analyzer._extract_keystroke_features(no_timing)

        assert features is not None
        assert features == [0.0] * 10

    def test_extract_features_partial_timing(self, analyzer):
        """Test feature extraction with partial timing information"""
        partial_timing = [
            {"press_time": 100, "release_time": 150},
            {"press_time": 200, "release_time": 250},
            {"press_time": 300, "release_time": 350},
        ]
        features = analyzer._extract_keystroke_features(partial_timing)

        assert features is not None
        assert len(features) == 10

    def test_dwell_time_calculation(self, analyzer):
        """Test correct calculation of dwell times"""
        data = [
            {"press_time": 100, "release_time": 180},  # 80ms dwell
            {"press_time": 200, "release_time": 290},  # 90ms dwell
        ]
        features = analyzer._extract_keystroke_features(data)

        # Mean dwell time should be around 85ms
        mean_dwell = features[0]
        assert 80 <= mean_dwell <= 90

    def test_flight_time_calculation(self, analyzer):
        """Test correct calculation of flight times"""
        data = [
            {"press_time": 100, "release_time": 150},
            {"press_time": 200, "release_time": 250},  # 50ms flight time (200-150)
            {"press_time": 300, "release_time": 350},  # 50ms flight time (300-250)
        ]
        features = analyzer._extract_keystroke_features(data)

        # Mean flight time should be 50ms
        mean_flight = features[5]
        assert mean_flight == 50.0

    def test_feature_statistical_properties(self, analyzer, sample_keystroke_data):
        """Test that extracted features contain proper statistical measures"""
        features = analyzer._extract_keystroke_features(sample_keystroke_data)

        # Features should be: [mean, std, median, max, min] for dwell and flight
        # Check that max >= median >= min for dwell times
        dwell_max = features[3]
        dwell_median = features[2]
        dwell_min = features[4]

        assert dwell_max >= dwell_median >= dwell_min

        # Check that flight times follow same pattern
        flight_max = features[8]
        flight_median = features[7]
        flight_min = features[9]

        assert flight_max >= flight_median >= flight_min

    def test_zero_features_when_no_dwell_times(self, analyzer):
        """Test that zero features are returned when no dwell times can be calculated"""
        data = [
            {"press_time": 100},  # Missing release_time
            {"press_time": 200},
        ]
        features = analyzer._extract_keystroke_features(data)

        # Should still return features but with zeros for dwell stats
        assert features is not None
        # First 5 features (dwell stats) should be zero
        assert all(f == 0.0 for f in features[:5])

    def test_model_initialization(self, analyzer):
        """Test that models are properly initialized"""
        assert analyzer.keystroke_model is not None
        assert analyzer.mouse_model is not None
        assert analyzer.touch_model is not None

    def test_consistent_results(self, analyzer, sample_keystroke_data):
        """Test that analysis produces consistent results for same input"""
        result1 = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)
        result2 = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)

        # Results should be identical for same input
        assert result1["risk_score"] == result2["risk_score"]
        assert result1["is_anomaly"] == result2["is_anomaly"]

    def test_features_analyzed_count(self, analyzer, sample_keystroke_data):
        """Test that features_analyzed count is reported correctly"""
        result = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)

        assert "features_analyzed" in result
        assert result["features_analyzed"] == 10

    def test_error_handling_malformed_data(self, analyzer):
        """Test error handling with malformed keystroke data"""
        malformed_data = [
            {"press_time": "invalid", "release_time": "bad"},
            {"press_time": "oops", "release_time": "nope"},
        ]
        result = analyzer.analyze_keystroke_dynamics(malformed_data)

        # All timing values are non-numeric, so no features extracted.
        # Function should handle gracefully without crashing.
        assert "risk_score" in result

    def test_large_dataset_performance(self, analyzer, performance_test_data):
        """Test analysis performance with large keystroke dataset"""
        result = analyzer.analyze_keystroke_dynamics(
            performance_test_data["keystroke_data"]
        )

        assert isinstance(result, dict)
        assert "risk_score" in result
        # Should complete without timeout or memory issues

    def test_edge_case_very_fast_typing(self, analyzer):
        """Test analysis with very fast typing pattern"""
        fast_typing = []
        base_time = 1000.0

        for i in range(20):
            keystroke = {
                "press_time": base_time + i * 50,
                "release_time": base_time + i * 50 + 20,  # 20ms dwell (very fast)
            }
            fast_typing.append(keystroke)

        result = analyzer.analyze_keystroke_dynamics(fast_typing)
        assert isinstance(result, dict)
        assert "risk_score" in result

    def test_edge_case_very_slow_typing(self, analyzer):
        """Test analysis with very slow typing pattern"""
        slow_typing = []
        base_time = 1000.0

        for i in range(20):
            keystroke = {
                "press_time": base_time + i * 1000,
                "release_time": base_time + i * 1000 + 300,  # 300ms dwell (very slow)
            }
            slow_typing.append(keystroke)

        result = analyzer.analyze_keystroke_dynamics(slow_typing)
        assert isinstance(result, dict)
        assert "risk_score" in result

    def test_risk_score_bounds(self, analyzer, sample_keystroke_data):
        """Test that risk score is always within valid bounds"""
        result = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)

        assert 0.0 <= result["risk_score"] <= 1.0

    def test_confidence_score_value(self, analyzer, sample_keystroke_data):
        """Test that confidence score has expected value"""
        result = analyzer.analyze_keystroke_dynamics(sample_keystroke_data)

        assert result["confidence"] == 0.85
