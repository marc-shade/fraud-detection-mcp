"""
Integration tests for complete fraud detection workflows
"""

import pytest
from datetime import datetime
from server_wrapper import (
    analyze_transaction,
    detect_behavioral_anomaly,
    assess_network_risk,
    generate_risk_score,
    explain_decision
)


class TestCompleteWorkflows:
    """Test complete fraud detection workflows"""

    def test_low_risk_transaction_workflow(self, sample_transaction_data):
        """Test complete workflow for low-risk transaction"""
        # Step 1: Analyze transaction
        result = analyze_transaction(sample_transaction_data)

        assert result['risk_level'] in ['LOW', 'MEDIUM']
        assert 'allow_transaction' in result['recommended_actions'] or \
               'monitor_closely' in result['recommended_actions']

    def test_high_risk_transaction_workflow(self, high_risk_transaction):
        """Test complete workflow for high-risk transaction"""
        # Step 1: Analyze transaction
        result = analyze_transaction(high_risk_transaction)

        assert result['risk_level'] in ['HIGH', 'CRITICAL']
        assert len(result['detected_anomalies']) > 0

        # Step 2: Generate comprehensive risk score
        comprehensive = generate_risk_score(high_risk_transaction)

        assert comprehensive['overall_risk_score'] > 0.4

        # Step 3: Explain the decision
        explanation = explain_decision(comprehensive)

        assert len(explanation['key_factors']) > 0
        assert 'decision_summary' in explanation

    def test_transaction_with_behavioral_workflow(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test workflow combining transaction and behavioral analysis"""
        # Step 1: Standalone behavioral analysis
        behavioral_result = detect_behavioral_anomaly(sample_behavioral_data)

        assert isinstance(behavioral_result, dict)
        assert 'overall_anomaly_score' in behavioral_result

        # Step 2: Combined analysis
        combined_result = analyze_transaction(
            sample_transaction_data,
            include_behavioral=True,
            behavioral_data=sample_behavioral_data
        )

        assert 'behavioral_analysis' in combined_result
        assert 'transaction_analysis' in combined_result

        # Step 3: Generate comprehensive score
        comprehensive = generate_risk_score(
            sample_transaction_data,
            behavioral_data=sample_behavioral_data
        )

        assert 'behavioral' in comprehensive['component_scores']
        assert 'transaction' in comprehensive['component_scores']

    def test_full_multi_modal_workflow(
        self, sample_transaction_data, sample_behavioral_data, sample_network_data
    ):
        """Test complete workflow with all detection methods"""
        # Step 1: Individual analyses
        transaction_result = analyze_transaction(sample_transaction_data)
        behavioral_result = detect_behavioral_anomaly(sample_behavioral_data)
        network_result = assess_network_risk(sample_network_data)

        # All should complete successfully
        assert isinstance(transaction_result, dict)
        assert isinstance(behavioral_result, dict)
        assert isinstance(network_result, dict)

        # Step 2: Comprehensive risk score
        comprehensive = generate_risk_score(
            sample_transaction_data,
            behavioral_data=sample_behavioral_data,
            network_data=sample_network_data
        )

        # Should have all three components
        assert len(comprehensive['component_scores']) == 3
        assert all(component in comprehensive['component_scores']
                  for component in ['transaction', 'behavioral', 'network'])

        # Step 3: Explain the decision
        explanation = explain_decision(comprehensive)

        # Should explain all components
        assert 'transaction' in explanation['algorithm_contributions']
        assert 'behavioral' in explanation['algorithm_contributions']
        assert 'network' in explanation['algorithm_contributions']

    def test_fraud_ring_detection_workflow(self, fraud_ring_network_data):
        """Test workflow for detecting fraud ring"""
        # Analyze network pattern
        result = assess_network_risk(fraud_ring_network_data)

        # Should detect suspicious patterns
        assert result['risk_score'] > 0.0
        assert len(result['risk_patterns']) > 0

        # Verify high connectivity was detected
        assert 'unusually_high_connectivity' in result['risk_patterns'] or \
               'potential_fraud_hub' in result['risk_patterns']

    def test_escalating_risk_workflow(self):
        """Test workflow where risk escalates through multiple signals"""
        # Start with borderline transaction
        transaction = {
            'amount': 5000.00,  # Moderate amount
            'merchant': 'Test Merchant',
            'location': 'United States',
            'timestamp': datetime.now().isoformat()
        }

        # Step 1: Transaction analysis (borderline)
        result1 = analyze_transaction(transaction)
        initial_risk = result1['overall_risk_score']

        # Step 2: Add suspicious behavioral pattern
        suspicious_behavioral = {
            'keystroke_dynamics': [
                {'press_time': i * 1000, 'release_time': i * 1000 + 500}
                for i in range(20)
            ]
        }

        result2 = analyze_transaction(
            transaction,
            include_behavioral=True,
            behavioral_data=suspicious_behavioral
        )

        # Risk might increase with behavioral anomaly
        assert isinstance(result2['overall_risk_score'], float)

        # Step 3: Add network risk
        network_data = {
            'entity_id': 'test_user',
            'connections': [
                {'entity_id': f'suspicious_{i}', 'strength': 0.9}
                for i in range(30)
            ]
        }

        comprehensive = generate_risk_score(
            transaction,
            behavioral_data=suspicious_behavioral,
            network_data=network_data
        )

        # Combined risk should reflect multiple signals
        assert comprehensive['overall_risk_score'] > 0.0

    def test_false_positive_mitigation_workflow(self, sample_transaction_data):
        """Test that normal patterns don't trigger false positives"""
        # Analyze clearly normal transaction
        result = analyze_transaction(sample_transaction_data)

        # Should be low risk with few/no anomalies
        assert result['risk_level'] in ['LOW', 'MEDIUM']

        # Recommend allowing the transaction
        if result['risk_level'] == 'LOW':
            assert 'allow_transaction' in result['recommended_actions']

    def test_explainability_workflow(self, high_risk_transaction):
        """Test complete explainability workflow"""
        # Generate risk assessment
        risk_assessment = generate_risk_score(high_risk_transaction)

        # Get explanation
        explanation = explain_decision(risk_assessment)

        # Verify explanation completeness
        assert explanation['decision_summary'] != ''
        assert len(explanation['key_factors']) > 0

        # Each factor should have proper structure
        for factor in explanation['key_factors']:
            assert 'factor' in factor
            assert 'impact' in factor
            assert 'description' in factor

        # Should explain why each algorithm contributed
        for component, contribution in explanation['algorithm_contributions'].items():
            assert contribution['contribution'] != ''
            assert 0.0 <= contribution['weight'] <= 1.0

    def test_progressive_disclosure_workflow(self):
        """Test progressive disclosure of information"""
        transaction = {'amount': 10000.00, 'payment_method': 'crypto'}

        # Level 1: Quick transaction analysis
        quick_result = analyze_transaction(transaction)
        assert 'overall_risk_score' in quick_result

        # Level 2: Comprehensive analysis
        comprehensive = generate_risk_score(transaction)
        assert 'component_scores' in comprehensive

        # Level 3: Detailed explanation
        explanation = explain_decision(comprehensive)
        assert 'algorithm_contributions' in explanation
        assert 'alternative_scenarios' in explanation

    def test_multi_transaction_workflow(self):
        """Test analyzing multiple transactions in sequence"""
        transactions = [
            {'amount': 100.00, 'merchant': 'Store A'},
            {'amount': 200.00, 'merchant': 'Store B'},
            {'amount': 50000.00, 'merchant': 'Store C', 'payment_method': 'crypto'},
        ]

        results = []
        for transaction in transactions:
            result = analyze_transaction(transaction)
            results.append(result)

        # All should complete
        assert len(results) == 3

        # Last transaction should have higher risk
        assert results[2]['overall_risk_score'] > results[0]['overall_risk_score']

    def test_real_time_analysis_workflow(self, sample_transaction_data):
        """Test workflow simulating real-time fraud detection"""
        import time

        start_time = time.time()

        # Real-time analysis should be fast
        result = analyze_transaction(sample_transaction_data)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 1 second for single transaction)
        assert elapsed_time < 1.0
        assert isinstance(result, dict)
        assert 'overall_risk_score' in result

    def test_batch_analysis_workflow(self):
        """Test analyzing multiple transactions efficiently"""
        transactions = [
            {'amount': float(i * 100), 'merchant': f'Merchant_{i}'}
            for i in range(10)
        ]

        results = [analyze_transaction(t) for t in transactions]

        # All should complete successfully
        assert len(results) == 10
        assert all('overall_risk_score' in r for r in results)

    def test_decision_audit_trail_workflow(self, sample_transaction_data):
        """Test that complete decision audit trail is maintained"""
        # Analyze transaction
        result = analyze_transaction(sample_transaction_data)

        # Verify audit trail components
        assert 'analysis_timestamp' in result
        assert 'model_version' in result
        assert 'transaction_analysis' in result

        # Generate comprehensive report
        comprehensive = generate_risk_score(sample_transaction_data)

        assert 'analysis_timestamp' in comprehensive
        assert 'analysis_components' in comprehensive

        # Get explanation
        explanation = explain_decision(comprehensive)

        assert 'explanation_timestamp' in explanation

        # All timestamps should be valid
        for timestamp in [
            result['analysis_timestamp'],
            comprehensive['analysis_timestamp'],
            explanation['explanation_timestamp']
        ]:
            dt = datetime.fromisoformat(timestamp)
            assert isinstance(dt, datetime)

    def test_recommendation_consistency_workflow(self, sample_transaction_data):
        """Test that recommendations are consistent with risk level"""
        result = analyze_transaction(sample_transaction_data)

        risk_level = result['risk_level']
        actions = result['recommended_actions']

        # Verify consistency
        if risk_level == 'CRITICAL':
            assert any('block' in action for action in actions)
        elif risk_level == 'HIGH':
            assert any('review' in action or 'verification' in action
                      for action in actions)
        elif risk_level == 'LOW':
            assert 'allow_transaction' in actions

    def test_confidence_reporting_workflow(
        self, sample_transaction_data, sample_behavioral_data
    ):
        """Test that confidence scores are properly reported"""
        # Single component
        result1 = generate_risk_score(sample_transaction_data)
        assert 'confidence' in result1
        assert 0.0 <= result1['confidence'] <= 1.0

        # Multiple components
        result2 = generate_risk_score(
            sample_transaction_data,
            behavioral_data=sample_behavioral_data
        )
        assert 'confidence' in result2
        assert 0.0 <= result2['confidence'] <= 1.0

    def test_anomaly_aggregation_workflow(self, high_risk_transaction):
        """Test aggregation of anomalies across components"""
        comprehensive = generate_risk_score(high_risk_transaction)

        # Should aggregate anomalies
        assert 'all_detected_anomalies' in comprehensive
        assert isinstance(comprehensive['all_detected_anomalies'], list)

        # Explanation should reference these anomalies
        explanation = explain_decision(comprehensive)

        if len(comprehensive['all_detected_anomalies']) > 0:
            # Key factors should include the anomalies
            factor_names = [f['factor'] for f in explanation['key_factors']]
            # At least some anomalies should be in factors
            assert any(anomaly in factor_names
                      for anomaly in comprehensive['all_detected_anomalies'])
