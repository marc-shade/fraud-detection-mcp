"""
Tests for network-based fraud ring detection
"""

import pytest
import networkx as nx
from server import NetworkAnalyzer


class TestNetworkAnalysis:
    """Test network-based fraud detection"""

    @pytest.fixture
    def analyzer(self):
        """Create network analyzer"""
        return NetworkAnalyzer()

    def test_analyze_normal_network(self, analyzer, sample_network_data):
        """Test analysis of normal network patterns"""
        result = analyzer.analyze_network_risk(sample_network_data)

        assert isinstance(result, dict)
        assert 'risk_score' in result
        assert 'network_metrics' in result
        assert 'risk_patterns' in result
        assert 'confidence' in result
        assert 'analysis_type' in result

        assert result['analysis_type'] == 'network_analysis'
        assert 0.0 <= result['risk_score'] <= 1.0

    def test_analyze_fraud_ring_network(self, analyzer, fraud_ring_network_data):
        """Test detection of fraud ring pattern"""
        result = analyzer.analyze_network_risk(fraud_ring_network_data)

        assert isinstance(result, dict)
        assert result['risk_score'] > 0.0
        # Should detect high connectivity or hub behavior
        assert len(result['risk_patterns']) > 0

    def test_missing_entity_id(self, analyzer):
        """Test analysis with missing entity_id"""
        data = {'connections': []}
        result = analyzer.analyze_network_risk(data)

        assert result['status'] == 'no_entity_id'
        assert result['risk_score'] == 0.0

    def test_empty_connections(self, analyzer):
        """Test analysis with no connections"""
        data = {
            'entity_id': 'user123',
            'connections': []
        }
        result = analyzer.analyze_network_risk(data)

        assert isinstance(result, dict)
        assert 'network_metrics' in result

    def test_update_graph_single_entity(self, analyzer):
        """Test graph update with single entity"""
        entity_id = 'test_user'
        connections = []

        analyzer._update_graph(entity_id, connections)

        assert entity_id in analyzer.transaction_graph.nodes()

    def test_update_graph_with_connections(self, analyzer, sample_network_data):
        """Test graph update with connections"""
        entity_id = sample_network_data['entity_id']
        connections = sample_network_data['connections']

        analyzer._update_graph(entity_id, connections)

        assert entity_id in analyzer.transaction_graph.nodes()
        # Check that edges were created
        assert len(list(analyzer.transaction_graph.edges(entity_id))) > 0

    def test_calculate_network_metrics_existing_entity(self, analyzer):
        """Test metric calculation for existing entity"""
        entity_id = 'test_user'
        analyzer.transaction_graph.add_node(entity_id)
        analyzer.transaction_graph.add_edge(entity_id, 'user2')
        analyzer.transaction_graph.add_edge(entity_id, 'user3')

        metrics = analyzer._calculate_network_metrics(entity_id)

        assert 'degree' in metrics
        assert 'clustering_coefficient' in metrics
        assert 'betweenness_centrality' in metrics
        assert 'closeness_centrality' in metrics

        assert metrics['degree'] == 2

    def test_calculate_network_metrics_nonexistent_entity(self, analyzer):
        """Test metric calculation for non-existent entity"""
        metrics = analyzer._calculate_network_metrics('nonexistent')

        assert metrics == {}

    def test_degree_centrality_calculation(self, analyzer):
        """Test correct calculation of degree centrality"""
        entity_id = 'hub'
        # Create star network with hub in center
        for i in range(10):
            analyzer.transaction_graph.add_edge(entity_id, f'user{i}')

        metrics = analyzer._calculate_network_metrics(entity_id)

        assert metrics['degree'] == 10

    def test_clustering_coefficient_calculation(self, analyzer):
        """Test clustering coefficient calculation"""
        # Create triangle (high clustering)
        analyzer.transaction_graph.add_edge('a', 'b')
        analyzer.transaction_graph.add_edge('b', 'c')
        analyzer.transaction_graph.add_edge('c', 'a')

        metrics = analyzer._calculate_network_metrics('a')

        # Triangle has perfect clustering
        assert metrics['clustering_coefficient'] == 1.0

    def test_detect_high_connectivity_risk(self, analyzer):
        """Test detection of unusually high connectivity"""
        metrics = {'degree': 60}
        patterns = analyzer._detect_risk_patterns('test_user', metrics)

        assert 'unusually_high_connectivity' in patterns

    def test_detect_fraud_hub_pattern(self, analyzer):
        """Test detection of potential fraud hub"""
        metrics = {'betweenness_centrality': 0.15}
        patterns = analyzer._detect_risk_patterns('test_user', metrics)

        assert 'potential_fraud_hub' in patterns

    def test_detect_tight_clustering_pattern(self, analyzer):
        """Test detection of tight clustering (fraud ring)"""
        metrics = {'clustering_coefficient': 0.9}
        patterns = analyzer._detect_risk_patterns('test_user', metrics)

        assert 'tight_clustering_pattern' in patterns

    def test_no_risk_patterns_normal_metrics(self, analyzer):
        """Test no risk patterns detected for normal metrics"""
        metrics = {
            'degree': 5,
            'betweenness_centrality': 0.05,
            'clustering_coefficient': 0.3
        }
        patterns = analyzer._detect_risk_patterns('test_user', metrics)

        assert len(patterns) == 0

    def test_calculate_network_risk_score_low_risk(self, analyzer):
        """Test risk score calculation for low-risk entity"""
        metrics = {
            'degree': 5,
            'betweenness_centrality': 0.02,
            'clustering_coefficient': 0.3
        }
        patterns = []

        risk_score = analyzer._calculate_network_risk_score(metrics, patterns)

        assert 0.0 <= risk_score < 0.3

    def test_calculate_network_risk_score_high_risk(self, analyzer):
        """Test risk score calculation for high-risk entity"""
        metrics = {
            'degree': 80,
            'betweenness_centrality': 0.3,
            'clustering_coefficient': 0.95
        }
        patterns = ['unusually_high_connectivity', 'potential_fraud_hub']

        risk_score = analyzer._calculate_network_risk_score(metrics, patterns)

        assert risk_score > 0.5

    def test_risk_score_degree_contribution(self, analyzer):
        """Test that degree contributes to risk score"""
        metrics_low = {'degree': 5, 'betweenness_centrality': 0}
        metrics_high = {'degree': 50, 'betweenness_centrality': 0}

        score_low = analyzer._calculate_network_risk_score(metrics_low, [])
        score_high = analyzer._calculate_network_risk_score(metrics_high, [])

        assert score_high > score_low

    def test_risk_score_betweenness_contribution(self, analyzer):
        """Test that betweenness centrality contributes to risk score"""
        metrics_low = {'degree': 5, 'betweenness_centrality': 0.01}
        metrics_high = {'degree': 5, 'betweenness_centrality': 0.3}

        score_low = analyzer._calculate_network_risk_score(metrics_low, [])
        score_high = analyzer._calculate_network_risk_score(metrics_high, [])

        assert score_high > score_low

    def test_risk_score_pattern_contribution(self, analyzer):
        """Test that detected patterns increase risk score"""
        metrics = {'degree': 10, 'betweenness_centrality': 0.05}

        score_no_patterns = analyzer._calculate_network_risk_score(metrics, [])
        score_with_patterns = analyzer._calculate_network_risk_score(
            metrics,
            ['unusually_high_connectivity', 'tight_clustering_pattern']
        )

        assert score_with_patterns > score_no_patterns

    def test_risk_score_capped_at_one(self, analyzer):
        """Test that risk score never exceeds 1.0"""
        metrics = {
            'degree': 1000,
            'betweenness_centrality': 1.0,
            'clustering_coefficient': 1.0
        }
        patterns = ['pattern1', 'pattern2', 'pattern3', 'pattern4', 'pattern5']

        risk_score = analyzer._calculate_network_risk_score(metrics, patterns)

        assert risk_score <= 1.0

    def test_edge_weight_storage(self, analyzer):
        """Test that edge weights are properly stored"""
        entity_id = 'user1'
        connections = [
            {'entity_id': 'user2', 'strength': 0.8, 'transaction_count': 5}
        ]

        analyzer._update_graph(entity_id, connections)

        edge_data = analyzer.transaction_graph.get_edge_data(entity_id, 'user2')
        assert edge_data is not None
        assert edge_data['weight'] == 0.8
        assert edge_data['transaction_count'] == 5

    def test_default_edge_values(self, analyzer):
        """Test default values when edge data is missing"""
        entity_id = 'user1'
        connections = [
            {'entity_id': 'user2'}  # Missing strength and transaction_count
        ]

        analyzer._update_graph(entity_id, connections)

        edge_data = analyzer.transaction_graph.get_edge_data(entity_id, 'user2')
        assert edge_data is not None
        assert edge_data['weight'] == 1.0
        assert edge_data['transaction_count'] == 1

    def test_graph_persistence(self, analyzer, sample_network_data):
        """Test that graph persists across multiple updates"""
        # Add first entity
        analyzer.analyze_network_risk(sample_network_data)

        initial_node_count = len(analyzer.transaction_graph.nodes())

        # Add another entity
        new_data = {
            'entity_id': 'new_user',
            'connections': [{'entity_id': 'user999'}]
        }
        analyzer.analyze_network_risk(new_data)

        # Graph should have grown
        assert len(analyzer.transaction_graph.nodes()) > initial_node_count

    def test_confidence_score_value(self, analyzer, sample_network_data):
        """Test that confidence score has expected value"""
        result = analyzer.analyze_network_risk(sample_network_data)

        assert result['confidence'] == 0.82

    def test_error_handling(self, analyzer):
        """Test error handling with malformed data"""
        malformed_data = {
            'entity_id': 'test',
            'connections': 'invalid'  # Should be a list
        }
        result = analyzer.analyze_network_risk(malformed_data)

        assert 'error' in result
        assert result['risk_score'] == 0.0
        assert result['status'] == 'error'

    def test_complete_analysis_flow(self, analyzer):
        """Test complete analysis flow from start to finish"""
        data = {
            'entity_id': 'test_entity',
            'connections': [
                {'entity_id': 'entity1', 'strength': 0.7, 'transaction_count': 3},
                {'entity_id': 'entity2', 'strength': 0.8, 'transaction_count': 5},
                {'entity_id': 'entity3', 'strength': 0.6, 'transaction_count': 2}
            ]
        }

        result = analyzer.analyze_network_risk(data)

        assert all(k in result for k in [
            'risk_score', 'network_metrics', 'risk_patterns',
            'confidence', 'analysis_type'
        ])
        assert isinstance(result['network_metrics'], dict)
        assert isinstance(result['risk_patterns'], list)

    def test_isolated_node_metrics(self, analyzer):
        """Test metrics for isolated node (no connections)"""
        analyzer.transaction_graph.add_node('isolated')

        metrics = analyzer._calculate_network_metrics('isolated')

        assert metrics['degree'] == 0
        assert metrics['clustering_coefficient'] == 0

    def test_large_network_performance(self, analyzer):
        """Test performance with large network"""
        # Create large network
        for i in range(100):
            analyzer.transaction_graph.add_node(f'user{i}')
            for j in range(i-5, i):
                if j >= 0:
                    analyzer.transaction_graph.add_edge(f'user{i}', f'user{j}')

        # Analysis should complete without timeout
        data = {
            'entity_id': 'user50',
            'connections': [{'entity_id': 'user51'}]
        }
        result = analyzer.analyze_network_risk(data)

        assert isinstance(result, dict)
        assert 'risk_score' in result
