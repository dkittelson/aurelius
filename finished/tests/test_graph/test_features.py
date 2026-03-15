"""Tests for graph feature engineering (PageRank, centrality, clustering)."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data


@pytest.fixture
def simple_graph():
    """A small 10-node graph with known structure for testing."""
    # Create a simple chain: 0->1->2->...->9 plus some cross edges
    edges_src = list(range(9)) + [0, 2, 5]
    edges_dst = list(range(1, 10)) + [5, 7, 9]
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    x = torch.rand(10, 16)  # 10 nodes, 16 features each
    y = torch.zeros(10, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def config():
    return {
        "features": {
            "pagerank_alpha": 0.85,
            "centrality_types": ["degree", "betweenness", "closeness"],
            "embedding_dim": 16,
        }
    }


class TestPageRank:

    def test_output_shape(self, simple_graph, config):
        """PageRank tensor should be [num_nodes, 1]."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        pr = eng.compute_pagerank(simple_graph)
        assert pr.shape == (simple_graph.num_nodes, 1)

    def test_values_are_positive(self, simple_graph, config):
        """All PageRank values must be > 0."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        pr = eng.compute_pagerank(simple_graph)
        assert (pr > 0).all(), "Some PageRank values are non-positive"

    def test_values_sum_to_approx_one(self, simple_graph, config):
        """PageRank values should sum to approximately 1."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        pr = eng.compute_pagerank(simple_graph)
        total = pr.sum().item()
        assert abs(total - 1.0) < 0.05, f"PageRank sums to {total}, expected ~1.0"


class TestCentrality:

    @pytest.mark.parametrize("kind", ["degree", "betweenness", "closeness"])
    def test_output_shape(self, simple_graph, config, kind):
        """Centrality tensor should be [num_nodes, 1]."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        c = eng.compute_centrality(simple_graph, kind)
        assert c.shape == (simple_graph.num_nodes, 1)

    @pytest.mark.parametrize("kind", ["degree", "betweenness", "closeness"])
    def test_values_in_range(self, simple_graph, config, kind):
        """All centrality values should be in [0, 1]."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        c = eng.compute_centrality(simple_graph, kind)
        assert c.min().item() >= 0.0, f"{kind} centrality has values < 0"
        assert c.max().item() <= 1.0, f"{kind} centrality has values > 1"

    def test_invalid_kind_raises(self, simple_graph, config):
        """Unknown centrality type should raise ValueError."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        with pytest.raises(ValueError, match="Unknown centrality type"):
            eng.compute_centrality(simple_graph, "eigenvalue")


class TestLocalClustering:

    def test_output_shape(self, simple_graph, config):
        """Clustering tensor should be [num_nodes, 1]."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        clust = eng.compute_local_clustering(simple_graph)
        assert clust.shape == (simple_graph.num_nodes, 1)

    def test_values_in_range(self, simple_graph, config):
        """Clustering coefficients should be in [0, 1]."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        clust = eng.compute_local_clustering(simple_graph)
        assert clust.min().item() >= 0.0
        assert clust.max().item() <= 1.0


class TestComputeAll:

    def test_feature_dimension_increases(self, simple_graph, config):
        """After compute_all, data.x should have more columns than before."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        original_dim = simple_graph.x.shape[1]
        result = eng.compute_all(simple_graph)
        new_dim = result.x.shape[1]
        assert new_dim > original_dim, (
            f"Feature dimension did not increase: {original_dim} -> {new_dim}"
        )

    def test_node_count_unchanged(self, simple_graph, config):
        """compute_all must not change the number of nodes."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        n_before = simple_graph.num_nodes
        result = eng.compute_all(simple_graph)
        assert result.num_nodes == n_before

    def test_expected_added_features(self, simple_graph, config):
        """
        Added features = 1 (PageRank) + 3 (centralities) + 1 (clustering) = 5.
        """
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        original_dim = simple_graph.x.shape[1]
        result = eng.compute_all(simple_graph)
        expected_dim = original_dim + 5  # 1 PageRank + 3 centralities + 1 clustering
        assert result.x.shape[1] == expected_dim, (
            f"Expected {expected_dim} features, got {result.x.shape[1]}"
        )

    def test_labels_preserved(self, simple_graph, config):
        """Labels (data.y) should be unchanged after feature engineering."""
        from src.graph.features import GraphFeatureEngineer
        eng = GraphFeatureEngineer(config)
        y_before = simple_graph.y.clone()
        result = eng.compute_all(simple_graph)
        assert torch.equal(result.y, y_before), "Labels changed after compute_all"
