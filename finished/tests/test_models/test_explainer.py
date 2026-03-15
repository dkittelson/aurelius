"""Tests for GNNExplainer wrapper."""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT
from src.models.explainer import AureliusExplainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(num_nodes=30, in_channels=16):
    torch.manual_seed(42)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def model():
    m = AureliusGAT(
        in_channels=16, hidden_channels=32, out_channels=2,
        num_heads=4, num_layers=2, dropout=0.0, jk_mode="cat",
    )
    m.eval()
    return m


@pytest.fixture
def explainer(model, graph):
    return AureliusExplainer(model, graph, epochs=10)  # few epochs for speed


# ---------------------------------------------------------------------------
# explain_node
# ---------------------------------------------------------------------------

class TestExplainNode:
    def test_returns_correct_keys(self, explainer):
        result = explainer.explain_node(0)
        assert "node_id" in result
        assert "important_edges" in result
        assert "important_features" in result
        assert "edge_mask" in result
        assert "feature_mask" in result

    def test_node_id_matches(self, explainer):
        result = explainer.explain_node(5)
        assert result["node_id"] == 5

    def test_edge_mask_values_in_range(self, explainer):
        result = explainer.explain_node(0)
        if len(result["edge_mask"]) > 0:
            assert result["edge_mask"].min() >= -0.01
            assert result["edge_mask"].max() <= 1.01

    def test_important_edges_have_structure(self, explainer):
        result = explainer.explain_node(0, top_k_edges=5)
        for edge in result["important_edges"]:
            assert "src" in edge
            assert "dst" in edge
            assert "importance" in edge
            assert isinstance(edge["src"], int)
            assert isinstance(edge["importance"], float)

    def test_important_features_have_structure(self, explainer):
        result = explainer.explain_node(0, top_k_features=5)
        for feat in result["important_features"]:
            assert "feature_index" in feat
            assert "importance" in feat

    def test_top_k_limits_results(self, explainer):
        result = explainer.explain_node(0, top_k_edges=3, top_k_features=2)
        assert len(result["important_edges"]) <= 3
        assert len(result["important_features"]) <= 2


# ---------------------------------------------------------------------------
# explain_cluster
# ---------------------------------------------------------------------------

class TestExplainCluster:
    def test_returns_correct_keys(self, explainer):
        result = explainer.explain_cluster([0, 1, 2])
        assert "node_ids" in result
        assert "important_edges" in result
        assert "summary" in result

    def test_node_ids_preserved(self, explainer):
        result = explainer.explain_cluster([3, 7, 12])
        assert result["node_ids"] == [3, 7, 12]

    def test_summary_is_nonempty(self, explainer):
        result = explainer.explain_cluster([0, 1])
        assert len(result["summary"]) > 0


# ---------------------------------------------------------------------------
# format_explanation
# ---------------------------------------------------------------------------

class TestFormatExplanation:
    def test_format_returns_nonempty_string(self, explainer):
        explanation = explainer.explain_node(0)
        text = explainer.format_explanation(explanation)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Node 0" in text
