"""Tests for AttentionTracer."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch_geometric.data import Data

from src.agents.attention_tracer import AttentionTracer, EvidencePath, AttentionProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(num_nodes: int, edges: list[tuple[int, int]]) -> Data:
    torch.manual_seed(0)
    x = torch.rand(num_nodes, 8)
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _make_mock_model(edge_index: torch.Tensor, weights: list[float], num_heads: int = 4):
    """Mock model whose get_attention_weights returns controlled values."""
    mock = MagicMock()
    # alpha shape: [num_edges, num_heads] — same weight across all heads
    alpha = torch.tensor([[w] * num_heads for w in weights], dtype=torch.float)
    mock.get_attention_weights.return_value = [(edge_index, alpha)]
    return mock


def _make_tracer(num_nodes, edges, weights):
    """Build an AttentionTracer with fully controlled attention weights."""
    data = _make_data(num_nodes, edges)
    model = _make_mock_model(data.edge_index, weights)
    return AttentionTracer(model, data)


# ---------------------------------------------------------------------------
# _build_cache
# ---------------------------------------------------------------------------

class TestBuildCache:
    def test_cache_populated_after_first_call(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.8, 0.6])
        tracer._build_cache()
        assert len(tracer._edge_weights) == 2
        assert tracer._cached is True

    def test_model_called_only_once(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.8, 0.6])
        tracer._build_cache()
        tracer._build_cache()  # second call — should hit cache
        tracer.model.get_attention_weights.assert_called_once()

    def test_edge_weights_values_correct(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.9, 0.5])
        tracer._build_cache()
        assert abs(tracer._edge_weights[(0, 1)] - 0.9) < 1e-5
        assert abs(tracer._edge_weights[(1, 2)] - 0.5) < 1e-5

    def test_reverse_index_built(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.9, 0.5])
        tracer._build_cache()
        # node 1 should have node 0 as incoming
        srcs = [src for src, _ in tracer._reverse_index[1]]
        assert 0 in srcs

    def test_forward_index_built(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.9, 0.5])
        tracer._build_cache()
        # node 0 should have node 1 as outgoing
        dsts = [dst for dst, _ in tracer._forward_index[0]]
        assert 1 in dsts


# ---------------------------------------------------------------------------
# trace_evidence_path
# ---------------------------------------------------------------------------

class TestTraceEvidencePath:
    def test_follows_highest_attention_edge(self):
        # 0→2 (0.9) and 1→2 (0.3) — tracer should pick 0 as predecessor of 2
        tracer = _make_tracer(3, [(0, 2), (1, 2)], [0.9, 0.3])
        result = tracer.trace_evidence_path(node_id=2, max_hops=1)
        assert result.path == [0, 2]

    def test_path_starts_at_source_ends_at_target(self):
        # Chain: 0→1→2, with weights 0.8, 0.7
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.8, 0.7])
        result = tracer.trace_evidence_path(node_id=2, max_hops=2)
        assert result.path[-1] == 2
        assert result.path[0] == 0

    def test_respects_max_hops(self):
        # Chain: 0→1→2→3→4, all weight 0.9
        tracer = _make_tracer(5, [(0,1),(1,2),(2,3),(3,4)], [0.9]*4)
        result = tracer.trace_evidence_path(node_id=4, max_hops=2)
        # With max_hops=2: target=4, walk back to 3, then 2 → path=[2,3,4]
        assert len(result.path) == 3
        assert result.path[-1] == 4

    def test_stops_at_node_with_no_incoming(self):
        # Node 0 has no incoming edges
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.8, 0.7])
        result = tracer.trace_evidence_path(node_id=2, max_hops=10)
        assert result.path[0] == 0  # stops naturally at the source

    def test_cumulative_attention_is_product_of_weights(self):
        tracer = _make_tracer(3, [(0, 1), (1, 2)], [0.8, 0.5])
        result = tracer.trace_evidence_path(node_id=2, max_hops=2)
        expected = 0.8 * 0.5
        assert abs(result.cumulative_attention - expected) < 1e-5

    def test_isolated_node_returns_single_node_path(self):
        tracer = _make_tracer(3, [(0, 1)], [0.9])
        result = tracer.trace_evidence_path(node_id=2, max_hops=3)
        assert result.path == [2]
        assert result.cumulative_attention == 0.0


# ---------------------------------------------------------------------------
# get_top_attention_edges
# ---------------------------------------------------------------------------

class TestGetTopAttentionEdges:
    def test_returns_edges_sorted_by_weight(self):
        tracer = _make_tracer(4, [(0,1),(1,2),(2,3)], [0.9, 0.4, 0.7])
        edges = tracer.get_top_attention_edges([0, 1, 2, 3])
        weights = [e["weight"] for e in edges]
        assert weights == sorted(weights, reverse=True)

    def test_filters_to_cluster_nodes_only(self):
        # Edges: 0→1 (in cluster), 2→3 (outside cluster)
        tracer = _make_tracer(4, [(0, 1), (2, 3)], [0.9, 0.8])
        edges = tracer.get_top_attention_edges([0, 1])
        assert len(edges) == 1
        assert edges[0]["src"] == 0 and edges[0]["dst"] == 1

    def test_top_k_respected(self):
        tracer = _make_tracer(5, [(0,1),(1,2),(2,3),(3,4)], [0.9,0.8,0.7,0.6])
        edges = tracer.get_top_attention_edges(list(range(5)), top_k=2)
        assert len(edges) == 2
        assert edges[0]["weight"] >= edges[1]["weight"]


# ---------------------------------------------------------------------------
# get_node_attention_profile
# ---------------------------------------------------------------------------

class TestGetNodeAttentionProfile:
    def test_incoming_and_outgoing_populated(self):
        # 0→1, 2→1 — node 1 has 2 incoming, 0 outgoing
        tracer = _make_tracer(3, [(0, 1), (2, 1)], [0.8, 0.6])
        profile = tracer.get_node_attention_profile(1)
        assert len(profile.incoming_attention) == 2
        assert len(profile.outgoing_attention) == 0

    def test_incoming_sorted_descending(self):
        tracer = _make_tracer(3, [(0, 2), (1, 2)], [0.3, 0.9])
        profile = tracer.get_node_attention_profile(2)
        weights = [e["weight"] for e in profile.incoming_attention]
        assert weights == sorted(weights, reverse=True)

    def test_totals_are_sums(self):
        tracer = _make_tracer(3, [(0, 2), (1, 2)], [0.4, 0.6])
        profile = tracer.get_node_attention_profile(2)
        assert abs(profile.total_incoming - 1.0) < 1e-5

    def test_isolated_node_has_zero_totals(self):
        tracer = _make_tracer(3, [(0, 1)], [0.9])
        profile = tracer.get_node_attention_profile(2)
        assert profile.total_incoming == 0.0
        assert profile.total_outgoing == 0.0
