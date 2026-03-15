"""Tests for TypologyDetector — 6 AML pattern detectors."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from src.agents.typology_detector import TypologyDetector, TypologyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(num_nodes: int, edges: list[tuple[int, int]]) -> Data:
    """Minimal PyG Data object."""
    torch.manual_seed(0)
    x = torch.rand(num_nodes, 8)
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _make_preds(num_nodes: int, high: list[int], val: float = 0.9) -> torch.Tensor:
    preds = torch.full((num_nodes,), 0.1)
    for n in high:
        preds[n] = val
    return preds


def _detector(num_nodes, edges, high_nodes=None, val=0.9):
    if high_nodes is None:
        high_nodes = list(range(num_nodes))
    data = _make_data(num_nodes, edges)
    preds = _make_preds(num_nodes, high_nodes, val)
    return TypologyDetector(data, preds, threshold=0.7)


# ---------------------------------------------------------------------------
# TypologyResult dataclass
# ---------------------------------------------------------------------------

class TestTypologyResult:
    def test_fields_accessible(self):
        r = TypologyResult(
            typology="smurfing",
            confidence=0.8,
            evidence_nodes=[0, 1, 2],
            evidence_edges=[(1, 0), (2, 0)],
            description="test",
            metrics={"in_degree": 2},
        )
        assert r.typology == "smurfing"
        assert r.confidence == 0.8
        assert r.evidence_nodes == [0, 1, 2]


# ---------------------------------------------------------------------------
# _build_cluster_subgraph
# ---------------------------------------------------------------------------

class TestBuildClusterSubgraph:
    def test_only_cluster_edges_included(self):
        # Edges: 0→1 (in cluster), 2→3 (outside cluster)
        det = _detector(4, [(0, 1), (2, 3)])
        G = det._build_cluster_subgraph([0, 1])
        assert set(G.nodes()) == {0, 1}
        assert list(G.edges()) == [(0, 1)]

    def test_illicit_prob_attribute_set(self):
        det = _detector(3, [(0, 1)])
        G = det._build_cluster_subgraph([0, 1, 2])
        for node in [0, 1, 2]:
            assert "illicit_prob" in G.nodes[node]
            assert isinstance(G.nodes[node]["illicit_prob"], float)


# ---------------------------------------------------------------------------
# Smurfing
# ---------------------------------------------------------------------------

class TestSmurfing:
    def test_detects_classic_smurfing(self):
        # Nodes 1,2,3,4 all send to node 0 — pure sink with 4 feeders
        edges = [(1, 0), (2, 0), (3, 0), (4, 0)]
        det = _detector(5, edges)
        G = det._build_cluster_subgraph([0, 1, 2, 3, 4])
        results = det._detect_smurfing(G)
        assert len(results) == 1
        r = results[0]
        assert r.typology == "smurfing"
        assert r.metrics["sink_node"] == 0
        assert r.metrics["in_degree"] == 4

    def test_no_smurfing_below_threshold(self):
        # Only 2 feeders — below in_degree>=3
        edges = [(1, 0), (2, 0)]
        det = _detector(3, edges)
        G = det._build_cluster_subgraph([0, 1, 2])
        results = det._detect_smurfing(G)
        assert results == []

    def test_confidence_capped_at_1(self):
        # 10+ feeders → confidence would exceed 1.0 without cap
        edges = [(i, 0) for i in range(1, 15)]
        det = _detector(15, edges)
        G = det._build_cluster_subgraph(list(range(15)))
        results = det._detect_smurfing(G)
        assert all(r.confidence <= 1.0 for r in results)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------

class TestFanOut:
    def test_detects_fan_out(self):
        # Node 0 sends to 4 destinations with no incoming edges
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        det = _detector(5, edges)
        G = det._build_cluster_subgraph(list(range(5)))
        results = det._detect_fan_out(G)
        assert len(results) == 1
        assert results[0].typology == "fan_out"
        assert results[0].metrics["hub_node"] == 0

    def test_no_fan_out_below_threshold(self):
        # Only 3 outgoing — below out_degree>=4
        edges = [(0, 1), (0, 2), (0, 3)]
        det = _detector(4, edges)
        G = det._build_cluster_subgraph(list(range(4)))
        assert det._detect_fan_out(G) == []


# ---------------------------------------------------------------------------
# Fan-in
# ---------------------------------------------------------------------------

class TestFanIn:
    def test_detects_fan_in(self):
        # Node 0 receives from 4 sources and has no outgoing
        edges = [(1, 0), (2, 0), (3, 0), (4, 0)]
        det = _detector(5, edges)
        G = det._build_cluster_subgraph(list(range(5)))
        results = det._detect_fan_in(G)
        assert len(results) == 1
        assert results[0].typology == "fan_in"
        assert results[0].metrics["collector_node"] == 0

    def test_no_fan_in_below_threshold(self):
        edges = [(1, 0), (2, 0), (3, 0)]  # only 3, needs >=4
        det = _detector(4, edges)
        G = det._build_cluster_subgraph(list(range(4)))
        assert det._detect_fan_in(G) == []


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_detects_simple_cycle(self):
        # 0→1→2→0
        edges = [(0, 1), (1, 2), (2, 0)]
        det = _detector(3, edges)
        G = det._build_cluster_subgraph([0, 1, 2])
        results = det._detect_round_trip(G)
        assert len(results) >= 1
        assert results[0].typology == "round_trip"
        assert results[0].metrics["cycle_length"] == 3

    def test_evidence_edges_form_cycle(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        det = _detector(3, edges)
        G = det._build_cluster_subgraph([0, 1, 2])
        results = det._detect_round_trip(G)
        r = results[0]
        # Each edge (u,v) in evidence_edges should be a consecutive pair in the cycle
        for (u, v) in r.evidence_edges:
            assert (u, v) in G.edges()

    def test_no_round_trip_in_dag(self):
        # Pure DAG — no cycles
        edges = [(0, 1), (1, 2), (2, 3)]
        det = _detector(4, edges)
        G = det._build_cluster_subgraph(list(range(4)))
        assert det._detect_round_trip(G) == []

    def test_description_not_empty(self):
        edges = [(0, 1), (1, 0)]
        det = _detector(2, edges)
        G = det._build_cluster_subgraph([0, 1])
        results = det._detect_round_trip(G)
        assert all(r.description for r in results)


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------

class TestLayering:
    def test_detects_layering_chain(self):
        # Linear chain: 0→1→2→3 (length 4 >= 3)
        edges = [(0, 1), (1, 2), (2, 3)]
        det = _detector(4, edges)
        G = det._build_cluster_subgraph(list(range(4)))
        results = det._detect_layering(G)
        assert len(results) >= 1
        assert results[0].typology == "layering"
        assert results[0].metrics["chain_length"] >= 3

    def test_short_path_excluded(self):
        # Path of length 2 (only 1 hop) — below cutoff of len>=3
        edges = [(0, 1)]
        det = _detector(2, edges)
        G = det._build_cluster_subgraph([0, 1])
        results = det._detect_layering(G)
        assert results == []


# ---------------------------------------------------------------------------
# Scatter-gather
# ---------------------------------------------------------------------------

class TestScatterGather:
    def test_detects_scatter_gather(self):
        # Hub=0 → intermediaries 1,2,3 → collector=4
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
        det = _detector(5, edges)
        G = det._build_cluster_subgraph(list(range(5)))
        results = det._detect_scatter_gather(G)
        assert len(results) >= 1
        r = results[0]
        assert r.typology == "scatter-gather"
        assert r.metrics["hub"] == 0
        assert r.metrics["collector"] == 4
        assert r.metrics["intermediary_count"] >= 2

    def test_no_scatter_gather_single_intermediary(self):
        # Only 1 shared intermediary — below threshold of >=2
        edges = [(0, 1), (0, 2), (1, 3), (2, 4)]  # no shared path to single collector
        det = _detector(5, edges)
        G = det._build_cluster_subgraph(list(range(5)))
        results = det._detect_scatter_gather(G)
        assert results == []


# ---------------------------------------------------------------------------
# detect_all
# ---------------------------------------------------------------------------

class TestDetectAll:
    def test_returns_list_of_typology_results(self):
        edges = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 2), (2, 0)]
        det = _detector(4, edges)
        results = det.detect_all(list(range(4)))
        assert isinstance(results, list)
        assert all(isinstance(r, TypologyResult) for r in results)

    def test_sorted_by_confidence_descending(self):
        edges = [(0, 1), (1, 2), (2, 0), (3, 0), (4, 0), (5, 0)]
        det = _detector(6, edges)
        results = det.detect_all(list(range(6)))
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_graph_returns_empty(self):
        det = _detector(3, [])
        results = det.detect_all([0, 1, 2])
        assert results == []
