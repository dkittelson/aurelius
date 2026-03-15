"""Tests for SuspiciousClusterDetector."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from torch_geometric.data import Data

from src.agents.cluster_detector import SuspiciousClusterDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(num_nodes: int, edges: list[tuple[int, int]]) -> Data:
    """Create a synthetic PyG Data object with given edges."""
    torch.manual_seed(42)
    x = torch.rand(num_nodes, 8)
    if edges:
        src, dst = zip(*edges)
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _make_predictions(num_nodes: int, high_nodes: list[int], high_val: float = 0.9, low_val: float = 0.1) -> torch.Tensor:
    """Create predictions tensor with high values at specified node indices."""
    preds = torch.full((num_nodes,), low_val)
    for n in high_nodes:
        preds[n] = high_val
    return preds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSuspiciousClusterDetector:

    def test_detect_empty_graph_no_suspicious_nodes(self):
        """When no nodes exceed threshold, returns empty list."""
        data = _make_graph(10, [(0, 1), (1, 2)])
        predictions = _make_predictions(10, high_nodes=[], low_val=0.3)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)
        result = detector.detect_from_predictions(data, predictions)
        assert result == []

    def test_detect_single_cluster(self):
        """Three connected suspicious nodes form one cluster."""
        # nodes 0, 1, 2 are suspicious and connected: 0→1→2
        data = _make_graph(10, [(0, 1), (1, 2), (3, 4)])
        predictions = _make_predictions(10, high_nodes=[0, 1, 2], high_val=0.9)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)
        result = detector.detect_from_predictions(data, predictions)
        assert len(result) == 1
        assert set(result[0]) == {0, 1, 2}

    def test_detect_multiple_clusters(self):
        """Two disconnected suspicious groups are returned as separate clusters."""
        # Group A: nodes 0, 1, 2 connected
        # Group B: nodes 5, 6, 7 connected
        # No edges between groups
        edges = [(0, 1), (1, 2), (5, 6), (6, 7)]
        data = _make_graph(10, edges)
        predictions = _make_predictions(10, high_nodes=[0, 1, 2, 5, 6, 7], high_val=0.9)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)
        result = detector.detect_from_predictions(data, predictions)
        assert len(result) == 2
        cluster_sets = [set(c) for c in result]
        assert {0, 1, 2} in cluster_sets
        assert {5, 6, 7} in cluster_sets

    def test_min_cluster_size_filters_small_clusters(self):
        """Clusters below min_cluster_size are excluded."""
        # Only 2 suspicious nodes connected — should be filtered with min_size=3
        data = _make_graph(10, [(0, 1)])
        predictions = _make_predictions(10, high_nodes=[0, 1], high_val=0.9)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=3)
        result = detector.detect_from_predictions(data, predictions)
        assert result == []

    def test_clusters_sorted_by_avg_confidence(self):
        """Most suspicious cluster (highest avg confidence) comes first."""
        # Cluster A: nodes 0, 1, 2 with prob 0.95
        # Cluster B: nodes 5, 6, 7 with prob 0.75
        edges = [(0, 1), (1, 2), (5, 6), (6, 7)]
        data = _make_graph(10, edges)
        preds = torch.full((10,), 0.1)
        preds[0] = preds[1] = preds[2] = 0.95
        preds[5] = preds[6] = preds[7] = 0.75
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)
        result = detector.detect_from_predictions(data, preds)
        assert len(result) == 2
        # First cluster should be the higher-confidence one
        assert set(result[0]) == {0, 1, 2}
        assert set(result[1]) == {5, 6, 7}

    def test_louvain_fallback_when_not_installed(self):
        """detect_with_community falls back to connected components if python-louvain missing."""
        data = _make_graph(10, [(0, 1), (1, 2)])
        predictions = _make_predictions(10, high_nodes=[0, 1, 2], high_val=0.9)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)

        with patch.dict("sys.modules", {"community": None}):
            result = detector.detect_with_community(data, predictions)

        # Falls back to connected components — still finds the cluster
        assert len(result) == 1
        assert set(result[0]) == {0, 1, 2}

    def test_louvain_detection(self):
        """detect_with_community finds communities when python-louvain is available."""
        edges = [(0, 1), (1, 2), (0, 2), (5, 6), (6, 7), (5, 7)]
        data = _make_graph(10, edges)
        predictions = _make_predictions(10, high_nodes=[0, 1, 2, 5, 6, 7], high_val=0.9)
        detector = SuspiciousClusterDetector(threshold=0.7, min_cluster_size=2)

        mock_community = MagicMock()
        # Louvain assigns community 0 to nodes 0,1,2 and community 1 to nodes 5,6,7
        mock_community.best_partition.return_value = {0: 0, 1: 0, 2: 0, 5: 1, 6: 1, 7: 1}

        with patch.dict("sys.modules", {"community": mock_community}):
            result = detector.detect_with_community(data, predictions)

        assert len(result) == 2

    def test_get_cluster_stats_all_keys_present(self):
        """get_cluster_stats returns dict with all required keys and correct types."""
        data = _make_graph(10, [(0, 1), (1, 2), (0, 2)])
        predictions = _make_predictions(10, high_nodes=[0, 1, 2], high_val=0.9)
        detector = SuspiciousClusterDetector()
        stats = detector.get_cluster_stats(data, [0, 1, 2], predictions)

        assert set(stats.keys()) == {"node_ids", "num_nodes", "num_edges", "avg_confidence", "max_confidence", "risk_level", "density"}
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 3       # 0→1, 1→2, 0→2
        assert isinstance(stats["avg_confidence"], float)
        assert isinstance(stats["max_confidence"], float)
        assert stats["risk_level"] in ("CRITICAL", "HIGH")  # nodes set to 0.9 — boundary
        assert 0.0 <= stats["density"] <= 1.0

    def test_score_to_risk_thresholds(self):
        """_score_to_risk maps confidence values to correct risk levels."""
        assert SuspiciousClusterDetector._score_to_risk(0.95) == "CRITICAL"
        assert SuspiciousClusterDetector._score_to_risk(0.90) == "CRITICAL"
        assert SuspiciousClusterDetector._score_to_risk(0.80) == "HIGH"
        assert SuspiciousClusterDetector._score_to_risk(0.75) == "HIGH"
        assert SuspiciousClusterDetector._score_to_risk(0.65) == "MEDIUM"
        assert SuspiciousClusterDetector._score_to_risk(0.60) == "MEDIUM"
        assert SuspiciousClusterDetector._score_to_risk(0.50) == "LOW"
        assert SuspiciousClusterDetector._score_to_risk(0.00) == "LOW"
