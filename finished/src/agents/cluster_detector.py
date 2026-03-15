"""Detects suspicious node clusters from GNN predictions.

Two detection strategies:
  1. Connected components — finds groups of high-risk nodes that directly transact
     with each other. Fast, interpretable.
  2. Louvain community detection — finds densely connected communities in the
     suspicious subgraph. Better at finding organised structures that spread
     across the network.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
from loguru import logger


class SuspiciousClusterDetector:
    """
    Identifies suspicious clusters from GNN node probability scores.

    Args:
        threshold: Minimum illicit probability for a node to be considered
                   suspicious (default 0.7). Nodes below this are ignored.
        min_cluster_size: Minimum number of nodes for a cluster to be reported.
    """

    def __init__(self, threshold: float = 0.7, min_cluster_size: int = 3):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    def detect_from_predictions(
        self,
        data: Data,
        predictions: torch.Tensor,
    ) -> list[list[int]]:
        """
        Find suspicious clusters using connected components.

        Steps:
          1. Filter to nodes with P(illicit) > threshold
          2. Extract the subgraph induced by those nodes
          3. Find connected components
          4. Filter by min_cluster_size
          5. Sort by average illicit probability (most suspicious first)

        Args:
            data:        PyG Data with edge_index.
            predictions: [num_nodes] float tensor — P(illicit) per node.

        Returns:
            List of clusters, each cluster is a list of global node indices.
            Sorted descending by average illicit probability.
        """
        probs = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        suspicious_mask = probs >= self.threshold
        suspicious_ids = np.where(suspicious_mask)[0]

        if len(suspicious_ids) == 0:
            logger.info(f"No nodes exceed threshold {self.threshold}.")
            return []

        logger.info(
            f"Found {len(suspicious_ids)} nodes above threshold {self.threshold}. "
            "Building suspicious subgraph..."
        )

        # Build induced subgraph of suspicious nodes
        suspicious_set = set(suspicious_ids.tolist())
        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], suspicious_ids) & np.isin(ei[1], suspicious_ids)
        sub_src = ei[0][mask]
        sub_dst = ei[1][mask]

        G = nx.DiGraph()
        G.add_nodes_from(suspicious_ids.tolist())
        G.add_edges_from(zip(sub_src.tolist(), sub_dst.tolist()))

        # Connected components (using undirected view for grouping)
        components = list(nx.weakly_connected_components(G))

        # Filter by size
        valid = [
            list(c) for c in components
            if len(c) >= self.min_cluster_size
        ]

        if not valid:
            logger.info(
                f"No clusters with >= {self.min_cluster_size} nodes found."
            )
            return []

        # Sort by average illicit probability (most suspicious first)
        valid.sort(key=lambda c: -np.mean(probs[c]))

        logger.info(
            f"Detected {len(valid)} suspicious clusters "
            f"(threshold={self.threshold}, min_size={self.min_cluster_size})"
        )
        return valid

    def detect_with_community(
        self,
        data: Data,
        predictions: torch.Tensor,
    ) -> list[list[int]]:
        """
        Alternative: Louvain community detection on the suspicious subgraph.

        Better than connected components for finding densely-organized structures
        that might not form a single connected component — e.g., multiple fan-out
        hubs that share common destination accounts.

        Requires: pip install python-louvain (community package)
        Falls back to connected components if not installed.
        """
        try:
            import community as community_louvain
        except ImportError:
            logger.warning(
                "python-louvain not installed. "
                "Falling back to connected components. "
                "Install with: pip install python-louvain"
            )
            return self.detect_from_predictions(data, predictions)

        probs = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        suspicious_mask = probs >= self.threshold
        suspicious_ids = np.where(suspicious_mask)[0]

        if len(suspicious_ids) == 0:
            return []

        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], suspicious_ids) & np.isin(ei[1], suspicious_ids)
        sub_src = ei[0][mask]
        sub_dst = ei[1][mask]

        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(suspicious_ids.tolist())
        G_undirected.add_edges_from(zip(sub_src.tolist(), sub_dst.tolist()))

        partition = community_louvain.best_partition(G_undirected)

        # Group by community label
        community_map: dict[int, list[int]] = {}
        for node, comm_id in partition.items():
            community_map.setdefault(comm_id, []).append(node)

        valid = [
            nodes for nodes in community_map.values()
            if len(nodes) >= self.min_cluster_size
        ]
        valid.sort(key=lambda c: -np.mean(probs[c]))

        logger.info(f"Louvain detected {len(valid)} suspicious communities.")
        return valid

    def get_cluster_stats(
        self,
        data: Data,
        cluster_node_ids: list[int],
        predictions: torch.Tensor,
    ) -> dict:
        """
        Compute summary statistics for a single cluster.

        Returns dict with: node_ids, num_nodes, num_edges, avg_confidence,
                           max_confidence, risk_level, density
        """
        probs = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        cluster_set = set(cluster_node_ids)

        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], cluster_node_ids) & np.isin(ei[1], cluster_node_ids)
        num_internal_edges = int(mask.sum())

        avg_conf = float(np.mean(probs[cluster_node_ids]))
        max_conf = float(np.max(probs[cluster_node_ids]))
        n = len(cluster_node_ids)

        # Density = actual edges / possible edges (excluding self-loops)
        max_possible = n * (n - 1)
        density = num_internal_edges / max_possible if max_possible > 0 else 0.0

        risk_level = self._score_to_risk(avg_conf)

        return {
            "node_ids": cluster_node_ids,
            "num_nodes": n,
            "num_edges": num_internal_edges,
            "avg_confidence": avg_conf,
            "max_confidence": max_conf,
            "risk_level": risk_level,
            "density": density,
        }

    @staticmethod
    def _score_to_risk(avg_confidence: float) -> str:
        if avg_confidence >= 0.90:
            return "CRITICAL"
        elif avg_confidence >= 0.75:
            return "HIGH"
        elif avg_confidence >= 0.60:
            return "MEDIUM"
        else:
            return "LOW"
