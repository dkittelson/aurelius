"""GNN Explainability for AML node predictions.

Wraps PyG's Explainer API to provide per-node edge and feature importance
masks. Critical for AML compliance — investigators need to understand
WHY a node was flagged as suspicious.

Provides:
  - Per-node edge importance (which transactions matter)
  - Per-node feature importance (which attributes matter)
  - Human-readable explanation summaries
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from loguru import logger

from src.models.gnn_model import AureliusGAT


class AureliusExplainer:
    """Wrapper around PyG's Explainer API for AML node explanations.

    Args:
        model: Trained AureliusGAT model.
        data:  Full graph Data object.
        epochs: Number of optimization epochs for GNNExplainer (default 200).
    """

    def __init__(
        self,
        model: AureliusGAT,
        data: Data,
        epochs: int = 200,
    ):
        self.model = model
        self.data = data
        self.model.eval()

        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain_node(
        self,
        node_id: int,
        top_k_edges: int = 10,
        top_k_features: int = 10,
    ) -> dict:
        """Generate explanation for a single node.

        Args:
            node_id:       Target node to explain.
            top_k_edges:   Number of top important edges to return.
            top_k_features: Number of top important features to return.

        Returns:
            Dict with node_id, important_edges, important_features,
            edge_mask, feature_mask.
        """
        explanation = self.explainer(
            self.data.x,
            self.data.edge_index,
            index=node_id,
        )

        # Edge importance
        edge_mask = explanation.edge_mask
        if edge_mask is not None:
            edge_mask_np = edge_mask.cpu().numpy()
            # Get top-k edges by importance
            top_edge_idx = np.argsort(edge_mask_np)[-top_k_edges:][::-1]
            important_edges = []
            for idx in top_edge_idx:
                if idx < self.data.edge_index.size(1):
                    src = int(self.data.edge_index[0, idx])
                    dst = int(self.data.edge_index[1, idx])
                    importance = float(edge_mask_np[idx])
                    if importance > 0:
                        important_edges.append({
                            "src": src,
                            "dst": dst,
                            "importance": round(importance, 4),
                        })
        else:
            edge_mask_np = np.array([])
            important_edges = []

        # Feature importance
        node_mask = explanation.node_mask
        if node_mask is not None:
            # node_mask is [N, F], we want the row for our target node
            if node_mask.dim() == 2:
                feature_mask_np = node_mask[node_id].cpu().numpy()
            else:
                feature_mask_np = node_mask.cpu().numpy()
            top_feat_idx = np.argsort(np.abs(feature_mask_np))[-top_k_features:][::-1]
            important_features = [
                {
                    "feature_index": int(idx),
                    "importance": round(float(feature_mask_np[idx]), 4),
                }
                for idx in top_feat_idx
                if abs(feature_mask_np[idx]) > 0
            ]
        else:
            feature_mask_np = np.array([])
            important_features = []

        return {
            "node_id": node_id,
            "important_edges": important_edges,
            "important_features": important_features,
            "edge_mask": edge_mask_np,
            "feature_mask": feature_mask_np,
        }

    def explain_cluster(
        self,
        node_ids: list[int],
        top_k_edges: int = 20,
    ) -> dict:
        """Aggregate explanations across a cluster of nodes.

        Combines edge importance masks from all cluster nodes to find
        the most important transaction edges for the cluster as a whole.

        Args:
            node_ids:    List of node IDs in the cluster.
            top_k_edges: Number of top edges to return.

        Returns:
            Dict with node_ids, important_edges, summary.
        """
        aggregated_edge_importance = np.zeros(self.data.edge_index.size(1))

        for nid in node_ids:
            try:
                explanation = self.explain_node(nid, top_k_edges=top_k_edges)
                if len(explanation["edge_mask"]) > 0:
                    aggregated_edge_importance += explanation["edge_mask"]
            except Exception as e:
                logger.warning(f"Failed to explain node {nid}: {e}")

        # Normalize
        max_val = aggregated_edge_importance.max()
        if max_val > 0:
            aggregated_edge_importance /= max_val

        # Top-k aggregated edges
        top_idx = np.argsort(aggregated_edge_importance)[-top_k_edges:][::-1]
        important_edges = []
        for idx in top_idx:
            importance = float(aggregated_edge_importance[idx])
            if importance > 0:
                important_edges.append({
                    "src": int(self.data.edge_index[0, idx]),
                    "dst": int(self.data.edge_index[1, idx]),
                    "importance": round(importance, 4),
                })

        return {
            "node_ids": node_ids,
            "important_edges": important_edges,
            "summary": self._format_cluster_summary(node_ids, important_edges),
        }

    def format_explanation(self, explanation: dict) -> str:
        """Format a node explanation as human-readable markdown."""
        nid = explanation["node_id"]
        edges = explanation["important_edges"]
        features = explanation["important_features"]

        lines = [
            f"### Explanation for Node {nid}",
            "",
            "**Top Important Edges (Transactions):**",
        ]

        if edges:
            for e in edges[:5]:
                lines.append(
                    f"- {e['src']} → {e['dst']} "
                    f"(importance: {e['importance']:.3f})"
                )
        else:
            lines.append("- No significant edges found")

        lines.extend(["", "**Top Important Features:**"])
        if features:
            for f in features[:5]:
                lines.append(
                    f"- Feature {f['feature_index']}: "
                    f"importance={f['importance']:.3f}"
                )
        else:
            lines.append("- No significant features found")

        return "\n".join(lines)

    @staticmethod
    def _format_cluster_summary(
        node_ids: list[int], important_edges: list[dict]
    ) -> str:
        lines = [
            f"Cluster explanation for {len(node_ids)} nodes.",
            f"Top {len(important_edges)} most important transaction edges:",
        ]
        for e in important_edges[:5]:
            lines.append(
                f"  {e['src']} → {e['dst']} (importance: {e['importance']:.3f})"
            )
        return "\n".join(lines)
