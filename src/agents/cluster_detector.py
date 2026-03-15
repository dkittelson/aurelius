import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from loguru import logger

class SuspiciousClusterDetector:
    def __init__(self, threshold: float = 0.7, min_cluster_size: int = 3):
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size


    def detect_from_predictions(self, data: Data, predictions: torch.Tensor) -> list[list[int]]:

        # --- Collect Suspicious Nodes ---
        probs = predictions.numpy()
        suspicious_mask = probs >= self.threshold
        suspicious_ids = np.where(suspicious_mask)[0]
        if len(suspicious_ids) == 0:
            logger.info(f"No nodes exceed threshold {self.threshold}.")
            return []

        # --- Collect Suspicious Edges ---
        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], suspicious_ids) & (np.isin(ei[1], suspicious_ids)) # find edges with suspicious src and dst
        sub_src = ei[0][mask] # suspicious src nodes
        sub_dst = ei[1][mask] # suspicious dst nodes

        # --- Build Suspicious Graph ---
        G = nx.DiGraph()
        G.add_nodes_from(suspicious_ids)
        G.add_edges_from(zip(sub_src, sub_dst))

        # --- Sort node pairs from highest to lowest suspiciousness ---
        components = list(nx.weakly_connected_components(G))
        valid = [list(c) for c in components if len(c) >= self.min_cluster_size]
        if not valid:
            logger.info(f"No clusters with >= {self.min_cluster_size} nodes found.")
            return []
        valid.sort(key=lambda c: -np.mean(probs[c]))
        logger.info(f"Detected {len(valid)} suspicious clusters (threshold={self.threshold}, min_size={self.min_cluster_size})")

        return valid

    def detect_with_community(self, data : Data, predictions: torch.Tensor) -> list[list[int]]:
        try:
            import community as community_louvain
        except ImportError:
            logger.warning("python-louvain not installed. Falling back to connected components.")
            return self.detect_from_predictions(data, predictions)

        # --- Collect Suspicious Nodes ---
        probs = predictions.numpy()
        suspicious_mask = probs >= self.threshold
        suspicious_ids = np.where(suspicious_mask)[0]
        if len(suspicious_ids) == 0:
            logger.info(f"No nodes exceed threshold {self.threshold}.")
            return []

        # --- Collect Suspicious Edges ---
        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], suspicious_ids) & (np.isin(ei[1], suspicious_ids)) # find edges with suspicious src and dst
        sub_src = ei[0][mask] # suspicious src nodes
        sub_dst = ei[1][mask] # suspicious dst nodes

        # --- Build Suspicious Graph ---
        G = nx.DiGraph()
        G.add_nodes_from(suspicious_ids)
        G.add_edges_from(zip(sub_src, sub_dst))
        G_undirected = G.to_undirected()
        partition = community_louvain.best_partition(G_undirected)
        community_map = {}
        for node, comm_id in partition.items():
            community_map.setdefault(comm_id, []).append(node)
        valid = [nodes for nodes in community_map.values() if len(nodes) >= self.min_cluster_size]
        valid.sort(key=lambda c: -np.mean(probs[c]))
        logger.info(f"Louvain detected {len(valid)} suspicious communities.")
        return valid
        
    def get_cluster_stats(self, data: Data, cluster_node_ids: list[int], predictions: torch.Tensor) -> dict:
        n = len(cluster_node_ids)
        probs = predictions.numpy()
        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], cluster_node_ids) & np.isin(ei[1], cluster_node_ids)
        num_internal_edges = int(mask.sum())
        mean_conf = float(np.mean(probs[cluster_node_ids]))
        max_conf = float(np.max(probs[cluster_node_ids]))


        return {
            "node_ids": cluster_node_ids,
            "num_nodes": n,
            "num_edges": num_internal_edges,       # edges internal to the cluster
            "avg_confidence": mean_conf,  # mean P(illicit) across cluster nodes
            "max_confidence": max_conf,  # max P(illicit)
            "risk_level": self._score_to_risk(mean_conf),      # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
            "density": num_internal_edges / (n * (n-1)) if n > 1 else 0.0,         # actual_edges / (n * (n-1))
        }
    
    @staticmethod
    def _score_to_risk(mean_conf: float) -> str:
        if mean_conf >= 0.90:
            return "CRITICAL"
        elif mean_conf >= 0.75:
            return "HIGH"
        elif mean_conf >= 0.60:
            return "MEDIUM"
        else:
            return "LOW"
    

        


 