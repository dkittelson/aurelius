"""Graph-level feature engineering: PageRank, centrality, structural embeddings."""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from loguru import logger


class GraphFeatureEngineer:
    """Compute structural graph features and attach to PyG Data."""

    def __init__(self, config: dict):
        self.alpha = config["features"]["pagerank_alpha"]
        self.centrality_types = config["features"]["centrality_types"]
        self.embedding_dim = config["features"]["embedding_dim"]

    def compute_pagerank(self, data: Data) -> torch.Tensor:
        """
        Compute PageRank scores for all nodes.
        Returns [num_nodes, 1] tensor.
        """
        G = to_networkx(data, to_undirected=False)
        pr = nx.pagerank(G, alpha=self.alpha)
        scores = np.array([pr.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32)
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def compute_centrality(self, data: Data, kind: str) -> torch.Tensor:
        """
        Compute centrality measure for all nodes.

        Args:
            kind: One of 'degree', 'betweenness', 'closeness'

        Returns [num_nodes, 1] tensor.
        """
        G = to_networkx(data, to_undirected=False)

        if kind == "degree":
            centrality = nx.degree_centrality(G)
        elif kind == "betweenness":
            # Approximate for large graphs
            k = min(500, G.number_of_nodes())
            centrality = nx.betweenness_centrality(G, k=k)
        elif kind == "closeness":
            centrality = nx.closeness_centrality(G)
        else:
            raise ValueError(f"Unknown centrality type: {kind}")

        scores = np.array(
            [centrality.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32
        )
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def compute_local_clustering(self, data: Data) -> torch.Tensor:
        """Compute local clustering coefficient per node. [num_nodes, 1]."""
        G = to_networkx(data, to_undirected=True)
        clustering = nx.clustering(G)
        scores = np.array(
            [clustering.get(i, 0.0) for i in range(data.num_nodes)], dtype=np.float32
        )
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def compute_all(self, data: Data) -> Data:
        """
        Compute all structural features and concatenate to data.x.

        Adds: PageRank + all configured centralities + local clustering.
        """
        logger.info("Computing structural features...")
        new_features = []

        logger.info("  Computing PageRank...")
        new_features.append(self.compute_pagerank(data))

        for ctype in self.centrality_types:
            logger.info(f"  Computing {ctype} centrality...")
            new_features.append(self.compute_centrality(data, ctype))

        logger.info("  Computing local clustering coefficient...")
        new_features.append(self.compute_local_clustering(data))

        # Concatenate all new features to existing node features
        extra = torch.cat(new_features, dim=1)
        data.x = torch.cat([data.x, extra], dim=1)

        logger.info(
            f"Added {extra.shape[1]} structural features. "
            f"New feature dimension: {data.x.shape[1]}"
        )
        return data

    def compute_structural_embeddings(self, data: Data, dim: int = None) -> torch.Tensor:
        """
        Compute Node2Vec-style structural embeddings.

        Uses torch_geometric.nn.Node2Vec for random-walk based embeddings.
        Returns [num_nodes, dim] tensor.
        """
        from torch_geometric.nn import Node2Vec

        dim = dim or self.embedding_dim

        device = torch.device("cpu")
        model = Node2Vec(
            data.edge_index,
            embedding_dim=dim,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        logger.info(f"Training Node2Vec embeddings (dim={dim})...")
        model.train()
        for epoch in range(50):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            embeddings = model()

        logger.info(f"Node2Vec embeddings computed: {embeddings.shape}")
        return embeddings
