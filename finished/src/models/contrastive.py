"""Contrastive / self-supervised learning for GNN pre-training.

Implements two approaches:
  1. DGI (Deep Graph Infomax) — maximize mutual information between
     node embeddings and a graph-level summary vector.
  2. GraphCL — contrastive learning between two augmented graph views
     using NT-Xent loss.

Both methods leverage the ~77% unlabeled nodes in Elliptic for representation
learning before supervised fine-tuning.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT


# ── Graph Augmentation ─────────────────────────────────────────────────────


class GraphAugmentor:
    """Stochastic graph augmentations for contrastive learning.

    Applies a random subset of: node feature masking, edge perturbation,
    and node dropping.
    """

    def __init__(
        self,
        node_drop_rate: float = 0.1,
        edge_perturb_rate: float = 0.1,
        feature_mask_rate: float = 0.2,
    ):
        self.node_drop_rate = node_drop_rate
        self.edge_perturb_rate = edge_perturb_rate
        self.feature_mask_rate = feature_mask_rate

    def augment(self, data: Data) -> Data:
        """Apply full augmentation pipeline (all three transforms)."""
        data = self._mask_features(data)
        data = self._perturb_edges(data)
        data = self._drop_nodes(data)
        return data

    def _drop_nodes(self, data: Data) -> Data:
        """Randomly drop nodes and their incident edges."""
        n = data.num_nodes
        keep_mask = torch.rand(n) >= self.node_drop_rate
        if keep_mask.sum() == 0:
            keep_mask[0] = True  # keep at least one node

        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        # Remap node IDs
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[keep_idx] = torch.arange(keep_idx.size(0))

        src, dst = data.edge_index
        edge_mask = keep_mask[src] & keep_mask[dst]
        new_edge_index = remap[data.edge_index[:, edge_mask]]

        return Data(
            x=data.x[keep_idx],
            edge_index=new_edge_index,
            y=data.y[keep_idx] if data.y is not None else None,
        )

    def _perturb_edges(self, data: Data) -> Data:
        """Randomly add and remove edges."""
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        n = data.num_nodes

        # Remove edges
        keep_mask = torch.rand(num_edges) >= self.edge_perturb_rate
        new_edge_index = edge_index[:, keep_mask]

        # Add random edges (same count as removed)
        num_add = num_edges - int(keep_mask.sum())
        if num_add > 0 and n > 1:
            new_src = torch.randint(0, n, (num_add,))
            new_dst = torch.randint(0, n, (num_add,))
            added = torch.stack([new_src, new_dst])
            new_edge_index = torch.cat([new_edge_index, added], dim=1)

        return Data(x=data.x, edge_index=new_edge_index, y=data.y)

    def _mask_features(self, data: Data) -> Data:
        """Randomly zero-out feature dimensions."""
        x = data.x.clone()
        mask = torch.rand(x.size(1)) < self.feature_mask_rate
        x[:, mask] = 0.0
        return Data(x=x, edge_index=data.edge_index, y=data.y)


# ── Contrastive Encoder ───────────────────────────────────────────────────


class ContrastiveEncoder(nn.Module):
    """Wraps AureliusGAT with a projection head for contrastive learning.

    Architecture:
        GATv2 encoder → JK embedding (384-dim) → MLP projection head → z
    The projection head is discarded after pre-training.
    """

    def __init__(self, encoder: AureliusGAT, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        emb_dim = encoder.jk_out_channels
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ELU(),
            nn.Linear(256, projection_dim),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Returns (embeddings [N, emb_dim], projections [N, proj_dim])."""
        logits, embeddings = self.encoder(
            x, edge_index, return_embeddings=True
        )
        projections = self.projector(embeddings)
        return embeddings, projections

    def get_encoder(self) -> AureliusGAT:
        """Return the pre-trained encoder (without projection head)."""
        return self.encoder


# ── DGI (Deep Graph Infomax) ──────────────────────────────────────────────


class Discriminator(nn.Module):
    """Bilinear discriminator for DGI: scores (node_emb, summary) pairs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_emb: Tensor, summary: Tensor) -> Tensor:
        """Score each node against the global summary. Returns [N] logits."""
        # node_emb: [N, D], summary: [D]
        return (node_emb @ self.weight * summary).sum(dim=1)


class DGILoss(nn.Module):
    """Deep Graph Infomax loss.

    Maximizes mutual information between node embeddings and a graph-level
    summary by discriminating real (positive) from corrupted (negative) nodes.

    Corruption strategy: random permutation of node features.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.disc = Discriminator(hidden_dim)

    def forward(
        self, pos_z: Tensor, neg_z: Tensor, summary: Tensor
    ) -> Tensor:
        """
        Args:
            pos_z:   [N, D] embeddings from real graph
            neg_z:   [N, D] embeddings from corrupted graph
            summary: [D] graph-level summary (mean of pos_z)

        Returns:
            Scalar BCE loss.
        """
        pos_scores = self.disc(pos_z, summary)
        neg_scores = self.disc(neg_z, summary)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        return pos_loss + neg_loss


# ── GraphCL (NT-Xent) ────────────────────────────────────────────────────


class GraphCLLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Used in GraphCL: two augmented views produce graph-level representations;
    we maximize agreement between views of the same graph.

    For node-level contrastive learning, we compute the loss between
    corresponding node embeddings in two augmented views.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Node-level NT-Xent between two views.

        Args:
            z1: [N, D] projections from view 1
            z2: [N, D] projections from view 2

        Returns:
            Scalar loss.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        # Similarity matrix: [2N, 2N]
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = z @ z.T / self.temperature  # [2N, 2N]

        # Mask out self-similarity
        mask = ~torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat(
            [torch.arange(N, 2 * N), torch.arange(N)], dim=0
        ).to(z.device)

        return F.cross_entropy(sim, labels)
