"""Multi-task learning for GNN-based AML detection.

Shared GNN backbone with multiple prediction heads:
  - Classification: illicit/licit node classification (primary)
  - Link prediction: predict existence of edges (auxiliary)
  - Degree regression: predict log(degree + 1) (auxiliary)

Auxiliary tasks act as regularizers, providing free supervision signals
that improve the shared backbone without requiring additional labels.

Task weighting uses uncertainty-based approach (Kendall et al., 2018):
  total_loss = sum( L_task / (2 * sigma_task^2) + log(sigma_task) )
where sigma_task is a learnable parameter per task.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.models.gnn_model import AureliusGAT


class MultiTaskHead(nn.Module):
    """Collection of prediction heads sharing embeddings from a GNN backbone.

    Heads:
      - classify:        Linear(emb_dim, out_channels) → illicit/licit logits
      - predict_links:   dot product decoder → link probability
      - predict_degree:  Linear(emb_dim, 1) → log(degree + 1)
    """

    def __init__(self, embedding_dim: int, out_channels: int = 2):
        super().__init__()
        self.cls_head = nn.Linear(embedding_dim, out_channels)
        self.deg_head = nn.Linear(embedding_dim, 1)

    def classify(self, embeddings: Tensor) -> Tensor:
        """[N, emb_dim] → [N, out_channels]"""
        return self.cls_head(embeddings)

    def predict_links(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        """Dot product link predictor. [E] probabilities."""
        return (z_src * z_dst).sum(dim=1)

    def predict_degree(self, embeddings: Tensor) -> Tensor:
        """[N, emb_dim] → [N] predicted log(degree + 1)."""
        return self.deg_head(embeddings).squeeze(-1)


class MultiTaskAureliusGAT(nn.Module):
    """AureliusGAT backbone + MultiTaskHead.

    Replaces the encoder's built-in classifier with multi-task heads.

    Args:
        encoder: Pre-built AureliusGAT instance.
        out_channels: Number of classes for classification head.
    """

    def __init__(self, encoder: AureliusGAT, out_channels: int = 2):
        super().__init__()
        self.encoder = encoder
        emb_dim = encoder.jk_out_channels
        self.heads = MultiTaskHead(emb_dim, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        tasks: list[str] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass returning predictions for requested tasks.

        Args:
            x:          [N, in_channels] node features
            edge_index: [2, E] edge index
            tasks:      List of task names. Default: ["classification"]
                        Options: "classification", "link_prediction", "degree"

        Returns:
            Dict of task_name → prediction tensor.
        """
        if tasks is None:
            tasks = ["classification"]

        # Get embeddings from encoder (skip its classifier)
        _, embeddings = self.encoder(
            x, edge_index, return_embeddings=True
        )

        results = {}

        if "classification" in tasks:
            results["classification"] = self.heads.classify(embeddings)

        if "link_prediction" in tasks:
            # Use actual edges as positive samples
            src, dst = edge_index
            results["link_prediction"] = self.heads.predict_links(
                embeddings[src], embeddings[dst]
            )

        if "degree" in tasks:
            results["degree"] = self.heads.predict_degree(embeddings)

        return results

    def get_embeddings(
        self, x: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Return JK-aggregated embeddings."""
        return self.encoder.get_embeddings(x, edge_index)


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Each task has a learnable log_sigma. The effective loss is:
        L_total = sum_task( L_task / (2 * exp(2 * log_sigma_task)) + log_sigma_task )

    This automatically balances task losses — tasks with higher uncertainty
    get lower weight.
    """

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = task_names
        # Initialize log_sigma to 0 (sigma = 1)
        self.log_sigmas = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in task_names}
        )

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        """Combine per-task losses with uncertainty weighting.

        Args:
            losses: dict of task_name → scalar loss tensor

        Returns:
            Weighted total loss (scalar).
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name in self.task_names:
            if name not in losses:
                continue
            log_sigma = self.log_sigmas[name].squeeze()
            # L / (2 * sigma^2) + log(sigma) = L / (2 * exp(2*log_sigma)) + log_sigma
            precision = torch.exp(-2 * log_sigma)
            total = total + precision * losses[name] + log_sigma
        return total


def compute_link_prediction_loss(
    predictions: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    num_neg_samples: int | None = None,
) -> Tensor:
    """BCE loss for link prediction with negative sampling.

    Args:
        predictions: [E] positive edge scores from heads.predict_links
        edge_index:  [2, E] actual edges
        num_nodes:   Total number of nodes (for negative sampling)
        num_neg_samples: Number of negative edges. Defaults to len(predictions).

    Returns:
        Scalar BCE loss.
    """
    num_pos = predictions.size(0)
    num_neg = num_neg_samples or num_pos
    device = predictions.device

    # Positive loss
    pos_loss = F.binary_cross_entropy_with_logits(
        predictions, torch.ones(num_pos, device=device)
    )

    # Negative sampling: random node pairs (likely not edges)
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_scores = torch.zeros(num_neg, device=device)  # placeholder

    neg_loss = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros(num_neg, device=device)
    )

    return (pos_loss + neg_loss) / 2


def compute_degree_targets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute log(degree + 1) targets for degree regression.

    Args:
        edge_index: [2, E]
        num_nodes:  Total number of nodes

    Returns:
        [num_nodes] tensor of log(degree + 1) values.
    """
    degree = torch.zeros(num_nodes, device=edge_index.device)
    src = edge_index[0]
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    return torch.log(degree + 1)
