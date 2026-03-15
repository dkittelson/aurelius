"""Adversarial robustness training for GNN-based AML detection.

Money launderers actively modify transaction patterns to evade detection.
Adversarial training makes the GNN robust by training against worst-case
perturbations — a minimax game.

Implements:
  - FeatureAttacker: PGD (Projected Gradient Descent) on node features
  - TopologyAttacker: Random edge flip baseline
  - AdversarialTrainer: Wraps standard training with adversarial examples
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from loguru import logger

from src.models.gnn_model import AureliusGAT
from src.pipeline.evaluate import evaluate_model


class FeatureAttacker:
    """PGD (Projected Gradient Descent) attacker on node features.

    Computes gradient of loss w.r.t. input features, then iteratively
    perturbs in the gradient direction (maximizing loss = fooling the model).
    Perturbation is projected back into the epsilon-ball at each step.
    """

    def __init__(self, epsilon: float = 0.01, steps: int = 3):
        self.epsilon = epsilon
        self.steps = steps

    def attack(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        y: Tensor,
        criterion: torch.nn.Module,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate adversarial features via PGD.

        Args:
            model:      Target model (must be in eval mode for attack).
            x:          [N, F] original node features.
            edge_index: [2, E] edges.
            y:          [N] labels.
            criterion:  Loss function.
            mask:       Optional boolean mask for which nodes to compute loss on.

        Returns:
            [N, F] perturbed features within epsilon-ball of original.
        """
        x_adv = x.clone().detach().requires_grad_(True)
        step_size = self.epsilon / max(self.steps, 1)

        for _ in range(self.steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            logits = model(x_adv, edge_index)
            if mask is not None:
                loss = criterion(logits[mask], y[mask])
            else:
                loss = criterion(logits, y)

            loss.backward()

            with torch.no_grad():
                # Step in gradient direction (maximize loss)
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + step_size * grad_sign

                # Project back into epsilon-ball
                delta = x_adv - x
                delta = delta.clamp(-self.epsilon, self.epsilon)
                x_adv = (x + delta).detach().requires_grad_(True)

        return x_adv.detach()


class TopologyAttacker:
    """Random edge flip attacker (baseline topology perturbation).

    Randomly adds and removes edges up to a budget. This is a simple
    baseline — gradient-based topology attacks (Metattack) require
    differentiable relaxation of discrete edge additions.

    Args:
        budget_fraction: Fraction of edges to perturb (add + remove).
    """

    def __init__(self, budget_fraction: float = 0.05):
        self.budget_fraction = budget_fraction

    def attack(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Perturb edge_index by randomly flipping edges.

        Args:
            edge_index: [2, E] original edges.
            num_nodes:  Number of nodes in the graph.

        Returns:
            [2, E'] perturbed edge_index.
        """
        num_edges = edge_index.size(1)
        budget = max(1, int(num_edges * self.budget_fraction))
        device = edge_index.device

        # Remove random edges
        num_remove = budget // 2
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        if num_remove > 0 and num_edges > num_remove:
            remove_idx = torch.randperm(num_edges, device=device)[:num_remove]
            keep_mask[remove_idx] = False
        new_edges = edge_index[:, keep_mask]

        # Add random edges
        num_add = budget - num_remove
        if num_add > 0 and num_nodes > 1:
            new_src = torch.randint(0, num_nodes, (num_add,), device=device)
            new_dst = torch.randint(0, num_nodes, (num_add,), device=device)
            added = torch.stack([new_src, new_dst])
            new_edges = torch.cat([new_edges, added], dim=1)

        return new_edges


class AdversarialTrainer:
    """Wraps a standard training step with adversarial examples.

    Per training batch:
      total_loss = (1 - alpha) * clean_loss + alpha * adversarial_loss

    The adversarial loss uses PGD-perturbed features on the same batch,
    forcing the model to be robust to worst-case feature perturbations.

    Args:
        model:        GNN model to train.
        epsilon:      PGD perturbation budget.
        alpha:        Weight of adversarial loss in total (0.0 = clean only).
        attack_steps: Number of PGD steps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.3,
        attack_steps: int = 3,
    ):
        self.model = model
        self.alpha = alpha
        self.feature_attacker = FeatureAttacker(
            epsilon=epsilon, steps=attack_steps
        )
        self.topology_attacker = TopologyAttacker(budget_fraction=0.05)

    def adversarial_step(
        self,
        x: Tensor,
        edge_index: Tensor,
        y: Tensor,
        criterion: torch.nn.Module,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute clean + adversarial loss for one batch.

        Args:
            x, edge_index, y: Batch data.
            criterion: Loss function.
            mask: Optional labeled-node mask.

        Returns:
            (total_loss, clean_loss, adv_loss) — all scalar tensors.
        """
        # Clean forward
        clean_logits = self.model(x, edge_index)
        if mask is not None:
            clean_loss = criterion(clean_logits[mask], y[mask])
        else:
            clean_loss = criterion(clean_logits, y)

        # Generate adversarial features
        self.model.eval()
        x_adv = self.feature_attacker.attack(
            self.model, x, edge_index, y, criterion, mask
        )
        self.model.train()

        # Adversarial forward
        adv_logits = self.model(x_adv, edge_index)
        if mask is not None:
            adv_loss = criterion(adv_logits[mask], y[mask])
        else:
            adv_loss = criterion(adv_logits, y)

        total_loss = (1 - self.alpha) * clean_loss + self.alpha * adv_loss
        return total_loss, clean_loss, adv_loss

    def evaluate_robustness(
        self,
        data: Data,
        mask: Tensor,
        device: torch.device,
        epsilons: list[float] | None = None,
    ) -> dict[float, dict]:
        """Evaluate model accuracy under various perturbation strengths.

        Args:
            data:     Full graph Data.
            mask:     Evaluation mask.
            device:   Torch device.
            epsilons: List of epsilon values to test.

        Returns:
            {epsilon: metrics_dict} mapping.
        """
        if epsilons is None:
            epsilons = [0.0, 0.005, 0.01, 0.02, 0.05]

        criterion = torch.nn.CrossEntropyLoss()
        results = {}

        for eps in epsilons:
            if eps == 0.0:
                metrics = evaluate_model(self.model, data, mask, device)
            else:
                attacker = FeatureAttacker(epsilon=eps, steps=3)
                data_dev = data.to(device)
                mask_dev = mask.to(device)
                labeled = data_dev.y[mask_dev] != -1

                x_adv = attacker.attack(
                    self.model, data_dev.x, data_dev.edge_index,
                    data_dev.y, criterion, mask_dev,
                )

                # Evaluate with adversarial features
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(x_adv, data_dev.edge_index)
                    probs = F.softmax(logits, dim=1)[:, 1]

                probs_sub = probs[mask_dev].cpu().numpy()
                y_sub = data_dev.y[mask_dev].cpu().numpy()
                known = y_sub != -1

                from sklearn.metrics import average_precision_score, f1_score
                if known.sum() > 0 and y_sub[known].sum() > 0:
                    auprc = average_precision_score(y_sub[known], probs_sub[known])
                    preds = (probs_sub[known] >= 0.5).astype(int)
                    f1 = f1_score(y_sub[known], preds, zero_division=0)
                else:
                    auprc, f1 = 0.0, 0.0

                metrics = {"auprc": auprc, "f1": f1}

            results[eps] = metrics
            logger.info(
                f"Robustness @ eps={eps:.4f}: "
                f"AUPRC={metrics.get('auprc', 0):.4f}"
            )

        return results
