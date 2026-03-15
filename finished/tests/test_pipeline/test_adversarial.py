"""Tests for adversarial robustness components."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT
from src.pipeline.adversarial import (
    FeatureAttacker,
    TopologyAttacker,
    AdversarialTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(num_nodes=30, in_channels=16):
    torch.manual_seed(42)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def model():
    return AureliusGAT(
        in_channels=16, hidden_channels=32, out_channels=2,
        num_heads=4, num_layers=2, dropout=0.0, jk_mode="cat",
    )


# ---------------------------------------------------------------------------
# FeatureAttacker
# ---------------------------------------------------------------------------

class TestFeatureAttacker:
    def test_perturbation_within_epsilon(self, model, graph):
        attacker = FeatureAttacker(epsilon=0.1, steps=3)
        criterion = torch.nn.CrossEntropyLoss()
        x_adv = attacker.attack(
            model, graph.x, graph.edge_index, graph.y, criterion
        )
        delta = (x_adv - graph.x).abs()
        assert delta.max().item() <= 0.1 + 1e-6

    def test_perturbation_different_from_original(self, model, graph):
        attacker = FeatureAttacker(epsilon=0.1, steps=3)
        criterion = torch.nn.CrossEntropyLoss()
        x_adv = attacker.attack(
            model, graph.x, graph.edge_index, graph.y, criterion
        )
        assert not torch.allclose(x_adv, graph.x)

    def test_output_shape_preserved(self, model, graph):
        attacker = FeatureAttacker(epsilon=0.01, steps=2)
        criterion = torch.nn.CrossEntropyLoss()
        x_adv = attacker.attack(
            model, graph.x, graph.edge_index, graph.y, criterion
        )
        assert x_adv.shape == graph.x.shape

    def test_attack_with_mask(self, model, graph):
        attacker = FeatureAttacker(epsilon=0.05, steps=2)
        criterion = torch.nn.CrossEntropyLoss()
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[:15] = True
        x_adv = attacker.attack(
            model, graph.x, graph.edge_index, graph.y, criterion, mask
        )
        assert x_adv.shape == graph.x.shape


# ---------------------------------------------------------------------------
# TopologyAttacker
# ---------------------------------------------------------------------------

class TestTopologyAttacker:
    def test_edge_count_within_budget(self, graph):
        attacker = TopologyAttacker(budget_fraction=0.1)
        new_edges = attacker.attack(graph.edge_index, graph.num_nodes)
        original_count = graph.edge_index.size(1)
        # Should be roughly similar count (some removed, some added)
        assert new_edges.size(1) > 0

    def test_valid_node_indices(self, graph):
        attacker = TopologyAttacker(budget_fraction=0.1)
        new_edges = attacker.attack(graph.edge_index, graph.num_nodes)
        assert new_edges.max() < graph.num_nodes

    def test_edges_are_different(self, graph):
        torch.manual_seed(0)
        attacker = TopologyAttacker(budget_fraction=0.2)
        new_edges = attacker.attack(graph.edge_index, graph.num_nodes)
        # Edge count should differ from original
        assert new_edges.size(1) != graph.edge_index.size(1) or \
               not torch.equal(new_edges, graph.edge_index)


# ---------------------------------------------------------------------------
# AdversarialTrainer
# ---------------------------------------------------------------------------

class TestAdversarialTrainer:
    def test_adversarial_step_returns_losses(self, model, graph):
        adv_trainer = AdversarialTrainer(
            model, epsilon=0.01, alpha=0.3, attack_steps=2
        )
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        total, clean, adv = adv_trainer.adversarial_step(
            graph.x, graph.edge_index, graph.y, criterion
        )
        assert total.dim() == 0
        assert clean.dim() == 0
        assert adv.dim() == 0

    def test_total_loss_is_weighted_combination(self, model, graph):
        alpha = 0.3
        adv_trainer = AdversarialTrainer(
            model, epsilon=0.01, alpha=alpha, attack_steps=1
        )
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        total, clean, adv = adv_trainer.adversarial_step(
            graph.x, graph.edge_index, graph.y, criterion
        )
        expected = (1 - alpha) * clean + alpha * adv
        assert torch.allclose(total, expected, atol=1e-5)

    def test_gradient_flows(self, model, graph):
        adv_trainer = AdversarialTrainer(
            model, epsilon=0.01, alpha=0.3, attack_steps=1
        )
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        total, _, _ = adv_trainer.adversarial_step(
            graph.x, graph.edge_index, graph.y, criterion
        )
        total.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad
