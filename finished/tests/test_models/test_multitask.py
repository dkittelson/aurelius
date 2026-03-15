"""Tests for multi-task learning components."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT
from src.models.multitask import (
    MultiTaskHead,
    MultiTaskAureliusGAT,
    MultiTaskLoss,
    compute_link_prediction_loss,
    compute_degree_targets,
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
def encoder():
    return AureliusGAT(
        in_channels=16, hidden_channels=32, out_channels=2,
        num_heads=4, num_layers=2, dropout=0.0, jk_mode="cat",
    )


@pytest.fixture
def mt_model(encoder):
    return MultiTaskAureliusGAT(encoder, out_channels=2)


# ---------------------------------------------------------------------------
# MultiTaskHead
# ---------------------------------------------------------------------------

class TestMultiTaskHead:
    def test_classify_shape(self):
        head = MultiTaskHead(embedding_dim=64, out_channels=2)
        emb = torch.randn(20, 64)
        out = head.classify(emb)
        assert out.shape == (20, 2)

    def test_predict_links_shape(self):
        head = MultiTaskHead(embedding_dim=64)
        z_src = torch.randn(30, 64)
        z_dst = torch.randn(30, 64)
        out = head.predict_links(z_src, z_dst)
        assert out.shape == (30,)

    def test_predict_degree_shape(self):
        head = MultiTaskHead(embedding_dim=64)
        emb = torch.randn(20, 64)
        out = head.predict_degree(emb)
        assert out.shape == (20,)


# ---------------------------------------------------------------------------
# MultiTaskAureliusGAT
# ---------------------------------------------------------------------------

class TestMultiTaskAureliusGAT:
    def test_classification_only(self, mt_model, graph):
        result = mt_model(graph.x, graph.edge_index, tasks=["classification"])
        assert "classification" in result
        assert result["classification"].shape == (graph.num_nodes, 2)

    def test_all_tasks(self, mt_model, graph):
        result = mt_model(
            graph.x, graph.edge_index,
            tasks=["classification", "link_prediction", "degree"],
        )
        assert "classification" in result
        assert "link_prediction" in result
        assert "degree" in result
        assert result["classification"].shape == (graph.num_nodes, 2)
        assert result["degree"].shape == (graph.num_nodes,)

    def test_default_tasks(self, mt_model, graph):
        result = mt_model(graph.x, graph.edge_index)
        assert "classification" in result
        assert "link_prediction" not in result


# ---------------------------------------------------------------------------
# MultiTaskLoss
# ---------------------------------------------------------------------------

class TestMultiTaskLoss:
    def test_loss_is_scalar(self):
        mt_loss = MultiTaskLoss(["classification", "degree"])
        losses = {
            "classification": torch.tensor(1.5),
            "degree": torch.tensor(0.8),
        }
        total = mt_loss(losses)
        assert total.dim() == 0

    def test_learnable_sigmas(self):
        mt_loss = MultiTaskLoss(["classification", "link_prediction"])
        assert "classification" in mt_loss.log_sigmas
        assert mt_loss.log_sigmas["classification"].requires_grad

    def test_gradient_flows_to_sigmas(self):
        mt_loss = MultiTaskLoss(["classification"])
        losses = {"classification": torch.tensor(1.0, requires_grad=True)}
        total = mt_loss(losses)
        total.backward()
        assert mt_loss.log_sigmas["classification"].grad is not None

    def test_missing_task_skipped(self):
        mt_loss = MultiTaskLoss(["classification", "degree"])
        losses = {"classification": torch.tensor(1.0)}
        total = mt_loss(losses)
        assert total.dim() == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_degree_targets_shape(self, graph):
        targets = compute_degree_targets(graph.edge_index, graph.num_nodes)
        assert targets.shape == (graph.num_nodes,)
        assert (targets >= 0).all()

    def test_link_prediction_loss_is_scalar(self):
        preds = torch.randn(20)
        edge_index = torch.randint(0, 10, (2, 20))
        loss = compute_link_prediction_loss(preds, edge_index, num_nodes=10)
        assert loss.dim() == 0
        assert loss.item() > 0
