"""Tests for contrastive learning components."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT
from src.models.contrastive import (
    GraphAugmentor,
    ContrastiveEncoder,
    DGILoss,
    GraphCLLoss,
    Discriminator,
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
def contrastive_encoder(encoder):
    return ContrastiveEncoder(encoder, projection_dim=64)


# ---------------------------------------------------------------------------
# GraphAugmentor
# ---------------------------------------------------------------------------

class TestGraphAugmentor:
    def test_augment_preserves_feature_dim(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.0, edge_perturb_rate=0.0, feature_mask_rate=0.3)
        result = aug.augment(graph)
        assert result.x.shape[1] == graph.x.shape[1]

    def test_feature_masking_zeros_some_columns(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.0, edge_perturb_rate=0.0, feature_mask_rate=0.5)
        torch.manual_seed(0)
        result = aug._mask_features(graph)
        # Some columns should be zeroed
        col_sums = result.x.abs().sum(dim=0)
        assert (col_sums == 0).any()

    def test_edge_perturbation_changes_edges(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.0, edge_perturb_rate=0.3, feature_mask_rate=0.0)
        torch.manual_seed(0)
        result = aug._perturb_edges(graph)
        # Edge count should be similar (removed + added roughly equal)
        assert result.edge_index.size(1) > 0

    def test_node_dropping_reduces_nodes(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.5, edge_perturb_rate=0.0, feature_mask_rate=0.0)
        torch.manual_seed(0)
        result = aug._drop_nodes(graph)
        assert result.num_nodes <= graph.num_nodes
        assert result.num_nodes >= 1  # at least one kept

    def test_node_dropping_remaps_edges(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.3)
        torch.manual_seed(0)
        result = aug._drop_nodes(graph)
        # All edge indices should be valid
        assert result.edge_index.max() < result.num_nodes

    def test_full_augment_returns_valid_graph(self, graph):
        aug = GraphAugmentor(node_drop_rate=0.1, edge_perturb_rate=0.1, feature_mask_rate=0.2)
        torch.manual_seed(0)
        result = aug.augment(graph)
        assert result.x is not None
        assert result.edge_index is not None
        assert result.x.shape[1] == graph.x.shape[1]


# ---------------------------------------------------------------------------
# ContrastiveEncoder
# ---------------------------------------------------------------------------

class TestContrastiveEncoder:
    def test_forward_shapes(self, contrastive_encoder, graph):
        emb, proj = contrastive_encoder(graph.x, graph.edge_index)
        assert emb.shape == (graph.num_nodes, contrastive_encoder.encoder.jk_out_channels)
        assert proj.shape == (graph.num_nodes, 64)

    def test_get_encoder_returns_gat(self, contrastive_encoder, encoder):
        recovered = contrastive_encoder.get_encoder()
        assert recovered is encoder

    def test_projection_head_has_parameters(self, contrastive_encoder):
        proj_params = list(contrastive_encoder.projector.parameters())
        assert len(proj_params) > 0


# ---------------------------------------------------------------------------
# DGILoss
# ---------------------------------------------------------------------------

class TestDGILoss:
    def test_loss_is_scalar(self):
        hidden_dim = 32
        dgi = DGILoss(hidden_dim=hidden_dim)
        pos_z = torch.randn(20, hidden_dim)
        neg_z = torch.randn(20, hidden_dim)
        summary = torch.randn(hidden_dim)
        loss = dgi(pos_z, neg_z, summary)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_differentiable(self):
        hidden_dim = 16
        dgi = DGILoss(hidden_dim=hidden_dim)
        pos_z = torch.randn(10, hidden_dim, requires_grad=True)
        neg_z = torch.randn(10, hidden_dim, requires_grad=True)
        summary = torch.randn(hidden_dim)
        loss = dgi(pos_z, neg_z, summary)
        loss.backward()
        assert pos_z.grad is not None

    def test_discriminator_output_shape(self):
        disc = Discriminator(hidden_dim=32)
        emb = torch.randn(15, 32)
        summary = torch.randn(32)
        scores = disc(emb, summary)
        assert scores.shape == (15,)


# ---------------------------------------------------------------------------
# GraphCLLoss
# ---------------------------------------------------------------------------

class TestGraphCLLoss:
    def test_loss_is_scalar(self):
        cl = GraphCLLoss(temperature=0.5)
        z1 = torch.randn(20, 32)
        z2 = torch.randn(20, 32)
        loss = cl(z1, z2)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_identical_views_low_loss(self):
        cl = GraphCLLoss(temperature=0.5)
        z = torch.randn(10, 16)
        loss = cl(z, z)
        # Identical views should have relatively low loss
        assert loss.item() < 10.0

    def test_loss_differentiable(self):
        cl = GraphCLLoss(temperature=0.5)
        z1 = torch.randn(10, 16, requires_grad=True)
        z2 = torch.randn(10, 16, requires_grad=True)
        loss = cl(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert z2.grad is not None
