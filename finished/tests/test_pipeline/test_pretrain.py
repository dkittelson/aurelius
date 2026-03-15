"""Tests for self-supervised pre-training pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT
from src.models.contrastive import ContrastiveEncoder
from src.pipeline.pretrain import PreTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(num_nodes=40, in_channels=16):
    torch.manual_seed(42)
    src = torch.randint(0, num_nodes, (80,))
    dst = torch.randint(0, num_nodes, (80,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def minimal_config(tmp_path):
    # Save a small graph to tmp_path for loading
    graph = _make_graph()
    torch.save(graph, tmp_path / "elliptic_graph.pt")
    return {
        "data": {"processed_dir": str(tmp_path)},
        "model": {
            "gnn": {
                "hidden_channels": 32,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.0,
                "out_channels": 2,
                "jk_mode": "cat",
                "residual": True,
            },
            "contrastive": {
                "projection_dim": 64,
                "temperature": 0.5,
                "node_drop_rate": 0.1,
                "edge_perturb_rate": 0.1,
                "feature_mask_rate": 0.2,
                "pretrain_epochs": 3,
                "pretrain_lr": 0.001,
            },
        },
    }


# ---------------------------------------------------------------------------
# PreTrainer setup
# ---------------------------------------------------------------------------

class TestPreTrainerSetup:
    def test_setup_data_loads_graph(self, minimal_config):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        assert pt.data is not None
        assert pt.data.num_nodes == 40

    def test_setup_encoder_creates_model(self, minimal_config):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        pt.setup_encoder()
        assert pt.contrastive_encoder is not None
        assert pt.augmentor is not None

    def test_setup_data_missing_file(self, minimal_config):
        minimal_config["data"]["processed_dir"] = "/nonexistent"
        pt = PreTrainer(minimal_config, device="cpu")
        with pytest.raises(FileNotFoundError):
            pt.setup_data("elliptic")


# ---------------------------------------------------------------------------
# DGI pre-training
# ---------------------------------------------------------------------------

class TestPretrainDGI:
    def test_dgi_returns_history(self, minimal_config):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        pt.setup_encoder()
        history = pt.pretrain_dgi(epochs=2)
        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_dgi_loss_decreases_or_finite(self, minimal_config):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        pt.setup_encoder()
        history = pt.pretrain_dgi(epochs=3)
        assert all(torch.isfinite(torch.tensor(l)) for l in history["loss"])


# ---------------------------------------------------------------------------
# GraphCL pre-training
# ---------------------------------------------------------------------------

class TestPretrainGraphCL:
    def test_graphcl_returns_history(self, minimal_config):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        pt.setup_encoder()
        history = pt.pretrain_graphcl(epochs=2)
        assert "loss" in history
        assert len(history["loss"]) == 2


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_roundtrip(self, minimal_config, tmp_path):
        pt = PreTrainer(minimal_config, device="cpu")
        pt.setup_data("elliptic")
        pt.setup_encoder()

        save_path = str(tmp_path / "pretrained.pt")
        pt.save_pretrained(save_path)
        assert Path(save_path).exists()

        # Load into a fresh model
        model = AureliusGAT(
            in_channels=16, hidden_channels=32, out_channels=2,
            num_heads=4, num_layers=2, dropout=0.0, jk_mode="cat",
        )
        PreTrainer.load_pretrained_into_model(model, save_path)
        # Verify weights match
        original_state = pt.contrastive_encoder.encoder.state_dict()
        loaded_state = model.state_dict()
        for key in original_state:
            if key in loaded_state:
                assert torch.allclose(original_state[key], loaded_state[key])
