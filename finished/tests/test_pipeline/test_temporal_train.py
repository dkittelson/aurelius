"""Tests for temporal training pipeline."""

import pytest
import torch
from torch_geometric.data import Data

from src.pipeline.temporal_train import TemporalTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_temporal_graph(num_nodes=60, in_channels=16, num_timesteps=5):
    """Create a synthetic graph with timestep assignments."""
    torch.manual_seed(42)
    src = torch.randint(0, num_nodes, (120,))
    dst = torch.randint(0, num_nodes, (120,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.randint(0, 2, (num_nodes,))

    # Assign timesteps evenly
    timesteps = torch.arange(num_nodes) % num_timesteps + 1

    # Masks based on timesteps
    train_mask = (timesteps <= 3) & (y != -1)
    val_mask = (timesteps == 4) & (y != -1)
    test_mask = (timesteps == 5) & (y != -1)

    return Data(
        x=x, edge_index=edge_index, y=y,
        timestep=timesteps.int(),
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        node_ids=torch.arange(num_nodes),
    )


@pytest.fixture
def minimal_config(tmp_path):
    graph = _make_temporal_graph()
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
            "training": {
                "epochs": 3,
                "lr": 0.001,
                "weight_decay": 5e-4,
                "patience": 20,
                "batch_size": 64,
                "num_neighbors": [5, 3],
            },
            "temporal": {
                "memory_dim": 16,
                "bptt_steps": 2,
                "temporal_dropout": 0.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalTrainerSetup:
    def test_setup_data_creates_snapshots(self, minimal_config):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        assert len(trainer.snapshots) == 5  # 5 timesteps
        assert len(trainer.global_ids_list) == 5

    def test_snapshots_have_valid_edges(self, minimal_config):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        for snap in trainer.snapshots:
            if snap.edge_index.numel() > 0:
                assert snap.edge_index.max() < snap.num_nodes

    def test_setup_model_creates_temporal_gat(self, minimal_config):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        trainer.setup_model()
        assert trainer.model is not None


class TestTemporalTraining:
    def test_train_returns_history(self, minimal_config):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        trainer.setup_model()
        history = trainer.train_temporal(epochs=2)
        assert "train_loss" in history
        assert "val_auprc" in history
        assert len(history["train_loss"]) == 2

    def test_train_loss_is_finite(self, minimal_config):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        trainer.setup_model()
        history = trainer.train_temporal(epochs=2)
        for loss in history["train_loss"]:
            assert torch.isfinite(torch.tensor(loss))

    def test_pretrained_encoder_loading(self, minimal_config, tmp_path):
        trainer = TemporalTrainer(minimal_config, device="cpu")
        trainer.setup_data("elliptic")
        trainer.setup_model()

        # Save encoder weights
        encoder_path = str(tmp_path / "pretrained.pt")
        torch.save(trainer.model.encoder.state_dict(), encoder_path)

        # Load into fresh trainer
        trainer2 = TemporalTrainer(minimal_config, device="cpu")
        trainer2.setup_data("elliptic")
        trainer2.setup_model()
        trainer2.load_pretrained_encoder(encoder_path)
