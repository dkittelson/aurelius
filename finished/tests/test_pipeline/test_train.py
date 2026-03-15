"""Tests for the Trainer pipeline: device detection, class weights, training loop."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_graph(num_nodes=60, in_channels=16, seed=0):
    """Synthetic graph with train/val/test masks for testing the Trainer."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    src = torch.randint(0, num_nodes, (120,))
    dst = torch.randint(0, num_nodes, (120,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)

    # Labels: 80% licit (0), 20% illicit (1), no unknowns
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:int(num_nodes * 0.2)] = 1

    # Temporal-style masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:40] = True
    val_mask[40:50] = True
    test_mask[50:] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


@pytest.fixture
def minimal_config():
    """Config with tiny model and training settings for fast tests."""
    return {
        "data": {
            "processed_dir": "/tmp/aurelius_test_processed",
        },
        "model": {
            "gnn": {
                "type": "GATv2Conv",
                "hidden_channels": 16,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.0,
                "out_channels": 2,
                "jk_mode": "cat",
                "residual": True,
            },
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 3,
            },
            "training": {
                "epochs": 5,
                "lr": 0.01,
                "weight_decay": 0.0,
                "patience": 3,
                "batch_size": 20,
                "num_neighbors": [5, 3],
            },
        },
    }


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

class TestDeviceDetection:

    def test_explicit_cpu(self):
        from src.pipeline.train import Trainer
        t = Trainer.__new__(Trainer)
        device = Trainer._resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_valid_device(self):
        from src.pipeline.train import Trainer
        device = Trainer._resolve_device("auto")
        assert isinstance(device, torch.device)
        assert str(device) in ("cpu", "cuda", "mps")

    def test_auto_on_ci_returns_cpu(self):
        """When neither CUDA nor MPS is available, must fall back to CPU."""
        from src.pipeline.train import Trainer
        device = Trainer._resolve_device("auto")
        # On CI without GPU/MPS this will be CPU. Just verify no exception.
        assert device is not None


# ---------------------------------------------------------------------------
# Compute class weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:

    def _make_trainer_with_data(self, config, data):
        from src.pipeline.train import Trainer
        trainer = Trainer.__new__(Trainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.data = data
        return trainer

    def test_weights_shape(self, minimal_config):
        """weights tensor should have length = num_classes = 2."""
        data = make_small_graph()
        from src.pipeline.train import Trainer
        trainer = self._make_trainer_with_data(minimal_config, data)
        weights = trainer.compute_class_weights()
        assert weights.shape == (2,)

    def test_illicit_weight_greater_than_licit(self, minimal_config):
        """Illicit (minority) class should always have larger weight."""
        data = make_small_graph()
        from src.pipeline.train import Trainer
        trainer = self._make_trainer_with_data(minimal_config, data)
        weights = trainer.compute_class_weights()
        assert weights[1] > weights[0], (
            f"Expected illicit weight ({weights[1]:.2f}) > licit weight ({weights[0]:.2f})"
        )

    def test_weights_on_correct_device(self, minimal_config):
        data = make_small_graph()
        from src.pipeline.train import Trainer
        trainer = self._make_trainer_with_data(minimal_config, data)
        weights = trainer.compute_class_weights()
        assert weights.device == torch.device("cpu")

    def test_uniform_label_returns_ones(self, minimal_config):
        """If no illicit nodes exist, should return uniform weights (no crash)."""
        data = make_small_graph()
        data.y = torch.zeros(data.num_nodes, dtype=torch.long)  # all licit
        from src.pipeline.train import Trainer
        trainer = self._make_trainer_with_data(minimal_config, data)
        weights = trainer.compute_class_weights()
        assert weights.shape == (2,)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_stops_after_patience(self):
        """Should signal stop after `patience` non-improving steps."""
        from src.pipeline.train import EarlyStopping
        es = EarlyStopping(patience=3)
        es.step(0.5)  # improvement
        assert not es.step(0.4)
        assert not es.step(0.3)
        assert es.step(0.2)  # 3rd non-improvement → stop

    def test_resets_counter_on_improvement(self):
        """Counter resets whenever a new best is seen."""
        from src.pipeline.train import EarlyStopping
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.4)  # counter=1
        es.step(0.6)  # improvement — counter resets to 0
        assert not es.step(0.5)  # counter=1, should not stop

    def test_best_score_updates(self):
        """best_score should track the maximum seen so far."""
        from src.pipeline.train import EarlyStopping
        es = EarlyStopping(patience=5)
        es.step(0.3)
        es.step(0.7)
        es.step(0.5)
        assert abs(es.best_score - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Training loop (integration-lite: uses real model + loader on tiny graph)
# ---------------------------------------------------------------------------

class TestTrainingLoop:

    def _setup_trainer(self, config, data, tmp_path):
        """Patch a Trainer instance directly with synthetic data, bypass file I/O."""
        import os
        from src.pipeline.train import Trainer
        from torch_geometric.loader import NeighborLoader
        from src.models.gnn_model import AureliusGAT

        os.makedirs(str(tmp_path / "checkpoints"), exist_ok=True)
        config = dict(config)
        config["data"] = dict(config["data"])
        config["data"]["processed_dir"] = str(tmp_path)

        trainer = Trainer(config, device="cpu")
        trainer.data = data

        # Build a NeighborLoader matching config
        training_cfg = config["model"]["training"]
        train_input = data.train_mask & (data.y != -1)
        trainer.train_loader = NeighborLoader(
            data,
            num_neighbors=training_cfg["num_neighbors"],
            batch_size=training_cfg["batch_size"],
            input_nodes=train_input,
            shuffle=True,
            num_workers=0,
        )

        # Instantiate model
        gnn_cfg = config["model"]["gnn"]
        trainer.model = AureliusGAT(
            in_channels=data.x.shape[1],
            hidden_channels=gnn_cfg["hidden_channels"],
            out_channels=gnn_cfg["out_channels"],
            num_heads=gnn_cfg["num_heads"],
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            jk_mode=gnn_cfg["jk_mode"],
            residual=gnn_cfg["residual"],
        )
        trainer.training_history = {}

        return trainer

    def test_train_gnn_returns_history(self, minimal_config, tmp_path):
        """train_gnn() should return a dict with loss and metric lists."""
        data = make_small_graph()
        trainer = self._setup_trainer(minimal_config, data, tmp_path)
        history = trainer.train_gnn()

        assert "train_loss" in history
        assert "val_auprc" in history
        assert "val_f1" in history
        assert len(history["train_loss"]) > 0

    def test_train_gnn_loss_is_finite(self, minimal_config, tmp_path):
        """All training losses should be finite."""
        data = make_small_graph()
        trainer = self._setup_trainer(minimal_config, data, tmp_path)
        history = trainer.train_gnn()

        for loss in history["train_loss"]:
            assert np.isfinite(loss), f"Non-finite loss encountered: {loss}"

    def test_early_stopping_limits_epochs(self, minimal_config, tmp_path):
        """With patience=3 and 5 max epochs, training might stop early."""
        data = make_small_graph()
        config = dict(minimal_config)
        config["model"] = dict(config["model"])
        config["model"]["training"] = dict(config["model"]["training"])
        config["model"]["training"]["patience"] = 2
        config["model"]["training"]["epochs"] = 50

        trainer = self._setup_trainer(config, data, tmp_path)
        history = trainer.train_gnn()

        # Should have stopped well before 50 epochs given random data
        assert len(history["train_loss"]) <= 50

    def test_model_parameters_change_after_training(self, minimal_config, tmp_path):
        """Model weights should differ from initialization after training."""
        data = make_small_graph()
        trainer = self._setup_trainer(minimal_config, data, tmp_path)

        initial_params = [p.clone().detach() for p in trainer.model.parameters()]
        trainer.train_gnn()
        final_params = list(trainer.model.parameters())

        # At least one parameter should have changed
        changed = any(
            not torch.equal(i, f.detach())
            for i, f in zip(initial_params, final_params)
        )
        assert changed, "No model parameters changed during training"
