"""End-to-end training pipeline for Aurelius GNN + XGBoost.

Pipeline:
  1. setup_data()   — load graph, build NeighborLoader dataloaders
  2. setup_model()  — instantiate AureliusGAT from config, move to device
  3. train_gnn()    — mini-batch training with early stopping on val AUPRC
  4. train_hybrid() — extract GNN embeddings, train XGBoost on combined features
  5. full_pipeline() — runs all four steps end-to-end
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score

from src.models.gnn_model import AureliusGAT
from src.models.classifier import HybridClassifier
from src.pipeline.evaluate import evaluate_model, get_node_probabilities


class EarlyStopping:
    """Tracks validation metric and stops training when it stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False


class Trainer:
    """
    Manages the full training pipeline: data loading, GNN training, XGBoost fitting.

    Args:
        config: Full YAML config dict (from src.config.get_yaml_config()).
        device: 'auto' (detects MPS/CUDA/CPU), or explicit 'cpu', 'cuda', 'mps'.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        logger.info(f"Trainer using device: {self.device}")

        self.data = None
        self.model: Optional[AureliusGAT] = None
        self.train_loader: Optional[NeighborLoader] = None
        self.hybrid_clf: Optional[HybridClassifier] = None
        self.training_history: dict = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_data(self, dataset: str = "elliptic") -> None:
        """
        Load processed graph and build NeighborLoader dataloaders.

        The NeighborLoader samples neighborhoods for mini-batch training.
        For each seed node (a training node), it samples:
          - 15 neighbors at hop 1
          - 10 neighbors at hop 2
          -  5 neighbors at hop 3
        This gives each seed node a local subgraph to do message passing on,
        without loading the entire 200K-node graph into a single batch.

        Args:
            dataset: 'elliptic' or 'ibm_aml'
        """
        processed_dir = self.config["data"]["processed_dir"]

        if dataset == "elliptic":
            pt_path = Path(processed_dir) / "elliptic_graph.pt"
        elif dataset == "ibm_aml":
            pt_path = Path(processed_dir) / "ibm_aml_graph.pt"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if not pt_path.exists():
            raise FileNotFoundError(
                f"Processed graph not found at {pt_path}. "
                "Run the graph builder first: python -m src.graph.builder"
            )

        logger.info(f"Loading graph from {pt_path}")
        self.data = torch.load(pt_path, weights_only=False)
        logger.info(
            f"Graph loaded: {self.data.num_nodes} nodes, {self.data.num_edges} edges"
        )

        training_cfg = self.config["model"]["training"]
        num_neighbors = training_cfg["num_neighbors"]
        batch_size = training_cfg["batch_size"]

        # Only seed the loader with labeled training nodes
        train_input = self.data.train_mask & (self.data.y != -1)

        self.train_loader = NeighborLoader(
            self.data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_input,
            shuffle=True,
            num_workers=0,  # 0 avoids multiprocessing issues on macOS
        )

        logger.info(
            f"NeighborLoader: {train_input.sum()} training seed nodes, "
            f"batch_size={batch_size}, num_neighbors={num_neighbors}"
        )

    def setup_model(self) -> None:
        """Instantiate AureliusGAT from config and move to device."""
        if self.data is None:
            raise RuntimeError("Call setup_data() before setup_model().")

        gnn_cfg = self.config["model"]["gnn"]

        self.model = AureliusGAT(
            in_channels=self.data.x.shape[1],
            hidden_channels=gnn_cfg["hidden_channels"],
            out_channels=gnn_cfg["out_channels"],
            num_heads=gnn_cfg["num_heads"],
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            jk_mode=gnn_cfg["jk_mode"],
            residual=gnn_cfg["residual"],
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"AureliusGAT instantiated: {num_params:,} parameters, "
            f"in_channels={self.data.x.shape[1]}, "
            f"embedding_dim={self.model.jk_out_channels}"
        )

    # ------------------------------------------------------------------
    # GNN Training
    # ------------------------------------------------------------------

    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.

        With ~2% illicit nodes, the loss would naturally minimize by
        predicting everything as licit. Weighting illicit 10x corrects this.

        Returns:
            weights: [num_classes] tensor on self.device
        """
        y = self.data.y
        labeled = y[y != -1]
        n_licit = int((labeled == 0).sum())
        n_illicit = int((labeled == 1).sum())

        if n_illicit == 0:
            logger.warning("No illicit nodes found. Using uniform weights.")
            return torch.ones(2, device=self.device)

        ratio = n_licit / n_illicit
        weights = torch.tensor([1.0, ratio], dtype=torch.float32, device=self.device)
        logger.info(
            f"Class weights: licit=1.0, illicit={ratio:.1f} "
            f"(n_licit={n_licit}, n_illicit={n_illicit})"
        )
        return weights

    def train_gnn(self) -> dict:
        """
        Mini-batch GNN training loop with early stopping on validation AUPRC.

        Training details:
          - Adam optimizer (lr=0.001, weight_decay=5e-4)
          - Weighted CrossEntropyLoss to handle class imbalance
          - ReduceLROnPlateau scheduler (factor=0.5, patience=10)
          - Gradient clipping (max_norm=1.0) for stability
          - Early stopping (patience=20 on val AUPRC)
          - Saves best checkpoint whenever val AUPRC improves

        Returns:
            history dict with lists: train_loss, val_auprc, val_f1
        """
        if self.model is None or self.data is None:
            raise RuntimeError("Call setup_data() and setup_model() first.")

        training_cfg = self.config["model"]["training"]
        epochs = training_cfg["epochs"]
        patience = training_cfg["patience"]
        checkpoint_dir = Path(self.config["data"]["processed_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_gnn.pt"

        class_weights = self.compute_class_weights()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_cfg["lr"],
            weight_decay=training_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )
        early_stop = EarlyStopping(patience=patience)

        history = {"train_loss": [], "val_auprc": [], "val_f1": []}

        logger.info(f"Starting GNN training for up to {epochs} epochs...")
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            # ---- Training ----
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                out = self.model(batch.x, batch.edge_index)

                # In NeighborLoader, the first batch.batch_size nodes are the
                # seed nodes (the actual training nodes). We only compute loss
                # on those, not on the sampled neighborhood nodes.
                seed_out = out[: batch.batch_size]
                seed_y = batch.y[: batch.batch_size]

                # Skip batches with no labeled seeds
                labeled_mask = seed_y != -1
                if labeled_mask.sum() == 0:
                    continue

                loss = criterion(seed_out[labeled_mask], seed_y[labeled_mask])
                loss.backward()

                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            # ---- Validation (full-graph inference) ----
            val_metrics = evaluate_model(
                self.model, self.data, self.data.val_mask, self.device
            )
            val_auprc = val_metrics["auprc"]
            val_f1 = val_metrics["f1"]

            scheduler.step(val_auprc)

            history["train_loss"].append(avg_loss)
            history["val_auprc"].append(val_auprc)
            history["val_f1"].append(val_f1)

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t_start
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"loss={avg_loss:.4f} | "
                    f"val_auprc={val_auprc:.4f} | "
                    f"val_f1={val_f1:.4f} | "
                    f"lr={lr:.2e} | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Save checkpoint on improvement
            if val_auprc >= early_stop.best_score:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_auprc": val_auprc,
                    },
                    checkpoint_path,
                )

            # Early stopping check
            if early_stop.step(val_auprc):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val_auprc={early_stop.best_score:.4f})"
                )
                break

        # Restore best weights
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Restored best checkpoint from epoch {ckpt['epoch']} "
                f"(val_auprc={ckpt['val_auprc']:.4f})"
            )

        self.training_history = history
        return history

    # ------------------------------------------------------------------
    # Hybrid Training
    # ------------------------------------------------------------------

    def train_hybrid(self) -> dict:
        """
        Train XGBoost on [original_features || GNN_embeddings].

        Steps:
          1. Run full-graph GNN inference to extract embeddings for all nodes.
          2. Concatenate with original node features.
          3. Train XGBoost on training set (labeled nodes only).
          4. Evaluate on test set.

        Returns:
            Test-set evaluation metrics dict.
        """
        if self.model is None or self.data is None:
            raise RuntimeError("Train GNN before calling train_hybrid().")

        logger.info("Extracting GNN embeddings for all nodes...")
        data_cpu = self.data.to("cpu")
        self.model.to("cpu")
        embeddings = self.model.get_embeddings(data_cpu.x, data_cpu.edge_index)
        self.model.to(self.device)

        embeddings_np = embeddings.numpy()
        features_np = data_cpu.x.numpy()
        labels_np = data_cpu.y.numpy()

        # Build combined feature matrix
        X_all = np.concatenate([features_np, embeddings_np], axis=1)

        # Extract labeled train/val/test subsets
        train_mask = (data_cpu.train_mask & (data_cpu.y != -1)).numpy()
        val_mask = (data_cpu.val_mask & (data_cpu.y != -1)).numpy()
        test_mask = (data_cpu.test_mask & (data_cpu.y != -1)).numpy()

        X_train = X_all[train_mask]
        y_train = labels_np[train_mask]
        X_val = X_all[val_mask]
        y_val = labels_np[val_mask]
        X_test = X_all[test_mask]
        y_test = labels_np[test_mask]

        logger.info(
            f"Hybrid features: {X_train.shape[1]} dims "
            f"({features_np.shape[1]} original + {embeddings_np.shape[1]} GNN). "
            f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
        )

        self.hybrid_clf = HybridClassifier(self.config["model"]["xgboost"])
        self.hybrid_clf.fit(X_train, y_train, X_val, y_val)

        test_metrics = self.hybrid_clf.evaluate(X_test, y_test)
        logger.info(
            f"Hybrid test AUPRC: {test_metrics['auprc']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}"
        )

        return test_metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_pipeline(self, dataset: str = "elliptic") -> dict:
        """
        Run the complete training pipeline end-to-end.

        Returns:
            dict with 'gnn_history' and 'hybrid_test_metrics'.
        """
        self.setup_data(dataset)
        self.setup_model()
        gnn_history = self.train_gnn()
        hybrid_metrics = self.train_hybrid()

        return {
            "gnn_history": gnn_history,
            "hybrid_test_metrics": hybrid_metrics,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Detect best available device when device='auto'."""
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
