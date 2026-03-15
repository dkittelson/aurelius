"""Training pipeline for Temporal GNN on sequential graph snapshots.

Key difference from standard Trainer:
- Processes snapshots sequentially (not shuffled)
- Uses timestep attribute to split full graph into per-timestep subgraphs
- Memory resets between epochs but persists across timesteps within an epoch
- Truncated BPTT to prevent gradient explosion over long sequences
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import Data as PyGData

from src.models.temporal_gnn import TemporalAureliusGAT
from src.pipeline.train import EarlyStopping


class TemporalTrainer:
    """Training pipeline for TemporalAureliusGAT on snapshot sequences."""

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        self.data = None
        self.snapshots: list = []
        self.global_ids_list: list = []
        self.model: Optional[TemporalAureliusGAT] = None
        logger.info(f"TemporalTrainer using device: {self.device}")

    def setup_data(self, dataset: str = "elliptic") -> None:
        """Load processed graph and split into temporal snapshots.

        Uses the `timestep` attribute on the graph to create per-timestep
        subgraphs. Each snapshot gets a `global_node_mapping` tensor that
        maps local indices back to global node IDs for the memory module.
        """
        processed_dir = self.config["data"]["processed_dir"]
        pt_path = Path(processed_dir) / f"{dataset}_graph.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Processed graph not found at {pt_path}")

        self.data = torch.load(pt_path, weights_only=False)
        logger.info(
            f"Loaded graph: {self.data.num_nodes} nodes, "
            f"{self.data.num_edges} edges"
        )

        # Split into per-timestep snapshots
        self.snapshots = []
        self.global_ids_list = []
        timesteps = self.data.timestep

        unique_ts = sorted(timesteps.unique().tolist())
        for t in unique_ts:
            mask = timesteps == t
            local_ids = mask.nonzero(as_tuple=True)[0]

            # Remap to contiguous local indices
            remap = torch.full((self.data.num_nodes,), -1, dtype=torch.long)
            remap[local_ids] = torch.arange(local_ids.size(0))

            # Filter edges to only those within this timestep
            src, dst = self.data.edge_index
            edge_mask = mask[src] & mask[dst]
            local_edge_index = remap[self.data.edge_index[:, edge_mask]]

            snap = PyGData(
                x=self.data.x[mask],
                edge_index=local_edge_index,
                y=self.data.y[mask],
            )
            self.snapshots.append(snap)
            self.global_ids_list.append(local_ids)

        logger.info(
            f"Split into {len(self.snapshots)} temporal snapshots "
            f"(timesteps {unique_ts[0]}-{unique_ts[-1]})"
        )

    def setup_model(self) -> None:
        """Instantiate TemporalAureliusGAT from config."""
        if self.data is None:
            raise RuntimeError("Call setup_data() first.")

        gnn_cfg = self.config["model"]["gnn"]
        temporal_cfg = self.config.get("model", {}).get("temporal", {})

        self.model = TemporalAureliusGAT(
            in_channels=self.data.x.shape[1],
            hidden_channels=gnn_cfg["hidden_channels"],
            out_channels=gnn_cfg["out_channels"],
            num_heads=gnn_cfg["num_heads"],
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            jk_mode=gnn_cfg["jk_mode"],
            memory_dim=temporal_cfg.get("memory_dim", 128),
            max_nodes=self.data.num_nodes + 1,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"TemporalAureliusGAT: {num_params:,} parameters")

    def train_temporal(self, epochs: Optional[int] = None) -> dict:
        """Training loop over temporal snapshots.

        Per epoch:
          1. Reset memory
          2. Forward through train snapshots (t=1..34) sequentially
          3. Accumulate loss on labeled nodes
          4. Truncated BPTT every K steps
          5. Validate on t=35..42 snapshots

        Returns:
            history dict with train_loss, val_auprc, val_f1 lists
        """
        if self.model is None:
            raise RuntimeError("Call setup_model() first.")

        training_cfg = self.config["model"]["training"]
        temporal_cfg = self.config.get("model", {}).get("temporal", {})

        epochs = epochs or training_cfg.get("epochs", 200)
        patience = training_cfg.get("patience", 20)
        bptt_steps = temporal_cfg.get("bptt_steps", 5)

        # Train: timesteps 1-34 (indices 0-33), Val: 35-42 (34-41)
        train_snaps = min(34, len(self.snapshots))
        val_end = min(42, len(self.snapshots))

        # Class weights
        y = self.data.y
        labeled = y[y != -1]
        n_licit = int((labeled == 0).sum())
        n_illicit = int((labeled == 1).sum())
        ratio = n_licit / max(n_illicit, 1)
        class_weights = torch.tensor(
            [1.0, ratio], dtype=torch.float32, device=self.device
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_cfg.get("lr", 0.001),
            weight_decay=training_cfg.get("weight_decay", 5e-4),
        )
        early_stop = EarlyStopping(patience=patience)

        checkpoint_dir = Path(self.config["data"]["processed_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_temporal_gnn.pt"

        history = {"train_loss": [], "val_auprc": [], "val_f1": []}
        logger.info(f"Starting temporal training for up to {epochs} epochs...")
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.model.memory.reset()

            total_loss = 0.0
            num_steps = 0

            # Forward through training snapshots
            for i in range(train_snaps):
                snap = self.snapshots[i].to(self.device)
                gids = self.global_ids_list[i].to(self.device)

                logits = self.model.forward_snapshot(snap, gids)

                # Loss only on labeled nodes
                labeled_mask = snap.y != -1
                if labeled_mask.sum() == 0:
                    continue

                loss = criterion(logits[labeled_mask], snap.y[labeled_mask])
                total_loss += loss.item()
                num_steps += 1

                # Truncated BPTT: accumulate then step
                loss.backward()

                if num_steps % bptt_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            # Final optimizer step for remaining gradients
            if num_steps % bptt_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / max(num_steps, 1)

            # Validation on snapshots 35-42
            val_metrics = self._evaluate_snapshots(
                train_snaps, val_end, criterion
            )

            history["train_loss"].append(avg_loss)
            history["val_auprc"].append(val_metrics["auprc"])
            history["val_f1"].append(val_metrics["f1"])

            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"loss={avg_loss:.4f} | "
                    f"val_auprc={val_metrics['auprc']:.4f} | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Checkpoint on improvement
            if val_metrics["auprc"] >= early_stop.best_score:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_auprc": val_metrics["auprc"],
                    },
                    checkpoint_path,
                )

            if early_stop.step(val_metrics["auprc"]):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val_auprc={early_stop.best_score:.4f})"
                )
                break

        # Restore best
        if checkpoint_path.exists():
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Restored best temporal checkpoint from epoch {ckpt['epoch']}"
            )

        return history

    def _evaluate_snapshots(
        self, start_idx: int, end_idx: int, criterion
    ) -> dict:
        """Evaluate on a range of snapshots. Memory continues from training."""
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for i in range(start_idx, min(end_idx, len(self.snapshots))):
                snap = self.snapshots[i].to(self.device)
                gids = self.global_ids_list[i].to(self.device)

                logits = self.model.forward_snapshot(snap, gids)
                probs = F.softmax(logits, dim=1)[:, 1]

                labeled = snap.y != -1
                if labeled.sum() > 0:
                    all_probs.append(probs[labeled].cpu().numpy())
                    all_labels.append(snap.y[labeled].cpu().numpy())

        if not all_probs:
            return {"auprc": 0.0, "f1": 0.0}

        probs_np = np.concatenate(all_probs)
        labels_np = np.concatenate(all_labels)

        if labels_np.sum() == 0:
            return {"auprc": 0.0, "f1": 0.0}

        auprc = average_precision_score(labels_np, probs_np)
        preds = (probs_np >= 0.5).astype(int)
        f1 = f1_score(labels_np, preds, zero_division=0)

        return {"auprc": float(auprc), "f1": float(f1)}

    def load_pretrained_encoder(self, path: str) -> None:
        """Load pre-trained weights into the encoder component."""
        if self.model is None:
            raise RuntimeError("Call setup_model() first.")

        state_dict = torch.load(path, weights_only=True)
        missing, unexpected = self.model.encoder.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"Loaded pre-trained encoder from {path}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
