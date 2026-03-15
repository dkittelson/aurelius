"""Self-supervised pre-training pipeline for Aurelius GNN.

Leverages the ~77% unlabeled nodes in Elliptic via DGI or GraphCL
to learn rich node representations before supervised fine-tuning.

Usage:
    from src.pipeline.pretrain import PreTrainer
    from src.config import load_yaml_config

    cfg = load_yaml_config()
    pt = PreTrainer(cfg)
    pt.setup_data("elliptic")
    pt.setup_encoder()
    history = pt.pretrain_dgi(epochs=100)
    pt.save_pretrained("data/processed/checkpoints/pretrained_encoder.pt")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from src.models.gnn_model import AureliusGAT
from src.models.contrastive import (
    ContrastiveEncoder,
    DGILoss,
    GraphAugmentor,
    GraphCLLoss,
)


class PreTrainer:
    """Self-supervised pre-training pipeline.

    Supports two methods:
    - DGI: Corrupt graph by permuting node features, discriminate real vs corrupt.
    - GraphCL: Create two augmented views, maximize agreement via NT-Xent.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        self.data = None
        self.contrastive_encoder: Optional[ContrastiveEncoder] = None
        self.augmentor: Optional[GraphAugmentor] = None
        logger.info(f"PreTrainer using device: {self.device}")

    def setup_data(self, dataset: str = "elliptic") -> None:
        """Load the full graph (ALL nodes, including unlabeled)."""
        processed_dir = self.config["data"]["processed_dir"]
        pt_path = Path(processed_dir) / f"{dataset}_graph.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Processed graph not found at {pt_path}")

        self.data = torch.load(pt_path, weights_only=False)
        logger.info(
            f"Loaded graph: {self.data.num_nodes} nodes, "
            f"{self.data.num_edges} edges (ALL nodes used for pre-training)"
        )

    def setup_encoder(self) -> None:
        """Create AureliusGAT encoder + ContrastiveEncoder wrapper."""
        if self.data is None:
            raise RuntimeError("Call setup_data() first.")

        gnn_cfg = self.config["model"]["gnn"]
        contrastive_cfg = self.config.get("model", {}).get("contrastive", {})

        encoder = AureliusGAT(
            in_channels=self.data.x.shape[1],
            hidden_channels=gnn_cfg["hidden_channels"],
            out_channels=gnn_cfg["out_channels"],
            num_heads=gnn_cfg["num_heads"],
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            jk_mode=gnn_cfg["jk_mode"],
            residual=gnn_cfg["residual"],
        )

        projection_dim = contrastive_cfg.get("projection_dim", 128)
        self.contrastive_encoder = ContrastiveEncoder(
            encoder, projection_dim=projection_dim
        ).to(self.device)

        # Setup augmentor for GraphCL
        self.augmentor = GraphAugmentor(
            node_drop_rate=contrastive_cfg.get("node_drop_rate", 0.1),
            edge_perturb_rate=contrastive_cfg.get("edge_perturb_rate", 0.1),
            feature_mask_rate=contrastive_cfg.get("feature_mask_rate", 0.2),
        )

        num_params = sum(
            p.numel() for p in self.contrastive_encoder.parameters()
        )
        logger.info(f"ContrastiveEncoder: {num_params:,} parameters")

    def pretrain_dgi(self, epochs: Optional[int] = None) -> dict:
        """DGI pre-training: discriminate real vs corrupted graph.

        Corruption: random permutation of node features.
        Summary: mean of real node embeddings.

        Returns:
            history dict with 'loss' list.
        """
        if self.contrastive_encoder is None or self.data is None:
            raise RuntimeError("Call setup_data() and setup_encoder() first.")

        contrastive_cfg = self.config.get("model", {}).get("contrastive", {})
        epochs = epochs or contrastive_cfg.get("pretrain_epochs", 100)
        lr = contrastive_cfg.get("pretrain_lr", 0.001)

        emb_dim = self.contrastive_encoder.encoder.jk_out_channels
        dgi_loss_fn = DGILoss(hidden_dim=emb_dim).to(self.device)

        all_params = list(self.contrastive_encoder.parameters()) + list(
            dgi_loss_fn.parameters()
        )
        optimizer = torch.optim.Adam(all_params, lr=lr)

        data = self.data.to(self.device)
        history = {"loss": []}

        logger.info(f"Starting DGI pre-training for {epochs} epochs...")
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            self.contrastive_encoder.train()
            dgi_loss_fn.train()
            optimizer.zero_grad()

            # Real embeddings
            pos_emb, _ = self.contrastive_encoder(data.x, data.edge_index)

            # Corrupted embeddings: permute node features
            perm = torch.randperm(data.num_nodes, device=self.device)
            neg_emb, _ = self.contrastive_encoder(
                data.x[perm], data.edge_index
            )

            # Graph summary = mean of real embeddings
            summary = pos_emb.mean(dim=0)

            loss = dgi_loss_fn(pos_emb, neg_emb, summary)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            history["loss"].append(loss.item())

            if epoch % 20 == 0 or epoch == 1:
                elapsed = time.time() - t_start
                logger.info(
                    f"DGI Epoch {epoch:03d}/{epochs} | "
                    f"loss={loss.item():.4f} | "
                    f"elapsed={elapsed:.0f}s"
                )

        logger.info(
            f"DGI pre-training complete. Final loss: {history['loss'][-1]:.4f}"
        )
        return history

    def pretrain_graphcl(self, epochs: Optional[int] = None) -> dict:
        """GraphCL pre-training: two augmented views + NT-Xent loss.

        Since augmentations change graph size (node dropping), we use
        feature masking + edge perturbation only (preserves node count).

        Returns:
            history dict with 'loss' list.
        """
        if self.contrastive_encoder is None or self.data is None:
            raise RuntimeError("Call setup_data() and setup_encoder() first.")

        contrastive_cfg = self.config.get("model", {}).get("contrastive", {})
        epochs = epochs or contrastive_cfg.get("pretrain_epochs", 100)
        lr = contrastive_cfg.get("pretrain_lr", 0.001)
        temperature = contrastive_cfg.get("temperature", 0.5)

        cl_loss_fn = GraphCLLoss(temperature=temperature)
        optimizer = torch.optim.Adam(
            self.contrastive_encoder.parameters(), lr=lr
        )

        # Use a node-preserving augmentor (no node dropping)
        safe_aug = GraphAugmentor(
            node_drop_rate=0.0,
            edge_perturb_rate=self.augmentor.edge_perturb_rate,
            feature_mask_rate=self.augmentor.feature_mask_rate,
        )

        data = self.data
        history = {"loss": []}

        logger.info(f"Starting GraphCL pre-training for {epochs} epochs...")
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            self.contrastive_encoder.train()
            optimizer.zero_grad()

            # Create two augmented views (on CPU, then move)
            view1 = safe_aug.augment(data).to(self.device)
            view2 = safe_aug.augment(data).to(self.device)

            _, z1 = self.contrastive_encoder(view1.x, view1.edge_index)
            _, z2 = self.contrastive_encoder(view2.x, view2.edge_index)

            loss = cl_loss_fn(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.contrastive_encoder.parameters(), max_norm=1.0
            )
            optimizer.step()

            history["loss"].append(loss.item())

            if epoch % 20 == 0 or epoch == 1:
                elapsed = time.time() - t_start
                logger.info(
                    f"GraphCL Epoch {epoch:03d}/{epochs} | "
                    f"loss={loss.item():.4f} | "
                    f"elapsed={elapsed:.0f}s"
                )

        logger.info(
            f"GraphCL pre-training complete. "
            f"Final loss: {history['loss'][-1]:.4f}"
        )
        return history

    def save_pretrained(self, path: str) -> None:
        """Save the pre-trained encoder weights (without projection head)."""
        if self.contrastive_encoder is None:
            raise RuntimeError("No encoder to save.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.contrastive_encoder.encoder.state_dict(), save_path
        )
        logger.info(f"Saved pre-trained encoder to {save_path}")

    @staticmethod
    def load_pretrained_into_model(
        model: AureliusGAT, checkpoint_path: str
    ) -> None:
        """Load pre-trained encoder weights into an existing model.

        Uses strict=False to handle any missing/extra keys gracefully.
        """
        state_dict = torch.load(checkpoint_path, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading pretrained: {missing}")
        if unexpected:
            logger.warning(
                f"Unexpected keys when loading pretrained: {unexpected}"
            )
        logger.info(f"Loaded pre-trained weights from {checkpoint_path}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
