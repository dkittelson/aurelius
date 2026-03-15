"""Temporal GNN: AureliusGAT encoder + per-node GRU memory.

Processes graph snapshots sequentially across timesteps, maintaining
a GRU memory per node that captures how transaction patterns evolve.
Critical for detecting AML typologies like layering that unfold
over multiple timesteps.

Architecture per timestep t:
  1. GNN embedding = AureliusGAT.encode(snapshot_t)
  2. memory_t = GRU(embedding, memory_{t-1})
  3. logits = classifier([embedding || memory_t])
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.data import Data

from src.models.gnn_model import AureliusGAT


class TemporalNodeMemory(nn.Module):
    """Per-node GRU memory that persists across timesteps.

    Maintains a memory bank for all possible node IDs.
    At each timestep, only active nodes get their memory updated.
    """

    def __init__(
        self,
        embedding_dim: int,
        memory_dim: int = 128,
        max_nodes: int = 250_000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.max_nodes = max_nodes

        self.gru_cell = nn.GRUCell(embedding_dim, memory_dim)
        # Memory bank — not a parameter, just persistent state
        self.register_buffer(
            "memory", torch.zeros(max_nodes, memory_dim)
        )

    def update(
        self, node_embeddings: Tensor, node_ids: Tensor
    ) -> Tensor:
        """Update memory for the given nodes and return their new memory.

        Args:
            node_embeddings: [N, embedding_dim] GNN outputs for active nodes
            node_ids:        [N] global node IDs

        Returns:
            [N, memory_dim] updated memory for active nodes
        """
        prev_memory = self.memory[node_ids]
        new_memory = self.gru_cell(node_embeddings, prev_memory)
        self.memory[node_ids] = new_memory.detach()  # detach to avoid BPTT across all timesteps
        return new_memory

    def get_memory(self, node_ids: Tensor) -> Tensor:
        """Retrieve current memory for given nodes."""
        return self.memory[node_ids]

    def reset(self) -> None:
        """Zero all memory (start of new sequence / epoch)."""
        self.memory.zero_()


class TemporalAureliusGAT(nn.Module):
    """Temporal GNN: AureliusGAT encoder + GRU node memory.

    For each snapshot:
      1. Encode nodes with shared GATv2 encoder → embeddings
      2. Update per-node GRU memory with new embeddings
      3. Classify based on [embedding || memory]

    Args:
        in_channels:     Input feature dimension per node.
        hidden_channels: GATv2 hidden channels (default 128).
        memory_dim:      GRU memory dimension (default 128).
        out_channels:    Number of classes (default 2).
        num_heads:       GATv2 attention heads.
        num_layers:      GATv2 layers.
        dropout:         Dropout rate.
        jk_mode:         Jumping Knowledge mode.
        max_nodes:       Maximum node ID for memory bank.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        memory_dim: int = 128,
        out_channels: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        jk_mode: str = "cat",
        max_nodes: int = 250_000,
    ):
        super().__init__()

        self.encoder = AureliusGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            jk_mode=jk_mode,
            residual=True,
        )

        emb_dim = self.encoder.jk_out_channels
        self.memory = TemporalNodeMemory(
            embedding_dim=emb_dim,
            memory_dim=memory_dim,
            max_nodes=max_nodes,
        )

        self.classifier = nn.Linear(emb_dim + memory_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward_snapshot(
        self,
        snapshot: Data,
        global_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Process one timestep snapshot. Updates internal memory.

        Args:
            snapshot:   Data with .x and .edge_index for this timestep.
            global_ids: [N] global node IDs mapping local→global.
                        If None, uses torch.arange(num_nodes).

        Returns:
            logits: [N, out_channels]
        """
        if global_ids is None:
            global_ids = torch.arange(
                snapshot.num_nodes, device=snapshot.x.device
            )

        # Get GNN embeddings (without classifier)
        _, embeddings = self.encoder(
            snapshot.x, snapshot.edge_index, return_embeddings=True
        )

        # Update memory and get new state
        new_memory = self.memory.update(embeddings, global_ids)

        # Classify on [embedding || memory]
        combined = torch.cat([embeddings, new_memory], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

    def forward_sequence(
        self,
        snapshots: list[Data],
        global_ids_list: Optional[list[Tensor]] = None,
        reset_memory: bool = True,
    ) -> list[Tensor]:
        """Process a sequence of snapshots. Returns per-snapshot logits.

        Args:
            snapshots:       List of Data objects, one per timestep.
            global_ids_list: List of [N_t] tensors mapping local→global IDs.
            reset_memory:    Whether to reset memory before processing.

        Returns:
            List of logit tensors, one per snapshot.
        """
        if reset_memory:
            self.memory.reset()

        logits_list = []
        for i, snap in enumerate(snapshots):
            gids = global_ids_list[i] if global_ids_list else None
            logits = self.forward_snapshot(snap, gids)
            logits_list.append(logits)

        return logits_list

    def get_embeddings(
        self,
        snapshots: list[Data],
        global_ids_list: Optional[list[Tensor]] = None,
    ) -> Tensor:
        """Run full sequence and return final [embedding || memory] for all seen nodes.

        Returns:
            [max_seen_id + 1, emb_dim + memory_dim] tensor
        """
        self.eval()
        self.memory.reset()

        with torch.no_grad():
            for i, snap in enumerate(snapshots):
                gids = global_ids_list[i] if global_ids_list else None
                self.forward_snapshot(snap, gids)

        # Collect embeddings for all nodes that have non-zero memory
        memory = self.memory.memory
        non_zero = memory.abs().sum(dim=1) > 0
        return memory[non_zero]
