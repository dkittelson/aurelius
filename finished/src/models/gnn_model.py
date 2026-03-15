"""GATv2-based Graph Neural Network for AML node classification.

Architecture:
  Input -> [GATv2Conv + BatchNorm + ELU + Dropout + Residual] x num_layers
        -> JumpingKnowledge (cat/max/lstm)
        -> Linear -> logits
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, JumpingKnowledge, HeteroConv
from torch.nn import Linear, BatchNorm1d, Dropout
from typing import Optional, Union


class AureliusGAT(torch.nn.Module):
    """
    Multi-layer GATv2 with Jumping Knowledge for AML node classification.

    Key design decisions:
    - per_head_channels = hidden_channels // num_heads so that each layer
      outputs exactly hidden_channels total (not hidden_channels * num_heads).
    - add_self_loops=False in each conv because we already add self-loops
      during graph construction in builder.py.
    - Residual connection: layer 0 uses a learned projection (in_channels ->
      hidden_channels); all subsequent layers use Identity since dimensions match.
    - JK 'cat' concatenates all layer outputs: final embedding dim = num_layers * hidden_channels.

    Args:
        in_channels:      Number of input features per node.
        hidden_channels:  Total channels per layer (= per_head * num_heads).
        out_channels:     Number of output classes (default 2: licit/illicit).
        num_heads:        Number of attention heads per GATv2 layer.
        num_layers:       Number of GATv2 layers.
        dropout:          Dropout probability applied inside convs and after activation.
        jk_mode:          Jumping Knowledge aggregation: 'cat', 'max', or 'lstm'.
        residual:         Whether to add residual skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        jk_mode: str = "cat",
        residual: bool = True,
    ):
        super().__init__()

        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
        )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.jk_mode = jk_mode
        self.residual = residual
        self._per_head = hidden_channels // num_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels

            # concat=True -> output is per_head * num_heads = hidden_channels
            conv = GATv2Conv(
                in_ch,
                self._per_head,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
                bias=True,
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(hidden_channels))

            if residual:
                # Project if dimensions differ (only layer 0 when in_channels != hidden_channels)
                if in_ch != hidden_channels:
                    self.residual_projs.append(
                        Linear(in_ch, hidden_channels, bias=False)
                    )
                else:
                    self.residual_projs.append(nn.Identity())

        self.dropout = Dropout(dropout)

        # Jumping Knowledge
        if jk_mode == "lstm":
            self.jk = JumpingKnowledge(
                jk_mode, channels=hidden_channels, num_layers=num_layers
            )
            self.jk_out_channels = hidden_channels
        else:
            self.jk = JumpingKnowledge(jk_mode)
            self.jk_out_channels = (
                num_layers * hidden_channels if jk_mode == "cat" else hidden_channels
            )

        self.classifier = Linear(self.jk_out_channels, out_channels)

    def _message_pass(self, x, edge_index, collect_attention=False):
        """
        Core forward loop. Returns (xs, attention_weights).
        xs: list of per-layer outputs, one per layer.
        attention_weights: list of (edge_index, alpha) tuples, only if collect_attention.
        """
        xs = []
        attention_weights = []
        h = x

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if collect_attention:
                h_new, attn = conv(h, edge_index, return_attention_weights=True)
                attention_weights.append(attn)  # (edge_index, alpha [E, heads])
            else:
                h_new = conv(h, edge_index)

            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)

            if self.residual:
                h_new = h_new + self.residual_projs[i](h)

            h = h_new
            xs.append(h)

        return xs, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass.

        Args:
            x:                 [num_nodes, in_channels]
            edge_index:        [2, num_edges]
            return_attention:  Also return per-layer attention weights.
            return_embeddings: Also return the JK embedding (before classifier).

        Returns (depending on flags):
            logits                             (default)
            (logits, attention_list)           (return_attention=True)
            (logits, embeddings)               (return_embeddings=True)
            (logits, embeddings, attention)    (both True)
        """
        xs, attention_weights = self._message_pass(
            x, edge_index, collect_attention=return_attention
        )
        embedding = self.jk(xs)
        logits = self.classifier(embedding)

        if return_embeddings and return_attention:
            return logits, embedding, attention_weights
        if return_embeddings:
            return logits, embedding
        if return_attention:
            return logits, attention_weights
        return logits

    def get_attention_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> list[tuple]:
        """
        Extract attention weights from every layer without running the classifier.

        Returns:
            List of (edge_index, alpha) tuples, one per layer.
            alpha shape: [num_edges, num_heads]
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self._message_pass(
                x, edge_index, collect_attention=True
            )
        return attention_weights

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the JK-aggregated node embeddings (penultimate layer output).
        Shape: [num_nodes, jk_out_channels]

        These embeddings are used as input features for the XGBoost classifier.
        """
        self.eval()
        with torch.no_grad():
            xs, _ = self._message_pass(x, edge_index, collect_attention=False)
            embedding = self.jk(xs)
        return embedding


class AureliusGATHetero(torch.nn.Module):
    """
    Heterogeneous GATv2 for the IBM AML dataset.

    Uses HeteroConv to apply separate GATv2Conv parameters for each edge type.
    Input projection layers unify all node-type feature dimensions before
    the first conv layer so all subsequent layers work in a uniform hidden space.

    Args:
        metadata:         (node_types, edge_types) from data.metadata()
        in_channels_dict: {node_type: feature_dim} for each node type.
        hidden_channels:  Uniform hidden dimension after projection.
        out_channels:     Number of output classes.
        num_heads:        Attention heads per layer.
        num_layers:       Number of HeteroConv layers.
        dropout:          Dropout rate.
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        assert hidden_channels % num_heads == 0
        node_types, edge_types = metadata
        per_head = hidden_channels // num_heads

        self.dropout_p = dropout
        self.node_types = node_types

        # Project each node type to the same hidden_channels dimension
        self.input_projs = nn.ModuleDict(
            {
                nt: Linear(in_channels_dict[nt], hidden_channels)
                for nt in node_types
                if nt in in_channels_dict
            }
        )

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                et: GATv2Conv(
                    hidden_channels,
                    per_head,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for et in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # BatchNorm per node type, per layer
        self.bns = nn.ModuleList(
            [
                nn.ModuleDict(
                    {nt: BatchNorm1d(hidden_channels) for nt in node_types}
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = Dropout(dropout)

        # Classifier applied to each node type independently
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(
        self, x_dict: dict, edge_index_dict: dict
    ) -> dict:
        """
        Forward pass over heterogeneous graph.

        Args:
            x_dict:          {node_type: [num_nodes, in_channels]}
            edge_index_dict: {edge_type: [2, num_edges]}

        Returns:
            {node_type: [num_nodes, out_channels]} logits for each node type.
        """
        # Project all node types to hidden_channels
        h_dict = {
            nt: F.elu(self.input_projs[nt](x))
            for nt, x in x_dict.items()
            if nt in self.input_projs
        }

        # Iterative message passing
        for conv, bns in zip(self.convs, self.bns):
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {
                nt: bns[nt](F.elu(self.dropout(h)))
                for nt, h in h_dict.items()
                if nt in bns
            }

        return {nt: self.classifier(h) for nt, h in h_dict.items()}

    def get_embeddings(
        self, x_dict: dict, edge_index_dict: dict
    ) -> dict:
        """Return pre-classifier embeddings for each node type."""
        self.eval()
        with torch.no_grad():
            h_dict = {
                nt: F.elu(self.input_projs[nt](x))
                for nt, x in x_dict.items()
                if nt in self.input_projs
            }
            for conv, bns in zip(self.convs, self.bns):
                h_dict = conv(h_dict, edge_index_dict)
                h_dict = {
                    nt: bns[nt](F.elu(self.dropout(h)))
                    for nt, h in h_dict.items()
                    if nt in bns
                }
        return h_dict
