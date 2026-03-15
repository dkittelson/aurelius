import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, JumpingKnowledge, HeteroConv
from torch.nn import Linear, BatchNorm1d, Dropout
from typing import Optional, Union

class AureliusGAT(nn.Module):
    def __init__(self, in_channels: int, 
                 hidden_channels: int = 128,
                 out_channels: int = 2,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 jk_mode="cat",
                 residual=True):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_p = dropout
        self._per_head = hidden_channels // num_heads
        self.residual = residual
        self.jk_mode = jk_mode
        self.residual_projs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # --- Build 3 GAT Layers and 3 BatchNorm Layers --- 
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(GATv2Conv(in_ch, self._per_head, heads=num_heads, concat=True, dropout=dropout, add_self_loops=False)) # attention layer
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if residual:
                if in_ch != hidden_channels:
                    self.residual_projs.append(Linear(in_ch, hidden_channels, bias=False))
                else:
                    self.residual_projs.append(nn.Identity())
        self.dropout = Dropout(dropout)

        if jk_mode == "lstm":
            self.jk = JumpingKnowledge(jk_mode, channels=hidden_channels, num_layers=num_layers)
            self.jk_out_channels = hidden_channels
        else:
            self.jk = JumpingKnowledge(jk_mode)
            self.jk_out_channels = num_layers * hidden_channels if jk_mode == "cat" else hidden_channels

        self.classifier = Linear(self.jk_out_channels, out_channels)

    def _message_pass(self, x, edge_index, collect_attention=False):
        xs = []
        attention_weights = []
        h = x

        # Forward pass through GAT backbone
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if collect_attention:
                h_new, attn = conv(h, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
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
    
    def forward(self, x, edge_index, return_attention=False, return_embeddings=False):
        # Forward pass
        xs, attention_weights = self._message_pass(x, edge_index, collect_attention=return_attention)

        # concatenate all 3 layers
        embedding = self.jk(xs)

        # get logits
        logits = self.classifier(embedding)

        if return_embeddings and return_attention:
            return logits, embedding, attention_weights
        if return_embeddings:
            return logits, embedding
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self, x, edge_index):
        """Return list of (edge_index, alpha) per GAT layer"""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self._message_pass(x, edge_index, collect_attention=True)
        return attention_weights

    def get_embeddings(self, x, edge_index):
        """Return JK-concatenated node embeddings"""
        self.eval()
        with torch.no_grad():
            xs, _ = self._message_pass(x, edge_index, collect_attention=False)
            return self.jk(xs)


                 