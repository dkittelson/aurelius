import torch
import torch.nn as nn
from src.models.gnn_model import AureliusGAT

class TemporalNodeMemory(nn.Module):
    def __init__(self, embedding_dim, memory_dim=128, max_nodes=240_000):
        super().__init__()

        # --- Gated Recurrent Unit: Decides what old/new memory stays ---
        # input -> embedding_dim: 384 of features describing each node 
        # output -> memory_dim: 128 features describing each node
        self.gru_cell = nn.GRUCell(embedding_dim, memory_dim)
        self.register_buffer("memory", torch.zeros(max_nodes, memory_dim))
        self.memory_dim = memory_dim
        self.max_nodes = max_nodes

    # --- Update old memory with new memory --- 
    def update(self, embeddings, node_ids):
        prev = self.memory[node_ids]
        new_memory = self.gru_cell(embeddings, prev)
        self.memory[node_ids] = new_memory.detach() # .detach() cuts gradient chain between timesteps
        return new_memory
    
    def reset(self):
        self.memory.zero_()
        
    def get_memory(self, node_ids):
        return self.memory[node_ids]

# Wrapper
class TemporalAureliusGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, memory_dim=128, out_channels=2,
                 num_heads=4, num_layers=3, dropout=0.3, jk_mode="cat", max_nodes=250_000):
        super().__init__()

        # Current snapshot
        self.encoder = AureliusGAT(in_channels=in_channels, 
                                   hidden_channels=hidden_channels, 
                                   out_channels=out_channels, 
                                   num_heads=num_heads, 
                                   num_layers=num_layers, 
                                   dropout=dropout,
                                   jk_mode=jk_mode,
                                   residual=True)
        
        # Historical summary
        self.memory = TemporalNodeMemory(embedding_dim=self.encoder.jk_out_channels, 
                                         memory_dim=memory_dim,
                                         max_nodes=max_nodes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classifier (input_size: current snapshot + historical data, output_size: 2)
        self.classifier = nn.Linear(self.encoder.jk_out_channels + memory_dim, out_channels)

    # Runs forward pass on all snapshots
    def forward_sequence(self, snapshots, global_ids_list=None, reset_memory=True):
        if reset_memory:
            self.memory.reset()
        logits_list = []
        for i, snap in enumerate(snapshots):
            gids = global_ids_list[i] if global_ids_list else None
            logits_list.append(self.forward_snapshot(snap, gids))
        return logits_list # 49 logit tensors (2 logits per node in each 49 snapshots)

    def forward_snapshot(self, snapshot, global_ids=None):

        if global_ids is None: 
            global_ids = torch.arange(snapshot.num_nodes, device=snapshot.x.device)

        # Get Current Embeddings
        _, embeddings = self.encoder(snapshot.x, snapshot.edge_index, return_embeddings=True)

        # Update memory
        new_memory = self.memory.update(embeddings, global_ids)

        # Combine current and historical memory side by side
        combined = torch.cat([embeddings, new_memory], dim=1)

        combined = self.dropout(combined)

        return self.classifier(combined)