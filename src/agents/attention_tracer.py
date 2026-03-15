from dataclasses import dataclass, field
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from loguru import logger

@dataclass
class EvidencePath:
    path: list[int]
    attention_weights: list[float]
    cumulative_attention: float

@dataclass
class AttentionProfile:
    node_id: int
    incoming_attention: list[dict]
    outgoing_attention: list[dict]
    total_incoming: float
    total_outgoing: float

class AttentionTracer:
    def __init__(self, model: torch.nn.Module, data: Data):
        self.model = model
        self.data = data
        self._edge_weights: dict[tuple[int, int], float] = {}
        self._reverse_index: dict[int, list[tuple[int, float]]] = {} 
        self._forward_index: dict[int, list[tuple[int, float]]] = {}  
        self._cached = False

    def _build_cache(self):
        """Extracts attention weights for every edge in graph"""
        
        if self._cached:
            return
        
        attn_list = self.model.get_attention_weights(self.data.x, self.data.edge_index)

        # Use last layer, mean across heads → shape [num_edges]
        _, alpha_last = attn_list[-1]
        alpha_mean = alpha_last.mean(dim=1).numpy()
        
        ei = self.data.edge_index.numpy()
        for i, (src, dst) in enumerate(zip(ei[0], ei[1])):
            src, dst = int(src), int(dst)
            w = float(alpha_mean[i])
            self._edge_weights[(src, dst)] = w
            self._reverse_index.setdefault(dst, []).append((src, w)) # incoming edge nodes
            self._forward_index.setdefault(src, []).append((dst, w)) # outgoing edge nodes
        
        self._cached = True
        logger.info(f"Attention cache built: {len(self._edge_weights)} edges.")

    def trace_evidence_path(self, node_id: int, max_hops: int = 3) -> EvidencePath:
            """Walks backwards through highest attention incoming edges"""

            self._build_cache()

            path = [node_id]
            weights = []
            visited = {node_id}

            current = node_id
            for _ in range(max_hops):
                candidates = self._reverse_index.get(current, []) # get all candidates
                candidates = [(src, w) for src, w in candidates if src not in visited]
                if not candidates:
                    break
                best_src, best_w = max(candidates, key=lambda x: x[1]) # pick highest attention
                path.append(best_src)
                weights.append(best_w)
                visited.add(best_src)
                current = best_src

            path.reverse()
            weights.reverse()

            cumulative = float(np.prod(weights)) if weights else 0.0

            return EvidencePath(
                path=path,
                attention_weights=weights,
                cumulative_attention=cumulative,
            )
    
    def get_top_attention_edges(self, node_ids: list[int], top_k: int = 20) -> list[dict]:
        """Find edges within a given cluster with highest attention weights and return the top-k"""

        self._build_cache()
        node_set = set(node_ids)
        edges = [
            {"src": src, "dst": dst, "weight": w}
            for (src, dst), w in self._edge_weights.items()
            if src in node_set and dst in node_set
        ]
        edges.sort(key=lambda e: -e["weight"])
        return edges[:top_k]
    
    def get_node_attention_profile(self, node_id: int) -> AttentionProfile:
        """Return complete picture of all its attention connections"""

        self._build_cache()
        incoming = sorted(
            [{"src": src, "weight": w} for src, w in self._reverse_index.get(node_id, [])],
            key=lambda x: -x["weight"]
        )
        outgoing = sorted(
        [{"dst": dst, "weight": w} for dst, w in self._forward_index.get(node_id, [])],
        key=lambda x: -x["weight"]
        )
        return AttentionProfile(
            node_id=node_id,
            incoming_attention=incoming,
            outgoing_attention=outgoing,
            total_incoming=sum(e["weight"] for e in incoming),
            total_outgoing=sum(e["weight"] for e in outgoing),
        )

