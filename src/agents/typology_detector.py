from dataclasses import dataclass
import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import itertools
from itertools import islice

@dataclass
class TypologyResult:
    typology: str
    confidence: float
    evidence_nodes: list[int]
    evidence_edges: list[tuple[int, int]]
    description: str
    metrics: dict

class TypologyDetector:
    def __init__(self, data: Data, predictions: torch.Tensor, threshold: float = 0.7):
        self.data = data
        self.predictions = predictions
        self.threshold = threshold
        self._probs = predictions.numpy()

    def detect_all(self, cluster_ids: list[int]) -> list[TypologyResult]:
        G = self._build_cluster_subgraph(cluster_ids)
        results = []
        results.extend(self._detect_smurfing(G))
        results.extend(self._detect_fan_in(G))
        results.extend(self._detect_fan_out(G))
        results.extend(self._detect_round_trip(G))
        results.extend(self._detect_layering(G))
        results.extend(self._detect_scatter_gather(G))
        results.sort(key=lambda r: -r.confidence)
        return results

    def _build_cluster_subgraph(self, cluster_ids: list[int]) -> nx.DiGraph:
        
        # --- Collect Edges ---
        ei = self.data.edge_index.numpy()
        mask = np.isin(ei[0], cluster_ids) & np.isin(ei[1], cluster_ids)
        sub_src = ei[0][mask]
        sub_dst = ei[1][mask]

        # --- Build Graph ---
        G = nx.DiGraph()
        G.add_nodes_from(cluster_ids)
        G.add_edges_from(zip(sub_src, sub_dst))

        # Add ilicit_prob as a node attribute
        for node in cluster_ids:
            G.nodes[node]["illicit_prob"] = float(self._probs[node])
        
        return G
    
    # --- Smurfing ---
    def _detect_smurfing(self, G: nx.DiGraph) -> list[TypologyResult]:
        results = []
        for node in G.nodes:
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if in_degree >= 3 and (out_degree == 0 or in_degree / out_degree >= 3.0):
                results.append(TypologyResult(
                    typology="smurfing",
                    confidence=min(1.0, in_degree / 10),
                    evidence_nodes=[node] + list(G.predecessors(node)),
                    evidence_edges=[(pred, node) for pred in G.predecessors(node)],
                    description=f"Smurfing detected: node {node} receives from {in_degree} feeders (in/out ratio: {in_degree}/{out_degree})",
                    metrics={"sink_node": node, "in_degree": in_degree, "out_degree": out_degree},
                ))
            
        return results
    
    # --- Fan-in ---
    def _detect_fan_in(self, G: nx.DiGraph) -> list[TypologyResult]:
        results = []
        for node in G.nodes:
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if in_degree >= 4 and (out_degree == 0 or out_degree / in_degree >= 3.0):
                results.append(TypologyResult(
                    typology="fan_in", 
                    confidence=min(1.0, in_degree / 12),
                    evidence_nodes=[node] + list(G.predecessors(node)),
                    evidence_edges=[(pred, node) for pred in G.predecessors(node)],
                    description=f"Fan-in detected: node {node} collects from {in_degree} sources",
                    metrics={"collector_node": node, "in_degree": in_degree, "out_degree": out_degree},
                ))
                
        return results
    
    # --- Fan-out ---
    def _detect_fan_out(self, G: nx.DiGraph) -> list[TypologyResult]:
        results = []
        for node in G.nodes:
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if out_degree >= 4 and (in_degree == 0 or out_degree / in_degree >= 3.0):
                results.append(TypologyResult(
                    typology="fan_out", 
                    confidence=min(1.0, out_degree / 12),
                    evidence_nodes=[node] + list(G.successors(node)),
                    evidence_edges=[(node, succ) for succ in G.successors(node)],
                    description=f"Fan-out detected: node {node} disperses to {out_degree} destinations",
                    metrics={"hub_node": node, "in_degree": in_degree, "out_degree": out_degree},
                ))
                
        return results
    
    # --- Round Trip ---
    def _detect_round_trip(self, G):
        cycles = list(islice(nx.simple_cycles(G), 100))
        results =[]
        for cycle in cycles:
            if len(cycle) > 8:
                continue
            results.append(TypologyResult(
                typology="round_trip",
                confidence=1.0-(len(cycle) / 10),
                evidence_nodes=cycle,
                evidence_edges=list(zip(cycle, cycle[1:] + [cycle[0]])),
                description=f"Round-trip cycle of {len(cycle)} nodes: {cycle}",
                metrics = {"cycle_length": len(cycle), "nodes": cycle}
            ))
        return results

    # --- Layering ---
    def _detect_layering(self, G):
        sources = [n for n in G.nodes if G.in_degree(n) == 0][:10]
        sinks = [n for n in G.nodes if G.out_degree(n) == 0][:10]
        results = []
        paths_found = 0

        for source in sources:
            for sink in sinks:
                if source == sink:
                    continue
                for path in islice(nx.all_simple_paths(G, source, sink, cutoff=10), 50):
                    if paths_found >= 50:
                        break
                    if len(path) < 3:
                        continue
                    paths_found += 1
                    results.append(TypologyResult(
                    typology="layering",
                    confidence=min(1.0, len(path) / 5),
                    evidence_nodes=path,
                    evidence_edges=list(zip(path, path[1:])),
                    description=f"Layering chain of {len(path)} hops from node {source} to {sink}",
                    metrics={"chain_length": len(path), "source": source, "sink": sink},
                ))
            if paths_found >= 50:
                break
        return results
        
    # -- Scatter-gather ---
    def _detect_scatter_gather(self, G: nx.DiGraph) -> list[TypologyResult]:
        results = []
        hubs = [n for n in G.nodes if G.out_degree(n) >= 3]
        collectors = [n for n in G.nodes if G.in_degree(n) >= 3]

        for hub in hubs:
            for collector in collectors:
                if hub == collector:
                    continue
                intermediaries = set(G.successors(hub)) & set(G.predecessors(collector))
                if len(intermediaries) >= 2:
                    evidence_nodes = [hub, collector] + list(intermediaries)
                    evidence_edges = (
                        [(hub, m) for m in intermediaries] + 
                        [(m, collector) for m in intermediaries]
                    )
                    results.append(TypologyResult(
                        typology="scatter-gather",
                        confidence=min(1.0, len(intermediaries) / 5),
                        evidence_nodes=evidence_nodes,
                        evidence_edges=evidence_edges,
                        description=f"Scatter-gather: hub {hub} disperses through {len(intermediaries)} intermediaries to collector {collector}",
                        metrics={"hub": hub, "collector": collector, "intermediary_count": len(intermediaries)},
                    ))
        return results

