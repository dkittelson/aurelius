"""Graph inspection routes — stats, neighbors, subgraph queries."""

from __future__ import annotations

from typing import Any

import torch
from fastapi import APIRouter, Depends, HTTPException
from torch_geometric.utils import to_networkx

from src.api.dependencies import AppState, get_app_state
from src.api.schemas import (
    GraphStatsResponse,
    NodeNeighborsRequest,
    NodeNeighborsResponse,
    SubgraphRequest,
    SubgraphResponse,
)

router = APIRouter(prefix="/graph", tags=["graph"])


def _require_graph(state: AppState) -> Any:
    if state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")
    return state.graph


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(state: AppState = Depends(get_app_state)):
    graph = _require_graph(state)
    y = graph.y
    num_illicit = int((y == 1).sum())
    num_licit = int((y == 0).sum())
    num_unknown = int((y == -1).sum()) if (y == -1).any() else 0
    labeled = num_illicit + num_licit
    illicit_rate = num_illicit / labeled if labeled > 0 else 0.0
    return GraphStatsResponse(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        num_illicit=num_illicit,
        num_licit=num_licit,
        num_unknown=num_unknown,
        illicit_rate=round(illicit_rate, 4),
    )


@router.post("/neighbors", response_model=NodeNeighborsResponse)
async def node_neighbors(req: NodeNeighborsRequest, state: AppState = Depends(get_app_state)):
    graph = _require_graph(state)
    node_id = req.node_id
    if node_id < 0 or node_id >= graph.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} out of range")

    edge_index = graph.edge_index
    # BFS up to req.hops hops
    frontier = {node_id}
    visited = {node_id}
    for _ in range(req.hops):
        src, dst = edge_index[0], edge_index[1]
        mask = torch.isin(src, torch.tensor(list(frontier)))
        new_neighbors = set(dst[mask].tolist()) - visited
        visited |= new_neighbors
        frontier = new_neighbors
        if not frontier:
            break

    neighbors = sorted(visited - {node_id})
    edge_mask = torch.isin(edge_index[0], torch.tensor(list(visited))) & \
                torch.isin(edge_index[1], torch.tensor(list(visited)))
    return NodeNeighborsResponse(
        node_id=node_id,
        neighbors=neighbors,
        edge_count=int(edge_mask.sum()),
    )


def _bfs_expand(edge_index: torch.Tensor, seeds: set[int], hops: int, max_nodes: int = 300) -> set[int]:
    """Bidirectional BFS expansion — follows both outgoing and incoming edges."""
    visited = set(seeds)
    frontier = set(seeds)
    src_t, dst_t = edge_index[0], edge_index[1]
    for _ in range(hops):
        if not frontier:
            break
        f_tensor = torch.tensor(sorted(frontier), dtype=torch.long)
        # Outgoing: seed → neighbor
        out_mask = torch.isin(src_t, f_tensor)
        # Incoming: neighbor → seed
        in_mask = torch.isin(dst_t, f_tensor)
        new_nodes = (
            set(dst_t[out_mask].tolist()) | set(src_t[in_mask].tolist())
        ) - visited
        visited |= new_nodes
        frontier = new_nodes
        if len(visited) >= max_nodes:
            break
    return visited


@router.post("/subgraph", response_model=SubgraphResponse)
async def subgraph(req: SubgraphRequest, state: AppState = Depends(get_app_state)):
    graph = _require_graph(state)
    seed_set = set(req.node_ids)
    if not seed_set:
        raise HTTPException(status_code=400, detail="node_ids must be non-empty")
    invalid = [n for n in seed_set if n < 0 or n >= graph.num_nodes]
    if invalid:
        raise HTTPException(status_code=404, detail=f"Invalid node ids: {invalid[:5]}")

    # Expand seeds to their neighborhood for richer graph context
    node_set = (
        _bfs_expand(graph.edge_index, seed_set, req.expand_hops)
        if req.expand_hops > 0
        else seed_set
    )

    probs = state.predictions
    seed_ids = set(req.node_ids)  # mark which nodes are the original cluster
    node_list = []
    for nid in sorted(node_set):
        info: dict[str, Any] = {
            "node_id": nid,
            "label": int(graph.y[nid]),
            "is_seed": nid in seed_ids,
        }
        if probs is not None:
            info["illicit_prob"] = round(float(probs[nid]), 4)
        node_list.append(info)

    edges: list[dict[str, int]] = []
    if req.include_edges:
        src_list = graph.edge_index[0].tolist()
        dst_list = graph.edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            if s in node_set and d in node_set and s != d:  # skip self-loops
                edges.append({"src": s, "dst": d})

    return SubgraphResponse(nodes=node_list, edges=edges)
