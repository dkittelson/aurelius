"""Explainability routes — GNNExplainer for node/cluster explanations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.dependencies import AppState, get_app_state
from src.api.schemas import (
    ExplainNodeRequest,
    ExplainClusterRequest,
    ImportantEdge,
    ImportantFeature,
    NodeExplanationResponse,
    ClusterExplanationResponse,
)
from src.models.explainer import AureliusExplainer

router = APIRouter(prefix="/explain", tags=["explain"])


def _get_explainer(state: AppState) -> AureliusExplainer:
    if state.model is None or state.graph is None:
        raise HTTPException(
            status_code=503, detail="Model or graph not loaded"
        )
    # Create explainer on-demand (not persisted — it's lightweight)
    return AureliusExplainer(state.model, state.graph, epochs=100)


@router.post("/node", response_model=NodeExplanationResponse)
async def explain_node(
    req: ExplainNodeRequest,
    state: AppState = Depends(get_app_state),
):
    """Generate GNNExplainer explanation for a single node."""
    if state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")
    if req.node_id < 0 or req.node_id >= state.graph.num_nodes:
        raise HTTPException(
            status_code=404, detail=f"Node {req.node_id} out of range"
        )

    explainer = _get_explainer(state)
    explanation = explainer.explain_node(
        req.node_id,
        top_k_edges=req.top_k_edges,
        top_k_features=req.top_k_features,
    )
    summary = explainer.format_explanation(explanation)

    return NodeExplanationResponse(
        node_id=explanation["node_id"],
        important_edges=[
            ImportantEdge(**e) for e in explanation["important_edges"]
        ],
        important_features=[
            ImportantFeature(**f) for f in explanation["important_features"]
        ],
        summary=summary,
    )


@router.post("/cluster", response_model=ClusterExplanationResponse)
async def explain_cluster(
    req: ExplainClusterRequest,
    state: AppState = Depends(get_app_state),
):
    """Aggregate GNNExplainer explanations for a cluster of nodes."""
    if state.graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")

    invalid = [
        n for n in req.node_ids
        if n < 0 or n >= state.graph.num_nodes
    ]
    if invalid:
        raise HTTPException(
            status_code=404, detail=f"Invalid node ids: {invalid[:5]}"
        )

    explainer = _get_explainer(state)
    result = explainer.explain_cluster(
        req.node_ids, top_k_edges=req.top_k_edges
    )

    return ClusterExplanationResponse(
        node_ids=result["node_ids"],
        important_edges=[
            ImportantEdge(**e) for e in result["important_edges"]
        ],
        summary=result["summary"],
    )
