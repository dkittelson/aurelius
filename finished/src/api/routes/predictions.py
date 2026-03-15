"""Prediction routes — per-node risk scores and top-k flagged nodes."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, get_app_state, run_inference
from src.api.schemas import (
    NodePrediction,
    PredictRequest,
    PredictResponse,
    TopKRequest,
    TopKResponse,
)
from src.agents.cluster_detector import SuspiciousClusterDetector

router = APIRouter(prefix="/predictions", tags=["predictions"])

_RISK_THRESHOLDS = [
    (0.90, "CRITICAL"),
    (0.75, "HIGH"),
    (0.60, "MEDIUM"),
    (0.0,  "LOW"),
]


def _score_to_risk(prob: float) -> str:
    for threshold, level in _RISK_THRESHOLDS:
        if prob >= threshold:
            return level
    return "LOW"


def _require_predictions(state: AppState) -> torch.Tensor:
    if state.predictions is None:
        raise HTTPException(status_code=503, detail="Model not loaded or inference not run")
    return state.predictions


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, state: AppState = Depends(get_app_state)):
    probs = _require_predictions(state)
    threshold = state.optimal_threshold

    if req.node_ids is not None:
        invalid = [n for n in req.node_ids if n < 0 or n >= state.graph.num_nodes]
        if invalid:
            raise HTTPException(status_code=404, detail=f"Invalid node ids: {invalid[:5]}")
        node_ids = req.node_ids
        selected_probs = probs[node_ids]
    else:
        node_ids = list(range(state.graph.num_nodes))
        selected_probs = probs

    predictions = [
        NodePrediction(
            node_id=nid,
            illicit_prob=round(float(p), 4),
            risk_level=_score_to_risk(float(p)),
        )
        for nid, p in zip(node_ids, selected_probs.tolist())
    ]
    num_flagged = sum(1 for p in predictions if p.illicit_prob >= threshold)
    return PredictResponse(
        predictions=predictions,
        num_flagged=num_flagged,
        threshold_used=threshold,
    )


@router.post("/top-k", response_model=TopKResponse)
async def top_k(req: TopKRequest, state: AppState = Depends(get_app_state)):
    probs = _require_predictions(state)
    above = (probs >= req.threshold).nonzero(as_tuple=True)[0]
    total_above = int(above.numel())

    # Sort descending by prob, take top k
    sorted_idx = above[probs[above].argsort(descending=True)]
    top_idx = sorted_idx[: req.k].tolist()

    nodes = [
        NodePrediction(
            node_id=nid,
            illicit_prob=round(float(probs[nid]), 4),
            risk_level=_score_to_risk(float(probs[nid])),
        )
        for nid in top_idx
    ]
    return TopKResponse(nodes=nodes, total_above_threshold=total_above)


@router.post("/refresh")
async def refresh_predictions(state: AppState = Depends(get_app_state)):
    """Re-run full-graph inference (e.g. after loading a new model checkpoint)."""
    if state.model is None or state.graph is None:
        raise HTTPException(status_code=503, detail="Model or graph not available")
    state.predictions = run_inference(state.graph, state.model, state.hybrid_clf)
    flagged = int((state.predictions > state.optimal_threshold).sum())
    return {"status": "ok", "num_flagged": flagged}
