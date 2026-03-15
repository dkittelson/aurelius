"""Forensic / cluster routes — detect suspicious clusters, generate AI reports."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import AppState, get_app_state
from src.api.schemas import (
    ClusterDetectRequest,
    ClusterDetectResponse,
    ClusterSummary,
    InvestigateRequest,
    InvestigateResponse,
)
from src.agents.cluster_detector import SuspiciousClusterDetector
from src.agents.forensic_bot import ForensicAgent, ForensicVectorStore

router = APIRouter(prefix="/forensic", tags=["forensic"])


def _require_state(state: AppState) -> AppState:
    if state.graph is None or state.predictions is None:
        raise HTTPException(status_code=503, detail="Graph / predictions not loaded")
    return state


@router.post("/clusters", response_model=ClusterDetectResponse)
async def detect_clusters(req: ClusterDetectRequest, state: AppState = Depends(get_app_state)):
    state = _require_state(state)
    detector = SuspiciousClusterDetector(
        threshold=req.threshold,
        min_cluster_size=req.min_cluster_size,
    )
    raw_clusters = detector.detect_from_predictions(state.graph, state.predictions)
    raw_clusters = raw_clusters[: req.max_clusters]

    summaries: list[ClusterSummary] = []
    for i, node_ids in enumerate(raw_clusters):
        stats = detector.get_cluster_stats(state.graph, node_ids, state.predictions)
        summaries.append(
            ClusterSummary(
                cluster_id=i,
                node_ids=stats["node_ids"],
                num_nodes=stats["num_nodes"],
                num_edges=stats["num_edges"],
                avg_confidence=round(stats["avg_confidence"], 4),
                max_confidence=round(stats["max_confidence"], 4),
                risk_level=stats["risk_level"],
                density=round(stats["density"], 4),
            )
        )

    total_suspicious = sum(len(c) for c in raw_clusters)
    return ClusterDetectResponse(clusters=summaries, total_suspicious_nodes=total_suspicious)


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate_cluster(req: InvestigateRequest, state: AppState = Depends(get_app_state)):
    state = _require_state(state)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set — forensic reports unavailable",
        )

    cfg = state.config
    agent_cfg = cfg.get("agent", {})
    faiss_path = cfg.get("faiss", {}).get("index_path", "data/faiss_index")

    vector_store = ForensicVectorStore(
        index_path=faiss_path,
        embedding_model=agent_cfg.get("rag", {}).get(
            "embedding_model", "models/text-embedding-004"
        ),
    )
    vector_store.load_index()

    agent = ForensicAgent(cfg, vector_store, gemini_api_key=api_key)

    # Validate node ids
    if not req.node_ids:
        raise HTTPException(status_code=400, detail="node_ids must be non-empty")
    invalid = [n for n in req.node_ids if n < 0 or n >= state.graph.num_nodes]
    if invalid:
        raise HTTPException(status_code=404, detail=f"Invalid node ids: {invalid[:5]}")

    report, context = agent.investigate_cluster(
        graph=state.graph,
        cluster_nodes=req.node_ids,
        cluster_id=req.cluster_id,
        model=state.model,
        node_probs=state.predictions,
        dataset=req.dataset,
    )

    probs_np = state.predictions[req.node_ids].numpy()
    avg_conf = float(probs_np.mean())

    detector = SuspiciousClusterDetector()
    stats = detector.get_cluster_stats(state.graph, req.node_ids, state.predictions)

    similar: list[dict[str, Any]] = []
    try:
        raw_similar = vector_store.search(report[:500], k=3)
        for s in raw_similar:
            similar.append({"metadata": s.get("metadata", {}), "text_snippet": s.get("text", "")[:200]})
    except Exception:
        pass

    return InvestigateResponse(
        cluster_id=req.cluster_id,
        report=report,
        risk_level=stats["risk_level"],
        avg_confidence=round(avg_conf, 4),
        similar_cases=similar,
    )
