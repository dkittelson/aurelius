"""Dashboard route — aggregated stats for the frontend overview panel."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import AppState, get_app_state
from src.api.schemas import DashboardStats
from src.agents.cluster_detector import SuspiciousClusterDetector

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def dashboard_stats(state: AppState = Depends(get_app_state)):
    graph = state.graph
    probs = state.predictions

    if graph is None:
        return DashboardStats(
            total_nodes=0, total_edges=0, flagged_nodes=0,
            critical_clusters=0, high_clusters=0, medium_clusters=0, low_clusters=0,
            model_val_auprc=None, dataset=state.dataset_name,
        )

    flagged = int((probs > 0.5).sum()) if probs is not None else 0

    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    if probs is not None:
        detector = SuspiciousClusterDetector(threshold=0.5, min_cluster_size=2)
        clusters = detector.detect_from_predictions(graph, probs)
        for node_ids in clusters:
            stats = detector.get_cluster_stats(graph, node_ids, probs)
            level = stats["risk_level"]
            counts[level] = counts.get(level, 0) + 1

    return DashboardStats(
        total_nodes=graph.num_nodes,
        total_edges=graph.num_edges,
        flagged_nodes=flagged,
        critical_clusters=counts["CRITICAL"],
        high_clusters=counts["HIGH"],
        medium_clusters=counts["MEDIUM"],
        low_clusters=counts["LOW"],
        model_val_auprc=state.val_auprc,
        dataset=state.dataset_name,
    )
