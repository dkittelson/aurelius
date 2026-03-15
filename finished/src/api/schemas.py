"""Pydantic request/response schemas for the Aurelius API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / generic
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset: str


# ---------------------------------------------------------------------------
# Graph routes
# ---------------------------------------------------------------------------

class GraphStatsResponse(BaseModel):
    num_nodes: int
    num_edges: int
    num_illicit: int
    num_licit: int
    num_unknown: int
    illicit_rate: float


class NodeNeighborsRequest(BaseModel):
    node_id: int
    hops: int = Field(default=1, ge=1, le=3)


class NodeNeighborsResponse(BaseModel):
    node_id: int
    neighbors: list[int]
    edge_count: int


class SubgraphRequest(BaseModel):
    node_ids: list[int]
    include_edges: bool = True
    expand_hops: int = Field(default=1, ge=0, le=2)


class SubgraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, int]]


# ---------------------------------------------------------------------------
# Predictions routes
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    node_ids: Optional[list[int]] = None   # None → predict all nodes
    dataset: str = "elliptic"


class NodePrediction(BaseModel):
    node_id: int
    illicit_prob: float
    risk_level: str


class PredictResponse(BaseModel):
    predictions: list[NodePrediction]
    num_flagged: int
    threshold_used: float


class TopKRequest(BaseModel):
    k: int = Field(default=20, ge=1, le=500)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class TopKResponse(BaseModel):
    nodes: list[NodePrediction]
    total_above_threshold: int


# ---------------------------------------------------------------------------
# Forensic / cluster routes
# ---------------------------------------------------------------------------

class ClusterDetectRequest(BaseModel):
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_cluster_size: int = Field(default=2, ge=1)
    max_clusters: int = Field(default=10, ge=1, le=50)


class ClusterSummary(BaseModel):
    cluster_id: int
    node_ids: list[int]
    num_nodes: int
    num_edges: int
    avg_confidence: float
    max_confidence: float
    risk_level: str
    density: float


class ClusterDetectResponse(BaseModel):
    clusters: list[ClusterSummary]
    total_suspicious_nodes: int


class InvestigateRequest(BaseModel):
    cluster_id: int
    node_ids: list[int]
    dataset: str = "elliptic"


class InvestigateResponse(BaseModel):
    cluster_id: int
    report: str
    risk_level: str
    avg_confidence: float
    similar_cases: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    total_nodes: int
    total_edges: int
    flagged_nodes: int
    critical_clusters: int
    high_clusters: int
    medium_clusters: int
    low_clusters: int
    model_val_auprc: Optional[float]
    dataset: str


# ---------------------------------------------------------------------------
# Explainability routes
# ---------------------------------------------------------------------------

class ExplainNodeRequest(BaseModel):
    node_id: int
    top_k_edges: int = Field(default=10, ge=1, le=50)
    top_k_features: int = Field(default=10, ge=1, le=50)


class ImportantEdge(BaseModel):
    src: int
    dst: int
    importance: float


class ImportantFeature(BaseModel):
    feature_index: int
    importance: float


class NodeExplanationResponse(BaseModel):
    node_id: int
    important_edges: list[ImportantEdge]
    important_features: list[ImportantFeature]
    summary: str


class ExplainClusterRequest(BaseModel):
    node_ids: list[int]
    top_k_edges: int = Field(default=20, ge=1, le=100)


class ClusterExplanationResponse(BaseModel):
    node_ids: list[int]
    important_edges: list[ImportantEdge]
    summary: str
