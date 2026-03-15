"""
Integration tests for FastAPI routes.

All graph/model/prediction state is injected via dependency override so tests
run fully offline without loading real data or model checkpoints.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from torch_geometric.data import Data

from src.api.main import create_app
from src.api.dependencies import AppState, get_app_state
from src.models.gnn_model import AureliusGAT


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_NODES = 50
IN_CHANNELS = 16


def _make_test_graph(num_nodes: int = NUM_NODES) -> Data:
    torch.manual_seed(0)
    src = torch.randint(0, num_nodes, (100,))
    dst = torch.randint(0, num_nodes, (100,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, IN_CHANNELS)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:5] = 1   # 5 illicit nodes
    y[5:10] = -1  # 5 unknown
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def _make_test_predictions(num_nodes: int = NUM_NODES) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(1)
    probs = torch.rand(num_nodes, generator=rng)
    probs[:5] = torch.tensor([0.92, 0.88, 0.85, 0.91, 0.95])  # high-risk first 5
    return probs


def _make_test_model() -> AureliusGAT:
    return AureliusGAT(
        in_channels=IN_CHANNELS,
        hidden_channels=16,
        out_channels=2,
        num_heads=4,
        num_layers=2,
    )


@pytest.fixture
def test_state() -> AppState:
    graph = _make_test_graph()
    model = _make_test_model()
    probs = _make_test_predictions()
    return AppState(
        graph=graph,
        model=model,
        predictions=probs,
        dataset_name="elliptic",
        val_auprc=0.72,
        config={
            "model": {"hidden_channels": 16, "num_heads": 4, "num_layers": 2},
            "training": {"checkpoint_dir": "data/processed/checkpoints"},
        },
    )


@pytest.fixture
def client(test_state: AppState) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_app_state] = lambda: test_state
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["dataset"] == "elliptic"

    def test_health_no_model(self, test_state: AppState):
        test_state.model = None
        app = create_app()
        app.dependency_overrides[get_app_state] = lambda: test_state
        c = TestClient(app)
        assert c.get("/health").json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# Graph routes
# ---------------------------------------------------------------------------

class TestGraphRoutes:
    def test_stats_shape(self, client: TestClient):
        resp = client.get("/api/v1/graph/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_nodes"] == NUM_NODES
        assert body["num_illicit"] == 5
        assert body["num_unknown"] == 5
        assert 0.0 <= body["illicit_rate"] <= 1.0

    def test_stats_illicit_rate_correct(self, client: TestClient):
        # 5 illicit / (5 illicit + 40 licit) = 0.111...
        resp = client.get("/api/v1/graph/stats")
        body = resp.json()
        expected = 5 / (NUM_NODES - 5)  # 5 illicit / 45 labeled
        assert abs(body["illicit_rate"] - expected) < 0.01

    def test_neighbors_valid_node(self, client: TestClient):
        resp = client.post("/api/v1/graph/neighbors", json={"node_id": 0, "hops": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == 0
        assert isinstance(body["neighbors"], list)

    def test_neighbors_invalid_node(self, client: TestClient):
        resp = client.post("/api/v1/graph/neighbors", json={"node_id": 9999, "hops": 1})
        assert resp.status_code == 404

    def test_subgraph_returns_nodes_and_edges(self, client: TestClient):
        resp = client.post("/api/v1/graph/subgraph", json={"node_ids": [0, 1, 2, 3], "include_edges": True})
        assert resp.status_code == 200
        body = resp.json()
        # BFS expansion (default expand_hops=1) returns seed nodes + neighbors
        assert len(body["nodes"]) >= 4
        assert isinstance(body["edges"], list)
        for node in body["nodes"]:
            assert "node_id" in node
            assert "illicit_prob" in node

    def test_subgraph_empty_raises(self, client: TestClient):
        resp = client.post("/api/v1/graph/subgraph", json={"node_ids": []})
        assert resp.status_code == 400

    def test_subgraph_invalid_node_raises(self, client: TestClient):
        resp = client.post("/api/v1/graph/subgraph", json={"node_ids": [9999]})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Prediction routes
# ---------------------------------------------------------------------------

class TestPredictionRoutes:
    def test_predict_all(self, client: TestClient):
        resp = client.post("/api/v1/predictions/predict", json={"dataset": "elliptic"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == NUM_NODES
        assert body["threshold_used"] == 0.5
        assert body["num_flagged"] >= 5  # at least the 5 high-prob nodes

    def test_predict_subset(self, client: TestClient):
        resp = client.post("/api/v1/predictions/predict", json={"node_ids": [0, 1, 2], "dataset": "elliptic"})
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3

    def test_predict_invalid_node(self, client: TestClient):
        resp = client.post("/api/v1/predictions/predict", json={"node_ids": [9999]})
        assert resp.status_code == 404

    def test_predict_risk_levels(self, client: TestClient):
        resp = client.post("/api/v1/predictions/predict", json={"node_ids": [0], "dataset": "elliptic"})
        node = resp.json()["predictions"][0]
        assert node["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        # node 0 has prob 0.92 → CRITICAL
        assert node["risk_level"] == "CRITICAL"

    def test_top_k_count(self, client: TestClient):
        resp = client.post("/api/v1/predictions/top-k", json={"k": 5, "threshold": 0.5})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) <= 5
        assert body["total_above_threshold"] >= 5

    def test_top_k_sorted_descending(self, client: TestClient):
        resp = client.post("/api/v1/predictions/top-k", json={"k": 10, "threshold": 0.0})
        nodes = resp.json()["nodes"]
        probs = [n["illicit_prob"] for n in nodes]
        assert probs == sorted(probs, reverse=True)

    def test_refresh_predictions(self, client: TestClient):
        resp = client.post("/api/v1/predictions/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Forensic routes
# ---------------------------------------------------------------------------

class TestForensicRoutes:
    def test_detect_clusters_returns_list(self, client: TestClient):
        resp = client.post(
            "/api/v1/forensic/clusters",
            json={"threshold": 0.7, "min_cluster_size": 2, "max_clusters": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "clusters" in body
        assert isinstance(body["clusters"], list)
        assert "total_suspicious_nodes" in body

    def test_detect_clusters_risk_levels_valid(self, client: TestClient):
        resp = client.post("/api/v1/forensic/clusters", json={"threshold": 0.5})
        for cluster in resp.json()["clusters"]:
            assert cluster["risk_level"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
            assert 0.0 <= cluster["density"] <= 1.0

    def test_investigate_no_api_key(self, client: TestClient):
        """Should 503 when GEMINI_API_KEY is not set."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            resp = client.post(
                "/api/v1/forensic/investigate",
                json={"cluster_id": 0, "node_ids": [0, 1, 2], "dataset": "elliptic"},
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Dashboard routes
# ---------------------------------------------------------------------------

class TestDashboardRoutes:
    def test_stats_keys(self, client: TestClient):
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.status_code == 200
        body = resp.json()
        required = {
            "total_nodes", "total_edges", "flagged_nodes",
            "critical_clusters", "high_clusters", "medium_clusters", "low_clusters",
            "model_val_auprc", "dataset",
        }
        assert required.issubset(body.keys())

    def test_stats_total_nodes_match(self, client: TestClient):
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.json()["total_nodes"] == NUM_NODES

    def test_stats_val_auprc(self, client: TestClient):
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.json()["model_val_auprc"] == pytest.approx(0.72)

    def test_stats_no_graph(self, test_state: AppState):
        test_state.graph = None
        test_state.predictions = None
        app = create_app()
        app.dependency_overrides[get_app_state] = lambda: test_state
        c = TestClient(app)
        body = c.get("/api/v1/dashboard/stats").json()
        assert body["total_nodes"] == 0
        assert body["flagged_nodes"] == 0
