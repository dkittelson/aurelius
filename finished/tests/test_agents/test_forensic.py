"""Tests for the forensic agent system.

All Gemini API calls and FAISS operations are mocked so tests run
offline without any API keys or GPU requirements.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_graph(num_nodes=30, in_channels=16, seed=42):
    torch.manual_seed(seed)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    self_loops = torch.arange(num_nodes)
    edge_index = torch.stack([
        torch.cat([src, self_loops]),
        torch.cat([dst, self_loops]),
    ])
    x = torch.rand(num_nodes, in_channels)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[:5] = 1
    return Data(x=x, edge_index=edge_index, y=y)


def make_predictions(num_nodes=30, seed=42):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0, 1, num_nodes).astype(np.float32)
    # Make first 5 nodes highly suspicious
    probs[:5] = rng.uniform(0.85, 0.99, 5)
    return torch.tensor(probs)


@pytest.fixture
def graph():
    return make_graph()


@pytest.fixture
def predictions():
    return make_predictions()


@pytest.fixture
def config():
    return {
        "agent": {
            "llm": {
                "model_name": "gemini-2.0-flash",
                "temperature": 0.3,
                "max_output_tokens": 4096,
            },
            "rag": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "top_k": 3,
                "embedding_model": "models/text-embedding-004",
            },
        }
    }


# ---------------------------------------------------------------------------
# SuspiciousClusterDetector
# ---------------------------------------------------------------------------

class TestSuspiciousClusterDetector:

    def test_detects_high_prob_nodes(self, graph, predictions):
        """Should detect clusters containing the high-probability nodes."""
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector(threshold=0.8, min_cluster_size=2)
        clusters = detector.detect_from_predictions(graph, predictions)
        # The first 5 nodes have prob > 0.85, should be detected
        if clusters:
            all_detected = [n for c in clusters for n in c]
            high_prob_nodes = (predictions.numpy() >= 0.8).nonzero()[0].tolist()
            overlap = set(all_detected) & set(high_prob_nodes)
            assert len(overlap) > 0

    def test_returns_list_of_lists(self, graph, predictions):
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector(threshold=0.7)
        clusters = detector.detect_from_predictions(graph, predictions)
        assert isinstance(clusters, list)
        for c in clusters:
            assert isinstance(c, list)
            assert all(isinstance(n, int) for n in c)

    def test_min_cluster_size_filter(self, graph, predictions):
        """All returned clusters must meet the minimum size."""
        from src.agents.cluster_detector import SuspiciousClusterDetector
        min_size = 3
        detector = SuspiciousClusterDetector(threshold=0.5, min_cluster_size=min_size)
        clusters = detector.detect_from_predictions(graph, predictions)
        for c in clusters:
            assert len(c) >= min_size

    def test_sorted_by_confidence(self, graph, predictions):
        """Clusters should be sorted descending by average illicit probability."""
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector(threshold=0.5, min_cluster_size=1)
        clusters = detector.detect_from_predictions(graph, predictions)
        if len(clusters) >= 2:
            probs = predictions.numpy()
            avg_confs = [np.mean(probs[c]) for c in clusters]
            assert avg_confs == sorted(avg_confs, reverse=True)

    def test_no_nodes_above_threshold(self, graph):
        """If no nodes exceed threshold, returns empty list."""
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector(threshold=0.99)
        # All-zero predictions — nothing should exceed 0.99
        zero_preds = torch.zeros(graph.num_nodes)
        clusters = detector.detect_from_predictions(graph, zero_preds)
        assert clusters == []

    def test_get_cluster_stats_keys(self, graph, predictions):
        """get_cluster_stats should return all required keys."""
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector()
        stats = detector.get_cluster_stats(graph, [0, 1, 2, 3], predictions)
        required = {"node_ids", "num_nodes", "num_edges", "avg_confidence",
                    "max_confidence", "risk_level", "density"}
        assert required.issubset(stats.keys())

    def test_risk_level_mapping(self):
        from src.agents.cluster_detector import SuspiciousClusterDetector
        assert SuspiciousClusterDetector._score_to_risk(0.95) == "CRITICAL"
        assert SuspiciousClusterDetector._score_to_risk(0.80) == "HIGH"
        assert SuspiciousClusterDetector._score_to_risk(0.65) == "MEDIUM"
        assert SuspiciousClusterDetector._score_to_risk(0.50) == "LOW"

    def test_density_in_range(self, graph, predictions):
        from src.agents.cluster_detector import SuspiciousClusterDetector
        detector = SuspiciousClusterDetector()
        stats = detector.get_cluster_stats(graph, list(range(10)), predictions)
        assert 0.0 <= stats["density"] <= 1.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class TestPromptTemplates:

    def test_investigation_prompt_formats(self):
        """INVESTIGATION_PROMPT should format without KeyError with valid context."""
        from src.agents.prompt_templates import INVESTIGATION_PROMPT

        context = {
            "cluster_id": 0,
            "dataset": "elliptic",
            "num_nodes": 10,
            "num_edges": 15,
            "avg_confidence": 0.87,
            "max_confidence": 0.95,
            "timestamp": "2026-01-01T00:00:00",
            "density": 0.33,
            "avg_clustering": 0.45,
            "diameter": 3,
            "hub_nodes": [1, 2, 3],
            "pagerank_leaders": [1, 4, 5],
            "node_details": "Node 1: P=0.95",
            "attention_analysis": "1 -> 2: 0.42",
            "rag_context": "No similar cases.",
        }
        result = INVESTIGATION_PROMPT.format(**context)
        assert "cluster_id" not in result  # all placeholders filled
        assert "0.87" in result            # avg_confidence present
        assert "elliptic" in result

    def test_format_node_details_empty(self):
        from src.agents.prompt_templates import format_node_details
        result = format_node_details([])
        assert "No node details" in result

    def test_format_node_details_caps_at_20(self):
        from src.agents.prompt_templates import format_node_details
        nodes = [{"node_id": i, "illicit_prob": 0.9, "degree": 3, "pagerank": 0.01}
                 for i in range(30)]
        result = format_node_details(nodes)
        assert "10 more nodes" in result  # 30-20=10

    def test_format_attention_analysis_empty(self):
        from src.agents.prompt_templates import format_attention_analysis
        result = format_attention_analysis([])
        assert "not available" in result

    def test_format_attention_analysis_caps_at_10(self):
        from src.agents.prompt_templates import format_attention_analysis
        edges = [{"src": i, "dst": i+1, "weight": 0.5} for i in range(20)]
        result = format_attention_analysis(edges)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 10


# ---------------------------------------------------------------------------
# ForensicVectorStore (mocked FAISS + Gemini embeddings)
#
# FAISS native search crashes on macOS ARM (faiss-cpu 1.13.2 bug), so we
# replace the FAISS index with a pure-Python mock for all these tests.
# ---------------------------------------------------------------------------

class _MockFaissIndex:
    """Pure-Python drop-in for faiss.IndexFlatL2 — no native calls."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: list[np.ndarray] = []

    @property
    def ntotal(self) -> int:
        return len(self._vecs)

    def add(self, vecs: np.ndarray) -> None:
        for v in vecs:
            self._vecs.append(v.copy())

    def search(self, query: np.ndarray, k: int):
        """Brute-force L2 search in Python."""
        q = query[0]
        k = min(k, len(self._vecs))
        dists = [float(np.sum((q - v) ** 2)) for v in self._vecs]
        order = np.argsort(dists)[:k]
        D = np.array([[dists[i] for i in order]], dtype=np.float32)
        I = np.array([order], dtype=np.int64)
        return D, I


def _make_mock_embedding(dim: int = 768) -> list[float]:
    """Return a random unit vector as a fake embedding."""
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _faiss_patches(index_path: str):
    """
    Context manager that patches faiss.IndexFlatL2, faiss.write_index,
    and faiss.read_index so no native FAISS code is executed.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        mock_index_instance = None

        def fake_flat_l2(dim):
            nonlocal mock_index_instance
            mock_index_instance = _MockFaissIndex(dim)
            return mock_index_instance

        def fake_write_index(idx, path):
            # Create stub file so load_index's existence check passes.
            # Real vector data is persisted via the JSON metadata.
            import pathlib
            pathlib.Path(path).touch()

        def fake_read_index(path):
            # Rebuild from the JSON metadata that _save() wrote
            import json, pathlib
            meta_path = pathlib.Path(path).with_suffix(".json")
            if not meta_path.exists():
                return _MockFaissIndex(768)
            with open(meta_path) as f:
                data = json.load(f)
            embs = data.get("embeddings", [])
            dim = len(embs[0]) if embs else 768
            idx = _MockFaissIndex(dim)
            for e in embs:
                idx.add(np.array([e], dtype=np.float32))
            return idx

        with patch("faiss.IndexFlatL2", side_effect=fake_flat_l2), \
             patch("faiss.write_index", side_effect=fake_write_index), \
             patch("faiss.read_index", side_effect=fake_read_index):
            yield

    return _ctx()


class TestForensicVectorStore:

    def test_add_and_search_roundtrip(self, tmp_path):
        """Add a report then search — should find it."""
        from src.agents.forensic_bot import ForensicVectorStore

        store = ForensicVectorStore(str(tmp_path / "test_index"))
        with _faiss_patches(str(tmp_path / "test_index")), \
             patch.object(store, "_get_embedding", side_effect=lambda t: _make_mock_embedding()):
            store.add_report("Suspicious layering pattern in cluster 5.", {"cluster_id": 5})
            results = store.search("layering pattern", k=1)

        assert len(results) == 1
        assert results[0]["metadata"]["cluster_id"] == 5

    def test_search_empty_store_returns_empty(self, tmp_path):
        """Searching an empty store should return []."""
        from src.agents.forensic_bot import ForensicVectorStore
        store = ForensicVectorStore(str(tmp_path / "empty_index"))
        results = store.search("any query")
        assert results == []

    def test_num_reports_increments(self, tmp_path):
        from src.agents.forensic_bot import ForensicVectorStore
        store = ForensicVectorStore(str(tmp_path / "idx"))
        with _faiss_patches(str(tmp_path / "idx")), \
             patch.object(store, "_get_embedding", return_value=_make_mock_embedding()):
            assert store.num_reports == 0
            store.add_report("Report 1", {"cluster_id": 1})
            assert store.num_reports == 1
            store.add_report("Report 2", {"cluster_id": 2})
            assert store.num_reports == 2

    def test_search_returns_k_results(self, tmp_path):
        from src.agents.forensic_bot import ForensicVectorStore
        store = ForensicVectorStore(str(tmp_path / "idx2"))
        with _faiss_patches(str(tmp_path / "idx2")), \
             patch.object(store, "_get_embedding", return_value=_make_mock_embedding()):
            for i in range(10):
                store.add_report(f"Report about cluster {i}", {"cluster_id": i})
            results = store.search("cluster pattern", k=3)
        assert len(results) == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        """Index saved to disk should load back with same document count."""
        from src.agents.forensic_bot import ForensicVectorStore

        store = ForensicVectorStore(str(tmp_path / "persist_idx"))
        with _faiss_patches(str(tmp_path / "persist_idx")), \
             patch.object(store, "_get_embedding", return_value=_make_mock_embedding()):
            store.add_report("Fan-out pattern detected.", {"cluster_id": 42, "risk_level": "HIGH"})

        store2 = ForensicVectorStore(str(tmp_path / "persist_idx"))
        with _faiss_patches(str(tmp_path / "persist_idx")):
            store2.load_index()
        assert store2.num_reports == 1
        assert store2._documents[0]["metadata"]["cluster_id"] == 42


# ---------------------------------------------------------------------------
# ForensicAgent (mocked Gemini + FAISS)
# ---------------------------------------------------------------------------

class TestForensicAgent:

    def _make_agent(self, config, tmp_path):
        from src.agents.forensic_bot import ForensicVectorStore, ForensicAgent

        store = ForensicVectorStore(str(tmp_path / "agent_idx"))

        # Patch genai at the module level before instantiation
        with patch("src.agents.forensic_bot.google") as mock_google, \
             patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockModel:

            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.return_value = MagicMock(
                text="**1. Executive Summary** Mock forensic report for testing."
            )
            MockModel.return_value = mock_model_instance

            import google.generativeai as genai
            with patch.object(genai, "configure"), \
                 patch.object(genai, "GenerativeModel", return_value=mock_model_instance):
                agent = ForensicAgent(config, store, gemini_api_key="test-key")
                agent.model = mock_model_instance

        return agent, store

    def test_extract_cluster_context_keys(self, graph, predictions, config, tmp_path):
        """extract_cluster_context should return all keys needed by INVESTIGATION_PROMPT."""
        from src.agents.forensic_bot import ForensicVectorStore, ForensicAgent
        from src.models.gnn_model import AureliusGAT

        store = ForensicVectorStore(str(tmp_path / "ctx_idx"))

        model = AureliusGAT(in_channels=16, hidden_channels=16, out_channels=2,
                            num_heads=4, num_layers=2)
        model.eval()

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockGM:
            MockGM.return_value = MagicMock()
            agent = ForensicAgent(config, store, gemini_api_key="test-key")

        context = agent.extract_cluster_context(
            graph, [0, 1, 2, 3, 4], model, predictions
        )

        required_keys = {
            "num_nodes", "num_edges", "avg_confidence", "max_confidence",
            "timestamp", "density", "avg_clustering", "diameter",
            "hub_nodes", "pagerank_leaders", "node_details", "attention_analysis",
            "node_details_list",
        }
        assert required_keys.issubset(context.keys())

    def test_context_num_nodes_matches_input(self, graph, predictions, config, tmp_path):
        from src.agents.forensic_bot import ForensicVectorStore, ForensicAgent
        from src.models.gnn_model import AureliusGAT

        store = ForensicVectorStore(str(tmp_path / "ctx2"))
        model = AureliusGAT(in_channels=16, hidden_channels=16, out_channels=2,
                            num_heads=4, num_layers=2)

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockGM:
            MockGM.return_value = MagicMock()
            agent = ForensicAgent(config, store, gemini_api_key="test-key")

        cluster_ids = [0, 1, 2, 3, 4]
        context = agent.extract_cluster_context(graph, cluster_ids, model, predictions)
        assert context["num_nodes"] == len(cluster_ids)

    def test_retrieve_similar_cases_empty_store(self, config, tmp_path):
        """Empty store should return a 'no similar cases' string."""
        from src.agents.forensic_bot import ForensicVectorStore, ForensicAgent

        store = ForensicVectorStore(str(tmp_path / "empty"))
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockGM:
            MockGM.return_value = MagicMock()
            agent = ForensicAgent(config, store, gemini_api_key="test-key")

        result = agent.retrieve_similar_cases("suspicious fan-out pattern")
        assert "No similar" in result

    def test_rate_limiting_enforced(self, config, tmp_path):
        """Second call within the rate limit window should be delayed."""
        import time
        from src.agents.forensic_bot import ForensicVectorStore, ForensicAgent

        store = ForensicVectorStore(str(tmp_path / "rl"))
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockGM:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="report")
            MockGM.return_value = mock_model
            agent = ForensicAgent(config, store, gemini_api_key="test-key")

        with patch("time.sleep") as mock_sleep:
            # Simulate last call was 1 second ago (less than the 4s minimum)
            agent._last_call_time = time.time() - 1.0
            agent._call_gemini("test prompt")
            # sleep should have been called
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            assert sleep_duration > 0
