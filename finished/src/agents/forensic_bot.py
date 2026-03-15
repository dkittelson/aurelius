"""Forensic investigation agent powered by Gemini + FAISS RAG.

Pipeline per cluster:
  1. Extract ego-graph context (structural metrics, attention weights, node stats)
  2. Retrieve similar historical cases from FAISS vector store
  3. Format INVESTIGATION_PROMPT with all context
  4. Call Gemini to generate a formal audit report
  5. Store the report back in FAISS for future retrieval
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from loguru import logger

from src.agents.prompt_templates import (
    SYSTEM_PROMPT,
    INVESTIGATION_PROMPT,
    COMPARISON_PROMPT,
    format_node_details,
    format_attention_analysis,
)
from src.agents.cluster_detector import SuspiciousClusterDetector


class ForensicVectorStore:
    """
    FAISS-backed vector store for forensic report RAG.

    Uses Google's text-embedding-004 model to embed report text,
    then stores in a FAISS flat index for similarity search.
    """

    def __init__(self, index_path: str, embedding_model: str = "models/text-embedding-004"):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self._index = None
        self._documents: list[dict] = []  # parallel list: doc text + metadata
        self._embeddings_cache: list[list[float]] = []

    def _get_embedding(self, text: str) -> list[float]:
        """Call Gemini embedding API for a single text."""
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
        )
        return result["embedding"]

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build a FAISS flat L2 index from a 2D numpy array of embeddings."""
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return index

    def create_index(self, documents: list[dict]) -> None:
        """
        Build FAISS index from a list of documents.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys.
        """
        if not documents:
            logger.warning("No documents provided to create_index.")
            return

        logger.info(f"Embedding {len(documents)} documents...")
        self._documents = documents
        embeddings = []
        for doc in documents:
            emb = self._get_embedding(doc["text"])
            embeddings.append(emb)
            self._embeddings_cache.append(emb)

        emb_array = np.array(embeddings, dtype=np.float32)
        self._index = self._build_faiss_index(emb_array)
        self._save()
        logger.info(f"FAISS index created with {len(documents)} documents.")

    def load_index(self) -> None:
        """Load existing FAISS index and documents from disk."""
        import faiss

        index_file = self.index_path.with_suffix(".faiss")
        meta_file = self.index_path.with_suffix(".json")

        if not index_file.exists():
            logger.info("No existing FAISS index found. Starting fresh.")
            return

        self._index = faiss.read_index(str(index_file))
        with open(meta_file) as f:
            data = json.load(f)
        self._documents = data["documents"]
        self._embeddings_cache = data["embeddings"]
        logger.info(f"Loaded FAISS index with {len(self._documents)} documents.")

    def add_report(self, report_text: str, metadata: dict) -> None:
        """
        Add a new forensic report to the FAISS index.

        Args:
            report_text: The full markdown report text.
            metadata: Dict with cluster_id, timestamp, risk_level, dataset, etc.
        """
        import faiss

        emb = self._get_embedding(report_text)
        emb_array = np.array([emb], dtype=np.float32)

        doc = {"text": report_text, "metadata": metadata}
        self._documents.append(doc)
        self._embeddings_cache.append(emb)

        if self._index is None:
            self._index = self._build_faiss_index(emb_array)
        else:
            self._index.add(emb_array)

        self._save()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Similarity search: find the k most similar reports to the query.

        Returns list of dicts with 'text', 'metadata', 'score' (L2 distance).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_emb = np.ascontiguousarray(
            [self._get_embedding(query)], dtype=np.float32
        )
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(dist),
            })
        return results

    def _save(self) -> None:
        """Persist FAISS index and document metadata to disk."""
        import faiss

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path.with_suffix(".faiss")))
        with open(self.index_path.with_suffix(".json"), "w") as f:
            json.dump(
                {"documents": self._documents, "embeddings": self._embeddings_cache},
                f,
            )

    @property
    def num_reports(self) -> int:
        return len(self._documents)


class ForensicAgent:
    """
    LLM-powered forensic analysis agent using Gemini 2.0 Flash.

    Orchestrates the full investigation pipeline:
      extract context → retrieve similar cases → generate report → store report
    """

    # Gemini free tier: 15 requests per minute
    _RATE_LIMIT_RPM = 15
    _MIN_SECONDS_BETWEEN_CALLS = 60 / _RATE_LIMIT_RPM  # 4 seconds

    def __init__(
        self,
        config: dict,
        vector_store: ForensicVectorStore,
        gemini_api_key: str,
    ):
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        self._genai = genai

        llm_cfg = config["agent"]["llm"]
        self.model = genai.GenerativeModel(
            model_name=llm_cfg["model_name"],
            system_instruction=SYSTEM_PROMPT,
        )
        self.generation_config = genai.GenerationConfig(
            temperature=llm_cfg["temperature"],
            max_output_tokens=llm_cfg["max_output_tokens"],
        )
        self.rag_cfg = config["agent"]["rag"]
        self.vector_store = vector_store
        self._last_call_time: float = 0.0

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    def extract_cluster_context(
        self,
        data: Data,
        cluster_node_ids: list[int],
        model: torch.nn.Module,
        predictions: torch.Tensor,
        dataset: str = "elliptic",
    ) -> dict:
        """
        Extract rich context from a cluster for the investigation prompt.

        Args:
            data:             Full PyG Data object.
            cluster_node_ids: List of global node indices in the cluster.
            model:            Trained AureliusGAT (for attention weights).
            predictions:      [num_nodes] P(illicit) tensor.
            dataset:          Dataset name for the report.

        Returns:
            Context dict ready to be unpacked into INVESTIGATION_PROMPT.
        """
        probs = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        cluster_arr = np.array(cluster_node_ids)

        # --- Build cluster subgraph (NetworkX) ---
        ei = data.edge_index.numpy()
        mask = np.isin(ei[0], cluster_arr) & np.isin(ei[1], cluster_arr)
        sub_src = ei[0][mask].tolist()
        sub_dst = ei[1][mask].tolist()

        G = nx.DiGraph()
        G.add_nodes_from(cluster_node_ids)
        G.add_edges_from(zip(sub_src, sub_dst))
        G_undirected = G.to_undirected()

        # --- Structural metrics ---
        n = len(cluster_node_ids)
        num_edges = G.number_of_edges()
        max_possible = n * (n - 1)
        density = num_edges / max_possible if max_possible > 0 else 0.0

        # Average clustering coefficient
        try:
            avg_clustering = nx.average_clustering(G_undirected)
        except Exception:
            avg_clustering = 0.0

        # Diameter (on largest connected component to avoid error on disconnected)
        try:
            lcc = max(nx.weakly_connected_components(G), key=len)
            lcc_subgraph = G.subgraph(lcc)
            diameter = nx.diameter(lcc_subgraph.to_undirected())
        except Exception:
            diameter = "N/A"

        # Hub nodes by degree
        degree_dict = dict(G.degree())
        hub_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:5]

        # PageRank leaders
        try:
            pr = nx.pagerank(G, alpha=0.85)
            pagerank_leaders = sorted(pr, key=pr.get, reverse=True)[:5]
        except Exception:
            pagerank_leaders = hub_nodes[:5]

        # --- Node details ---
        node_details_list = []
        for nid in cluster_node_ids:
            node_details_list.append({
                "node_id": nid,
                "illicit_prob": float(probs[nid]),
                "degree": degree_dict.get(nid, 0),
                "pagerank": pr.get(nid, 0.0) if isinstance(pr, dict) else 0.0,
            })
        node_details_list.sort(key=lambda x: -x["illicit_prob"])

        # --- Attention weights (top edges) ---
        attention_edges = []
        try:
            model.eval()
            with torch.no_grad():
                attn_list = model.get_attention_weights(data.x, data.edge_index)
            if attn_list:
                # Use last layer attention, mean across heads
                _, alpha_last = attn_list[-1]
                alpha_mean = alpha_last.mean(dim=1).numpy()  # [num_edges]
                # Filter to edges within the cluster
                for i, (s, d) in enumerate(zip(ei[0], ei[1])):
                    if s in set(cluster_node_ids) and d in set(cluster_node_ids):
                        attention_edges.append({
                            "src": int(s), "dst": int(d),
                            "weight": float(alpha_mean[i])
                        })
                attention_edges.sort(key=lambda x: -x["weight"])
        except Exception as e:
            logger.debug(f"Could not extract attention weights: {e}")

        return {
            "cluster_id": None,  # filled in by caller
            "dataset": dataset,
            "num_nodes": n,
            "num_edges": num_edges,
            "avg_confidence": float(np.mean(probs[cluster_arr])),
            "max_confidence": float(np.max(probs[cluster_arr])),
            "timestamp": datetime.utcnow().isoformat(),
            "density": density,
            "avg_clustering": avg_clustering,
            "diameter": diameter,
            "hub_nodes": hub_nodes,
            "pagerank_leaders": pagerank_leaders,
            "node_details_list": node_details_list,
            "attention_edges": attention_edges,
            "node_details": format_node_details(node_details_list),
            "attention_analysis": format_attention_analysis(attention_edges),
        }

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_cases(self, cluster_summary: str) -> str:
        """
        Search FAISS for the most similar past investigations.

        Returns a formatted string ready to insert into the prompt.
        """
        k = self.rag_cfg.get("top_k", 5)
        results = self.vector_store.search(cluster_summary, k=k)

        if not results:
            return "No similar historical cases found in the knowledge base."

        lines = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            lines.append(
                f"**Case {i}** (Cluster {meta.get('cluster_id', '?')}, "
                f"Risk: {meta.get('risk_level', '?')}, "
                f"Dataset: {meta.get('dataset', '?')}):\n"
                f"{r['text'][:400]}..."
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def investigate_cluster(
        self,
        data: Data,
        cluster_node_ids: list[int],
        model: torch.nn.Module,
        predictions: torch.Tensor,
        cluster_id: int,
        dataset: str = "elliptic",
    ) -> str:
        """
        Full investigation pipeline for a single cluster.

        1. Extract cluster context
        2. Retrieve similar cases from FAISS
        3. Format prompt and call Gemini
        4. Store report in FAISS
        5. Return markdown report

        Args:
            cluster_id: An integer ID for this cluster (for tracking).
        """
        # Step 1: Extract context
        context = self.extract_cluster_context(
            data, cluster_node_ids, model, predictions, dataset
        )
        context["cluster_id"] = cluster_id

        # Step 2: Retrieve similar cases
        cluster_summary = (
            f"AML cluster with {context['num_nodes']} nodes, "
            f"avg illicit probability {context['avg_confidence']:.2f}, "
            f"density {context['density']:.3f}, "
            f"diameter {context['diameter']}"
        )
        rag_context = self.retrieve_similar_cases(cluster_summary)
        context["rag_context"] = rag_context

        # Step 3: Format prompt and call Gemini
        prompt = INVESTIGATION_PROMPT.format(**context)
        report = self._call_gemini(prompt)

        # Step 4: Store in FAISS for future retrieval
        self.vector_store.add_report(
            report_text=report,
            metadata={
                "cluster_id": cluster_id,
                "dataset": dataset,
                "risk_level": SuspiciousClusterDetector._score_to_risk(
                    context["avg_confidence"]
                ),
                "timestamp": context["timestamp"],
                "num_nodes": context["num_nodes"],
            },
        )

        return report

    def compare_clusters(
        self,
        data: Data,
        cluster_a_ids: list[int],
        cluster_b_ids: list[int],
        model: torch.nn.Module,
        predictions: torch.Tensor,
    ) -> str:
        """Compare two clusters to determine if they're part of the same operation."""
        probs = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions

        def _summarize(ids):
            avg = float(np.mean(probs[ids]))
            return (
                f"Nodes: {ids[:10]}{'...' if len(ids) > 10 else ''}, "
                f"count={len(ids)}, avg_P(illicit)={avg:.3f}"
            )

        prompt = COMPARISON_PROMPT.format(
            cluster_a_summary=_summarize(cluster_a_ids),
            cluster_b_summary=_summarize(cluster_b_ids),
        )
        return self._call_gemini(prompt)

    def batch_investigate(
        self,
        data: Data,
        clusters: list[list[int]],
        model: torch.nn.Module,
        predictions: torch.Tensor,
        dataset: str = "elliptic",
    ) -> list[str]:
        """
        Investigate multiple clusters with rate limiting for Gemini free tier (15 RPM).

        Args:
            clusters: List of cluster node-ID lists (sorted most suspicious first).

        Returns:
            List of markdown report strings, one per cluster.
        """
        reports = []
        for i, cluster_ids in enumerate(clusters):
            logger.info(
                f"Investigating cluster {i+1}/{len(clusters)} "
                f"({len(cluster_ids)} nodes)..."
            )
            report = self.investigate_cluster(
                data, cluster_ids, model, predictions,
                cluster_id=i, dataset=dataset
            )
            reports.append(report)
            logger.info(f"Cluster {i+1} report generated ({len(report)} chars).")

        return reports

    # ------------------------------------------------------------------
    # Gemini call with rate limiting
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini with rate limiting to stay within free tier (15 RPM).

        Enforces a minimum gap of 4 seconds between calls.
        """
        elapsed = time.time() - self._last_call_time
        if elapsed < self._MIN_SECONDS_BETWEEN_CALLS:
            sleep_time = self._MIN_SECONDS_BETWEEN_CALLS - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            self._last_call_time = time.time()
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
