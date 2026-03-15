# Aurelius — Graph-Native Financial Intelligence & Forensic Reasoning

**TEACHING MODE: Claude is acting as a high-end, award-winning teacher guiding Devan through building this project. Guide, explain, and help him learn — don't just write the code for him.**

AML detection system: GATv2 + Temporal GNN + XGBoost hybrid + Gemini forensic agent.

**STATUS: Rebuilding `finished/` from scratch in `src/`, `tests/`, `frontend/src/`, then extending with new agentic features not in `finished/`.**

## Project Scope

### ✅ Phases 1–7: ML Pipeline (complete)
- Graph construction (Elliptic/IBM AML), PageRank/centrality features
- GATv2 (3-layer, 128-hidden, 4-head, JK-cat, residuals) → trained standalone
- TemporalAureliusGAT: GATv2 encoder + GRU memory (128-dim) across 49 timesteps
- HybridClassifier: 677-dim features [165 raw || 384 GNN emb || 128 GRU mem] → XGBoost
- Best results: **F1=0.87, AUPRC=0.89**, threshold=0.95
- Checkpoints: `data/processed/checkpoints/best_temporal_gnn.pt`, `best_xgboost.json`

### 🔲 Phases 8–13: Agentic Investigation System (building now)

**Phase 8** — `src/agents/cluster_detector.py` (from `finished/`)
- `SuspiciousClusterDetector`: connected components + Louvain on high-risk subgraph
- `get_cluster_stats()` → density, risk_level, avg/max confidence

**Phase 9** — `src/agents/typology_detector.py` ⭐ NEW
- `TypologyDetector.detect_all(cluster_ids)` → `list[TypologyResult]`
- 6 algorithmic pattern detectors via NetworkX (no LLM guessing):
  - Smurfing: `in_degree≥3, in/out≥3.0`; Fan-out: `out_degree≥4, out/in≥3.0`
  - Fan-in: mirror of fan-out; Round-trip: `nx.simple_cycles()` length≤8
  - Layering: `nx.all_simple_paths(cutoff=10)` all-suspicious chain
  - Scatter-gather: fan-out hub → shared downstream fan-in collector
- `TypologyResult`: `typology, confidence, evidence_nodes, evidence_edges, description, metrics`

**Phase 10** — `src/agents/attention_tracer.py` ⭐ NEW
- Prereq: add `get_attention_weights()` + `get_embeddings()` to `src/models/gnn_model.py`
- `AttentionTracer(model, data)`: lazy-caches GATv2 attention from one forward pass
- `trace_evidence_path(node, max_hops=3)` → greedy backward walk on highest-attention edges
- Returns `EvidencePath(path, attention_weights, cumulative_attention)` — explains *why* a node was flagged
- `get_top_attention_edges(node_ids, top_k)` + `get_node_attention_profile(node_id)`

**Phase 11** — `src/agents/forensic_agent.py` ⭐ NEW
- `ForensicAgent.investigate_cluster()`: Gemini ReAct loop (max 8 iterations, function calling)
- 7 registered tools: `detect_typologies`, `trace_attention_path`, `get_node_profile`, `get_neighborhood`, `get_cluster_structure`, `search_similar_cases`, `expand_investigation`
- Returns `InvestigationResult(report, tool_calls_log, typologies, evidence_paths, iterations)`

**Phase 12** — FastAPI: new endpoints `/forensic/typologies`, `/forensic/attention-trace`, `/forensic/investigate`

**Phase 13** — React: `InvestigationTimeline`, `TypologyBadge`, `AttentionPathView`, attention-weighted graph edges

## Architecture

```
src/graph/builder.py        ← PyG graph construction (Elliptic + IBM AML)
src/graph/features.py       ← PageRank, centrality, clustering (+5 features/node)
src/models/gnn_model.py     ← GATv2 (3-layer, 128-hidden, 4-head, JK-cat, residuals)
src/models/temporal_gnn.py  ← TemporalNodeMemory (GRU) + TemporalAureliusGAT
src/models/classifier.py    ← HybridClassifier (GNN emb → XGBoost)
src/pipeline/train.py       ← Trainer (NeighborLoader, EarlyStopping, checkpoint)
src/pipeline/temporal_train.py ← TemporalTrainer (truncated BPTT, snapshot loop)
src/pipeline/evaluate.py    ← evaluate_model, find_optimal_threshold, report gen
src/agents/cluster_detector.py   ← SuspiciousClusterDetector
src/agents/typology_detector.py  ← 6 AML pattern detectors
src/agents/attention_tracer.py   ← GATv2 attention → evidence paths
src/agents/forensic_agent.py     ← Gemini ReAct agent (function calling)
src/agents/tools.py              ← ToolRegistry + FunctionDeclarations
src/api/                    ← FastAPI backend
frontend/                   ← React + Vite + TypeScript + Tailwind v4 + ForceGraph2D
```

## Config (`config.yaml`) — CRITICAL

Training params under `model.training`, NOT top-level.

```yaml
model:
  gnn: { hidden_channels: 128, num_heads: 4, num_layers: 3, dropout: 0.3, jk_mode: cat, residual: true }
  training: { epochs: 200, lr: 0.001, patience: 20, batch_size: 2048, num_neighbors: [15, 10, 5] }
  temporal: { memory_dim: 128, bptt_steps: 5 }
  xgboost: { n_estimators: 500, max_depth: 8, learning_rate: 0.05, eval_metric: aucpr }
agent:
  llm: { model_name: gemini-2.0-flash, temperature: 0.3 }
  rag: { top_k: 5, embedding_model: models/text-embedding-004 }
```

## Known Issues

| Issue | Fix |
|-------|-----|
| FAISS `search()` aborts on macOS ARM | `np.ascontiguousarray()` + mock in tests |
| numpy `bool_` deprecation in temporal masks | Wrap with `bool()` |
| `ReduceLROnPlateau(verbose=False)` removed PyTorch 2.8 | Remove `verbose` kwarg |
| pyg-lib required for NeighborLoader | `pip3 install pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html` |

## Tests

Run: `python3 -m pytest tests/ -q` — target ~150 tests total

```
tests/test_config.py, test_graph/, test_models/, test_pipeline/,
test_agents/ (cluster, typology ~12, attention ~8, forensic ~10), test_api/
```

## How to Run

```bash
python3 scripts/train_best_pipeline.py   # train full GNN+XGBoost pipeline
python3 -m src.api.main                  # API server
cd frontend && npm run dev               # React frontend
GEMINI_API_KEY=... docker compose up     # production
```

## API Routes (`/api/v1`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, model_loaded |
| GET | `/graph/stats` | Node/edge counts, illicit rate |
| POST | `/graph/neighbors` | k-hop BFS neighbors |
| POST | `/predictions/predict` | Per-node illicit probabilities |
| POST | `/predictions/top-k` | Highest-risk nodes |
| POST | `/forensic/clusters` | Detect suspicious clusters |
| POST | `/forensic/typologies` | Algorithmic AML pattern detection |
| POST | `/forensic/attention-trace` | GATv2 evidence path for a node |
| POST | `/forensic/investigate` | Full Gemini ReAct report |
| GET | `/dashboard/stats` | Aggregated overview |
