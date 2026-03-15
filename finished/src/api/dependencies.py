"""
Shared application state loaded once at startup and injected via FastAPI deps.

Usage in a route:
    async def my_route(state: AppState = Depends(get_app_state)):
        ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from src.config import load_yaml_config
from src.models.classifier import HybridClassifier
from src.models.gnn_model import AureliusGAT
from src.models.temporal_gnn import TemporalAureliusGAT


# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    graph: Optional[Data] = None
    model: Optional[AureliusGAT | TemporalAureliusGAT] = None
    hybrid_clf: Optional[HybridClassifier] = None
    optimal_threshold: float = 0.5
    predictions: Optional[torch.Tensor] = None   # float probs, shape (N,)
    dataset_name: str = "elliptic"
    val_auprc: Optional[float] = None
    config: dict = field(default_factory=dict)
    # Temporal snapshot data — needed for inference
    _snapshots: list = field(default_factory=list)
    _global_ids_list: list = field(default_factory=list)


# Module-level singleton
_state: AppState = AppState()


def get_app_state() -> AppState:
    """FastAPI dependency — returns the singleton app state."""
    return _state


# ---------------------------------------------------------------------------
# Loader called from app lifespan
# ---------------------------------------------------------------------------

def load_graph(dataset: str = "elliptic") -> Data:
    """Load pre-built PyG graph from disk."""
    processed_dir = Path("data/processed")
    path_map = {
        "elliptic": processed_dir / "elliptic_graph.pt",
        "ibm_aml": processed_dir / "ibm_aml_graph.pt",
    }
    pt_path = path_map.get(dataset)
    if pt_path is None or not pt_path.exists():
        raise FileNotFoundError(f"Graph file not found for dataset '{dataset}': {pt_path}")
    graph = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    logger.info(f"Loaded {dataset} graph — {graph.num_nodes} nodes, {graph.num_edges} edges")
    return graph


def load_model(graph: Data, config: dict) -> tuple[AureliusGAT, Optional[float]]:
    """Load base GNN model from best_gnn.pt checkpoint (fallback)."""
    model_cfg = config.get("model", {})
    gnn_cfg = model_cfg.get("gnn", model_cfg)  # handle both nested and flat
    in_channels = graph.x.shape[1]
    model = AureliusGAT(
        in_channels=in_channels,
        hidden_channels=gnn_cfg.get("hidden_channels", 128),
        out_channels=2,
        num_heads=gnn_cfg.get("num_heads", 4),
        num_layers=gnn_cfg.get("num_layers", 3),
        dropout=gnn_cfg.get("dropout", 0.3),
        jk_mode=gnn_cfg.get("jk_mode", "cat"),
    )

    ckpt_dir = Path("data/processed/checkpoints")
    ckpt_path = ckpt_dir / "best_gnn.pt"
    val_auprc: Optional[float] = None

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        val_auprc = ckpt.get("val_auprc")
        logger.info(
            f"Loaded GNN checkpoint from epoch {ckpt.get('epoch')} — val AUPRC={val_auprc:.4f}"
            if val_auprc else "Loaded GNN checkpoint (no val AUPRC recorded)"
        )
    else:
        logger.warning(f"No checkpoint at {ckpt_path}. Model uses random weights.")

    model.eval()
    return model, val_auprc


def load_temporal_pipeline(
    graph: Data,
    config: dict,
) -> tuple[Optional[TemporalAureliusGAT], Optional[HybridClassifier], float, Optional[float]]:
    """Load the Temporal GNN + XGBoost hybrid pipeline (best performance).

    Returns:
        (temporal_model, hybrid_clf, optimal_threshold, eval_auprc)
        Returns (None, None, 0.5, None) if checkpoints not found.
    """
    ckpt_dir = Path("data/processed/checkpoints")
    temporal_ckpt = ckpt_dir / "best_temporal_gnn.pt"
    xgb_ckpt = ckpt_dir / "best_xgboost.json"
    pipeline_cfg_path = ckpt_dir / "pipeline_config.json"

    if not temporal_ckpt.exists() or not xgb_ckpt.exists():
        missing = []
        if not temporal_ckpt.exists():
            missing.append("best_temporal_gnn.pt")
        if not xgb_ckpt.exists():
            missing.append("best_xgboost.json")
        logger.warning(
            f"Temporal+XGBoost pipeline not available — missing: {missing}. "
            "Run scripts/train_best_pipeline.py to train it."
        )
        return None, None, 0.5, None

    # Load pipeline config (threshold + metadata)
    optimal_threshold = 0.95  # default from benchmark
    eval_auprc: Optional[float] = None
    if pipeline_cfg_path.exists():
        with open(pipeline_cfg_path) as f:
            pcfg = json.load(f)
        optimal_threshold = pcfg.get("optimal_threshold", 0.95)
        eval_auprc = pcfg.get("eval_auprc")
        logger.info(
            f"Pipeline config: threshold={optimal_threshold:.2f}, "
            f"eval_auprc={eval_auprc}, eval_f1={pcfg.get('eval_f1')}"
        )

    # Load Temporal GNN
    model_cfg = config.get("model", {})
    gnn_cfg = model_cfg.get("gnn", model_cfg)
    in_channels = graph.x.shape[1]

    base_encoder = AureliusGAT(
        in_channels=in_channels,
        hidden_channels=gnn_cfg.get("hidden_channels", 128),
        out_channels=2,
        num_heads=gnn_cfg.get("num_heads", 4),
        num_layers=gnn_cfg.get("num_layers", 3),
        dropout=gnn_cfg.get("dropout", 0.3),
        jk_mode=gnn_cfg.get("jk_mode", "cat"),
    )
    temporal_model = TemporalAureliusGAT(encoder=base_encoder)

    ckpt = torch.load(str(temporal_ckpt), map_location="cpu", weights_only=False)
    temporal_model.load_state_dict(ckpt["model_state_dict"])
    temporal_model.eval()
    logger.info(
        f"Loaded temporal GNN checkpoint from epoch {ckpt.get('epoch')} — "
        f"val AUPRC={ckpt.get('val_auprc', 'N/A')}"
    )

    # Load XGBoost
    hybrid_clf = HybridClassifier.load(str(xgb_ckpt))

    return temporal_model, hybrid_clf, optimal_threshold, eval_auprc


def _build_snapshots(graph: Data) -> tuple[list, list]:
    """Split full graph into per-timestep snapshot Data objects.

    Replicates TemporalTrainer.setup_data() snapshot logic without needing config.

    Returns:
        (snapshots, global_ids_list)
    """
    from torch_geometric.data import Data as PyGData

    timesteps = graph.timestep
    unique_ts = sorted(timesteps.unique().tolist())
    snapshots = []
    global_ids_list = []

    for t in unique_ts:
        mask = timesteps == t
        local_ids = mask.nonzero(as_tuple=True)[0]

        remap = torch.full((graph.num_nodes,), -1, dtype=torch.long)
        remap[local_ids] = torch.arange(local_ids.size(0))

        src, dst = graph.edge_index
        edge_mask = mask[src] & mask[dst]
        local_edge_index = remap[graph.edge_index[:, edge_mask]]

        snap = PyGData(
            x=graph.x[mask],
            edge_index=local_edge_index,
            y=graph.y[mask],
        )
        snapshots.append(snap)
        global_ids_list.append(local_ids)

    return snapshots, global_ids_list


def _extract_temporal_features(
    graph: Data,
    temporal_model: TemporalAureliusGAT,
) -> np.ndarray:
    """Extract [165 orig || 384 GNN || 128 mem] = 677-dim features for all nodes.

    Replays all snapshots in timestep order to build up GRU memory state.
    """
    device = torch.device("cpu")
    num_nodes = graph.num_nodes
    emb_dim = temporal_model.encoder.jk_out_channels

    snapshots, global_ids_list = _build_snapshots(graph)

    node_embeddings = torch.zeros(num_nodes, emb_dim)
    temporal_model.eval()
    temporal_model.memory.reset()

    with torch.no_grad():
        for i in range(len(snapshots)):
            snap = snapshots[i].to(device)
            gids = global_ids_list[i].to(device)
            _, embeddings = temporal_model.encoder(snap.x, snap.edge_index, return_embeddings=True)
            temporal_model.memory.update(embeddings, gids)
            node_embeddings[gids.cpu()] = embeddings.cpu()

    node_memory = temporal_model.memory.memory[:num_nodes].cpu()

    features_np = graph.x.cpu().numpy()
    embeddings_np = node_embeddings.numpy()
    memory_np = node_memory.numpy()
    return np.concatenate([features_np, embeddings_np, memory_np], axis=1)


@torch.no_grad()
def run_inference(
    graph: Data,
    model: AureliusGAT | TemporalAureliusGAT,
    hybrid_clf: Optional[HybridClassifier] = None,
) -> torch.Tensor:
    """Run inference and return per-node illicit probabilities.

    Uses Temporal+XGBoost pipeline if hybrid_clf is provided (F1=0.87, AUPRC=0.89).
    Falls back to plain GNN inference otherwise (F1=0.59, AUPRC=0.87).
    """
    if hybrid_clf is not None and isinstance(model, TemporalAureliusGAT):
        logger.info("Running Temporal+XGBoost inference (best pipeline)")
        X_all = _extract_temporal_features(graph, model)
        proba = hybrid_clf.predict_proba(X_all)[:, 1]
        return torch.tensor(proba, dtype=torch.float32)

    # Fallback: plain GNN forward pass
    logger.info("Running GNN-only inference (fallback)")
    logits = model(graph.x, graph.edge_index)  # (N, 2)
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return probs


def initialize_state(dataset: str = "elliptic") -> None:
    """Called once at app startup to populate global _state."""
    global _state
    cfg = load_yaml_config()
    _state.config = cfg
    _state.dataset_name = dataset

    try:
        graph = load_graph(dataset)
        _state.graph = graph
    except FileNotFoundError as e:
        logger.error(f"Could not load graph: {e}")
        return

    # Try best pipeline first (Temporal + XGBoost)
    temporal_model, hybrid_clf, optimal_threshold, eval_auprc = load_temporal_pipeline(
        graph, cfg
    )

    if temporal_model is not None and hybrid_clf is not None:
        _state.model = temporal_model
        _state.hybrid_clf = hybrid_clf
        _state.optimal_threshold = optimal_threshold
        _state.val_auprc = eval_auprc
        logger.info(
            f"Using Temporal+XGBoost pipeline "
            f"(AUPRC={eval_auprc}, threshold={optimal_threshold:.2f})"
        )
    else:
        # Fallback to base GNN
        try:
            model, val_auprc = load_model(graph, cfg)
            _state.model = model
            _state.val_auprc = val_auprc
            _state.optimal_threshold = 0.5
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            return

    try:
        _state.predictions = run_inference(graph, _state.model, _state.hybrid_clf)
        threshold = _state.optimal_threshold
        logger.info(
            f"Inference done — {int((_state.predictions > threshold).sum())} nodes "
            f"flagged (p>{threshold:.2f})"
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
