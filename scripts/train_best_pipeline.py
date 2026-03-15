"""Train and save the best Aurelius pipeline: Temporal GNN + XGBoost Hybrid.

Best benchmark results (F1=0.8706, AUPRC=0.8883 on held-out eval set):
  - Temporal GNN: AureliusGAT encoder + per-node GRU memory across 49 timesteps
  - XGBoost: trained on 677-dim features [165 orig || 384 GNN emb || 128 GRU mem]
  - Optimal threshold: 0.95 (tuned on calibration, validated on held-out eval)

Saved artifacts:
  data/processed/checkpoints/best_temporal_gnn.pt    (saved by TemporalTrainer)
  data/processed/checkpoints/best_xgboost.json       (XGBoost model)
  data/processed/checkpoints/pipeline_config.json    (threshold + metadata)

Usage:
    python3 scripts/train_best_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml_config
from src.models.classifier import HybridClassifier
from src.pipeline.evaluate import find_optimal_threshold
from src.pipeline.temporal_train import TemporalTrainer

SEED = 42
CHECKPOINT_DIR = Path("data/processed/checkpoints")
XGB_SAVE_PATH = CHECKPOINT_DIR / "best_xgboost.json"
PIPELINE_CONFIG_PATH = CHECKPOINT_DIR / "pipeline_config.json"


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def extract_temporal_features(model, tt: TemporalTrainer) -> tuple[np.ndarray, np.ndarray]:
    """Run all snapshots through the trained temporal GNN and collect hybrid features.

    Returns:
        X_all: [num_nodes, 677] float array (165 orig + 384 GNN + 128 mem)
        labels: [num_nodes] int array
    """
    data = tt.data
    device = tt.device
    num_nodes = data.num_nodes
    emb_dim = model.encoder.jk_out_channels

    node_embeddings = torch.zeros(num_nodes, emb_dim)
    model.eval()
    model.memory.reset()

    logger.info("Extracting temporal embeddings from all snapshots...")
    with torch.no_grad():
        for i in range(len(tt.snapshots)):
            snap = tt.snapshots[i].to(device)
            gids = tt.global_ids_list[i].to(device)
            _, embeddings = model.encoder(snap.x, snap.edge_index, return_embeddings=True)
            model.memory.update(embeddings, gids)
            node_embeddings[gids.cpu()] = embeddings.cpu()

    node_memory = model.memory.memory[:num_nodes].cpu()

    features_np = data.x.cpu().numpy()
    embeddings_np = node_embeddings.numpy()
    memory_np = node_memory.numpy()
    X_all = np.concatenate([features_np, embeddings_np, memory_np], axis=1)
    labels_np = data.y.cpu().numpy()

    logger.info(
        f"Hybrid features: {X_all.shape[1]} dims "
        f"({features_np.shape[1]} orig + {embeddings_np.shape[1]} GNN + {memory_np.shape[1]} mem)"
    )
    return X_all, labels_np


def main() -> None:
    set_seed()
    cfg = load_yaml_config()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Train Temporal GNN ----
    logger.info("=" * 60)
    logger.info("Step 1: Training Temporal GNN")
    logger.info("=" * 60)
    t0 = time.time()

    tt = TemporalTrainer(cfg, device="cpu")
    tt.setup_data("elliptic")
    tt.setup_model()
    history = tt.train_temporal()

    gnn_time = time.time() - t0
    best_epoch = int(np.argmax(history["val_auprc"])) + 1
    best_gnn_auprc = float(max(history["val_auprc"]))
    logger.info(f"GNN training done in {gnn_time:.0f}s — best epoch {best_epoch}, AUPRC={best_gnn_auprc:.4f}")
    logger.info(f"Checkpoint saved to {CHECKPOINT_DIR / 'best_temporal_gnn.pt'}")

    model = tt.model
    data = tt.data

    # ---- Step 2: Extract temporal embeddings ----
    X_all, labels_np = extract_temporal_features(model, tt)

    # ---- Step 3: Build train/val splits ----
    train_mask = (data.train_mask & (data.y != -1)).numpy()
    val_mask = (data.val_mask & (data.y != -1)).numpy()

    val_indices = np.where(val_mask)[0]
    np.random.seed(SEED)
    np.random.shuffle(val_indices)
    half = len(val_indices) // 2
    cal_indices = val_indices[:half]
    eval_indices = val_indices[half:]

    X_train, y_train = X_all[train_mask], labels_np[train_mask]
    X_cal, y_cal = X_all[cal_indices], labels_np[cal_indices]
    X_eval, y_eval = X_all[eval_indices], labels_np[eval_indices]

    logger.info(
        f"Train: {len(y_train)} | Cal: {len(y_cal)} (illicit: {int(y_cal.sum())}) | "
        f"Eval: {len(y_eval)} (illicit: {int(y_eval.sum())})"
    )

    # ---- Step 4: Train XGBoost ----
    logger.info("=" * 60)
    logger.info("Step 2: Training XGBoost on temporal embeddings")
    logger.info("=" * 60)
    t1 = time.time()

    xgb_cfg = cfg["model"]["xgboost"]
    hybrid_clf = HybridClassifier(xgb_cfg)
    hybrid_clf.fit(X_train, y_train, X_cal, y_cal)
    xgb_time = time.time() - t1

    # ---- Step 5: Threshold calibration on cal set ----
    cal_proba = hybrid_clf.predict_proba(X_cal)[:, 1]
    best_threshold, cal_f1 = find_optimal_threshold(y_cal, cal_proba)
    cal_auprc = average_precision_score(y_cal, cal_proba)
    logger.info(f"Calibration: AUPRC={cal_auprc:.4f}, threshold={best_threshold:.2f}, F1={cal_f1:.4f}")

    # ---- Step 6: Honest eval on held-out set ----
    from sklearn.metrics import f1_score, precision_score, recall_score
    eval_proba = hybrid_clf.predict_proba(X_eval)[:, 1]
    eval_auprc = average_precision_score(y_eval, eval_proba)
    eval_preds = (eval_proba >= best_threshold).astype(int)
    eval_f1 = f1_score(y_eval, eval_preds, zero_division=0)
    eval_precision = precision_score(y_eval, eval_preds, zero_division=0)
    eval_recall = recall_score(y_eval, eval_preds, zero_division=0)
    logger.info(
        f"Eval (held-out): AUPRC={eval_auprc:.4f}, "
        f"F1={eval_f1:.4f}, P={eval_precision:.4f}, R={eval_recall:.4f}"
    )

    # ---- Step 7: Save XGBoost + pipeline config ----
    hybrid_clf.save(str(XGB_SAVE_PATH))

    pipeline_cfg = {
        "pipeline": "temporal_xgboost",
        "optimal_threshold": best_threshold,
        "feature_dims": int(X_all.shape[1]),
        "gnn_checkpoint": "best_temporal_gnn.pt",
        "xgb_checkpoint": "best_xgboost.json",
        "eval_auprc": round(eval_auprc, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "best_gnn_epoch": best_epoch,
        "gnn_train_time_s": round(gnn_time, 1),
        "xgb_train_time_s": round(xgb_time, 1),
    }
    PIPELINE_CONFIG_PATH.write_text(json.dumps(pipeline_cfg, indent=2))
    logger.info(f"Pipeline config saved to {PIPELINE_CONFIG_PATH}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  GNN checkpoint: {CHECKPOINT_DIR / 'best_temporal_gnn.pt'}")
    logger.info(f"  XGBoost model:  {XGB_SAVE_PATH}")
    logger.info(f"  Pipeline cfg:   {PIPELINE_CONFIG_PATH}")
    logger.info(f"  Eval AUPRC:     {eval_auprc:.4f}")
    logger.info(f"  Eval F1:        {eval_f1:.4f}  (threshold={best_threshold:.2f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
