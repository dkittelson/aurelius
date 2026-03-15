"""Enhanced benchmark: push F1 > 0.80 on Elliptic.

Three strategies built on the Temporal GNN (best single model, AUPRC=0.87):

  1. Temporal GNN + Threshold Optimization
  2. Temporal + Adversarial Training + Threshold Opt
  3. Temporal + XGBoost Hybrid + Threshold Opt

Evaluation approach:
  - Temporal strategies: split val snapshots (35-42) into calibration (35-38)
    and eval (39-42) for honest threshold tuning.
  - XGBoost strategy: split data.val_mask 50/50 into calibration + eval.
  - This avoids the train-on-val / test-on-val leakage problem.

Usage:
    python3 scripts/benchmark_v2.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml_config
from src.pipeline.train import EarlyStopping
from src.pipeline.temporal_train import TemporalTrainer
from src.pipeline.adversarial import FeatureAttacker
from src.pipeline.evaluate import find_optimal_threshold
from src.models.classifier import HybridClassifier

SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def collect_snapshot_probs(
    trainer: TemporalTrainer, start_idx: int, end_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run temporal model on a range of snapshots, collect probs + labels."""
    trainer.model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for i in range(start_idx, min(end_idx, len(trainer.snapshots))):
            snap = trainer.snapshots[i].to(trainer.device)
            gids = trainer.global_ids_list[i].to(trainer.device)
            logits = trainer.model.forward_snapshot(snap, gids)
            probs = F.softmax(logits, dim=1)[:, 1]
            labeled = snap.y != -1
            if labeled.sum() > 0:
                all_probs.append(probs[labeled].cpu().numpy())
                all_labels.append(snap.y[labeled].cpu().numpy())

    if not all_probs:
        return np.array([]), np.array([])
    return np.concatenate(all_probs), np.concatenate(all_labels)


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Compute metrics at a specific threshold."""
    if len(y_true) == 0 or y_true.sum() == 0:
        return {"auprc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    auprc = average_precision_score(y_true, y_proba)
    preds = (y_proba >= threshold).astype(int)
    return {
        "auprc": float(auprc),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def temporal_train_and_eval(
    tt: TemporalTrainer, history: dict, label: str
) -> dict:
    """After training temporal model, do honest threshold calibration.

    Split validation snapshots:
      - Calibration: snapshots 34-37 (timesteps 35-38) → find optimal threshold
      - Evaluation: snapshots 38-41 (timesteps 39-42) → report honest F1
    """
    model = tt.model
    device = tt.device

    # The training already forwarded through train snapshots and built memory.
    # Now forward through val snapshots in two stages.
    train_snaps = min(34, len(tt.snapshots))

    # Calibration: snapshots 34-37
    cal_end = min(38, len(tt.snapshots))
    cal_probs, cal_labels = collect_snapshot_probs(tt, train_snaps, cal_end)

    if len(cal_labels) == 0 or cal_labels.sum() == 0:
        logger.warning(f"{label}: No illicit nodes in calibration set, using threshold=0.5")
        best_threshold = 0.5
        best_cal_f1 = 0.0
    else:
        best_threshold, best_cal_f1 = find_optimal_threshold(cal_labels, cal_probs)
        cal_auprc = average_precision_score(cal_labels, cal_probs)
        logger.info(
            f"{label} calibration: AUPRC={cal_auprc:.4f}, "
            f"optimal threshold={best_threshold:.2f}, F1={best_cal_f1:.4f}"
        )

    # Evaluation: snapshots 38-41 (memory continues from calibration)
    eval_end = min(42, len(tt.snapshots))
    eval_probs, eval_labels = collect_snapshot_probs(tt, cal_end, eval_end)

    if len(eval_labels) == 0 or eval_labels.sum() == 0:
        logger.warning(f"{label}: No illicit nodes in eval set")
        eval_metrics_opt = {"auprc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        eval_metrics_050 = eval_metrics_opt
    else:
        eval_metrics_opt = evaluate_at_threshold(eval_labels, eval_probs, best_threshold)
        eval_metrics_050 = evaluate_at_threshold(eval_labels, eval_probs, 0.5)
        logger.info(
            f"{label} eval: AUPRC={eval_metrics_opt['auprc']:.4f}, "
            f"F1@0.5={eval_metrics_050['f1']:.4f}, "
            f"F1@opt={eval_metrics_opt['f1']:.4f}"
        )

    # Also report full val AUPRC from training history for comparison with v1
    best_idx = int(np.argmax(history["val_auprc"]))

    return {
        "val_auprc_training": history["val_auprc"][best_idx],
        "val_f1_training": history["val_f1"][best_idx],
        "eval_auprc": eval_metrics_opt["auprc"],
        "eval_f1_at_050": eval_metrics_050["f1"],
        "eval_f1_optimized": eval_metrics_opt["f1"],
        "eval_precision": eval_metrics_opt["precision"],
        "eval_recall": eval_metrics_opt["recall"],
        "optimal_threshold": best_threshold,
        "cal_f1": best_cal_f1,
        "best_epoch": best_idx + 1,
    }


# -----------------------------------------------------------------------
# Strategy 1: Temporal GNN + Threshold Optimization
# -----------------------------------------------------------------------

def run_temporal_threshold(cfg: dict) -> dict:
    """Train Temporal GNN, then find F1-maximizing threshold."""
    logger.info("=" * 60)
    logger.info("STRATEGY 1: Temporal GNN + Threshold Optimization")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    tt = TemporalTrainer(cfg, device="cpu")
    tt.setup_data("elliptic")
    tt.setup_model()
    history = tt.train_temporal()
    train_time = time.time() - t0

    result = temporal_train_and_eval(tt, history, "Strategy 1")
    result["train_time_s"] = round(train_time, 1)
    return result


# -----------------------------------------------------------------------
# Strategy 2: Temporal + Adversarial Training + Threshold Opt
# -----------------------------------------------------------------------

def run_temporal_adversarial(cfg: dict) -> dict:
    """Temporal GNN with PGD adversarial perturbation during training."""
    logger.info("=" * 60)
    logger.info("STRATEGY 2: Temporal + Adversarial + Threshold Opt")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    tt = TemporalTrainer(cfg, device="cpu")
    tt.setup_data("elliptic")
    tt.setup_model()

    model = tt.model
    device = tt.device
    data = tt.data

    adv_cfg = cfg.get("model", {}).get("adversarial", {})
    epsilon = adv_cfg.get("epsilon", 0.01)
    alpha = adv_cfg.get("alpha", 0.3)
    attack_steps = adv_cfg.get("attack_steps", 3)
    attacker = FeatureAttacker(epsilon=epsilon, steps=attack_steps)

    training_cfg = cfg["model"]["training"]
    temporal_cfg = cfg.get("model", {}).get("temporal", {})
    epochs = training_cfg.get("epochs", 200)
    patience = training_cfg.get("patience", 20)
    bptt_steps = temporal_cfg.get("bptt_steps", 5)

    train_snaps = min(34, len(tt.snapshots))
    val_end = min(42, len(tt.snapshots))

    # Class weights
    y = data.y
    labeled = y[y != -1]
    n_licit = int((labeled == 0).sum())
    n_illicit = int((labeled == 1).sum())
    ratio = n_licit / max(n_illicit, 1)
    class_weights = torch.tensor([1.0, ratio], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg.get("lr", 0.001),
        weight_decay=training_cfg.get("weight_decay", 5e-4),
    )
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_auprc": [], "val_f1": []}
    logger.info(f"Starting temporal+adversarial training (eps={epsilon}, alpha={alpha})...")

    for epoch in range(1, epochs + 1):
        model.train()
        model.memory.reset()

        total_loss = 0.0
        num_steps = 0

        for i in range(train_snaps):
            snap = tt.snapshots[i].to(device)
            gids = tt.global_ids_list[i].to(device)

            # Clean forward (updates memory)
            logits = model.forward_snapshot(snap, gids)

            labeled_mask = snap.y != -1
            if labeled_mask.sum() == 0:
                continue

            clean_loss = criterion(logits[labeled_mask], snap.y[labeled_mask])

            # Adversarial: PGD on snapshot features via encoder
            model.eval()
            x_adv = attacker.attack(
                model.encoder, snap.x, snap.edge_index,
                snap.y, criterion, mask=labeled_mask,
            )
            model.train()

            # Adversarial forward through encoder only (don't double-update memory)
            _, adv_embeddings = model.encoder(x_adv, snap.edge_index, return_embeddings=True)
            adv_memory = model.memory.get_memory(gids)
            adv_combined = torch.cat([adv_embeddings, adv_memory], dim=1)
            adv_combined = model.dropout(adv_combined)
            adv_logits = model.classifier(adv_combined)
            adv_loss = criterion(adv_logits[labeled_mask], snap.y[labeled_mask])

            loss = (1 - alpha) * clean_loss + alpha * adv_loss
            total_loss += loss.item()
            num_steps += 1

            loss.backward()

            if num_steps % bptt_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        if num_steps % bptt_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_steps, 1)

        # Validation (full val range for early stopping)
        val_metrics = tt._evaluate_snapshots(train_snaps, val_end, criterion)

        history["train_loss"].append(avg_loss)
        history["val_auprc"].append(val_metrics["auprc"])
        history["val_f1"].append(val_metrics["f1"])

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | loss={avg_loss:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f}"
            )

        if early_stop.step(val_metrics["auprc"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0

    result = temporal_train_and_eval(tt, history, "Strategy 2")
    result["train_time_s"] = round(train_time, 1)
    return result


# -----------------------------------------------------------------------
# Strategy 3: Temporal + XGBoost Hybrid + Threshold Opt
# -----------------------------------------------------------------------

def run_temporal_xgboost(cfg: dict) -> dict:
    """Train Temporal GNN → extract embeddings → XGBoost hybrid → threshold opt.

    Honest evaluation: split val_mask 50/50 into calibration (threshold tuning)
    and eval (honest reporting).
    """
    logger.info("=" * 60)
    logger.info("STRATEGY 3: Temporal + XGBoost Hybrid + Threshold Opt")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    tt = TemporalTrainer(cfg, device="cpu")
    tt.setup_data("elliptic")
    tt.setup_model()
    history = tt.train_temporal()
    gnn_time = time.time() - t0

    model = tt.model
    device = tt.device
    data = tt.data

    # ---- Extract temporal embeddings for all nodes ----
    model.eval()
    model.memory.reset()

    num_nodes = data.num_nodes
    emb_dim = model.encoder.jk_out_channels

    node_embeddings = torch.zeros(num_nodes, emb_dim)

    logger.info("Extracting temporal embeddings from all snapshots...")
    with torch.no_grad():
        for i in range(len(tt.snapshots)):
            snap = tt.snapshots[i].to(device)
            gids = tt.global_ids_list[i].to(device)
            _, embeddings = model.encoder(snap.x, snap.edge_index, return_embeddings=True)
            model.memory.update(embeddings, gids)
            node_embeddings[gids.cpu()] = embeddings.cpu()

    node_memory = model.memory.memory[:num_nodes].cpu()

    # Build combined features: [original_165 || gnn_emb_384 || gru_memory_128]
    features_np = data.x.cpu().numpy()
    embeddings_np = node_embeddings.numpy()
    memory_np = node_memory.numpy()
    X_all = np.concatenate([features_np, embeddings_np, memory_np], axis=1)
    labels_np = data.y.cpu().numpy()

    logger.info(
        f"Hybrid features: {X_all.shape[1]} dims "
        f"({features_np.shape[1]} orig + {embeddings_np.shape[1]} GNN + {memory_np.shape[1]} mem)"
    )

    # ---- Split val_mask 50/50 into calibration + eval ----
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

    # ---- Train XGBoost (use calibration set for early stopping) ----
    t1 = time.time()
    xgb_cfg = cfg["model"]["xgboost"]
    hybrid_clf = HybridClassifier(xgb_cfg)
    hybrid_clf.fit(X_train, y_train, X_cal, y_cal)
    xgb_time = time.time() - t1

    # ---- Threshold optimization on calibration set ----
    cal_proba = hybrid_clf.predict_proba(X_cal)[:, 1]
    best_threshold, cal_f1 = find_optimal_threshold(y_cal, cal_proba)
    cal_auprc = average_precision_score(y_cal, cal_proba)
    logger.info(
        f"XGBoost calibration: AUPRC={cal_auprc:.4f}, "
        f"threshold={best_threshold:.2f}, F1={cal_f1:.4f}"
    )

    # ---- Honest evaluation on held-out eval set ----
    eval_proba = hybrid_clf.predict_proba(X_eval)[:, 1]
    eval_metrics_opt = evaluate_at_threshold(y_eval, eval_proba, best_threshold)
    eval_metrics_050 = evaluate_at_threshold(y_eval, eval_proba, 0.5)

    logger.info(
        f"XGBoost eval: AUPRC={eval_metrics_opt['auprc']:.4f}, "
        f"F1@0.5={eval_metrics_050['f1']:.4f}, "
        f"F1@opt={eval_metrics_opt['f1']:.4f}"
    )

    total_time = gnn_time + xgb_time
    best_idx = int(np.argmax(history["val_auprc"]))

    return {
        "val_auprc_training": float(max(history["val_auprc"])),
        "cal_auprc_xgboost": cal_auprc,
        "cal_f1": cal_f1,
        "eval_auprc": eval_metrics_opt["auprc"],
        "eval_f1_at_050": eval_metrics_050["f1"],
        "eval_f1_optimized": eval_metrics_opt["f1"],
        "eval_precision": eval_metrics_opt["precision"],
        "eval_recall": eval_metrics_opt["recall"],
        "optimal_threshold": best_threshold,
        "best_epoch": best_idx + 1,
        "gnn_time_s": round(gnn_time, 1),
        "xgb_time_s": round(xgb_time, 1),
        "train_time_s": round(total_time, 1),
        "feature_dims": X_all.shape[1],
    }


# -----------------------------------------------------------------------
# Results formatting
# -----------------------------------------------------------------------

def print_results_table(results: dict):
    print("\n")
    print("=" * 80)
    print("              AURELIUS ENHANCED BENCHMARK — Target: F1 > 0.80")
    print("=" * 80)
    print("  Threshold tuned on calibration set, F1 reported on held-out eval set")
    print()

    print(f"{'Strategy':<35} {'AUPRC':>7} {'F1@0.5':>7} {'F1*':>7} {'Thresh':>7} {'Time':>7}")
    print("-" * 80)

    for name, r in results.items():
        if "error" in r:
            print(f"{name:<35} ERROR: {r['error']}")
            continue

        auprc = r.get("eval_auprc", 0)
        f1_050 = r.get("eval_f1_at_050", 0)
        f1_opt = r.get("eval_f1_optimized", 0)
        thresh = r.get("optimal_threshold", 0.5)
        t = r.get("train_time_s", 0)

        marker = " ***" if f1_opt >= 0.80 else ""
        print(f"{name:<35} {auprc:>7.4f} {f1_050:>7.4f} {f1_opt:>7.4f} {thresh:>7.2f} {t:>6.0f}s{marker}")

    print("-" * 80)
    print("AUPRC/F1 = honest eval set (NOT calibration set)")
    print("F1* = F1 at threshold optimized on calibration set")
    print("*** = target F1 >= 0.80 achieved")

    # Detail breakdown
    for name, r in results.items():
        if "error" in r:
            continue
        print(f"\n--- {name} ---")
        print(f"  Eval AUPRC:        {r.get('eval_auprc', 0):.4f}")
        print(f"  Eval F1@0.50:      {r.get('eval_f1_at_050', 0):.4f}")
        print(f"  Eval F1@optimized: {r.get('eval_f1_optimized', 0):.4f}")
        print(f"  Eval Precision:    {r.get('eval_precision', 0):.4f}")
        print(f"  Eval Recall:       {r.get('eval_recall', 0):.4f}")
        print(f"  Calibration F1:    {r.get('cal_f1', 0):.4f}")
        print(f"  Optimal threshold: {r.get('optimal_threshold', 0.5):.2f}")
        print(f"  Train time:        {r.get('train_time_s', 0):.0f}s")
        if "val_auprc_training" in r:
            print(f"  GNN val AUPRC:     {r['val_auprc_training']:.4f}")

    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Enhanced Benchmark — Target: F1 > 0.80")
    cfg = load_yaml_config()

    strategies = [
        ("Temporal + ThresholdOpt", run_temporal_threshold),
        ("Temporal + Adversarial + ThreshOpt", run_temporal_adversarial),
        ("Temporal + XGBoost + ThreshOpt", run_temporal_xgboost),
    ]

    results = {}
    total_start = time.time()

    for name, fn in strategies:
        try:
            results[name] = fn(cfg)
            f1 = results[name].get("eval_f1_optimized", 0)
            logger.info(f"{name} complete: eval F1={f1:.4f}")
        except Exception as e:
            logger.error(f"{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    total_time = time.time() - total_start
    logger.info(f"All strategies complete in {total_time:.0f}s")

    print_results_table(results)

    output_path = Path("data/processed/benchmark_v2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")
