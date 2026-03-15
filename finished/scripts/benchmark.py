"""Benchmark all 5 ML improvement phases against the baseline on Elliptic.

Experiments:
  0. Baseline GNN (3-layer GATv2, JK-cat)
  1. DGI Pre-trained + Fine-tune
  2. Temporal GNN (GRU memory across timesteps)
  3. Multi-Task Learning (classification + link prediction + degree)
  4. Adversarial Training (PGD feature attack)

Usage:
    python3 scripts/benchmark.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml_config
from src.pipeline.train import Trainer, EarlyStopping
from src.pipeline.pretrain import PreTrainer
from src.pipeline.temporal_train import TemporalTrainer
from src.pipeline.adversarial import AdversarialTrainer
from src.pipeline.evaluate import evaluate_model
from src.models.gnn_model import AureliusGAT
from src.models.multitask import (
    MultiTaskAureliusGAT,
    MultiTaskLoss,
    compute_link_prediction_loss,
    compute_degree_targets,
)

SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------
# Experiment 0: Baseline GNN
# -----------------------------------------------------------------------

def run_baseline(cfg: dict) -> dict:
    """Train vanilla GATv2 — the control experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 0: Baseline GNN")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    trainer = Trainer(cfg, device="cpu")
    trainer.setup_data("elliptic")
    trainer.setup_model()
    history = trainer.train_gnn()
    train_time = time.time() - t0

    best_idx = int(np.argmax(history["val_auprc"]))
    return {
        "best_val_auprc": history["val_auprc"][best_idx],
        "best_val_f1": history["val_f1"][best_idx],
        "best_epoch": best_idx + 1,
        "total_epochs": len(history["val_auprc"]),
        "train_time_s": round(train_time, 1),
    }


# -----------------------------------------------------------------------
# Experiment 1: DGI Pre-trained + Fine-tune
# -----------------------------------------------------------------------

def run_dgi_pretrained(cfg: dict) -> dict:
    """DGI self-supervised pre-training on all 203K nodes, then fine-tune."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: DGI Pre-trained + Fine-tune")
    logger.info("=" * 60)
    set_seed()

    ckpt_path = "data/processed/checkpoints/dgi_pretrained.pt"

    # Phase 1: Pre-train
    t0 = time.time()
    pt = PreTrainer(cfg, device="cpu")
    pt.setup_data("elliptic")
    pt.setup_encoder()
    pretrain_history = pt.pretrain_dgi(epochs=100)
    pt.save_pretrained(ckpt_path)
    pretrain_time = time.time() - t0

    # Phase 2: Fine-tune
    t1 = time.time()
    trainer = Trainer(cfg, device="cpu")
    trainer.setup_data("elliptic")
    trainer.setup_model()
    PreTrainer.load_pretrained_into_model(trainer.model, ckpt_path)
    history = trainer.train_gnn()
    finetune_time = time.time() - t1

    best_idx = int(np.argmax(history["val_auprc"]))
    return {
        "best_val_auprc": history["val_auprc"][best_idx],
        "best_val_f1": history["val_f1"][best_idx],
        "best_epoch": best_idx + 1,
        "total_epochs": len(history["val_auprc"]),
        "pretrain_time_s": round(pretrain_time, 1),
        "finetune_time_s": round(finetune_time, 1),
        "train_time_s": round(pretrain_time + finetune_time, 1),
        "pretrain_final_loss": pretrain_history["loss"][-1],
    }


# -----------------------------------------------------------------------
# Experiment 2: Temporal GNN
# -----------------------------------------------------------------------

def run_temporal(cfg: dict) -> dict:
    """Temporal GNN with per-node GRU memory across 49 timesteps."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Temporal GNN")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    tt = TemporalTrainer(cfg, device="cpu")
    tt.setup_data("elliptic")
    tt.setup_model()
    history = tt.train_temporal()
    train_time = time.time() - t0

    best_idx = int(np.argmax(history["val_auprc"]))
    return {
        "best_val_auprc": history["val_auprc"][best_idx],
        "best_val_f1": history["val_f1"][best_idx],
        "best_epoch": best_idx + 1,
        "total_epochs": len(history["val_auprc"]),
        "train_time_s": round(train_time, 1),
    }


# -----------------------------------------------------------------------
# Experiment 3: Multi-Task Learning
# -----------------------------------------------------------------------

def run_multitask(cfg: dict) -> dict:
    """Multi-task: classification + link prediction + degree regression."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Multi-Task Learning")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()

    # Reuse Trainer for data loading
    trainer = Trainer(cfg, device="cpu")
    trainer.setup_data("elliptic")
    trainer.setup_model()

    device = trainer.device
    data = trainer.data

    # Wrap encoder in multi-task model
    mt_model = MultiTaskAureliusGAT(
        encoder=trainer.model, out_channels=2
    ).to(device)

    tasks = ["classification", "link_prediction", "degree"]
    mt_loss_fn = MultiTaskLoss(tasks).to(device)

    class_weights = trainer.compute_class_weights()
    cls_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    all_params = list(mt_model.parameters()) + list(mt_loss_fn.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    training_cfg = cfg["model"]["training"]
    epochs = training_cfg["epochs"]
    patience = training_cfg["patience"]
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_auprc": [], "val_f1": []}

    # Pre-compute degree targets
    data_dev = data.to(device)
    degree_targets = compute_degree_targets(data_dev.edge_index, data_dev.num_nodes)

    logger.info(f"Starting multi-task training for up to {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        mt_model.train()
        mt_loss_fn.train()

        total_loss = 0.0
        num_batches = 0

        for batch in trainer.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = mt_model(
                batch.x, batch.edge_index,
                tasks=tasks,
            )

            # Classification loss (seed nodes only)
            seed_logits = outputs["classification"][:batch.batch_size]
            seed_y = batch.y[:batch.batch_size]
            labeled = seed_y != -1
            if labeled.sum() == 0:
                continue

            loss_cls = cls_criterion(seed_logits[labeled], seed_y[labeled])

            # Link prediction loss
            loss_lp = compute_link_prediction_loss(
                outputs["link_prediction"],
                batch.edge_index,
                batch.num_nodes,
            )

            # Degree loss
            batch_degree_targets = compute_degree_targets(
                batch.edge_index, batch.num_nodes
            )
            loss_deg = F.mse_loss(outputs["degree"], batch_degree_targets)

            # Combine with uncertainty weighting
            loss = mt_loss_fn({
                "classification": loss_cls,
                "link_prediction": loss_lp,
                "degree": loss_deg,
            })

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Validate (classification only, using the underlying encoder)
        # MultiTaskAureliusGAT's forward returns a dict, but evaluate_model
        # expects model(x, edge_index) → logits. Use the classification path.
        val_metrics = _evaluate_multitask_model(mt_model, data, data.val_mask, device)

        scheduler.step(val_metrics["auprc"])

        history["train_loss"].append(avg_loss)
        history["val_auprc"].append(val_metrics["auprc"])
        history["val_f1"].append(val_metrics["f1"])

        if epoch % 10 == 0 or epoch == 1:
            sigmas = {
                k: torch.exp(v).item()
                for k, v in mt_loss_fn.log_sigmas.items()
            }
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f} | "
                f"sigmas={sigmas}"
            )

        if early_stop.step(val_metrics["auprc"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    best_idx = int(np.argmax(history["val_auprc"]))

    # Log final sigma values
    final_sigmas = {
        k: round(torch.exp(v).item(), 4)
        for k, v in mt_loss_fn.log_sigmas.items()
    }

    return {
        "best_val_auprc": history["val_auprc"][best_idx],
        "best_val_f1": history["val_f1"][best_idx],
        "best_epoch": best_idx + 1,
        "total_epochs": len(history["val_auprc"]),
        "train_time_s": round(train_time, 1),
        "final_task_sigmas": final_sigmas,
    }


def _evaluate_multitask_model(mt_model, data, mask, device):
    """Evaluate MultiTaskAureliusGAT on classification task."""
    mt_model.eval()
    data_dev = data.to(device)
    mask_dev = mask.to(device)

    with torch.no_grad():
        outputs = mt_model(data_dev.x, data_dev.edge_index, tasks=["classification"])
        logits = outputs["classification"]
        probs = F.softmax(logits, dim=1)[:, 1]

    probs_sub = probs[mask_dev].cpu().numpy()
    y_sub = data_dev.y[mask_dev].cpu().numpy()
    known = y_sub != -1
    probs_sub = probs_sub[known]
    y_sub = y_sub[known]

    if len(y_sub) == 0 or y_sub.sum() == 0:
        return {"auprc": 0.0, "f1": 0.0}

    auprc = average_precision_score(y_sub, probs_sub)
    preds = (probs_sub >= 0.5).astype(int)
    f1 = f1_score(y_sub, preds, zero_division=0)
    return {"auprc": float(auprc), "f1": float(f1)}


# -----------------------------------------------------------------------
# Experiment 4: Adversarial Training
# -----------------------------------------------------------------------

def run_adversarial(cfg: dict) -> dict:
    """Adversarial training with PGD feature attacks."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Adversarial Training")
    logger.info("=" * 60)
    set_seed()

    t0 = time.time()
    trainer = Trainer(cfg, device="cpu")
    trainer.setup_data("elliptic")
    trainer.setup_model()

    device = trainer.device
    data = trainer.data
    model = trainer.model

    adv_cfg = cfg.get("model", {}).get("adversarial", {})
    epsilon = adv_cfg.get("epsilon", 0.01)
    alpha = adv_cfg.get("alpha", 0.3)
    attack_steps = adv_cfg.get("attack_steps", 3)

    adv_trainer = AdversarialTrainer(
        model, epsilon=epsilon, alpha=alpha, attack_steps=attack_steps
    )

    class_weights = trainer.compute_class_weights()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    training_cfg = cfg["model"]["training"]
    epochs = training_cfg["epochs"]
    patience = training_cfg["patience"]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_auprc": [], "val_f1": []}

    logger.info(
        f"Starting adversarial training (eps={epsilon}, alpha={alpha}) "
        f"for up to {epochs} epochs..."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in trainer.train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            seed_y = batch.y[:batch.batch_size]
            labeled = seed_y != -1
            if labeled.sum() == 0:
                continue

            # Build a mask for the full batch (only seed + labeled)
            full_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
            labeled_indices = torch.where(labeled)[0]
            full_mask[labeled_indices] = True

            total, clean, adv = adv_trainer.adversarial_step(
                batch.x, batch.edge_index, batch.y, criterion, mask=full_mask
            )

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        val_metrics = evaluate_model(model, data, data.val_mask, device)
        scheduler.step(val_metrics["auprc"])

        history["train_loss"].append(avg_loss)
        history["val_auprc"].append(val_metrics["auprc"])
        history["val_f1"].append(val_metrics["f1"])

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f}"
            )

        if early_stop.step(val_metrics["auprc"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    best_idx = int(np.argmax(history["val_auprc"]))

    # Robustness evaluation
    logger.info("Evaluating robustness under PGD attacks...")
    robustness = adv_trainer.evaluate_robustness(
        data, data.val_mask, device,
        epsilons=[0.0, 0.001, 0.005, 0.01, 0.05],
    )
    robustness_clean = {
        str(k): {"auprc": round(v.get("auprc", 0), 4), "f1": round(v.get("f1", 0), 4)}
        for k, v in robustness.items()
    }

    return {
        "best_val_auprc": history["val_auprc"][best_idx],
        "best_val_f1": history["val_f1"][best_idx],
        "best_epoch": best_idx + 1,
        "total_epochs": len(history["val_auprc"]),
        "train_time_s": round(train_time, 1),
        "robustness": robustness_clean,
    }


# -----------------------------------------------------------------------
# Results formatting
# -----------------------------------------------------------------------

def print_results_table(results: dict):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 72)
    print("                   AURELIUS BENCHMARK RESULTS")
    print("=" * 72)
    print(f"{'Variant':<20} {'AUPRC':>8} {'F1':>8} {'Best Ep':>9} {'Time (s)':>10}")
    print("-" * 72)

    for name, r in results.items():
        auprc = r.get("best_val_auprc", 0)
        f1 = r.get("best_val_f1", 0)
        ep = r.get("best_epoch", 0)
        t = r.get("train_time_s", 0)
        print(f"{name:<20} {auprc:>8.4f} {f1:>8.4f} {ep:>9d} {t:>10.1f}")

    print("=" * 72)

    # Baseline comparison
    baseline_auprc = results.get("Baseline", {}).get("best_val_auprc", 0)
    if baseline_auprc > 0:
        print(f"\nBaseline AUPRC: {baseline_auprc:.4f}")
        print("Improvements over baseline:")
        for name, r in results.items():
            if name == "Baseline":
                continue
            delta = r.get("best_val_auprc", 0) - baseline_auprc
            pct = (delta / baseline_auprc) * 100 if baseline_auprc > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {name:<18} {sign}{delta:.4f} ({sign}{pct:.1f}%)")

    # Adversarial robustness
    adv = results.get("Adversarial", {})
    if "robustness" in adv:
        print("\nAdversarial Robustness Curve:")
        for eps, metrics in adv["robustness"].items():
            print(f"  eps={eps}: AUPRC={metrics['auprc']:.4f}  F1={metrics['f1']:.4f}")

    # Multi-task sigmas
    mt = results.get("MultiTask", {})
    if "final_task_sigmas" in mt:
        print(f"\nMulti-Task Learned Sigmas: {mt['final_task_sigmas']}")

    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Aurelius Benchmark Suite")
    logger.info(f"Seed: {SEED}, Device: cpu")

    cfg = load_yaml_config()

    experiments = [
        ("Baseline", run_baseline),
        ("DGI+Finetune", run_dgi_pretrained),
        ("Temporal", run_temporal),
        ("MultiTask", run_multitask),
        ("Adversarial", run_adversarial),
    ]

    results = {}
    total_start = time.time()

    for name, fn in experiments:
        try:
            results[name] = fn(cfg)
            logger.info(f"{name} complete: AUPRC={results[name]['best_val_auprc']:.4f}")
        except Exception as e:
            logger.error(f"{name} FAILED: {e}")
            results[name] = {"error": str(e)}

    total_time = time.time() - total_start
    logger.info(f"All experiments complete in {total_time:.0f}s")

    print_results_table(results)

    # Save to JSON
    output_path = Path("data/processed/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")
