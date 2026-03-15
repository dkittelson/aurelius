"""Evaluation utilities for the GNN model.

These functions run inference on a fixed graph split (train/val/test)
and return standardized metric dicts used by both the Trainer and the API.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger


def evaluate_model(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run GNN inference and compute evaluation metrics on the masked node subset.

    Uses full-graph inference (not mini-batched) — safe for Elliptic (~200K nodes,
    137 MB feature matrix). For very large graphs, use evaluate_model_batched().

    Args:
        model:  Trained AureliusGAT.
        data:   PyG Data object with .x, .edge_index, .y.
        mask:   Boolean tensor selecting nodes to evaluate.
        device: Torch device to run inference on.

    Returns:
        dict with auprc, f1, precision, recall, confusion_matrix,
                  classification_report, num_evaluated
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]  # P(illicit)

    # Apply split mask and filter unknown labels
    mask = mask.to(device)
    probs_subset = probs[mask].cpu().numpy()
    y_subset = data.y[mask].cpu().numpy()

    known = y_subset != -1
    probs_subset = probs_subset[known]
    y_subset = y_subset[known]

    if len(y_subset) == 0:
        logger.warning("No labeled nodes in evaluation subset.")
        return _empty_metrics()

    if y_subset.sum() == 0:
        logger.warning("No positive (illicit) samples in evaluation subset — AUPRC undefined.")
        return _empty_metrics()

    auprc = average_precision_score(y_subset, probs_subset)
    preds = (probs_subset >= 0.5).astype(int)
    f1 = f1_score(y_subset, preds, zero_division=0)
    precision = precision_score(y_subset, preds, zero_division=0)
    recall = recall_score(y_subset, preds, zero_division=0)
    cm = confusion_matrix(y_subset, preds)
    report = classification_report(
        y_subset, preds, target_names=["licit", "illicit"], zero_division=0
    )

    return {
        "auprc": float(auprc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm,
        "classification_report": report,
        "num_evaluated": int(known.sum()),
    }


def get_node_probabilities(
    model: torch.nn.Module,
    data,
    device: torch.device,
) -> np.ndarray:
    """
    Run full-graph inference and return P(illicit) for every node.

    Returns:
        probs: [num_nodes] float numpy array
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]

    return probs.cpu().numpy()


def generate_evaluation_report(metrics: dict, split: str = "test") -> str:
    """
    Format evaluation metrics as a markdown report string.

    Args:
        metrics: Output of evaluate_model().
        split:   Which split was evaluated ('train', 'val', 'test').

    Returns:
        Markdown string.
    """
    lines = [
        f"## Aurelius GNN Evaluation — {split.capitalize()} Set",
        "",
        f"| Metric    | Value  |",
        f"|-----------|--------|",
        f"| AUPRC     | {metrics.get('auprc', 0):.4f} |",
        f"| F1        | {metrics.get('f1', 0):.4f} |",
        f"| Precision | {metrics.get('precision', 0):.4f} |",
        f"| Recall    | {metrics.get('recall', 0):.4f} |",
        f"| Evaluated | {metrics.get('num_evaluated', 0)} nodes |",
        "",
        "### Classification Report",
        "```",
        metrics.get("classification_report", "N/A"),
        "```",
    ]

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        lines += [
            "",
            "### Confusion Matrix",
            "```",
            f"          Pred Licit  Pred Illicit",
            f"True Licit    {cm[0][0]:6d}        {cm[0][1]:6d}",
            f"True Illicit  {cm[1][0]:6d}        {cm[1][1]:6d}",
            "```",
        ]

    return "\n".join(lines)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find the decision threshold that maximizes the given metric.

    With class-imbalanced data (11.6% illicit), the default 0.5 threshold
    is suboptimal. This scans thresholds from 0.05 to 0.95 and returns
    the one maximizing F1 (or precision/recall).

    Args:
        y_true:  Ground truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        metric:  'f1', 'precision', or 'recall'.

    Returns:
        (best_threshold, best_score)
    """
    metric_fns = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    }
    fn = metric_fns.get(metric, metric_fns["f1"])

    best_threshold = 0.5
    best_score = 0.0

    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_proba >= t).astype(int)
        score = fn(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, best_score


def _empty_metrics() -> dict:
    return {
        "auprc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "confusion_matrix": None,
        "classification_report": "N/A",
        "num_evaluated": 0,
    }
