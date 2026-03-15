"""Hybrid GNN + XGBoost classifier for AML detection.

Two-stage approach:
  1. GNN produces rich node embeddings capturing graph structure.
  2. XGBoost takes [original_features || gnn_embeddings] as tabular input,
     combining structural and transactional signals.

This outperforms either model alone:
  - XGBoost excels at sharp thresholds on tabular features (amounts, ratios).
  - GNN embeddings encode neighborhood patterns XGBoost cannot see directly.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from loguru import logger


class HybridClassifier:
    """
    Two-stage AML classifier: GNN embeddings + XGBoost.

    Usage:
        clf = HybridClassifier(xgb_params)
        X_train = clf.prepare_features(original_features, gnn_embeddings)
        clf.fit(X_train, y_train, X_val, y_val)
        metrics = clf.evaluate(X_test, y_test)
    """

    def __init__(self, xgb_params: dict):
        self.xgb_params = xgb_params
        self.model: xgb.XGBClassifier | None = None

    def prepare_features(
        self,
        original_features: np.ndarray,
        gnn_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate original node features with GNN embeddings.

        Args:
            original_features: [num_nodes, num_original_features]
            gnn_embeddings:    [num_nodes, embedding_dim]

        Returns:
            [num_nodes, num_original_features + embedding_dim]
        """
        assert original_features.shape[0] == gnn_embeddings.shape[0], (
            f"Row mismatch: original={original_features.shape[0]}, "
            f"embeddings={gnn_embeddings.shape[0]}"
        )
        return np.concatenate([original_features, gnn_embeddings], axis=1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost with early stopping on validation AUPRC.

        Class imbalance is handled via scale_pos_weight = n_negative / n_positive.
        This tells XGBoost to treat each illicit sample as scale_pos_weight
        licit samples, correcting for the ~2% illicit rate in Elliptic.
        """
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

        if n_pos == 0:
            raise ValueError("No positive (illicit) samples in training set.")

        scale_pos_weight = n_neg / n_pos
        logger.info(
            f"XGBoost class weight: {scale_pos_weight:.2f} "
            f"(neg={n_neg}, pos={n_pos})"
        )

        params = {
            "n_estimators": self.xgb_params.get("n_estimators", 500),
            "max_depth": self.xgb_params.get("max_depth", 8),
            "learning_rate": self.xgb_params.get("learning_rate", 0.05),
            "subsample": self.xgb_params.get("subsample", 0.8),
            "colsample_bytree": self.xgb_params.get("colsample_bytree", 0.8),
            "eval_metric": self.xgb_params.get("eval_metric", "aucpr"),
            "early_stopping_rounds": self.xgb_params.get("early_stopping_rounds", 50),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",  # fastest algorithm
            "objective": "binary:logistic",
            "verbosity": 0,
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Log best iteration info
        best_iteration = getattr(self.model, "best_iteration", None)
        if best_iteration is not None:
            logger.info(f"XGBoost best iteration: {best_iteration}")

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (0=licit, 1=illicit)."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.
        Shape: [num_nodes, 2] where [:, 1] is P(illicit).
        """
        self._check_fitted()
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Full evaluation on a labeled dataset.

        Primary metric is AUPRC (Area Under Precision-Recall Curve).
        Accuracy is deliberately omitted — it's misleading when only 2%
        of nodes are illicit (a model predicting all-licit gets 98% accuracy).

        Returns dict with: auprc, f1, precision, recall, confusion_matrix,
                           classification_report
        """
        self._check_fitted()
        proba = self.predict_proba(X)[:, 1]
        preds = self.predict(X)

        auprc = average_precision_score(y, proba)
        f1 = f1_score(y, preds, zero_division=0)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)
        report = classification_report(
            y, preds, target_names=["licit", "illicit"], zero_division=0
        )

        logger.info(
            f"XGBoost evaluation — AUPRC: {auprc:.4f}, "
            f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        return {
            "auprc": float(auprc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def feature_importance(self) -> dict:
        """
        Return XGBoost feature importances (gain-based).

        'Gain' measures the average information gain of each feature when
        it is used in a split. Features with high gain are actually useful
        for the splits, unlike 'weight' which just counts how often a
        feature appears regardless of its contribution.
        """
        self._check_fitted()
        scores = self.model.get_booster().get_score(importance_type="gain")
        # Sort descending by importance
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save the XGBoost model to disk (JSON format)."""
        self._check_fitted()
        self.model.save_model(path)
        logger.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: str, xgb_params: dict | None = None) -> "HybridClassifier":
        """Load a previously saved XGBoost model from disk."""
        import xgboost as xgb
        instance = cls(xgb_params or {})
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(path)
        logger.info(f"XGBoost model loaded from {path}")
        return instance

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Classifier has not been trained. Call fit() first.")
