import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

class HybridClassifier:
    def __init__(self, xgb_params: dict):
        self.xgb_params = xgb_params
        self.model = None

    def prepare_features(self, original_features, gnn_embeddings) -> np.ndarray:
        assert original_features.shape[0] == gnn_embeddings.shape[0]

        # concatenate raw features per node + output embeddings from static gnn
        return np.concatenate([original_features, gnn_embeddings], axis=1)

    def fit(self, X_train, y_train, X_val, y_val):

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0:
            raise ValueError("No positive (illicit) samples in training set.")
        scale_pos_weight = n_neg / n_pos

        params = {
            "n_estimators": self.xgb_params.get("n_estimators", 500),
            "max_depth": self.xgb_params.get("max_depth", 8),
            "learning_rate": self.xgb_params.get("learning_rate", 0.05),
            "subsample": self.xgb_params.get("subsample", 0.8),
            "colsample_bytree": self.xgb_params.get("colsample_bytree", 0.8),
            "eval_metric": self.xgb_params.get("eval_metric", "aucpr"),
            "early_stopping_rounds": self.xgb_params.get("early_stopping_rounds", 50),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "verbosity": 0,
        }
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        self._check_fitted()
        proba = self.predict_proba(X)[:, 1]
        preds = self.predict(X)
        return {
            "auprc": float(average_precision_score(y, proba)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds),
            "classification_report": classification_report(y, preds, target_names=["licit", "illicit"], zero_division=0),
        }

    def feature_importance(self) -> dict:
        self._check_fitted()
        scores = self.model.get_booster().get_score(importance_type="gain")
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        self._check_fitted()
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str, xgb_params: dict = None) -> "HybridClassifier":
        instance = cls(xgb_params or {})
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(path)
        return instance

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Classifier has not been trained. Call fit() first.")

   



