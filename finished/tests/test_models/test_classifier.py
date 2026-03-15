"""Tests for the HybridClassifier (XGBoost on GNN embeddings + original features)."""

import pytest
import numpy as np


@pytest.fixture
def xgb_params():
    """Minimal XGBoost params for fast testing."""
    return {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 5,
    }


@pytest.fixture
def synthetic_data():
    """
    200-sample synthetic dataset: 16 original features + 8-dim embeddings.
    Labels are imbalanced (10% positive) to mimic real AML data.
    """
    rng = np.random.default_rng(42)
    n_samples = 200
    original_features = rng.standard_normal((n_samples, 16)).astype(np.float32)
    gnn_embeddings = rng.standard_normal((n_samples, 8)).astype(np.float32)
    # 10% positive class
    y = rng.integers(0, 2, n_samples)
    y[:int(n_samples * 0.1)] = 1
    y[int(n_samples * 0.1):] = 0
    rng.shuffle(y)

    split = int(n_samples * 0.7)
    val_split = int(n_samples * 0.85)
    return {
        "original_features": original_features,
        "gnn_embeddings": gnn_embeddings,
        "y": y,
        "split": split,
        "val_split": val_split,
    }


class TestPrepareFeatures:

    def test_concatenated_shape(self, synthetic_data, xgb_params):
        """Combined features should have original_dim + embedding_dim columns."""
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        X = clf.prepare_features(
            synthetic_data["original_features"],
            synthetic_data["gnn_embeddings"]
        )
        expected_cols = 16 + 8
        assert X.shape == (200, expected_cols)

    def test_row_count_preserved(self, synthetic_data, xgb_params):
        """Row count must match original."""
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        X = clf.prepare_features(
            synthetic_data["original_features"],
            synthetic_data["gnn_embeddings"]
        )
        assert X.shape[0] == 200

    def test_mismatched_rows_raises(self, xgb_params):
        """Mismatched row count should raise AssertionError."""
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        with pytest.raises(AssertionError):
            clf.prepare_features(
                np.zeros((100, 16), dtype=np.float32),
                np.zeros((50, 8), dtype=np.float32),
            )


class TestFitAndPredict:

    @pytest.fixture
    def trained_clf(self, synthetic_data, xgb_params):
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        d = synthetic_data
        X = clf.prepare_features(d["original_features"], d["gnn_embeddings"])
        X_train = X[: d["split"]]
        y_train = d["y"][: d["split"]]
        X_val = X[d["split"]: d["val_split"]]
        y_val = d["y"][d["split"]: d["val_split"]]
        clf.fit(X_train, y_train, X_val, y_val)
        return clf, X, d["y"]

    def test_fit_does_not_raise(self, synthetic_data, xgb_params):
        """Fitting should succeed without error."""
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        d = synthetic_data
        X = clf.prepare_features(d["original_features"], d["gnn_embeddings"])
        clf.fit(X[: d["split"]], d["y"][: d["split"]],
                X[d["split"]: d["val_split"]], d["y"][d["split"]: d["val_split"]])

    def test_predict_shape(self, trained_clf, synthetic_data):
        clf, X, y = trained_clf
        preds = clf.predict(X)
        assert preds.shape == (200,)

    def test_predict_binary_labels(self, trained_clf):
        clf, X, y = trained_clf
        preds = clf.predict(X)
        unique = set(preds.tolist())
        assert unique.issubset({0, 1})

    def test_predict_proba_shape(self, trained_clf):
        clf, X, y = trained_clf
        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_predict_proba_sums_to_one(self, trained_clf):
        """Each row of predict_proba should sum to 1."""
        clf, X, y = trained_clf
        proba = clf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(200), atol=1e-5)

    def test_predict_without_fit_raises(self, xgb_params):
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(np.zeros((10, 24), dtype=np.float32))


class TestEvaluate:

    @pytest.fixture
    def trained_clf_and_test(self, synthetic_data, xgb_params):
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        d = synthetic_data
        X = clf.prepare_features(d["original_features"], d["gnn_embeddings"])
        clf.fit(X[: d["split"]], d["y"][: d["split"]],
                X[d["split"]: d["val_split"]], d["y"][d["split"]: d["val_split"]])
        X_test = X[d["val_split"]:]
        y_test = d["y"][d["val_split"]:]
        return clf, X_test, y_test

    def test_evaluate_returns_all_keys(self, trained_clf_and_test):
        clf, X_test, y_test = trained_clf_and_test
        metrics = clf.evaluate(X_test, y_test)
        required_keys = {"auprc", "f1", "precision", "recall",
                         "confusion_matrix", "classification_report"}
        assert required_keys.issubset(metrics.keys())

    def test_metrics_are_floats(self, trained_clf_and_test):
        clf, X_test, y_test = trained_clf_and_test
        metrics = clf.evaluate(X_test, y_test)
        for key in ["auprc", "f1", "precision", "recall"]:
            assert isinstance(metrics[key], float), f"{key} is not float"

    def test_auprc_in_range(self, trained_clf_and_test):
        clf, X_test, y_test = trained_clf_and_test
        metrics = clf.evaluate(X_test, y_test)
        assert 0.0 <= metrics["auprc"] <= 1.0

    def test_confusion_matrix_shape(self, trained_clf_and_test):
        clf, X_test, y_test = trained_clf_and_test
        metrics = clf.evaluate(X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert cm.shape == (2, 2)


class TestFeatureImportance:

    def test_returns_non_empty_dict(self, synthetic_data, xgb_params):
        from src.models.classifier import HybridClassifier
        clf = HybridClassifier(xgb_params)
        d = synthetic_data
        X = clf.prepare_features(d["original_features"], d["gnn_embeddings"])
        clf.fit(X[: d["split"]], d["y"][: d["split"]],
                X[d["split"]: d["val_split"]], d["y"][d["split"]: d["val_split"]])
        importance = clf.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
