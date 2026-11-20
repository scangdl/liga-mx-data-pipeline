"""Over 2.5 Goals Model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from liga_mx.config import get_config, get_path
from liga_mx.models.base import ModelBase, ModelBundle
from liga_mx.features import add_advanced_features


class Over25Model(ModelBase):
    """Over 2.5 goals prediction model."""

    def __init__(self):
        cfg = get_config()
        super().__init__(
            name="over25",
            target="over25",
            feature_set=cfg.features.extended_features,
            config=cfg.models.over25,
        )
        # Force CPU training to avoid CUDA driver issues
        self._xgb_params = {"tree_method": "hist", "device": "cpu"}

    def train(self, features_path: Path | None = None) -> dict[str, float]:
        """Train the Over 2.5 model."""
        if features_path is None:
            features_path = get_path("features_parquet")

        _log(f"Loading features from {features_path}")
        df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)

        # Add advanced features
        _log("Adding advanced features...")
        df = add_advanced_features(df)

        # Prepare data
        X_train_s, y_train, X_test_s, y_test, pool, is_test = self._prepare_data(df)

        # Train XGBoost
        _log("Training XGBoost...")
        xgb_params = self.config.get("xgb_params", {})
        # Merge with CPU params to avoid CUDA issues
        xgb_params = {**self._xgb_params, **xgb_params}
        xgb_model = XGBClassifier(
            **xgb_params,
        )
        xgb_model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

        self.model = xgb_model

        # Evaluation
        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score

        p = xgb_model.predict_proba(X_test_s)[:, 1]
        results = {
            "auc": roc_auc_score(y_test, p),
            "brier": brier_score_loss(y_test, p),
            "logloss": log_loss(y_test, p, labels=[0, 1]),
            "acc": accuracy_score(y_test, (p >= 0.5).astype(int)),
        }

        # Feature importance
        imp = pd.Series(xgb_model.feature_importances_, index=self.feature_set).sort_values(ascending=False)
        _log(f"Top 15 features:\n{imp.head(15).round(4).to_string()}")

        # Save metrics
        self.bundle.metrics = results
        self.bundle.version = self.config.get("version", "v1")

        # Save test predictions
        test_out = pool[is_test][["date", "home", "away", "over25"]].copy()
        test_out["p_model"] = np.round(p, 4)
        test_path = get_path("data_dir") / "over25_test_predictions.parquet"
        test_out.to_parquet(test_path, index=False)
        _log(f"Test predictions saved: {test_path}")

        _log("Training complete!")
        for k, v in results.items():
            _log(f"  {k}: {v:.4f}")

        return results

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict Over 2.5 probability."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        X = X[self.feature_set].fillna(pd.Series(self.bundle.impute_means))
        Xs = self.bundle.scaler.transform(X) if self.bundle.scaler is not None else X.values
        return self.model.predict_proba(Xs)[:, 1]

    def save(self, path: Path | None = None) -> Path:
        """Save model bundle."""
        if path is None:
            path = get_path("over25_model")

        bundle_data = {
            "model": self.model,
            "scaler": self.bundle.scaler,
            "features": self.bundle.features,
            "impute_means": self.bundle.impute_means,
            "train_cut": self.bundle.train_cut,
            "version": self.bundle.version,
            "metrics": self.bundle.metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle_data, f)

        _log(f"Saved Over25 model to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "Over25Model":
        """Load model from disk."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        instance = cls.__new__(cls)
        instance.name = "over25"
        instance.target = "over25"
        instance.feature_set = bundle["features"]
        instance.config = get_config().models.over25

        instance.model = bundle["model"]
        instance.bundle = ModelBundle(
            features=bundle["features"],
            scaler=bundle["scaler"],
            impute_means=bundle["impute_means"],
            train_cut=bundle["train_cut"],
            version=bundle["version"],
            metrics=bundle["metrics"],
        )
        return instance


def _log(msg: str) -> None:
    print(msg, flush=True)


__all__ = ["Over25Model"]