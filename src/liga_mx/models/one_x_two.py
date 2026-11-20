"""1X2 Model (Match Result: Home/Draw/Away)."""

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


class OneXTwoModel(ModelBase):
    """1X2 match result prediction model (multiclass)."""

    def __init__(self):
        cfg = get_config()
        super().__init__(
            name="1x2",
            target="res_1x2",
            feature_set=cfg.features.extended_features,
            config=cfg.models.one_x_two,
        )
        self.target_map = {"H": 0, "D": 1, "A": 2}
        self.inv_target_map = {v: k for k, v in self.target_map.items()}
        # Force CPU training to avoid CUDA driver issues
        self._xgb_params = {"tree_method": "hist", "device": "cpu"}

    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Override to handle multiclass target."""
        pool = df[df[self.target].notna()].copy()
        pool["y"] = pool[self.target].map(self.target_map)

        X = pool[self.feature_set].copy()
        y = pool["y"].values
        dates = pool["date"]

        cut = pd.Timestamp(self.bundle.train_cut)
        is_test = dates >= cut

        X_train, y_train = X[~is_test], y[~is_test]
        X_test, y_test = X[is_test], y[is_test]

        self._means = X_train.mean()
        self.bundle.impute_means = self._means.to_dict()
        X_train = X_train.fillna(self._means)
        X_test = X_test.fillna(self._means)

        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)
        self.bundle.scaler = self._scaler

        return (X_train_s, y_train, X_test_s, y_test, pool, is_test)

    def train(self, features_path: Path | None = None) -> dict[str, float]:
        """Train the 1X2 model."""
        if features_path is None:
            features_path = get_path("features_parquet")

        _log(f"Loading features from {features_path}")
        df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)

        # Add advanced features
        _log("Adding advanced features...")
        df = add_advanced_features(df)

        # Prepare data
        X_train_s, y_train, X_test_s, y_test, pool, is_test = self._prepare_data(df)

        # Train XGBoost multiclass
        _log("Training XGBoost (multiclass)...")
        xgb_params = self.config.get("xgb_params", {}).copy()
        # early_stopping_rounds must be a constructor param in newer XGBoost
        early_stopping = xgb_params.pop("early_stopping_rounds", 40)
        # Merge with CPU params to avoid CUDA issues
        xgb_params = {**self._xgb_params, **xgb_params}
        xgb_model = XGBClassifier(
            early_stopping_rounds=early_stopping,
            **xgb_params,
        )
        xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False
        )

        self.model = xgb_model

        # Evaluation
        from sklearn.metrics import log_loss, accuracy_score, classification_report

        p = xgb_model.predict_proba(X_test_s)
        pred = p.argmax(axis=1)

        results = {
            "accuracy": accuracy_score(y_test, pred),
            "logloss": log_loss(y_test, p, labels=[0, 1, 2]),
        }

        # Per-class metrics
        report = classification_report(y_test, pred, target_names=["H", "D", "A"], output_dict=True)
        for cls in ["H", "D", "A"]:
            r = report[cls]
            results[f"{cls}_precision"] = r["precision"]
            results[f"{cls}_recall"] = r["recall"]
            results[f"{cls}_f1"] = r["f1-score"]

        # Baseline: implied odds (market)
        test_pool = pool[is_test].copy()
        # Would need implied odds - skip for now

        # Feature importance
        imp = pd.Series(xgb_model.feature_importances_, index=self.feature_set).sort_values(ascending=False)
        _log(f"Top 15 features:\n{imp.head(15).round(4).to_string()}")

        # Save metrics
        self.bundle.metrics = results
        self.bundle.version = self.config.get("version", "v1")

        # Save test predictions
        test_out = test_pool[["date", "home", "away", "res_1x2"]].copy()
        test_out["p_H"] = np.round(p[:, 0], 4)
        test_out["p_D"] = np.round(p[:, 1], 4)
        test_out["p_A"] = np.round(p[:, 2], 4)
        test_out["pred"] = pd.Series(pred).map(self.inv_target_map).values
        test_path = get_path("data_dir") / "test_predictions_1x2.parquet"
        test_out.to_parquet(test_path, index=False)
        _log(f"Test predictions saved: {test_path}")

        _log("Training complete!")
        _log(f"  Accuracy: {results['accuracy']:.3f}")
        _log(f"  LogLoss: {results['logloss']:.4f}")
        for cls in ["H", "D", "A"]:
            _log(f"  {cls}: P={results[f'{cls}_precision']:.3f} R={results[f'{cls}_recall']:.3f} F1={results[f'{cls}_f1']:.3f}")

        return results

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1X2 probabilities [p_H, p_D, p_A]."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        X = X[self.feature_set].fillna(pd.Series(self.bundle.impute_means))
        Xs = self.bundle.scaler.transform(X) if self.bundle.scaler is not None else X.values
        return self.model.predict_proba(Xs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 1X2 class (H/D/A)."""
        proba = self.predict_proba(X)
        return np.array([self.inv_target_map[i] for i in proba.argmax(axis=1)])

    def save(self, path: Path | None = None) -> Path:
        """Save model bundle."""
        if path is None:
            path = get_path("model_1x2")

        bundle_data = {
            "model": self.model,
            "scaler": self.bundle.scaler,
            "features": self.bundle.features,
            "impute_means": self.bundle.impute_means,
            "train_cut": self.bundle.train_cut,
            "version": self.bundle.version,
            "metrics": self.bundle.metrics,
            "target_map": self.target_map,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle_data, f)

        _log(f"Saved 1X2 model to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "OneXTwoModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        instance = cls.__new__(cls)
        instance.name = "1x2"
        instance.target = "res_1x2"
        instance.feature_set = bundle["features"]
        instance.config = get_config().models.one_x_two
        instance.target_map = bundle["target_map"]
        instance.inv_target_map = {v: k for k, v in instance.target_map.items()}

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


__all__ = ["OneXTwoModel"]