"""Base model classes for Liga MX Pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from liga_mx.config import get_config, get_path


class ModelBundle(BaseModel):
    """Container for model artifacts."""
    model: Any = None
    scaler: StandardScaler | None = None
    features: list[str] = []
    impute_means: dict[str, float] = {}
    train_cut: str = ""
    version: str = "v1"
    metrics: dict[str, float] = {}
    ensemble_weights: dict[str, float] | None = None

    class Config:
        arbitrary_types_allowed = True


class ModelBase:
    """Base class for all models."""

    def __init__(self, name: str, target: str, feature_set: list[str], config: dict):
        self.name = name
        self.target = target
        self.feature_set = feature_set
        self.config = config
        self.bundle = ModelBundle(
            features=feature_set,
            train_cut=config.get("train_cut_date", "2024-07-01"),
            version=config.get("version", "v1"),
        )
        self._scaler = StandardScaler()
        self._means: pd.Series | None = None

    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare features and target from DataFrame."""
        pool = df[df[self.target].notna()].copy()
        X = pool[self.feature_set].copy()
        y = pool[self.target].astype(int).values
        dates = pool["date"]

        # Temporal split
        cut = pd.Timestamp(self.bundle.train_cut)
        is_test = dates >= cut

        X_train, y_train = X[~is_test], y[~is_test]
        X_test, y_test = X[is_test], y[is_test]

        # Impute with train means
        self._means = X_train.mean()
        self.bundle.impute_means = self._means.to_dict()
        X_train = X_train.fillna(self._means)
        X_test = X_test.fillna(self._means)

        # Scale
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)
        self.bundle.scaler = self._scaler

        return (X_train_s, y_train, X_test_s, y_test, pool, is_test)

    def save(self, path: Path | None = None) -> Path:
        """Save model bundle to disk."""
        if path is None:
            path = get_path(f"{self.name}_model")
        self.bundle.model = self.model
        with open(path, "wb") as f:
            pickle.dump(self.bundle.model_dump(), f)
        print(f"Saved {self.name} model to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelBase":
        """Load model bundle from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Reconstruct instance
        instance = cls.__new__(cls)
        instance.bundle = ModelBundle(**data)
        instance.model = data["model"]
        instance._scaler = data["scaler"]
        instance._means = pd.Series(data["impute_means"])
        return instance

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (to be implemented by subclasses)."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return proba.argmax(axis=1) if proba.ndim > 1 else (proba >= 0.5).astype(int)


__all__ = ["ModelBase", "ModelBundle"]