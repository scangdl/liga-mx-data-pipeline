"""BTTS Model (Both Teams To Score) — v3 ensemble."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

from liga_mx.config import get_config, get_path
from liga_mx.models.base import ModelBase, ModelBundle
from liga_mx.features import StateBuilder, add_advanced_features, poisson_btts_prob


class BTTSModel(ModelBase):
    """BTTS prediction model with ensemble (XGBoost + LR + LightGBM)."""

    def __init__(self):
        cfg = get_config()
        super().__init__(
            name="btts",
            target="btts",
            feature_set=cfg.features.extended_features,
            config=cfg.models.btts,
        )
        self.ensemble_weights = cfg.models.btts.get("ensemble_weights", {"xgb": 0.85, "lr": 0.10, "lgb": 0.05})
        self.calibration = cfg.models.btts.get("calibration", "platt")
        self._lr_model: LogisticRegression | None = None
        self._lgb_model: Any = None
        # Force CPU training to avoid CUDA driver issues
        self._xgb_params = {"tree_method": "hist", "device": "cpu"}

    def train(self, features_path: Path | None = None) -> dict[str, float]:
        """Train the BTTS ensemble model."""
        if features_path is None:
            features_path = get_path("features_parquet")

        _log(f"Loading features from {features_path}")
        df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)

        # Add advanced features
        _log("Adding advanced features...")
        df = add_advanced_features(df)

        # Prepare data
        X_train_s, y_train, X_test_s, y_test, pool, is_test = self._prepare_data(df)

        # Split train for validation
        n_train = len(X_train_s)
        val_size = max(200, int(n_train * 0.15))
        X_tr, X_val = X_train_s[:-val_size], X_train_s[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        _log(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_s)}")

        # Train XGBoost
        _log("Training XGBoost...")
        xgb_params = self.config.get("xgb_params", {})
        # Merge with CPU params to avoid CUDA issues
        xgb_params = {**self._xgb_params, **xgb_params}
        xgb_model = XGBClassifier(
            **xgb_params,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb = xgb_model.predict_proba(X_test_s)[:, 1]

        # Train LogisticRegression
        _log("Training LogisticRegression...")
        lr_model = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight="balanced")
        lr_model.fit(X_tr, y_tr)
        p_lr = lr_model.predict_proba(X_test_s)[:, 1]

        # Train LightGBM if available
        lgb_model = None
        p_lgb = None
        if HAS_LGBM:
            _log("Training LightGBM...")
            lgb_model = LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="binary",
                metric="logloss",
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            )
            lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
            p_lgb = lgb_model.predict_proba(X_test_s)[:, 1]

        # Poisson baseline
        _log("Poisson baseline...")
        lam_mask = pool.loc[is_test, "lam_h"].notna() & pool.loc[is_test, "lam_a"].notna()
        p_poisson = np.full(len(y_test), np.nan)
        if lam_mask.any():
            p_poisson[lam_mask] = (
                1 - poisson.pmf(0, pool.loc[is_test, "lam_h"][lam_mask])
            ) * (
                1 - poisson.pmf(0, pool.loc[is_test, "lam_a"][lam_mask])
            )

        # Optimize ensemble weights
        _log("Optimizing ensemble weights...")
        models = {"xgb": p_xgb, "lr": p_lr}
        if p_lgb is not None:
            models["lgb"] = p_lgb

        model_names = list(models.keys())
        preds = np.column_stack([models[name] for name in model_names])

        def weight_loss(w: np.ndarray) -> float:
            w = np.abs(w)
            w = w / w.sum()
            p_ens = (preds * w).sum(axis=1)
            from sklearn.metrics import brier_score_loss
            return brier_score_loss(y_test, p_ens)

        w0 = np.ones(len(model_names)) / len(model_names)
        res = minimize(weight_loss, w0, method="Nelder-Mead", options={"maxiter": 2000})
        best_weights = np.abs(res.x) / np.abs(res.x).sum()
        ensemble_weights = dict(zip(model_names, best_weights.round(4)))
        _log(f"Optimal weights: {ensemble_weights}")

        p_ens = (preds * best_weights).sum(axis=1)

        # Platt calibration on validation predictions
        if self.calibration == "platt":
            _log("Fitting Platt calibration...")
            p_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
            p_val_lr = lr_model.predict_proba(X_val)[:, 1]
            p_val_lgb = lgb_model.predict_proba(X_val)[:, 1] if lgb_model else None

            p_val_ens = ensemble_weights["xgb"] * p_val_xgb + ensemble_weights["lr"] * p_val_lr
            if p_val_lgb is not None and "lgb" in ensemble_weights:
                p_val_ens += ensemble_weights["lgb"] * p_val_lgb

            platt = LogisticRegression(max_iter=1000)
            platt.fit(p_val_ens.reshape(-1, 1), y_val)

            # Apply calibration
            p_ens_cal = platt.predict_proba(p_ens.reshape(-1, 1))[:, 1]
            p_ens = p_ens_cal

            # Save calibration model
            self._lr_model = platt
        else:
            self._lr_model = None

        # Store models
        self.model = xgb_model
        self._lr_model = lr_model
        self._lgb_model = lgb_model

        # Evaluation
        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score

        results = {}
        for name, p in {"xgb": p_xgb, "lr": p_lr, "ensemble": p_ens}.items():
            if p_lgb is not None and name == "lgb":
                continue
            mask = ~np.isnan(p)
            if mask.sum() == 0:
                continue
            results[name] = {
                "auc": roc_auc_score(y_test[mask], p[mask]),
                "brier": brier_score_loss(y_test[mask], p[mask]),
                "logloss": log_loss(y_test[mask], p[mask], labels=[0, 1]),
                "acc": accuracy_score(y_test[mask], (p[mask] >= 0.5).astype(int)),
            }

        if p_lgb is not None:
            mask = ~np.isnan(p_lgb)
            results["lgb"] = {
                "auc": roc_auc_score(y_test[mask], p_lgb[mask]),
                "brier": brier_score_loss(y_test[mask], p_lgb[mask]),
                "logloss": log_loss(y_test[mask], p_lgb[mask], labels=[0, 1]),
                "acc": accuracy_score(y_test[mask], (p_lgb[mask] >= 0.5).astype(int)),
            }

        # Feature importance
        imp = pd.Series(xgb_model.feature_importances_, index=self.feature_set).sort_values(ascending=False)
        _log(f"Top 20 features:\n{imp.head(20).round(4).to_string()}")

        # Save metrics
        self.bundle.metrics = {
            "test_brier": results["ensemble"]["brier"],
            "test_auc": results["ensemble"]["auc"],
            "test_logloss": results["ensemble"]["logloss"],
            "test_acc": results["ensemble"]["acc"],
        }
        self.bundle.ensemble_weights = ensemble_weights
        self.bundle.version = self.config.get("version", "v3_improved")

        # Save test predictions
        test_out = pool[is_test][["date", "home", "away", "btts"]].copy()
        for name, p in {"xgb": p_xgb, "lr": p_lr, "ensemble": p_ens}.items():
            test_out[f"p_{name}"] = np.round(p, 4)
        if p_lgb is not None:
            test_out["p_lgb"] = np.round(p_lgb, 4)
        if not np.all(np.isnan(p_poisson)):
            test_out["p_poisson"] = np.round(p_poisson, 4)

        test_path = get_path("data_dir") / "btts_test_predictions_v3.parquet"
        test_out.to_parquet(test_path, index=False)
        _log(f"Test predictions saved: {test_path}")

        _log("Training complete!")
        for name, r in results.items():
            _log(f"  {name:10s} AUC={r['auc']:.4f} Brier={r['brier']:.4f} LogLoss={r['logloss']:.4f} Acc={r['acc']:.3f}")

        return self.bundle.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict BTTS probability using ensemble."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        # Ensure features match
        X = X[self.feature_set].fillna(pd.Series(self.bundle.impute_means))
        Xs = self.bundle.scaler.transform(X) if self.bundle.scaler is not None else X.values

        # XGBoost
        p_xgb = self.model.predict_proba(Xs)[:, 1]

        # LogisticRegression
        p_lr = self._lr_model.predict_proba(Xs)[:, 1] if self._lr_model is not None else None

        # LightGBM
        p_lgb = self._lgb_model.predict_proba(Xs)[:, 1] if self._lgb_model is not None else None

        # Ensemble
        weights = self.bundle.ensemble_weights
        p_ens = weights["xgb"] * p_xgb + weights["lr"] * (p_lr if p_lr is not None else 0)
        if p_lgb is not None and "lgb" in weights:
            p_ens += weights["lgb"] * p_lgb

        # Apply Platt calibration if available
        if hasattr(self, "_platt") and self._platt is not None:
            p_ens = self._platt.predict_proba(p_ens.reshape(-1, 1))[:, 1]

        return p_ens

    def save(self, path: Path | None = None) -> Path:
        """Save full ensemble bundle."""
        if path is None:
            path = get_path("btts_model_v3")

        bundle_data = {
            "xgb_model": self.model,
            "lr_model": self._lr_model,
            "lgb_model": self._lgb_model,
            "ensemble_weights": self.bundle.ensemble_weights,
            "scaler": self.bundle.scaler,
            "features": self.bundle.features,
            "impute_means": self.bundle.impute_means,
            "train_cut": self.bundle.train_cut,
            "version": self.bundle.version,
            "metrics": self.bundle.metrics,
            "platt_calibration": getattr(self, "_platt", None),
        }

        with open(path, "wb") as f:
            pickle.dump(bundle_data, f)

        _log(f"Saved BTTS ensemble to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "BTTSModel":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        instance = cls.__new__(cls)
        instance.name = "btts"
        instance.target = "btts"
        instance.feature_set = bundle["features"]
        instance.config = get_config().models.btts
        instance.ensemble_weights = bundle["ensemble_weights"]
        instance.calibration = get_config().models.btts.get("calibration", "platt")

        instance.model = bundle["xgb_model"]
        instance._lr_model = bundle["lr_model"]
        instance._lgb_model = bundle["lgb_model"]
        instance.bundle = ModelBundle(
            features=bundle["features"],
            scaler=bundle["scaler"],
            impute_means=bundle["impute_means"],
            train_cut=bundle["train_cut"],
            version=bundle["version"],
            metrics=bundle["metrics"],
        )
        instance.bundle.ensemble_weights = bundle["ensemble_weights"]

        return instance


def _log(msg: str) -> None:
    print(msg, flush=True)


__all__ = ["BTTSModel"]