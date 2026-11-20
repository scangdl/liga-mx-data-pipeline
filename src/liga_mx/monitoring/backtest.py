"""Monthly Backtest — Automated model performance tracking."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from liga_mx.config import get_config, get_path
from liga_mx.features import add_advanced_features
from liga_mx.models import BTTSModel

log = structlog.get_logger(__name__)


class MonthlyBacktest:
    """
    Monthly automated backtest for model decay detection.

    Runs on 1st of each month 06:00.
    Evaluates model on last 30 days (falls back to 60 if needed).
    Compares vs historical baseline and alerts on degradation.
    """

    def __init__(self):
        self.cfg = get_config().monitoring.monthly_backtest

    def run(self) -> dict[str, Any]:
        """Run monthly backtest and return results."""
        log.info("=" * 60)
        log.info(f"MONTHLY BACKTEST — {datetime.now().strftime('%Y-%m-%d')}")
        log.info("=" * 60)

        # Load model
        model_path = get_path("btts_model_v3")
        if not model_path.exists():
            model_path = get_path("btts_model")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        version = bundle.get("version", "v2")
        if version == "v3_improved" or "ensemble_weights" in bundle:
            model = bundle["xgb_model"]
            scaler = bundle["scaler"]
            FEATS = bundle["features"]
            means = pd.Series(bundle["impute_means"])
            log.info(f"Model: {version} (ensemble)")
        else:
            model = bundle["model"]
            scaler = bundle["scaler"]
            FEATS = bundle["features"]
            means = pd.Series(bundle["impute_means"])
            log.info(f"Model: {version}")

        # Load features
        df = pd.read_parquet(get_path("features_parquet")).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = add_advanced_features(df)

        # Last complete month
        end_date = df["date"].max()
        start_date = end_date - timedelta(days=30)
        test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

        if test_mask.sum() < self.cfg.get("min_matches", 10):
            log.info(f"Few matches in last month ({test_mask.sum()}), extending to 60 days...")
            start_date = end_date - timedelta(days=60)
            test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

        test_df = df[test_mask].copy()
        log.info(f"Period: {start_date.date()} - {end_date.date()} | Matches: {len(test_df)}")

        if len(test_df) < self.cfg.get("min_matches", 10):
            log.warning("Insufficient data for backtest")
            return {"status": "insufficient_data"}

        # Prepare features
        X = test_df[FEATS].fillna(means)
        y = test_df["btts"].astype(int).values
        Xs = scaler.transform(X) if scaler is not None else X.values

        # Predict
        p = model.predict_proba(Xs)[:, 1]

        # Metrics
        brier = brier_score_loss(y, p)
        ll = log_loss(y, p, labels=[0, 1])
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else 0.5
        acc = ((p >= 0.5).astype(int) == y).mean()
        btts_rate = y.mean()

        # Baseline (historical before period)
        base_rate = df[df["date"] < start_date]["btts"].mean()
        base_brier = brier_score_loss(y, np.full(len(y), base_rate))

        log.info(f"\nMetrics for {start_date.date()} - {end_date.date()}:")
        log.info(f"  BTTS rate: {btts_rate:.1%}")
        log.info(f"  Model Brier: {brier:.4f} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Acc@0.5: {acc:.1%}")
        log.info(f"  Base rate (hist): {base_rate:.1%} | Base Brier: {base_brier:.4f}")
        log.info(f"  Improvement vs base: {base_brier - brier:+.4f} Brier ({((base_brier - brier)/base_brier)*100:+.1f}%)")

        # Compare with history
        hist_path = get_path("monthly_backtest_history")
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            log.info(f"\nHistorical ({len(hist)} months):")
            log.info(f"  Brier mean: {hist['brier'].mean():.4f} ± {hist['brier'].std():.4f}")
            log.info(f"  AUC mean: {hist['auc'].mean():.4f}")

            # Alerts
            if brier > hist["brier"].mean() * (1 + self.cfg.get("brier_degradation_alert", 0.05)):
                log.warning("  ⚠️ ALERT: Brier degrades >5% vs historical mean!")

            if brier > hist["brier"].mean() * (1 + self.cfg.get("brier_degradation_critical", 0.10)):
                log.error("  🚨 CRITICAL ALERT: Brier degrades >10% — consider retraining")

        # Save record
        record = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "period_start": start_date.date().isoformat(),
            "period_end": end_date.date().isoformat(),
            "n_matches": int(len(test_df)),
            "btts_rate": float(btts_rate),
            "brier": float(brier),
            "logloss": float(ll),
            "auc": float(auc),
            "acc_05": float(acc),
            "base_rate": float(base_rate),
            "base_brier": float(base_brier),
            "improvement_pct": float((base_brier - brier) / base_brier * 100),
            "model_version": version,
        }

        if hist_path.exists():
            hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
        else:
            hist = pd.DataFrame([record])
        hist.to_csv(hist_path, index=False)
        log.info(f"\nHistory updated: {hist_path}")

        # Detail file for audit
        detail = test_df[["date", "home", "away", "btts"]].copy()
        detail["p_model"] = np.round(p, 4)
        detail["correct_05"] = ((detail["p_model"] >= 0.5).astype(int) == detail["btts"]).astype(int)
        detail_file = get_path("predictions_dir") / f"monthly_backtest_{datetime.now().strftime('%Y-%m')}.csv"
        detail.to_csv(detail_file, index=False, encoding="utf-8-sig")
        log.info(f"Detail saved: {detail_file}")

        log.info("=" * 60)
        log.info("MONTHLY BACKTEST COMPLETED")
        log.info("=" * 60)

        return {"status": "completed", "metrics": record}


if __name__ == "__main__":
    backtest = MonthlyBacktest()
    backtest.run()