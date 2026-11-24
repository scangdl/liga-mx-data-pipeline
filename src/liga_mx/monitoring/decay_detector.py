"""Decay Detector — Weekly model health monitoring."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import brier_score_loss, roc_auc_score

from liga_mx.config import get_config, get_path
from liga_mx.features import add_advanced_features
from liga_mx.models import BTTSModel
from liga_mx.notifications.twilio_whatsapp import get_twilio_notifier

log = structlog.get_logger(__name__)


class DecayDetector:
    """
    Weekly decay detection for BTTS model.

    Compares rolling 4-week metrics vs 6-month historical baseline.
    Alerts on:
    - Brier score degradation > 10% vs mean
    - Brier > mean + 2σ
    - AUC < 0.52 (near random)
    - AUC degradation > 5% vs mean
    - Reliability (calibration) > 0.05
    - BTTS rate shift > 10% (data drift)
    """

    def __init__(self):
        self.cfg = get_config().monitoring.decay_detector

    def _send_telegram_alert(self, message: str, parse_mode: str | None = "HTML") -> bool:
        """Send alert via Telegram if configured."""
        try:
            from liga_mx.notifications.telegram import get_notifier
            return get_notifier().send(message, parse_mode=parse_mode)
        except Exception as e:
            log.warning(f"Telegram not available: {e}")
            return False

    def _send_whatsapp_alert(self, message: str) -> bool:
        """Send alert via WhatsApp (Twilio) if configured."""
        try:
            return get_twilio_notifier().send(message)
        except Exception as e:
            log.warning(f"WhatsApp not available: {e}")
            return False

    def _send_alerts(self, message: str) -> None:
        """Send to all configured channels."""
        self._send_telegram_alert(message)
        self._send_whatsapp_alert(message)

    def run(self) -> dict[str, Any]:
        """Run decay detection and return results."""
        log.info("=" * 60)
        log.info(f"DECAY DETECTOR — {datetime.now().strftime('%Y-%m-%d')}")
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
        from liga_mx.features import StateBuilder
        df = pd.read_parquet(get_path("features_parquet")).sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = add_advanced_features(df)

        # Rolling 4 weeks (last 28 days with results)
        end_date = df[df["btts"].notna()]["date"].max()
        start_date = end_date - timedelta(days=28)
        test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

        if test_mask.sum() < self.cfg.get("min_matches", 20):
            log.info(f"Few recent matches ({test_mask.sum()}), extending to 60 days...")
            start_date = end_date - timedelta(days=60)
            test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

        test_df = df[test_mask].copy()
        log.info(f"Rolling period: {start_date.date()} - {end_date.date()} | {len(test_df)} matches")

        if len(test_df) < self.cfg.get("min_matches", 20):
            log.warning("Insufficient data for decay evaluation")
            return {"status": "insufficient_data"}

        # Predict
        X = test_df[FEATS].fillna(means)
        y = test_df["btts"].astype(int).values
        Xs = scaler.transform(X) if scaler is not None else X.values
        p = model.predict_proba(Xs)[:, 1]

        # Metrics
        brier = brier_score_loss(y, p)
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else 0.5
        acc = ((p >= 0.5).astype(int) == y).mean()

        # Reliability (calibration)
        bins = np.linspace(0, 1, 11)
        bin_ids = np.digitize(p, bins) - 1
        reliability = 0.0
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() > 0:
                avg_pred = p[mask].mean()
                avg_true = y[mask].mean()
                reliability += mask.sum() / len(p) * abs(avg_pred - avg_true)

        log.info(f"Rolling 4w: Brier={brier:.4f}, AUC={auc:.4f}, Acc={acc:.1%}, Reliability={reliability:.4f}")

        # Load historical baseline
        hist_path = get_path("monthly_backtest_history")
        if not hist_path.exists():
            log.warning("No monthly history, skipping comparison")
            return {"status": "no_history"}

        hist = pd.read_csv(hist_path)
        if len(hist) < 3:
            log.warning("History too short (<3 months)")
            return {"status": "short_history"}

        # Baseline: last 6 months
        recent = hist.tail(6)
        brier_mean = recent["brier"].mean()
        brier_std = recent["brier"].std()
        auc_mean = recent["auc"].mean()

        log.info(f"\nBaseline (last {len(recent)} months):")
        log.info(f"  Brier: {brier_mean:.4f} ± {brier_std:.4f}")
        log.info(f"  AUC: {auc_mean:.4f}")

        # Check alerts
        alerts = []

        # 1. Brier degrades > 10% vs mean
        if brier > brier_mean * (1 + self.cfg.get("brier_degradation_threshold", 0.10)):
            alerts.append({
                "level": "critical",
                "msg": f"🚨 BRIER DEGRADA {((brier/brier_mean)-1)*100:.1f}% vs media ({brier:.4f} vs {brier_mean:.4f})"
            })

        # 2. Brier > mean + 2σ
        if brier > brier_mean + 2 * brier_std:
            alerts.append({
                "level": "warning",
                "msg": f"⚠️ BRIER > 2σ: {brier:.4f} (media {brier_mean:.4f}, std {brier_std:.4f})"
            })

        # 3. AUC collapses
        if auc < self.cfg.get("auc_critical", 0.52):
            alerts.append({
                "level": "critical",
                "msg": f"🚨 AUC CRÍTICO: {auc:.4f} (casi aleatorio)"
            })

        # 4. AUC degrades > 5%
        if auc < auc_mean * (1 - self.cfg.get("auc_degradation_threshold", 0.05)):
            alerts.append({
                "level": "warning",
                "msg": f"⚠️ AUC baja {((auc_mean/auc)-1)*100:.1f}% vs media ({auc:.4f} vs {auc_mean:.4f})"
            })

        # 5. Calibration fails
        if reliability > self.cfg.get("reliability_threshold", 0.05):
            alerts.append({
                "level": "warning",
                "msg": f"⚠️ CALIBRACIÓN: reliability={reliability:.4f} (esperado < {self.cfg.get('reliability_threshold', 0.05)})"
            })

        # 6. BTTS rate shift (data drift)
        current_btts = y.mean()
        hist_btts = recent["btts_rate"].mean()
        if abs(current_btts - hist_btts) > self.cfg.get("btts_rate_drift_threshold", 0.10):
            alerts.append({
                "level": "info",
                "msg": f"📊 DATA DRIFT: BTTS rate {current_btts:.1%} vs histórico {hist_btts:.1%}"
            })

        # Output alerts
        if alerts:
            log.info("\n" + "=" * 60)
            log.info("ALERTAS DETECTADAS:")
            for a in alerts:
                log.info(f"  {a['msg']}")
            log.info("=" * 60)

            # Send alerts to all configured channels
            msg = f"⚠️ DECAY DETECTOR — {datetime.now().strftime('%Y-%m-%d')}\n\n" + "\n".join(a["msg"] for a in alerts)
            msg += f"\n\nRolling: Brier={brier:.4f}, AUC={auc:.4f}, Rel={reliability:.4f}"
            msg += f"\nBaseline: Brier={brier_mean:.4f}±{brier_std:.4f}, AUC={auc_mean:.4f}"
            # Send as plain text to avoid HTML parsing issues with emojis
            self._send_telegram_alert(msg, parse_mode=None)
            self._send_whatsapp_alert(msg)
        else:
            log.info("✅ Sin alertas — modelo estable")

        # Save weekly snapshot
        snap_path = get_path("decay_snapshots")
        snap = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "period_start": start_date.date().isoformat(),
            "period_end": end_date.date().isoformat(),
            "n": len(test_df),
            "brier": brier,
            "auc": auc,
            "acc": acc,
            "reliability": reliability,
            "btts_rate": float(y.mean()),
            "alerts": [a["msg"] for a in alerts],
        }

        if snap_path.exists():
            snaps = pd.read_csv(snap_path)
            snaps = pd.concat([snaps, pd.DataFrame([snap])], ignore_index=True)
        else:
            snaps = pd.DataFrame([snap])
        snaps.to_csv(snap_path, index=False)
        log.info(f"Snapshot saved: {snap_path}")

        return {
            "status": "completed",
            "metrics": {"brier": brier, "auc": auc, "acc": acc, "reliability": reliability},
            "baseline": {"brier_mean": brier_mean, "brier_std": brier_std, "auc_mean": auc_mean},
            "alerts": alerts,
        }


if __name__ == "__main__":
    detector = DecayDetector()
    detector.run()