#!/usr/bin/env python3
"""
monthly_backtest.py — Backtest mensual automatizado para detectar decay del modelo.

Ejecuta: cron 1ro de cada mes 06:00
  /usr/bin/python3 /home/petyt/futbolmx/monthly_backtest.py

Salida: predictions/monthly_backtest_YYYY-MM.csv + alerta si Brier degrada > 5%
"""
import json
import pickle
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
PRED = FUTBOL / "predictions"
PRED.mkdir(exist_ok=True)

def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 60)
    log(f"BACKTEST MENSUAL — {datetime.now().strftime('%Y-%m-%d')}")
    log("=" * 60)

    # 1. Cargar modelo actual
    with open(DATA / "btts_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    version = bundle.get("version", "v2")
    if version == "v3_improved" or "ensemble_weights" in bundle:
        model = bundle["xgb_model"]  # primary
        scaler = bundle["scaler"]
        FEATS = bundle["features"]
        means = pd.Series(bundle["impute_means"])
        log(f"Modelo: {version} (ensemble)")
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
        FEATS = bundle["features"]
        means = pd.Series(bundle["impute_means"])
        log(f"Modelo: {version}")

    # 2. Cargar master y features
    df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Último mes completo
    end_date = df["date"].max()
    start_date = end_date - timedelta(days=30)
    test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

    if test_mask.sum() < 10:
        log(f"Pocos partidos en último mes ({test_mask.sum()}), extendiendo a 60 días...")
        start_date = end_date - timedelta(days=60)
        test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

    test_df = df[test_mask].copy()
    log(f"Periodo: {start_date.date()} - {end_date.date()} | Partidos: {len(test_df)}")

    # 3. Preparar features (solo las que espera el modelo)
    from improve_btts_v2 import add_advanced_features
    test_df = add_advanced_features(test_df)
    X = test_df[FEATS].copy()
    y = test_df["btts"].astype(int).values

    # Imputar con medias de train
    X = X.fillna(means)
    Xs = scaler.transform(X) if scaler is not None else X.values

    # 4. Predicciones
    p = model.predict_proba(Xs)[:, 1]

    # 5. Métricas
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else 0.5
    acc = ((p >= 0.5).astype(int) == y).mean()
    btts_rate = y.mean()

    # Baseline
    base_rate = df[df["date"] < start_date]["btts"].mean()
    base_brier = brier_score_loss(y, np.full(len(y), base_rate))

    log(f"\nMétricas último mes:")
    log(f"  BTTS rate: {btts_rate:.1%}")
    log(f"  Model Brier: {brier:.4f} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Acc@0.5: {acc:.1%}")
    log(f"  Base rate (hist): {base_rate:.1%} | Base Brier: {base_brier:.4f}")
    log(f"  Mejora vs base: {base_brier - brier:+.4f} Brier ({((base_brier - brier)/base_brier)*100:+.1f}%)")

    # 6. Comparar con histórico (últimos 6 meses)
    hist_file = PRED / "monthly_backtest_history.csv"
    if hist_file.exists():
        hist = pd.read_csv(hist_file)
        log(f"\nHistórico ({len(hist)} meses):")
        log(f"  Brier medio: {hist['brier'].mean():.4f} ± {hist['brier'].std():.4f}")
        log(f"  AUC medio: {hist['auc'].mean():.4f}")
        # Alerta si degrada > 5% vs media histórica
        if brier > hist['brier'].mean() * 1.05:
            log("  ⚠️ ALERTA: Brier degrada >5% vs media histórica!")
        if brier > hist['brier'].mean() * 1.10:
            log("  🚨 ALERTA CRÍTICA: Brier degrada >10% — considerar reentreno")

    # 7. Guardar registro
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

    if hist_file.exists():
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])
    hist.to_csv(hist_file, index=False)
    log(f"\nHistorial actualizado: {hist_file}")

    # 8. Detalle por partido (para auditoría)
    detail = test_df[["date", "home", "away", "btts"]].copy()
    detail["p_model"] = np.round(p, 4)
    detail["correct_05"] = ((detail["p_model"] >= 0.5).astype(int) == detail["btts"]).astype(int)
    detail_file = PRED / f"monthly_backtest_{datetime.now().strftime('%Y-%m')}.csv"
    detail.to_csv(detail_file, index=False, encoding="utf-8-sig")
    log(f"Detalle: {detail_file}")

    log("=" * 60)
    log("BACKTEST MENSUAL COMPLETADO")
    log("=" * 60)


if __name__ == "__main__":
    main()