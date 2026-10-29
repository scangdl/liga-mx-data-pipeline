#!/usr/bin/env python3
"""
decay_detector.py — Detección semanal de decay del modelo.

Ejecuta: cron lunes 06:00 (antes de liquidar)
  /usr/bin/python3 /home/petyt/futbolmx/decay_detector.py

Compara métricas rolling 4 semanas vs baseline histórico.
Alerta por Telegram si:
  - Brier sube > 10% vs media 6 meses
  - AUC baja < 0.52 (casi aleatorio)
  - Calibración falla (reliability > 0.05)
"""
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
PRED = FUTBOL / "predictions"

def log(msg):
    print(msg, flush=True)


def send_telegram_alert(message: str):
    """Envía alerta via Telegram si está configurado."""
    # El bot ya está en el entorno, usar hermes o bot directo
    try:
        # Usar el bot de Hermes si está disponible
        import subprocess
        result = subprocess.run(
            ["hermes", "send", "telegram", message],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            log(f"Alerta enviada por Telegram")
        else:
            log(f"Fallo al enviar alerta: {result.stderr}")
    except Exception as e:
        log(f"Telegram no disponible: {e}")


def main():
    log("=" * 60)
    log(f"DECAY DETECTOR — {datetime.now().strftime('%Y-%m-%d')}")
    log("=" * 60)

    # Cargar modelo
    with open(DATA / "btts_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    version = bundle.get("version", "v2")
    if version == "v3_improved" or "ensemble_weights" in bundle:
        model = bundle["xgb_model"]
        scaler = bundle["scaler"]
        FEATS = bundle["features"]
        means = pd.Series(bundle["impute_means"])
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
        FEATS = bundle["features"]
        means = pd.Series(bundle["impute_means"])

    # Cargar features master
    from improve_btts_v2 import add_advanced_features
    df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = add_advanced_features(df)

    # Rolling 4 semanas (últimos 28 días con resultados)
    end_date = df[df["btts"].notna()]["date"].max()
    start_date = end_date - timedelta(days=28)
    test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

    if test_mask.sum() < 20:
        log(f"Pocos partidos recientes ({test_mask.sum()}), extendiendo a 60 días...")
        start_date = end_date - timedelta(days=60)
        test_mask = (df["date"] >= start_date) & (df["date"] <= end_date) & df["btts"].notna()

    test_df = df[test_mask].copy()
    log(f"Periodo rolling: {start_date.date()} - {end_date.date()} | {len(test_df)} partidos")

    if len(test_df) < 20:
        log("Insuficientes datos para evaluar decay")
        return

    X = test_df[FEATS].fillna(means)
    y = test_df["btts"].astype(int).values
    Xs = scaler.transform(X) if scaler is not None else X.values
    p = model.predict_proba(Xs)[:, 1]

    # Métricas rolling
    brier = brier_score_loss(y, p)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else 0.5
    acc = ((p >= 0.5).astype(int) == y).mean()

    # Calibración (reliability diagram simplificado)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(p, bins) - 1
    reliability = 0.0
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() > 0:
            avg_pred = p[mask].mean()
            avg_true = y[mask].mean()
            reliability += mask.sum() / len(p) * abs(avg_pred - avg_true)

    log(f"Rolling 4 semanas: Brier={brier:.4f}, AUC={auc:.4f}, Acc={acc:.1%}, Reliability={reliability:.4f}")

    # Cargar histórico
    hist_file = PRED / "monthly_backtest_history.csv"
    if not hist_file.exists():
        log("Sin historial mensual, saltando comparación")
        return

    hist = pd.read_csv(hist_file)
    if len(hist) < 3:
        log("Historial insuficiente (<3 meses)")
        return

    # Baseline: últimos 6 meses (o todos si menos)
    recent = hist.tail(6)
    brier_mean = recent["brier"].mean()
    brier_std = recent["brier"].std()
    auc_mean = recent["auc"].mean()

    log(f"\nBaseline (últimos {len(recent)} meses):")
    log(f"  Brier: {brier_mean:.4f} ± {brier_std:.4f}")
    log(f"  AUC: {auc_mean:.4f}")

    # Alertas
    alerts = []

    # 1. Brier degrada > 10% vs media
    if brier > brier_mean * 1.10:
        alerts.append(f"🚨 BRIER DEGRADA {((brier/brier_mean)-1)*100:.1f}% vs media ({brier:.4f} vs {brier_mean:.4f})")

    # 2. Brier > media + 2 std
    if brier > brier_mean + 2 * brier_std:
        alerts.append(f"⚠️ BRIER > 2σ: {brier:.4f} (media {brier_mean:.4f}, std {brier_std:.4f})")

    # 3. AUC colapsa
    if auc < 0.52:
        alerts.append(f"🚨 AUC CRÍTICO: {auc:.4f} (casi aleatorio)")

    # 4. AUC degrada > 5%
    if auc < auc_mean * 0.95:
        alerts.append(f"⚠️ AUC baja {((auc_mean/auc)-1)*100:.1f}% vs media ({auc:.4f} vs {auc_mean:.4f})")

    # 5. Calibración falla
    if reliability > 0.05:
        alerts.append(f"⚠️ CALIBRACIÓN: reliability={reliability:.4f} (esperado < 0.05)")

    # 6. BTTS rate shift (data drift)
    current_btts = y.mean()
    hist_btts = recent["btts_rate"].mean()
    if abs(current_btts - hist_btts) > 0.10:
        alerts.append(f"📊 DATA DRIFT: BTTS rate {current_btts:.1%} vs histórico {hist_btts:.1%}")

    if alerts:
        msg = f"⚠️ DECAY DETECTOR — {datetime.now().strftime('%Y-%m-%d')}\n\n" + "\n".join(alerts)
        msg += f"\n\nRolling: Brier={brier:.4f}, AUC={auc:.4f}, Rel={reliability:.4f}"
        msg += f"\nBaseline: Brier={brier_mean:.4f}±{brier_std:.4f}, AUC={auc_mean:.4f}"
        log("\n" + "=" * 60)
        log("ALERTAS DETECTADAS:")
        for a in alerts:
            log(f"  {a}")
        log("=" * 60)

        # Enviar por Telegram
        send_telegram_alert(msg)
    else:
        log("✅ Sin alertas — modelo estable")

    # Guardar snapshot semanal
    snap_file = PRED / "decay_snapshots.csv"
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
    }
    if snap_file.exists():
        snaps = pd.read_csv(snap_file)
        snaps = pd.concat([snaps, pd.DataFrame([snap])], ignore_index=True)
    else:
        snaps = pd.DataFrame([snap])
    snaps.to_csv(snap_file, index=False)
    log(f"Snapshot guardado: {snap_file}")


if __name__ == "__main__":
    main()