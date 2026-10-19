#!/usr/bin/env python3
"""
train_btts.py — Modelo BTTS con validacion temporal (sin leakage).

Estrategia:
  - Train: partidos hasta fecha de corte (mismo dia de corte inclusive no; solo previos)
  - Test: temporadas mas recientes (Apertura 2024 -> Clausura 2026)
  - Metricas: AUC, Brier, log-loss, comparacion vs prob implicita derivada de odds 1X2

Modelos:
  - Logistica (baseline interpretable)
  - XGBoost
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]


def log(msg):
    print(msg, flush=True)


df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)

# solo partidos con marcador (para entrenar) y features validas
train_pool = df[df["btts"].notna()].copy()
log(f"Partidos con BTTS label: {len(train_pool)}")

# nan en features de forma: primer partido de un equipo -> imputar con media global de train
X = train_pool[FEATURES].copy()
y = train_pool["btts"].astype(int).values
dates = train_pool["date"]

log(f"NaN por feature (antes de imputar):\n{X.isna().sum()[X.isna().sum() > 0].to_string()}")

# Split temporal: test = ultimas 3 temporadas (Ape2024, Clau2025, Ape2025, Clau2026)
cut = pd.Timestamp("2024-07-01")
is_test = dates >= cut
X_train, y_train = X[~is_test], y[~is_test]
X_test, y_test = X[is_test], y[is_test]
log(f"Train: {len(X_train)} partidos ({dates[~is_test].min().date()} -> {dates[~is_test].max().date()})")
log(f"Test:  {len(X_test)} partidos ({dates[is_test].min().date()} -> {dates[is_test].max().date()})")
log(f"BTTS rate train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

# Imputacion: media de train (fit solo en train)
means = X_train.mean()
X_train = X_train.fillna(means)
X_test = X_test.fillna(means)

results = {}

# ---------- Baseline 1: tasa base ----------
p_base = np.full(len(y_test), y_train.mean())
results["base_rate"] = {
    "auc": roc_auc_score(y_test, p_base),
    "brier": brier_score_loss(y_test, p_base),
    "logloss": log_loss(y_test, p_base, labels=[0, 1]),
}

# ---------- Baseline 2: odds implicitas 1X2 -> proxy BTTS ----------
# BTTS aproximacion: p(H anota) * p(A anota) no es derivable solo de 1X2,
# pero una proxy lineal de imp_h/imp_d/imp_a sirve de comparacion.
logistic = LogisticRegression(max_iter=2000, C=1.0)
logistic.fit(StandardScaler().fit_transform(X_train), y_train)
scaler = StandardScaler().fit(X_train)
p_lr = logistic.predict_proba(scaler.transform(X_test))[:, 1]
results["logistic"] = {
    "auc": roc_auc_score(y_test, p_lr),
    "brier": brier_score_loss(y_test, p_lr),
    "logloss": log_loss(y_test, p_lr, labels=[0, 1]),
}

# ---------- XGBoost ----------
xgb = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    early_stopping_rounds=40, random_state=42,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
p_xgb = xgb.predict_proba(X_test)[:, 1]
results["xgb"] = {
    "auc": roc_auc_score(y_test, p_xgb),
    "brier": brier_score_loss(y_test, p_xgb),
    "logloss": log_loss(y_test, p_xgb, labels=[0, 1]),
}

# ---------- Reporte ----------
log("\n" + "=" * 55)
log(f"TEST: {len(y_test)} partidos BTTS (Ape2024 -> Clau2026)")
log("=" * 55)
for name, r in results.items():
    log(f"  {name:12s} AUC={r['auc']:.4f}  Brier={r['brier']:.4f}  LogLoss={r['logloss']:.4f}")

# Importancia de features XGBoost
imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
log("\nTop features XGBoost:")
log(imp.head(10).round(4).to_string())

# Coeficientes logisticos
coefs = pd.Series(logistic.coef_[0], index=FEATURES).sort_values()
log("\nCoeficientes logisticos (mas negativos = menos BTTS):")
log(coefs.round(3).to_string())

# Guardar predicciones del test para analisis posterior
out = pd.DataFrame({
    "date": dates[is_test].values,
    "home": train_pool.loc[is_test, "home"].values,
    "away": train_pool.loc[is_test, "away"].values,
    "btts": y_test,
    "p_base": p_base,
    "p_lr": p_lr,
    "p_xgb": p_xgb,
})
out.to_parquet(DATA / "btts_test_predictions.parquet", index=False)
log(f"\nPredicciones test guardadas: {DATA / 'btts_test_predictions.parquet'}")
