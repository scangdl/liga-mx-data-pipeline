#!/usr/bin/env python3
"""
blind_test.py — Test a ciegas del modelo BTTS en una jornada historica.

Protocolo (sellado antes de ver resultados):
  1. Se elige una JORNADA COMPLETA pasada (fecha de corte).
  2. El modelo se entrena SOLO con partidos anteriores al corte
     (features de esos partidos tampoco ven nada posterior).
  3. Se reconstruye el estado de equipos con partidos previos al corte
     y se predicen los 9 partidos de esa jornada, igual que en produccion
     (btts_jornada.csv). Las predicciones quedan SELLADAS (hash).
  4. Solo entonces se comparan contra los marcadores reales.

Sin leakage: ni entrenamiento ni features usan informacion posterior al corte.
"""
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
sys.path.insert(0, str(FUTBOL))

FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]

# ================= CONFIG =================
# Jornada 10 del Clausura 2026: partidos del 3-5 de abril de 2026.
# El corte es ANTES del primer partido: nada de esa jornada se uso para entrenar.
CUT = pd.Timestamp("2026-04-03")
JORNADA_LABEL = "J10 Clausura 2026 (3-5 abril 2026)"
OUT = FUTBOL / "predictions" / "blind_test_j10_clausura2026"
OUT.mkdir(exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ============================================================
# 1) Reconstruir features con SOLO partidos previos al corte
# ============================================================
feats = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)

train_pool = feats[(feats["btts"].notna()) & (feats["date"] < CUT)].copy()
log(f"[1] Partidos de entrenamiento disponibles: {len(train_pool)} "
    f"({train_pool['date'].min().date()} -> {train_pool['date'].max().date()})")

# la jornada objetivo: partidos reales de esas fechas (para predecir)
jornada = feats[(feats["date"] >= CUT) & (feats["date"] <= CUT + pd.Timedelta(days=3))].copy()
log(f"[1] Partidos objetivo ({JORNADA_LABEL}): {len(jornada)}")
assert len(jornada) >= 7, "jornada vacia o muy corta — revisar fechas"
# ocultar el resultado inmediatamente: lo guardamos aparte
truth = jornada[["date", "home", "away", "hs", "as_", "btts"]].copy()
X_j = jornada[FEATURES].copy()

# ============================================================
# 2) Entrenar modelos SOLO con datos < CUT
# ============================================================
X_all = train_pool[FEATURES]
y_all = train_pool["btts"].astype(int).values
means = X_all.mean()
X_all = X_all.fillna(means)

scaler = StandardScaler().fit(X_all)
Xs = scaler.transform(X_all)

log("[2] Entrenando XGBoost solo con datos previos al corte...")
xgb = XGBClassifier(
    n_estimators=350, max_depth=3, learning_rate=0.035,
    subsample=0.85, colsample_bytree=0.8, eval_metric="logloss",
    random_state=42, n_jobs=2,
)
xgb.fit(Xs, y_all)

log("[2] Entrenando Logistica (comparacion)...")
logr = LogisticRegression(max_iter=2000, C=0.6)
logr.fit(Xs, y_all)

# ============================================================
# 3) Predecir la jornada (estado pre-partido igual que produccion)
#    y SELLAR las predicciones antes de ver nada
# ============================================================
Xj = X_j.fillna(means)
Xj_s = scaler.transform(Xj)

p_xgb = xgb.predict_proba(Xj_s)[:, 1]
p_lr = logr.predict_proba(Xj_s)[:, 1]
p_base = float(y_all.mean())

sealed = pd.DataFrame({
    "date": jornada["date"].dt.date.astype(str),
    "home": jornada["home"], "away": jornada["away"],
    "p_btts_xgb": np.round(p_xgb, 4),
    "fair_odds": np.round(1.0 / p_xgb, 2),
    "p_btts_lr": np.round(p_lr, 4),
    "p_base_rate": np.round(p_base, 4),
})
sealed_path = OUT / "predicciones_selladas.csv"
sealed.to_csv(sealed_path, index=False)

h = hashlib.sha256(sealed.to_csv(index=False).encode()).hexdigest()[:16]
log(f"[3] PREDICCIONES SELLADAS -> {sealed_path}")
log(f"[3] SHA256 del archivo sellado: {h}")
log("[3] (a partir de aqui ya se consultan los resultados reales)")

# ============================================================
# 4) Verdad y evaluacion
# ============================================================
y_true = truth["btts"].astype(int).values
res = pd.DataFrame({
    "partido": truth["home"] + " vs " + truth["away"],
    "marcador": truth["hs"].astype(int).astype(str) + "-" + truth["as_"].astype(int).astype(str),
    "btts_real": y_true,
    "p_xgb": np.round(p_xgb, 4),
    "p_lr": np.round(p_lr, 4),
    "p_base": np.round(p_base, 4),
})
res["acierto_05_xgb"] = ((res["p_xgb"] >= 0.5).astype(int) == y_true).astype(int)
res.to_csv(OUT / "resultados.csv", index=False)

print()
print("=" * 78)
print(f"TEST A CIEGAS — {JORNADA_LABEL}")
print("=" * 78)
print(res.to_string(index=False))
print()
log(f"n = {len(y_true)} | BTTS reales: {y_true.sum()}/{len(y_true)} ({y_true.mean():.1%})")
log(f"BTTS rate train (hasta corte): {p_base:.1%}")
for name, p in [("XGB (modelo en produccion)", p_xgb), ("Logistica", p_lr), ("Base rate", np.full(len(y_true), p_base))]:
    log(f"{name:<28} Brier: {brier_score_loss(y_true, p):.4f} | LogLoss: {log_loss(y_true, p, labels=[0,1]):.4f}"
        + (f" | AUC: {roc_auc_score(y_true, p):.4f}" if len(set(y_true)) == 2 else " | AUC: n/a (1 clase)"))
log(f"Accuracy@0.5 XGB: {res['acierto_05_xgb'].mean():.1%} ({int(res['acierto_05_xgb'].sum())}/{len(res)})")
print()
log(f"Verdad usada (sellada antes): archivo {OUT / 'resultados.csv'}")
log(f"Archivo sellado: {sealed_path} (sha256 {h})")
