#!/usr/bin/env python3
"""
blind_test_ape24.py — Test a ciegas jornada por jornada del Apertura 2024.

Para cada fecha única (jornada):
  1. Corte = día anterior al primer partido de esa fecha
  2. Entrena modelo SOLO con partidos < corte
  3. Predice los partidos de esa fecha (estado previo al corte)
  4. Sella predicciones (hash) y compara contra realidad
"""
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
OUT = FUTBOL / "predictions" / "blind_test_ape24_jornadas"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]

XGB_PARAMS = dict(
    n_estimators=350, max_depth=3, learning_rate=0.035,
    subsample=0.85, colsample_bytree=0.8, eval_metric="logloss",
    random_state=42, n_jobs=2,
)

LOGREG_PARAMS = dict(max_iter=2000, C=0.6, random_state=42)

def log(msg):
    print(msg, flush=True)

# ============================================================
# Cargar features y master
# ============================================================
feats = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
master = pd.read_parquet(DATA / "liga_mx_master.parquet")
master['date'] = pd.to_datetime(master['date'])

# Fechas únicas del Apertura 2024 (ordenadas)
ape24_dates = master[(master['torneo']=='Apertura') & (master['anio_torneo']==2024)]['date'].sort_values().unique()
ape24_dates = pd.Series(ape24_dates).dt.date.unique()
log(f"Fechas (jornadas) en Apertura 2024: {len(ape24_dates)}")

all_results = []

for matchday_idx, matchday_date in enumerate(ape24_dates):
    matchday_date = pd.Timestamp(matchday_date)
    cut = matchday_date - pd.Timedelta(days=1)  # corte = día anterior
    
    log(f"\n{'='*60}")
    log(f"JORNADA {matchday_idx+1}/{len(ape24_dates)} — {matchday_date.date()} (corte: {cut.date()})")
    log(f"{'='*60}")
    
    # Partidos de esta jornada
    jornada = feats[feats['date'] == matchday_date].copy()
    if len(jornada) == 0:
        log("  Sin partidos en features para esta fecha, saltando...")
        continue
    
    # Entrenamiento: solo partidos < corte
    train_pool = feats[(feats['btts'].notna()) & (feats['date'] < cut)].copy()
    if len(train_pool) < 100:
        log(f"  Muy pocos datos de entrenamiento ({len(train_pool)}), saltando...")
        continue
    
    log(f"  Train: {len(train_pool)} partidos ({train_pool['date'].min().date()} -> {train_pool['date'].max().date()})")
    log(f"  Test:  {len(jornada)} partidos")
    
    # Ocultar verdad
    truth = jornada[['date', 'home', 'away', 'hs', 'as_', 'btts']].copy()
    X_j = jornada[FEATURES].copy()
    y_true = truth['btts'].astype(int).values
    
    # Entrenar
    X_all = train_pool[FEATURES]
    y_all = train_pool['btts'].astype(int).values
    means = X_all.mean()
    X_all_filled = X_all.fillna(means)
    
    scaler = StandardScaler().fit(X_all_filled)
    Xs = scaler.transform(X_all_filled)
    
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(Xs, y_all)
    
    logr = LogisticRegression(**LOGREG_PARAMS)
    logr.fit(Xs, y_all)
    
    p_base = float(y_all.mean())
    
    # Predecir jornada
    Xj = X_j.fillna(means)
    Xj_s = scaler.transform(Xj)
    
    p_xgb = xgb.predict_proba(Xj_s)[:, 1]
    p_lr = logr.predict_proba(Xj_s)[:, 1]
    
    # Sellado
    sealed = pd.DataFrame({
        "date": jornada["date"].dt.date.astype(str),
        "home": jornada["home"], "away": jornada["away"],
        "p_btts_xgb": np.round(p_xgb, 4),
        "fair_odds": np.round(1.0 / p_xgb, 2),
        "p_btts_lr": np.round(p_lr, 4),
        "p_base_rate": np.round(p_base, 4),
    })
    sealed_path = OUT / f"predicciones_j{matchday_idx+1:02d}_{matchday_date.date()}.csv"
    sealed.to_csv(sealed_path, index=False)
    
    h = hashlib.sha256(sealed.to_csv(index=False).encode()).hexdigest()[:16]
    log(f"  Sellado: {sealed_path.name} (sha256 {h})")
    
    # Evaluación
    res = pd.DataFrame({
        "partido": truth["home"] + " vs " + truth["away"],
        "marcador": truth["hs"].astype(int).astype(str) + "-" + truth["as_"].astype(int).astype(str),
        "btts_real": y_true,
        "p_xgb": np.round(p_xgb, 4),
        "p_lr": np.round(p_lr, 4),
        "p_base": np.round(p_base, 4),
    })
    res["acierto_05_xgb"] = ((res["p_xgb"] >= 0.5).astype(int) == y_true).astype(int)
    res["acierto_05_lr"] = ((res["p_lr"] >= 0.5).astype(int) == y_true).astype(int)
    res_path = OUT / f"resultados_j{matchday_idx+1:02d}_{matchday_date.date()}.csv"
    res.to_csv(res_path, index=False)
    
    # Métricas
    n = len(y_true)
    btts_real = y_true.sum()
    acc_xgb = res['acierto_05_xgb'].mean()
    acc_lr = res['acierto_05_lr'].mean()
    brier_xgb = brier_score_loss(y_true, p_xgb)
    brier_lr = brier_score_loss(y_true, p_lr)
    brier_base = brier_score_loss(y_true, np.full(n, p_base))
    
    log(f"  n={n} | BTTS reales: {btts_real}/{n} ({btts_real/n:.1%}) | Base rate train: {p_base:.1%}")
    log(f"  XGB  — Acc@0.5: {acc_xgb:.1%} | Brier: {brier_xgb:.4f}")
    log(f"  LogR — Acc@0.5: {acc_lr:.1%} | Brier: {brier_lr:.4f}")
    log(f"  Base — Brier: {brier_base:.4f}")
    
    # Guardar para resumen final
    all_results.append({
        "jornada": matchday_idx + 1,
        "fecha": str(matchday_date.date()),
        "n_partidos": n,
        "btts_reales": int(btts_real),
        "p_base_train": p_base,
        "acc_xgb": acc_xgb,
        "acc_lr": acc_lr,
        "brier_xgb": brier_xgb,
        "brier_lr": brier_lr,
        "brier_base": brier_base,
        "sha256": h,
    })

# ============================================================
# Resumen global
# ============================================================
log("\n" + "="*70)
log("RESUMEN GLOBAL — APERTURA 2024 (jornada a jornada)")
log("="*70)

summary = pd.DataFrame(all_results)
print(summary.to_string(index=False))

# Agregados
total_n = summary['n_partidos'].sum()
total_btts = summary['btts_reales'].sum()
avg_acc_xgb = np.average(summary['acc_xgb'], weights=summary['n_partidos'])
avg_acc_lr = np.average(summary['acc_lr'], weights=summary['n_partidos'])
avg_brier_xgb = np.average(summary['brier_xgb'], weights=summary['n_partidos'])
avg_brier_lr = np.average(summary['brier_lr'], weights=summary['n_partidos'])
avg_brier_base = np.average(summary['brier_base'], weights=summary['n_partidos'])

log(f"\nTOTAL partidos: {total_n} | BTTS reales: {total_btts} ({total_btts/total_n:.1%})")
log(f"XGB  — Acc@0.5 ponderado: {avg_acc_xgb:.1%} | Brier ponderado: {avg_brier_xgb:.4f}")
log(f"LogR — Acc@0.5 ponderado: {avg_acc_lr:.1%} | Brier ponderado: {avg_brier_lr:.4f}")
log(f"Base — Brier ponderado: {avg_brier_base:.4f}")

# Guardar resumen
summary.to_csv(OUT / "resumen_global.csv", index=False)
log(f"\nResumen guardado en: {OUT / 'resumen_global.csv'}")