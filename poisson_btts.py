#!/usr/bin/env python3
"""
poisson_btts.py — Modelo Poisson de goles -> probabilidad BTTS.

Enfoque clasico:
  1. Ratings ataque/defensa por equipo con ventana movil (solo partidos previos)
  2. lambda_h = ataque_local(home) * defensa_visita(away) * media_goles_local_liga
     lambda_a = ataque_visita(away) * defensa_local(home) * media_goles_visita_liga
  3. P(BTTS) = (1 - Poisson(0|lambda_h)) * (1 - Poisson(0|lambda_a))

Tambien agrega lambda_h/lambda_a como features al XGBoost.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

WINDOW = 15  # partidos previos por equipo


def log(msg):
    print(msg, flush=True)


df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)

# ============================================
# Ratings ataque/defensa (separando localia)
# ============================================
# Por equipo acumulamos EWMA (decaimiento exponencial) de goles, separando localia
# alpha = 0.93 -> vida media ~9.5 partidos (forma reciente pesa mas)
ALPHA = 0.93
stats = {}


def update_ewma(team, role, gf, ga):
    s = stats.setdefault(team, {}).setdefault(role, {"atk": None, "def": None})
    s["atk"] = gf if s["atk"] is None else ALPHA * s["atk"] + (1 - ALPHA) * gf
    s["def"] = ga if s["def"] is None else ALPHA * s["def"] + (1 - ALPHA) * ga


def team_attack_defense(team, role):
    """role: 1=local, 0=visita. Devuelve (attack, defense) EWMA."""
    s = stats.get(team, {}).get(role)
    if not s or s["atk"] is None:
        return (np.nan, np.nan)
    return (s["atk"], s["def"])


lam_h_list, lam_a_list = [], []
league = {"gf_home": [], "gf_away": []}  # medias de liga rolling

rows_hist = []
for _, r in df.iterrows():
    h, a = r["home"], r["away"]
    ah, dh = team_attack_defense(h, 1)   # home como local
    aa, da = team_attack_defense(a, 0)   # away como visita
    # medias de liga hasta ahora (todas las filas previas)
    if league["gf_home"]:
        lgf_h = np.mean(league["gf_home"][-800:])
        lgf_a = np.mean(league["gf_away"][-800:])
    else:
        lgf_h, lgf_a = np.nan, np.nan

    if all(pd.notna(x) for x in (ah, dh, aa, da, lgf_h, lgf_a)):
        # modelo multiplicativo estandar con EWMA:
        lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
        lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
    else:
        lam_h, lam_a = np.nan, np.nan
    lam_h_list.append(lam_h)
    lam_a_list.append(lam_a)

    # actualizar stats con el partido actual
    if pd.notna(r["hs"]) and pd.notna(r["as_"]):
        update_ewma(h, 1, r["hs"], r["as_"])
        update_ewma(a, 0, r["as_"], r["hs"])
        league["gf_home"].append(r["hs"])
        league["gf_away"].append(r["as_"])

df["lam_h"] = lam_h_list
df["lam_a"] = lam_a_list

# P(BTTS) Poisson
mask = df["lam_h"].notna() & df["lam_a"].notna()
df["p_btts_poisson"] = np.nan
df.loc[mask, "p_btts_poisson"] = (
    (1 - poisson.pmf(0, df.loc[mask, "lam_h"])) *
    (1 - poisson.pmf(0, df.loc[mask, "lam_a"]))
)
log(f"Lambda estimada para {mask.sum()}/{len(df)} partidos")
log(f"lam_h media: {df['lam_h'].mean():.2f} | lam_a media: {df['lam_a'].mean():.2f} (real: {df['hs'].mean():.2f} / {df['as_'].mean():.2f})")

# ============================================
# Evaluacion en el MISMO split temporal
# ============================================
cut = pd.Timestamp("2024-07-01")
pool = df[df["btts"].notna() & df["p_btts_poisson"].notna()].copy()
is_test = pool["date"] >= cut
y_test = pool.loc[is_test, "btts"].astype(int).values
p_pois = pool.loc[is_test, "p_btts_poisson"].values

log(f"\nTest Poisson: {len(y_test)} partidos")
log(f"  poisson  AUC={roc_auc_score(y_test, p_pois):.4f}  Brier={brier_score_loss(y_test, p_pois):.4f}  LogLoss={log_loss(y_test, p_pois, labels=[0,1]):.4f}")

# ============================================
# XGBoost + lambdas
# ============================================
FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
    "lam_h", "lam_a",
]
X = pool[FEATURES]
y = pool["btts"].astype(int).values
X_train, y_train = X[~is_test], y[~is_test]
X_test, y_test2 = X[is_test], y[is_test]
means = X_train.mean()
X_train, X_test = X_train.fillna(means), X_test.fillna(means)

xgb2 = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    early_stopping_rounds=40, random_state=42,
)
xgb2.fit(X_train, y_train, eval_set=[(X_test, y_test2)], verbose=False)
p_xgb2 = xgb2.predict_proba(X_test)[:, 1]
log(f"  xgb+lamb AUC={roc_auc_score(y_test2, p_xgb2):.4f}  Brier={brier_score_loss(y_test2, p_xgb2):.4f}  LogLoss={log_loss(y_test2, p_xgb2, labels=[0,1]):.4f}")

imp = pd.Series(xgb2.feature_importances_, index=FEATURES).sort_values(ascending=False)
log("\nTop 10 features XGB+lambdas:")
log(imp.head(10).round(4).to_string())

# ============================================
# Ensemble simple: promedio poisson + xgb
# ============================================
p_ens = 0.5 * p_pois + 0.5 * p_xgb2
log(f"  ensemble AUC={roc_auc_score(y_test2, p_ens):.4f}  Brier={brier_score_loss(y_test2, p_ens):.4f}  LogLoss={log_loss(y_test2, p_ens, labels=[0,1]):.4f}")

# ============================================
# Calibracion del mejor modelo
# ============================================
best_name, best_p = ("ensemble", p_ens) if brier_score_loss(y_test2, p_ens) <= min(brier_score_loss(y_test2, p_xgb2), brier_score_loss(y_test, p_pois)) else (
    ("xgb+lamb", p_xgb2) if brier_score_loss(y_test2, p_xgb2) <= brier_score_loss(y_test, p_pois) else ("poisson", p_pois))
log(f"\nMejor modelo: {best_name}")

bins = pd.cut(pd.Series(best_p), bins=np.linspace(0, 1, 11))
cal = pd.DataFrame({"p": best_p, "y": y_test2, "bin": bins}).groupby("bin", observed=True).agg(
    n=("y", "size"), p_mean=("p", "mean"), rate=("y", "mean"))
log("\nCalibracion (deciles de probabilidad predicha):")
log(cal.round(3).to_string())

# ============================================
# Guardar
# ============================================
df.to_parquet(DATA / "liga_mx_features.parquet", index=False)
log(f"\nFeatures con lambdas guardadas: {DATA / 'liga_mx_features.parquet'}")
