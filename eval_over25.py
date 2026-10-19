#!/usr/bin/env python3
"""
eval_over25.py — Walk-forward para Over 2.5 goles.

Configs:
  base_rate, poisson (cola Poisson con lambdas EWMA),
  xgb_plain, xgb_lam (BASE+lam), xgb_lam_p (+ p_over_poisson)
Ganador = menor Brier promedio en 4 folds temporales.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

BASE = [
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

# p_over_poisson: suma de Poissons independientes es Poisson(lam_h+lam_a)
m = df["lam_h"].notna() & df["lam_a"].notna()
df["p_over_poisson"] = np.nan
df.loc[m, "p_over_poisson"] = 1 - poisson.cdf(2, df.loc[m, "lam_h"] + df.loc[m, "lam_a"])

pool = df[df["over25"].notna()].copy()
log(f"Pool: {len(pool)} partidos con etiqueta O2.5 | rate={pool['over25'].mean():.3f}")
log(f"  con lambda: {pool['lam_h'].notna().sum()} | con odds: {pool['has_odds'].sum()}")

cuts = [
    ("Ape2022-Clau2023", pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-01")),
    ("Ape2023-Clau2024", pd.Timestamp("2023-07-01"), pd.Timestamp("2024-07-01")),
    ("Ape2024-Clau2025", pd.Timestamp("2024-07-01"), pd.Timestamp("2025-07-01")),
    ("Ape2025-Clau2026", pd.Timestamp("2025-07-01"), pd.Timestamp("2026-07-01")),
]


def xgb_fit_predict(Xtr, ytr, Xte, yte):
    mod = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        early_stopping_rounds=40, random_state=42)
    mod.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    return mod.predict_proba(Xte)[:, 1]


configs = ["base_rate", "poisson", "xgb_plain", "xgb_lam", "xgb_lam_p"]
met = {c: {"brier": [], "auc": []} for c in configs}

for fname, start, end in cuts:
    te = (pool["date"] >= start) & (pool["date"] < end)
    tr = pool["date"] < start
    if te.sum() == 0 or tr.sum() == 0:
        continue
    ytr = pool.loc[tr, "over25"].astype(int).values
    yte = pool.loc[te, "over25"].astype(int).values
    base_rate = ytr.mean()

    met["base_rate"]["brier"].append(brier_score_loss(yte, np.full(len(yte), base_rate)))
    met["base_rate"]["auc"].append(roc_auc_score(yte, np.full(len(yte), base_rate)))

    # poisson standalone (donde haya lambda)
    pm = te & pool["p_over_poisson"].notna()
    if pm.sum():
        pp = pool.loc[pm, "p_over_poisson"].values
        yy = pool.loc[pm, "over25"].astype(int).values
        met["poisson"]["brier"].append(brier_score_loss(yy, pp))
        met["poisson"]["auc"].append(roc_auc_score(yy, pp))

    for name, feats in [
        ("xgb_plain", BASE),
        ("xgb_lam", BASE + ["lam_h", "lam_a"]),
        ("xgb_lam_p", BASE + ["lam_h", "lam_a", "p_over_poisson"]),
    ]:
        dtr, dte = pool.loc[tr], pool.loc[te]
        Xtr, Xte = dtr[feats], dte[feats]
        mean = Xtr.mean()
        p = xgb_fit_predict(Xtr.fillna(mean), ytr, Xte.fillna(mean), yte)
        met[name]["brier"].append(brier_score_loss(yte, p))
        met[name]["auc"].append(roc_auc_score(yte, p))

    log(f"fold {fname}: n={te.sum()} | base={met['base_rate']['brier'][-1]:.4f} "
        f"poisson={met['poisson']['brier'][-1] if met['poisson']['brier'] else float('nan'):.4f} "
        f"plain={met['xgb_plain']['brier'][-1]:.4f} lam={met['xgb_lam']['brier'][-1]:.4f} "
        f"lam_p={met['xgb_lam_p']['brier'][-1]:.4f}")

log("\n" + "=" * 60)
rows = [{"config": c, "brier": np.mean(met[c]["brier"]), "brier_std": np.std(met[c]["brier"]),
         "auc": np.mean(met[c]["auc"])} for c in configs]
res = pd.DataFrame(rows).sort_values("brier")
log(res.round(4).to_string(index=False))
log(f"\nGANADOR: {res.iloc[0]['config']}")

# Entrenar modelo final con TODO lo jugado y guardar
winner = res.iloc[0]["config"]
feat_map = {"xgb_plain": BASE, "xgb_lam": BASE + ["lam_h", "lam_a"],
            "xgb_lam_p": BASE + ["lam_h", "lam_a", "p_over_poisson"],
            "poisson": BASE, "base_rate": BASE}
import pickle
FEATS_W = feat_map[winner]
NEEDS_LAM = winner in ("xgb_lam", "xgb_lam_p")
if NEEDS_LAM:
    tr_all = pool[pool["lam_h"].notna()].copy()
else:
    tr_all = pool.copy()
mean = tr_all[FEATS_W].mean()
Xa = tr_all[FEATS_W].fillna(mean)
ya = tr_all["over25"].astype(int).values
final = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                       random_state=42)
final.fit(Xa, ya)
with open(DATA / "over25_model.pkl", "wb") as f:
    pickle.dump({"model": final, "features": FEATS_W, "means": mean.to_dict(),
                 "winner": winner, "cv": res.round(4).to_dict()}, f)
log(f"Modelo final O2.5 ({winner}) entrenado con {len(tr_all)} partidos -> data/over25_model.pkl")
