#!/usr/bin/env python3
"""
add_altitude.py — Completa altitud por estadio local de cada equipo y
evalua si agregarla (+ dif de altitud) mejora el modelo en walk-forward.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"


def log(msg):
    print(msg, flush=True)


df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)

# 1. Mapeo estadio -> altitud (del parquet unificado viejo)
old = pd.read_parquet("/home/petyt/liga_mx_data/liga_mx_unified_2018_2025.parquet")
stadium_alt = old.dropna(subset=["stadium"]).drop_duplicates("stadium").set_index("stadium")["altitude"].to_dict()

# 2. Estadio local por equipo: la venue mas frecuente en partidos de local
alt_by_team = {}
with_alt = df.dropna(subset=["venue"]).copy()
with_alt["alt"] = with_alt["venue"].map(stadium_alt)
with_alt = with_alt.dropna(subset=["alt"])
home_venue = with_alt.groupby(["home", "venue", "alt"]).size().reset_index(name="n")
# por equipo: quedarse con la venue mas frecuente
top_venue = home_venue.sort_values("n", ascending=False).groupby("home").head(1)
alt_by_team = dict(zip(top_venue["home"], top_venue["alt"]))
log(f"Equipos con altitud de sede: {len(alt_by_team)}")
log(str(dict(sorted(alt_by_team.items(), key=lambda x: -x[1])[:6])))

# 3. Asignar al dataset: altitud del local y del visitante (sede del partido = sede del local)
df["alt_home"] = df["home"].map(alt_by_team)
df["alt_away"] = df["away"].map(alt_by_team)
df["alt_venue"] = df["alt_home"]  # el partido se juega en casa del local
df["alt_diff"] = (df["alt_home"] - df["alt_away"]).fillna(0)
log(f"Cobertura altitud: {df['alt_venue'].notna().sum()}/{len(df)}")

# ============================================
# 4. Walk-forward: xgb_ewma ± altitud
# ============================================
BASE = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]
LAM = ["lam_h", "lam_a"]  # ya guardadas (EWMA de la corrida anterior)

pool = df[df["btts"].notna() & df["lam_h"].notna()].copy()
cuts = [
    pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-01"),
    pd.Timestamp("2024-07-01"), pd.Timestamp("2025-07-01"),
]

def run_cv(feats):
    out = {"brier": [], "auc": []}
    for start in cuts:
        te = (pool["date"] >= start) & (pool["date"] < start + pd.Timedelta(days=365))
        tr = pool["date"] < start
        ytr = pool.loc[tr, "btts"].astype(int).values
        yte = pool.loc[te, "btts"].astype(int).values
        Xtr, Xte = pool.loc[tr, feats], pool.loc[te, feats]
        m = Xtr.mean()
        model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              early_stopping_rounds=40, random_state=42)
        model.fit(Xtr.fillna(m), ytr, eval_set=[(Xte.fillna(m), yte)], verbose=False)
        p = model.predict_proba(Xte.fillna(m))[:, 1]
        out["brier"].append(brier_score_loss(yte, p))
        out["auc"].append(roc_auc_score(yte, p))
    return np.mean(out["brier"]), np.mean(out["auc"])

b1, a1 = run_cv(BASE + LAM)
b2, a2 = run_cv(BASE + LAM + ["alt_venue", "alt_diff"])
log(f"\nxgb_ewma sin altitud : Brier={b1:.4f} AUC={a1:.4f}")
log(f"xgb_ewma con altitud : Brier={b2:.4f} AUC={a2:.4f}")
if b2 < b1:
    log("-> altitud AYUDA, se incluye")
    df.to_parquet(DATA / "liga_mx_features.parquet", index=False)
    log(f"Guardado con altitud: {DATA / 'liga_mx_features.parquet'}")
else:
    log("-> altitud NO ayuda, se descarta")
