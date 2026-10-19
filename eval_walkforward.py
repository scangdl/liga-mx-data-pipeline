#!/usr/bin/env python3
"""
eval_walkforward.py — Seleccion robusta de configuracion con walk-forward CV.

Compara en 4 folds temporales:
  a) base_rate
  b) XGB sin lambdas
  c) XGB + lambdas career (promedios historicos por rol)
  d) XGB + lambdas EWMA (alpha=0.93)
  e) XGB + lambdas "cross" (formula original que escalo grande)
  f) Poisson standalone (con lambdas career)

Ganador = menor Brier promedio. Salida: config elegida impresa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

BASE_FEATURES = [
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


# ---------- computar las 3 variantes de lambda ----------
def compute_lambdas(df, mode):
    stats = {}
    league = {"gf_home": [], "gf_away": []}
    lam_h_list, lam_a_list = [], []

    if mode == "ewma":
        ALPHA = 0.93

        def upd(t, role, gf, ga):
            s = stats.setdefault(t, {}).setdefault(role, [None, None])
            s[0] = gf if s[0] is None else ALPHA * s[0] + (1 - ALPHA) * gf
            s[1] = ga if s[1] is None else ALPHA * s[1] + (1 - ALPHA) * ga

        def get(t, role):
            s = stats.get(t, {}).get(role)
            return (np.nan, np.nan) if not s or s[0] is None else (s[0], s[1])
    else:
        def upd(t, role, gf, ga):
            s = stats.setdefault(t, {}).setdefault(role, [0, 0, 0])
            s[0] += gf
            s[1] += ga
            s[2] += 1

        def get(t, role):
            s = stats.get(t, {}).get(role)
            return (np.nan, np.nan) if not s or s[2] == 0 else (s[0] / s[2], s[1] / s[2])

    for _, r in df.iterrows():
        h, a = r["home"], r["away"]
        ah, dh = get(h, 1)
        aa, da = get(a, 0)
        if league["gf_home"]:
            lgf_h = np.mean(league["gf_home"][-800:])
            lgf_a = np.mean(league["gf_away"][-800:])
        else:
            lgf_h, lgf_a = np.nan, np.nan

        if all(pd.notna(x) for x in (ah, dh, aa, da, lgf_h, lgf_a)):
            if mode == "cross":
                lam_h = (ah * da / lgf_a) * lgf_h
                lam_a = (aa * dh / lgf_h) * lgf_a
            else:  # career / ewma
                lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
                lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
        else:
            lam_h, lam_a = np.nan, np.nan
        lam_h_list.append(lam_h)
        lam_a_list.append(lam_a)

        if pd.notna(r["hs"]) and pd.notna(r["as_"]):
            upd(h, 1, r["hs"], r["as_"])
            upd(a, 0, r["as_"], r["hs"])
            league["gf_home"].append(r["hs"])
            league["gf_away"].append(r["as_"])
    return pd.Series(lam_h_list, index=df.index), pd.Series(lam_a_list, index=df.index)


lam_variants = {}
for mode in ("career", "ewma", "cross"):
    lh, la = compute_lambdas(df, mode)
    lam_variants[mode] = (lh, la)
    log(f"lambdas[{mode}]: media lam_h={lh.mean():.2f} lam_a={la.mean():.2f}, disponibles={lh.notna().sum()}")

pool = df[df["btts"].notna()].copy()

# ---------- folds walk-forward ----------
# 4 folds: cada fold entrena con todo lo anterior al corte y evalua ~1 temporada
cuts = [
    ("Ape2022-Clau2023", pd.Timestamp("2022-07-01"), pd.Timestamp("2023-07-01")),
    ("Ape2023-Clau2024", pd.Timestamp("2023-07-01"), pd.Timestamp("2024-07-01")),
    ("Ape2024-Clau2025", pd.Timestamp("2024-07-01"), pd.Timestamp("2025-07-01")),
    ("Ape2025-Clau2026", pd.Timestamp("2025-07-01"), pd.Timestamp("2026-07-01")),
]


def xgb_fit(Xtr, ytr, Xte, yte):
    m = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        early_stopping_rounds=40, random_state=42,
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    return m.predict_proba(Xte)[:, 1]


configs = ["base_rate", "xgb_plain", "xgb_career", "xgb_ewma", "xgb_cross", "poisson_career"]
metrics = {c: {"brier": [], "auc": [], "ll": []} for c in configs}

for fold_name, start, end in cuts:
    te_mask = (pool["date"] >= start) & (pool["date"] < end)
    tr_mask = pool["date"] < start
    if te_mask.sum() == 0 or tr_mask.sum() == 0:
        continue
    ytr = pool.loc[tr_mask, "btts"].astype(int).values
    yte = pool.loc[te_mask, "btts"].astype(int).values
    base_rate = ytr.mean()

    metrics["base_rate"]["brier"].append(brier_score_loss(yte, np.full(len(yte), base_rate)))
    metrics["base_rate"]["auc"].append(roc_auc_score(yte, np.full(len(yte), base_rate)))
    metrics["base_rate"]["ll"].append(log_loss(yte, np.full(len(yte), base_rate), labels=[0, 1]))

    # XGB plain
    Xtr = pool.loc[tr_mask, BASE_FEATURES]
    Xte = pool.loc[te_mask, BASE_FEATURES]
    means = Xtr.mean()
    p = xgb_fit(Xtr.fillna(means), ytr, Xte.fillna(means), yte)
    metrics["xgb_plain"]["brier"].append(brier_score_loss(yte, p))
    metrics["xgb_plain"]["auc"].append(roc_auc_score(yte, p))
    metrics["xgb_plain"]["ll"].append(log_loss(yte, p, labels=[0, 1]))

    # XGB + cada variante de lambda
    for mode in ("career", "ewma", "cross"):
        lh, la = lam_variants[mode]
        feats = BASE_FEATURES + ["lam_h", "lam_a"]
        dtr = pool.loc[tr_mask].copy()
        dte = pool.loc[te_mask].copy()
        dtr["lam_h"], dtr["lam_a"] = lh[tr_mask], la[tr_mask]
        dte["lam_h"], dte["lam_a"] = lh[te_mask], la[te_mask]
        Xtr, Xte = dtr[feats], dte[feats]
        means = Xtr.mean()
        p = xgb_fit(Xtr.fillna(means), ytr, Xte.fillna(means), yte)
        metrics[f"xgb_{mode}"]["brier"].append(brier_score_loss(yte, p))
        metrics[f"xgb_{mode}"]["auc"].append(roc_auc_score(yte, p))
        metrics[f"xgb_{mode}"]["ll"].append(log_loss(yte, p, labels=[0, 1]))

    # Poisson standalone (career)
    lh, la = lam_variants["career"]
    m = te_mask & lh.notna() & la.notna()
    if m.sum():
        pp = (1 - poisson.pmf(0, lh[m])) * (1 - poisson.pmf(0, la[m]))
        yy = pool.loc[m, "btts"].astype(int).values
        metrics["poisson_career"]["brier"].append(brier_score_loss(yy, pp))
        metrics["poisson_career"]["auc"].append(roc_auc_score(yy, pp))
        metrics["poisson_career"]["ll"].append(log_loss(yy, pp, labels=[0, 1]))

    log(f"fold {fold_name}: n_test={te_mask.sum()}, base Brier={metrics['base_rate']['brier'][-1]:.4f}, "
        f"plain={metrics['xgb_plain']['brier'][-1]:.4f}, career={metrics['xgb_career']['brier'][-1]:.4f}, "
        f"ewma={metrics['xgb_ewma']['brier'][-1]:.4f}, cross={metrics['xgb_cross']['brier'][-1]:.4f}")

# ---------- resumen ----------
log("\n" + "=" * 70)
log("RESUMEN WALK-FORWARD (media ± desv sobre 4 folds)")
log("=" * 70)
rows = []
for c in configs:
    b = np.array(metrics[c]["brier"])
    a = np.array(metrics[c]["auc"])
    ll = np.array(metrics[c]["ll"])
    rows.append({"config": c, "brier_mean": b.mean(), "brier_std": b.std(),
                 "auc_mean": a.mean(), "logloss_mean": ll.mean()})
res = pd.DataFrame(rows).sort_values("brier_mean")
log(res.round(4).to_string(index=False))

winner = res.iloc[0]["config"]
log(f"\nGANADOR: {winner}")
