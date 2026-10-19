#!/usr/bin/env python3
"""
eval_new_features.py — Compara el modelo con features nuevas vs el viejo
en el MISMO split temporal (test = desde jul/2024), walk-forward honesto.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FUTBOL = "/home/petyt/futbolmx"
df = pd.read_parquet(f"{FUTBOL}/data/liga_mx_features.parquet").sort_values("date").reset_index(drop=True)

OLD = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]
NEW = [
    "pos_table_h", "pos_table_a", "ppg_table_h", "ppg_table_a",
    "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
    "sos_h", "sos_a", "cong_h", "cong_a",
]
ALL = OLD + NEW

pool = df[df["btts"].notna()].copy()
cut = pd.Timestamp("2024-07-01")
is_test = pool["date"] >= cut

print(f"train: {(~is_test).sum()} | test: {is_test.sum()} "
      f"(test {pool[is_test]['date'].min().date()} -> {pool[is_test]['date'].max().date()})")
print()

def run(feats, label):
    X = pool[feats].copy()
    y = pool["btts"].astype(int).values
    # imputar SOLO con train means
    tr = ~is_test
    means = X[tr].mean()
    X = X.fillna(means)
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[~tr], y[~tr]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    xgb = XGBClassifier(
        n_estimators=350, max_depth=3, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=2,
    )
    xgb.fit(Xtr_s, ytr)
    p = xgb.predict_proba(Xte_s)[:, 1]

    base_p = np.full(len(yte), ytr.mean())
    print(f"== {label} ==")
    print(f"   XGB      Brier {brier_score_loss(yte, p):.4f} | LogLoss {log_loss(yte, p, labels=[0,1]):.4f} | AUC {roc_auc_score(yte, p):.4f}")
    print(f"   baseline Brier {brier_score_loss(yte, base_p):.4f} | AUC {roc_auc_score(yte, base_p):.4f}")

    # importancia de las nuevas
    if label != "VIEJO":
        imp = pd.Series(xgb.feature_importances_, index=feats).sort_values(ascending=False)
        print("   top features:")
        for k, v in imp.head(12).items():
            print(f"      {k:<18} {v:.4f}")
    print()
    return p, yte

run(OLD, "VIEJO (21 features, odds imputadas 2012-18)")
run(ALL, "NUEVO (33 features, odds 100% + tabla/H2H/SOS/congestion)")
