#!/usr/bin/env python3
"""
retrain_production.py — Re-entrena el modelo BTTS de produccion con las 33 features
(odds 100% + tabla + H2H + SOS + congestion) y lo guarda como btts_model.pkl.
Split: train hasta jun/2024, test jul/2024+ (validacion temporal estandar).
"""
import pickle
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
    # nuevas
    "pos_table_h", "pos_table_a", "ppg_table_h", "ppg_table_a",
    "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
    "sos_h", "sos_a", "cong_h", "cong_a",
]

df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
pool = df[df["btts"].notna()].copy()
cut = pd.Timestamp("2024-07-01")
is_test = pool["date"] >= cut

X = pool[FEATURES].copy()
y = pool["btts"].astype(int).values
means = X[~is_test].mean()
X = X.fillna(means)

X_train, y_train = X[~is_test], y[~is_test]
X_test, y_test = X[is_test], y[is_test]

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

xgb = XGBClassifier(
    n_estimators=350, max_depth=3, learning_rate=0.035,
    subsample=0.85, colsample_bytree=0.8, eval_metric="logloss",
    random_state=42, n_jobs=2,
)
xgb.fit(X_train_s, y_train)

p = xgb.predict_proba(X_test_s)[:, 1]
print(f"Produccion re-entrenada (33 features):")
print(f"  Test: {len(y_test)} partidos | Brier {brier_score_loss(y_test, p):.4f} | "
      f"LogLoss {log_loss(y_test, p, labels=[0,1]):.4f} | AUC {roc_auc_score(y_test, p):.4f}")

# backup del viejo y guardar nuevo
(DATA / "btts_model_v1.pkl").write_bytes((DATA / "btts_model.pkl").read_bytes())
with open(DATA / "btts_model.pkl", "wb") as f:
    pickle.dump({"model": xgb, "scaler": scaler, "features": FEATURES,
                 "impute_means": means, "train_cut": str(cut), "version": "v2"}, f)
print(f"  Guardado: btts_model.pkl (v2, backup del v1 en btts_model_v1.pkl)")

# registrar el test set para trazabilidad
test_out = pool[is_test][["date", "home", "away", "btts"]].copy()
test_out["p_xgb_v2"] = np.round(p, 4)
test_out.to_parquet(DATA / "btts_test_predictions.parquet", index=False)
print(f"  Test predictions actualizadas: btts_test_predictions.parquet")
