#!/usr/bin/env python3
"""
train_1x2.py — Modelo 1X2 (multiclass) con validación temporal.

Target: res_1x2 (H=0, D=1, A=2)
Features: igual que BTTS + odds implícitas 1X2 normalizadas
Split: train hasta jun/2024, test jul/2024+
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, classification_report
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
    "imp_h", "imp_d", "imp_a",  # probabilidades implícitas normalizadas
    "pos_table_h", "pos_table_a", "ppg_table_h", "ppg_table_a",
    "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
    "sos_h", "sos_a", "cong_h", "cong_a",
    "lam_h", "lam_a",
]

def log(msg):
    print(msg, flush=True)

def normalize_1x2(odd_h, odd_d, odd_a):
    """Normaliza 3 cuotas 1X2 a probabilidades implícitas sin margen."""
    ih, idd, ia = 1.0 / odd_h, 1.0 / odd_d, 1.0 / odd_a
    s = ih + idd + ia
    return ih / s, idd / s, ia / s


# Load data
df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
log(f"Total partidos: {len(df)}")

# Solo partidos con resultado y odds válidas
pool = df[df["res_1x2"].notna()].copy()

# Map target: H=0, D=1, A=2
target_map = {"H": 0, "D": 1, "A": 2}
pool["y"] = pool["res_1x2"].map(target_map)

# Odds implícitas: usar Pinnacle closing (PSC) si existe, sino max, sino avg
def get_imp_odds(row):
    for prefix in ["psc", "max", "avg", "bfe", "b365"]:
        h, d, a = row.get(f"odd_h_{prefix}"), row.get(f"odd_d_{prefix}"), row.get(f"odd_a_{prefix}")
        if pd.notna(h) and pd.notna(d) and pd.notna(a) and h > 1.01 and d > 1.01 and a > 1.01:
            return normalize_1x2(h, d, a)
    return (np.nan, np.nan, np.nan)

log("Calculando probabilidades implícitas...")
imp = pool.apply(get_imp_odds, axis=1, result_type="expand")
pool["imp_h"], pool["imp_d"], pool["imp_a"] = imp[0], imp[1], imp[2]

log(f"Partidos con odds implícitas: {pool['imp_h'].notna().sum()}")

# Features + target
X = pool[FEATURES].copy()
y = pool["y"].values
dates = pool["date"]

# Split temporal: test = ultimas 3 temporadas (Ape2024, Clau2025, Ape2025, Clau2026)
cut = pd.Timestamp("2024-07-01")
is_test = dates >= cut
X_train, y_train = X[~is_test], y[~is_test]
X_test, y_test = X[is_test], y[is_test]
log(f"Train: {len(X_train)} ({dates[~is_test].min().date()} -> {dates[~is_test].max().date()})")
log(f"Test:  {len(X_test)} ({dates[is_test].min().date()} -> {dates[is_test].max().date()})")

# Imputar con media de train
means = X_train.mean()
X_train = X_train.fillna(means)
X_test = X_test.fillna(means)

# Scale
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# Train XGBoost multiclass
xgb = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
    objective="multi:softprob", num_class=3,
    early_stopping_rounds=40, random_state=42, n_jobs=2,
)
xgb.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

# Predict
p = xgb.predict_proba(X_test_s)
pred = p.argmax(axis=1)

# Metrics
log(f"\n{'='*60}")
log(f"TEST 1X2: {len(y_test)} partidos (Ape2024 -> presente)")
log(f"{'='*60}")
log(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
log(f"LogLoss:  {log_loss(y_test, p, labels=[0,1,2]):.4f}")

# Per-class
report = classification_report(y_test, pred, target_names=["H", "D", "A"], output_dict=True)
for cls in ["H", "D", "A"]:
    r = report[cls]
    log(f"  {cls}: prec={r['precision']:.3f} rec={r['recall']:.3f} f1={r['f1-score']:.3f} support={int(r['support'])}")

# Baseline: implied odds (market)
# Get implied probs for test set
test_pool = pool[is_test].copy()
p_mkt = test_pool[["imp_h", "imp_d", "imp_a"]].values
mkt_pred = p_mkt.argmax(axis=1)
log(f"\nMarket (implied odds) Accuracy: {accuracy_score(y_test, mkt_pred):.3f}")
log(f"Market LogLoss: {log_loss(y_test, p_mkt, labels=[0,1,2]):.4f}")

# Feature importance
imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
log(f"\nTop 15 features:")
log(imp.head(15).round(4).to_string())

# Save model
with open(DATA / "model_1x2.pkl", "wb") as f:
    pickle.dump({
        "model": xgb, "scaler": scaler, "features": FEATURES,
        "impute_means": means, "train_cut": str(cut), "version": "v1",
        "target_map": {0: "H", 1: "D", 2: "A"},
    }, f)
log(f"\nGuardado: {DATA / 'model_1x2.pkl'}")

# Save test predictions
test_out = pool[is_test][["date", "home", "away", "res_1x2"]].copy()
test_out["p_H"] = np.round(p[:, 0], 4)
test_out["p_D"] = np.round(p[:, 1], 4)
test_out["p_A"] = np.round(p[:, 2], 4)
test_out["pred"] = pd.Series(pred).map({0: "H", 1: "D", 2: "A"}).values
test_out.to_parquet(DATA / "test_predictions_1x2.parquet", index=False)
log(f"Predicciones test: {DATA / 'test_predictions_1x2.parquet'}")