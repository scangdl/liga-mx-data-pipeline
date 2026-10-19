"""BACKTEST HONESTO DEL MODELO O2.5 LIGA MX — respondiendo las 4 objeciones del usuario:
1) Curva de calibracion por deciles (walk-forward)
2) AUC/Brier por fold (estabilidad)
3) Umbral de edge: ¿la señal sobrevive a 9-10%?
4) Reportar limitaciones: sin cuotas historicas O/U no hay P&L real ni CLV — se declara.

Estrategia: walk-forward de 4 folds temporales (como eval_walkforward.py original).
Prediccion p_over del modelo en cada fold TEST. La "cuota justa" del modelo = 1/p.
La "cuota de mercado" para simular P&L: prob empirica del decil en TRAIN + margen 6% (supuesto declarado)
-> permite estimar si el edge del modelo SOBREVIVE cuando el mercado es tan bueno como el propio base rate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import brier_score_loss, roc_auc_score

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

# ---------- datos ----------
df = pd.read_parquet(DATA / "liga_mx_master.parquet")
feats = pd.read_parquet(DATA / "liga_mx_features.parquet")  # regenerado por make_features.py
feats = feats.dropna(subset=["over25"]).reset_index(drop=True)

with open(DATA / "over25_model.pkl", "rb") as f:
    bundle = pickle.load(f)
FEATS = bundle["features"]
means = pd.Series(bundle["means"])

X = feats[FEATS].fillna(means)
y = feats["over25"].astype(int).values
dates = pd.to_datetime(feats["date"]).values

# ---------- walk-forward 4 folds ----------
n = len(feats)
folds = np.array_split(np.arange(n), 4)  # orden temporal: el master viene cronologico
all_p = np.full(n, np.nan)
all_fold = np.full(n, -1)

print("=" * 78)
print("WALK-FORWARD 4 FOLDS — modelo O2.5 (21 feats, XGB)")
print("=" * 78)
for k, test_idx in enumerate(folds):
    train_idx = np.concatenate(folds[:k]) if k > 0 else None
    if train_idx is None or len(train_idx) < 300:
        continue
    from xgboost import XGBClassifier  # import tardio
    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    model.fit(X.iloc[train_idx], y[train_idx])
    p = model.predict_proba(X.iloc[test_idx])[:, 1]
    all_p[test_idx] = p
    all_fold[test_idx] = k
    auc = roc_auc_score(y[test_idx], p)
    br = brier_score_loss(y[test_idx], p)
    base = max(y[train_idx].mean(), 1 - y[train_idx].mean())
    print(f"Fold {k}: train={len(train_idx):5d} test={len(test_idx):4d} | "
          f"AUC={auc:.3f} | Brier={br:.4f} | base={base:.3f} | over_rate_test={y[test_idx].mean():.3f}")

mask = ~np.isnan(all_p)
p_hat, y_true, fold_id = all_p[mask], y[mask], all_fold[mask]
print()
auc_g = roc_auc_score(y_true, p_hat)
print(f"GLOBAL (folds 1-3 test): n={mask.sum()} | AUC={auc_g:.4f} | Brier={brier_score_loss(y_true, p_hat):.4f}")

# ---------- CURVA DE CALIBRACION (deciles) ----------
print()
print("=" * 78)
print("CURVA DE CALIBRACION — deciles de p_hat vs tasa real de Over 2.5 (todas las predicciones wf)")
print("=" * 78)
q = pd.qcut(p_hat, 10, duplicates="drop")
cal = pd.DataFrame({"p": p_hat, "y": y_true, "decil": q})
tabla = cal.groupby("decil", observed=True).agg(
    n=("y", "size"), p_prom=("p", "mean"), tasa_real=("y", "mean")
).reset_index()
tabla["gap"] = tabla["tasa_real"] - tabla["p_prom"]
print(f"{'decil':<22} {'n':>5} {'p_prom':>7} {'real':>7} {'gap':>7}")
for _, r in tabla.iterrows():
    bar = "#" * int(round(r["tasa_real"] * 40))
    print(f"{str(r['decil']):<22} {int(r['n']):>5} {r['p_prom']:>7.3f} {r['tasa_real']:>7.3f} {r['gap']:>+7.3f}  {bar}")

# ---------- ESTRATEGIA con umbral de edge vs mercado = base rate del fold + margen ----------
print()
print("=" * 78)
print("SIMULACION DE ESTRATEGIA (supuesto: cuota de mercado = base_rate_train*(1+6% margen))")
print("  — NO es P&L real (no hay cuotas O/U historicas); mide si el edge del modelo sobrevive")
print("=" * 78)
feats_mask = feats.iloc[np.where(mask)[0]].copy()
feats_mask["p"] = p_hat
feats_mask["fold"] = fold_id
MARGEN = 0.06

for umbral_edge in (0.04, 0.06, 0.09, 0.12):
    rows_out = []
    for k in sorted(feats_mask["fold"].unique()):
        tr_mask = (all_fold == k - 1)
        if not tr_mask.any():
            continue
        base_rate_train = y[tr_mask].mean()
        cuota_mercado = 1 / (base_rate_train * (1 + MARGEN))  # cuota over con margen
        te = feats_mask[feats_mask["fold"] == k]
        edge = te["p"] * cuota_mercado - 1
        sel = te[edge > umbral_edge]
        for _, r in sel.iterrows():
            won = r["over25"]
            stake = 50.0  # stake plano para comparacion limpia entre umbrales
            pl = (stake * cuota_mercado - stake) if won else -stake
            rows_out.append(pl)
    if rows_out:
        arr = np.array(rows_out)
        roi = arr.sum() / (len(arr) * 50) * 100
        wins = (arr > 0).sum()
        print(f"edge>{umbral_edge:.2f}: apuestas={len(arr):4d} | ganadas={wins:4d} "
              f"({wins/len(arr)*100:.1f}%) | P&L={arr.sum():+9.1f} | ROI={roi:+.2f}%")
    else:
        print(f"edge>{umbral_edge:.2f}: 0 apuestas")

print()
print("LIMITACIONES DECLARADAS:")
print("- Sin cuotas historicas O/U 2.5 reales: el P&L de arriba es SIMULADO (base rate + margen fijo).")
print("- CLV imposible con datos actuales; el Odds API snapshot solo guarda la cuota al momento del place.")
print("  Fix propuesto: guardar la cuota de CIERRE (ultimo fetch previo al partido) por trade -> CLV real.")
print("- n folds 4, temporal — aceptable para estabilidad; no es validacion exhaustiva.")

# ============================================================
# RECALIBRACION (Platt scaling) dentro del walk-forward — responde la objecion de calibracion
# ============================================================
print()
print("=" * 78)
print("RECALIBRADO CON PLATT (por fold, sobre p crudo) — la unica version honesta del edge")
print("=" * 78)
from sklearn.linear_model import LogisticRegression

p_cal = np.full(n, np.nan)
for k, test_idx in enumerate(folds):
    train_idx = np.concatenate(folds[:k]) if k > 0 else None
    if train_idx is None or len(train_idx) < 300:
        continue
    # entrenar Platt con las predicciones wf de los folds ANTERIORES (solo indices con p ya calculada)
    tr_ok = [i for i in train_idx if not np.isnan(all_p[i])]
    if len(tr_ok) < 300:
        continue
    lr = LogisticRegression()
    Xp = all_p[tr_ok].reshape(-1, 1)
    lr.fit(Xp, y[tr_ok])
    p_cal[test_idx] = lr.predict_proba(all_p[test_idx].reshape(-1, 1))[:, 1]

mc = ~np.isnan(p_cal)
q2 = pd.qcut(p_cal[mc], 10, duplicates="drop")
cal2 = pd.DataFrame({"p": p_cal[mc], "y": y[mc], "decil": q2})
tab2 = cal2.groupby("decil", observed=True).agg(n=("y", "size"), p_prom=("p", "mean"), tasa_real=("y", "mean")).reset_index()
tab2["gap"] = tab2["tasa_real"] - tab2["p_prom"]
print(f"{'decil':<22} {'n':>5} {'p_prom':>7} {'real':>7} {'gap':>7}")
for _, r in tab2.iterrows():
    print(f"{str(r['decil']):<22} {int(r['n']):>5} {r['p_prom']:>7.3f} {r['tasa_real']:>7.3f} {r['gap']:>+7.3f}")
print()
print("Brier recalibrado:", round(brier_score_loss(y[mc], p_cal[mc]), 4))
print("AUC recalibrado  :", round(roc_auc_score(y[mc], p_cal[mc]), 4), "(igual que crudo — Platt no cambia el ranking)")

# Estrategia recalibrada
feats_cal = feats.iloc[np.where(mc)[0]].copy()
feats_cal["p"] = p_cal[mc]
feats_cal["fold"] = all_fold[mc]
for umbral_edge in (0.04, 0.06, 0.09, 0.12):
    rows_out = []
    for k in sorted(feats_cal["fold"].unique()):
        tr_mask = (all_fold == k - 1)
        if not tr_mask.any():
            continue
        base_rate_train = y[tr_mask].mean()
        cuota_mercado = 1 / (base_rate_train * (1 + MARGEN))
        te = feats_cal[feats_cal["fold"] == k]
        edge = te["p"] * cuota_mercado - 1
        sel = te[edge > umbral_edge]
        for _, r in sel.iterrows():
            stake = 50.0
            pl = (stake * cuota_mercado - stake) if r["over25"] else -stake
            rows_out.append(pl)
    if rows_out:
        arr = np.array(rows_out)
        roi = arr.sum() / (len(arr) * 50) * 100
        wins = (arr > 0).sum()
        print(f"[CAL] edge>{umbral_edge:.2f}: apuestas={len(arr):4d} | ganadas={wins:4d} "
              f"({wins/len(arr)*100:.1f}%) | P&L={arr.sum():+9.1f} | ROI={roi:+.2f}%")
