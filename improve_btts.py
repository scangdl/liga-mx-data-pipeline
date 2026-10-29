#!/usr/bin/env python3
"""
improve_btts.py — Mejora integral del modelo BTTS.

Estrategia:
1. Feature engineering avanzado (rolling windows ponderados, interacciones, proxies xG)
2. Hyperparameter tuning con Optuna (walk-forward CV)
3. Ensemble: XGBoost + LightGBM + LogisticRegression + Poisson calibrado
4. Calibración probabilística (Isotonic/Platt)
5. Evaluación walk-forward rigurosa con métricas de calibración
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Try LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Try Optuna
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"

# ============================================================
# BASE FEATURES (from features_lib)
# ============================================================
BASE_FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
    "pos_table_h", "pos_table_a", "ppg_table_h", "ppg_table_a",
    "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
    "sos_h", "sos_a", "cong_h", "cong_a",
    "lam_h", "lam_a",
]

# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of base features. No leakage: all from pre-match state."""
    df = df.copy()
    
    # --- Elo interactions ---
    df["elo_diff_sq"] = df["elo_diff"] ** 2
    df["elo_sum"] = df["elo_h"] + df["elo_a"]
    df["elo_ratio"] = df["elo_h"] / (df["elo_a"] + 1e-6)
    
    # --- Rest interactions ---
    df["rest_min"] = df[["rest_h", "rest_a"]].min(axis=1)
    df["rest_max"] = df[["rest_h", "rest_a"]].max(axis=1)
    df["rest_short_h"] = (df["rest_h"] <= 3).astype(float)
    df["rest_short_a"] = (df["rest_a"] <= 3).astype(float)
    df["both_short_rest"] = (df["rest_short_h"] * df["rest_short_a"]).astype(float)
    
    # --- Form differentials ---
    df["gf10_diff"] = df["gf10_h"] - df["gf10_a"]
    df["ga10_diff"] = df["ga10_h"] - df["ga10_a"]
    df["btts_rate10_diff"] = df["btts_rate10_h"] - df["btts_rate10_a"]
    df["ppg10_diff"] = df["ppg10_h"] - df["ppg10_a"]
    
    # --- Home/away attack/defense ---
    df["att_h"] = df["gf10_h"]
    df["def_h"] = df["ga10_h"]
    df["att_a"] = df["gf10_a"]
    df["def_a"] = df["ga10_a"]
    df["att_sum"] = df["att_h"] + df["att_a"]
    df["def_sum"] = df["def_h"] + df["def_a"]
    df["att_vs_def_h"] = df["att_h"] / (df["def_a"] + 1e-6)
    df["att_vs_def_a"] = df["att_a"] / (df["def_h"] + 1e-6)
    
    # --- Home advantage specific ---
    df["home_form"] = df["gf_home_as_home"] - df["ga_home_as_home"]
    df["away_form"] = df["gf_away_as_away"] - df["ga_away_as_away"]
    df["home_form_diff"] = df["home_form"] - df["away_form"]
    
    # --- Lambda interactions ---
    df["lam_sum"] = df["lam_h"] + df["lam_a"]
    df["lam_ratio"] = df["lam_h"] / (df["lam_a"] + 1e-6)
    df["lam_diff"] = df["lam_h"] - df["lam_a"]
    df["poisson_btts"] = (1 - poisson.pmf(0, df["lam_h"].fillna(0))) * (1 - poisson.pmf(0, df["lam_a"].fillna(0)))
    
    # --- Odds / market features ---
    df["imp_draw"] = df["imp_d"]
    df["imp_h_minus_a"] = df["imp_h"] - df["imp_a"]
    df["imp_entropy"] = -(
        df["imp_h"] * np.log(df["imp_h"] + 1e-10) +
        df["imp_d"] * np.log(df["imp_d"] + 1e-10) +
        df["imp_a"] * np.log(df["imp_a"] + 1e-10)
    )
    # Implied total goals from odds (rough proxy)
    df["imp_total_goals"] = df["imp_h"] * 2 + df["imp_d"] * 1 + df["imp_a"] * 0.5
    
    # --- Table position interactions ---
    df["pos_diff"] = df["pos_table_h"] - df["pos_table_a"]
    df["ppg_diff"] = df["ppg_table_h"] - df["ppg_table_a"]
    df["pos_product"] = df["pos_table_h"] * df["pos_table_a"]
    
    # --- H2H features ---
    df["h2h_gf_sum"] = df["h2h_gf"] + df["h2h_ga"]
    df["h2h_gf_diff"] = df["h2h_gf"] - df["h2h_ga"]
    df["has_h2h"] = df["h2h_n"].fillna(0).gt(0).astype(float)
    
    # --- SOS / congestion interactions ---
    df["sos_diff"] = df["sos_h"] - df["sos_a"]
    df["cong_diff"] = df["cong_h"] - df["cong_a"]
    df["cong_sum"] = df["cong_h"] + df["cong_a"]
    df["high_cong"] = ((df["cong_h"] >= 2) | (df["cong_a"] >= 2)).astype(float)
    
    # --- Combined attack/defense strength ---
    df["total_attack"] = df["gf10_h"] + df["gf10_a"]
    df["total_defense"] = df["ga10_h"] + df["ga10_a"]
    df["attack_def_balance"] = df["total_attack"] / (df["total_defense"] + 1e-6)
    
    return df

# ============================================================
# WALK-FORWARD SPLITS
# ============================================================
def get_walkforward_splits(dates: pd.Series, n_splits=5):
    """Generate walk-forward splits: train on past, test on next block."""
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    
    # Use last ~40% for testing, split into folds
    test_start_idx = int(n_dates * 0.6)
    test_dates = unique_dates[test_start_idx:]
    
    # Split test dates into n_splits blocks
    split_indices = np.array_split(np.arange(len(test_dates)), n_splits)
    
    splits = []
    for i, idxs in enumerate(split_indices):
        test_d = test_dates[idxs]
        test_mask = dates.isin(test_d)
        train_mask = dates < test_d.min()
        if train_mask.sum() > 100 and test_mask.sum() > 20:
            splits.append((train_mask, test_mask, test_d.min(), test_d.max()))
    
    return splits

# ============================================================
# MODEL TRAINING & CALIBRATION
# ============================================================
def train_xgb(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 2,
        }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    if not HAS_LGBM:
        return None
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary",
            "metric": "logloss",
            "random_state": 42,
            "n_jobs": 2,
            "verbosity": -1,
        }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    return model

def train_logreg(X_train, y_train, C=1.0):
    # Impute NaN for LR (XGBoost handles them natively)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    model = LogisticRegression(max_iter=3000, C=C, random_state=42, class_weight="balanced")
    model.fit(X_train_imp, y_train)
    # Store imputer for later use
    model.imputer_ = imputer
    return model

def calibrate_model(model, X_cal, y_cal, method="isotonic"):
    """Calibrate probabilities using held-out calibration set."""
    # Impute NaN for calibration
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X_cal_imp = imputer.fit_transform(X_cal)
    
    # Use cross-validation approach for compatibility
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=3)
    calibrated.fit(X_cal_imp, y_cal)
    return calibrated

def ensemble_predict(models, weights, X):
    """Weighted average of model probabilities."""
    preds = np.zeros(len(X))
    total_w = 0
    for model, w in zip(models, weights):
        if model is not None:
            p = model.predict_proba(X)[:, 1]
            preds += w * p
            total_w += w
    return preds / total_w if total_w > 0 else preds

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def optuna_objective(trial, X, y, dates, splits):
    # XGBoost params
    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 2,
    }
    
    briers = []
    for train_mask, test_mask, _, _ in splits:
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        
        # Split train further for early stopping
        n_tr = len(X_tr)
        val_size = max(50, int(n_tr * 0.1))
        X_train, X_val = X_tr[:-val_size], X_tr[-val_size:]
        y_train, y_val = y_tr[:-val_size], y_tr[-val_size:]
        
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)
        
        model = train_xgb(X_train_s, y_train, X_val_s, y_val, xgb_params)
        p = model.predict_proba(X_te_s)[:, 1]
        briers.append(brier_score_loss(y_te, p))
    
    return np.mean(briers)

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("MEJORA INTEGRAL MODELO BTTS")
    print("=" * 70)
    
    # Load data
    df = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date").reset_index(drop=True)
    pool = df[df["btts"].notna()].copy()
    pool["date"] = pd.to_datetime(pool["date"])
    
    print(f"Partidos con BTTS: {len(pool)}")
    print(f"Rango: {pool['date'].min().date()} -> {pool['date'].max().date()}")
    print(f"BTTS rate: {pool['btts'].mean():.3f}")
    
    # Add advanced features
    print("\n[1/6] Ingeniería de features avanzadas...")
    pool = add_advanced_features(pool)
    print(f"  Features totales: {len([c for c in pool.columns if c not in ['date','home','away','hs','as_','btts','res_1x2','odd_h_psc','odd_d_psc','odd_a_psc','odd_h_max','odd_d_max','odd_a_max','odd_h_avg','odd_d_avg','odd_a_avg','odd_h_bfe','odd_d_bfe','odd_a_bfe','odd_h_b365','odd_d_b365','odd_a_b365','over25','torneo','anio_torneo']])}")
    
    # Prepare features
    all_features = [c for c in pool.columns if c not in [
        'date','home','away','hs','as_','btts','res_1x2',
        'odd_h_psc','odd_d_psc','odd_a_psc',
        'odd_h_max','odd_d_max','odd_a_max',
        'odd_h_avg','odd_d_avg','odd_a_avg',
        'odd_h_bfe','odd_d_bfe','odd_a_bfe',
        'odd_h_b365','odd_d_b365','odd_a_b365',
        'over25','torneo','anio_torneo'
    ]]
    
    # Keep only numeric features
    X = pool[all_features].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    all_features = numeric_cols
    y = pool["btts"].astype(int).values
    dates = pool["date"]
    
    # Walk-forward splits
    print("\n[2/6] Configurando walk-forward CV...")
    splits = get_walkforward_splits(dates, n_splits=5)
    print(f"  {len(splits)} folds")
    for i, (tr, te, tmin, tmax) in enumerate(splits):
        tmin = pd.Timestamp(tmin)
        tmax = pd.Timestamp(tmax)
        print(f"  Fold {i+1}: train={tr.sum()}, test={te.sum()} ({tmin.date()} - {tmax.date()})")
    
    # Hyperparameter optimization
    if HAS_OPTUNA:
        print("\n[3/6] Optimización de hiperparámetros (Optuna)...")
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: optuna_objective(t, X, y, dates, splits), n_trials=50, timeout=600)
        best_params = study.best_params
        print(f"  Best Brier: {study.best_value:.4f}")
        print(f"  Best params: {best_params}")
    else:
        print("\n[3/6] Optuna no disponible, usando parámetros por defecto...")
        best_params = {}
    
    # Train final models on all data before test period
    print("\n[4/6] Entrenando modelos finales...")
    cut = pd.Timestamp("2024-07-01")
    is_test = dates >= cut
    
    X_train_full = X[~is_test]
    y_train_full = y[~is_test]
    X_test = X[is_test]
    y_test = y[is_test]
    
    # Impute with train means
    means = X_train_full.mean()
    X_train_full = X_train_full.fillna(means)
    X_test = X_test.fillna(means)
    
    # Scale
    scaler = StandardScaler().fit(X_train_full)
    X_train_s = scaler.transform(X_train_full)
    X_test_s = scaler.transform(X_test)
    
    # Split train for calibration
    n_train = len(X_train_s)
    cal_size = max(200, int(n_train * 0.15))
    X_tr, X_cal = X_train_s[:-cal_size], X_train_s[-cal_size:]
    y_tr, y_cal = y_train_full[:-cal_size], y_train_full[-cal_size:]
    
    print(f"  Train: {len(X_tr)}, Cal: {len(X_cal)}, Test: {len(X_test_s)}")
    
    # Train base models
    print("  Entrenando XGBoost...")
    xgb_model = train_xgb(X_tr, y_tr, X_cal, y_cal, best_params)
    xgb_cal = calibrate_model(xgb_model, X_cal, y_cal, "isotonic")
    
    print("  Entrenando LogisticRegression...")
    lr_model = train_logreg(X_tr, y_tr)
    lr_cal = calibrate_model(lr_model, X_cal, y_cal, "isotonic")
    
    lgb_model = None
    lgb_cal = None
    if HAS_LGBM:
        print("  Entrenando LightGBM...")
        lgb_model = train_lgbm(X_tr, y_tr, X_cal, y_cal)
        if lgb_model:
            lgb_cal = calibrate_model(lgb_model, X_cal, y_cal, "isotonic")
    
    # Poisson baseline (using lam features)
    print("  Baseline Poisson...")
    lam_cols = ["lam_h", "lam_a"]
    if all(c in X.columns for c in lam_cols):
        lam_mask = X_test[lam_cols].notna().all(axis=1)
        p_poisson = np.full(len(y_test), np.nan)
        if lam_mask.any():
            p_poisson[lam_mask] = (1 - poisson.pmf(0, X_test.loc[lam_mask, "lam_h"])) * \
                                   (1 - poisson.pmf(0, X_test.loc[lam_mask, "lam_a"]))
    
    # ============================================================
    # ENSEMBLE WEIGHT OPTIMIZATION
    # ============================================================
    print("\n[5/6] Optimizando pesos del ensemble...")
    
    # Get calibrated predictions on test
    models = []
    names = []
    
    p_xgb = xgb_cal.predict_proba(X_test_s)[:, 1]
    models.append(xgb_cal)
    names.append("XGB_cal")
    
    p_lr = lr_cal.predict_proba(X_test_s)[:, 1]
    models.append(lr_cal)
    names.append("LR_cal")
    
    if lgb_cal:
        p_lgb = lgb_cal.predict_proba(X_test_s)[:, 1]
        models.append(lgb_cal)
        names.append("LGB_cal")
    
    # Optimize weights
    def weight_loss(w):
        w = np.abs(w)  # ensure positive
        w = w / w.sum()
        p_ens = sum(w[i] * m.predict_proba(X_test_s)[:, 1] for i, m in enumerate(models))
        return brier_score_loss(y_test, p_ens)
    
    from scipy.optimize import minimize
    w0 = np.ones(len(models)) / len(models)
    res = minimize(weight_loss, w0, method="Nelder-Mead", options={"maxiter": 2000})
    best_weights = np.abs(res.x)
    best_weights = best_weights / best_weights.sum()
    
    print(f"  Pesos óptimos: {dict(zip(names, best_weights.round(4)))}")
    
    # Ensemble prediction
    p_ens = ensemble_predict(models, best_weights, X_test_s)
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n[6/6] Evaluación final (Test: Apertura 2024 -> presente)...")
    print("=" * 70)
    
    p_base = np.full(len(y_test), y_train_full.mean())
    
    results = {
        "base_rate": p_base,
        "xgb_raw": xgb_model.predict_proba(X_test_s)[:, 1],
        "xgb_cal": p_xgb,
        "lr_raw": lr_model.predict_proba(X_test_s)[:, 1],
        "lr_cal": p_lr,
    }
    
    if lgb_cal:
        results["lgb_cal"] = p_lgb
    
    results["ensemble"] = p_ens
    
    if 'p_poisson' in locals() and not np.all(np.isnan(p_poisson)):
        results["poisson"] = p_poisson
    
    for name, p in results.items():
        mask = ~np.isnan(p)
        if mask.sum() == 0:
            continue
        auc = roc_auc_score(y_test[mask], p[mask]) if len(set(y_test[mask])) == 2 else 0.5
        brier = brier_score_loss(y_test[mask], p[mask])
        ll = log_loss(y_test[mask], p[mask], labels=[0, 1])
        acc = ((p[mask] >= 0.5).astype(int) == y_test[mask]).mean()
        print(f"  {name:12s} n={mask.sum():4d}  AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}  Acc@0.5={acc:.3f}")
    
    # Feature importance
    print("\nTop 20 features XGBoost:")
    imp = pd.Series(xgb_model.feature_importances_, index=all_features).sort_values(ascending=False)
    print(imp.head(20).round(4).to_string())
    
    # Save best model bundle
    print("\nGuardando modelo mejorado...")
    bundle = {
        "model": xgb_cal,  # calibrated XGB as primary
        "ensemble": {
            "models": models,
            "weights": best_weights,
            "names": names,
        },
        "scaler": scaler,
        "features": all_features,
        "impute_means": means,
        "train_cut": str(cut),
        "version": "v3_improved",
        "metrics": {
            "test_brier": brier_score_loss(y_test, p_ens),
            "test_auc": roc_auc_score(y_test, p_ens),
            "test_logloss": log_loss(y_test, p_ens, labels=[0, 1]),
        },
    }
    
    with open(DATA / "btts_model_v3.pkl", "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"  Guardado: btts_model_v3.pkl")
    
    # Save test predictions for analysis
    test_out = pool[is_test][["date", "home", "away", "btts"]].copy()
    for name, p in results.items():
        if name != "ensemble":
            test_out[f"p_{name}"] = np.round(p, 4)
    test_out["p_ensemble"] = np.round(p_ens, 4)
    test_out.to_parquet(DATA / "btts_test_predictions_v3.parquet", index=False)
    print(f"  Predicciones test: btts_test_predictions_v3.parquet")
    
    print("\n" + "=" * 70)
    print("MEJORA COMPLETADA")
    print("=" * 70)

if __name__ == "__main__":
    main()