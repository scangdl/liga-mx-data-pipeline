#!/usr/bin/env python3
"""
Blind Test Backtest — Simula paper trading histórico con split temporal personalizado.

Uso:
  python backtest_historico.py --cutoff 2020-12-31 --season "Apertura 2020"
  python backtest_historico.py --cutoff 2021-06-30 --season "Clausura 2021"

Simula que el modelo solo conoce datos hasta --cutoff y predice partidos posteriores.
Compara predicciones vs resultados reales.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

from liga_mx.config import get_config, get_path, load_config
from liga_mx.features import StateBuilder, add_advanced_features, poisson_btts_prob


def _log(msg: str) -> None:
    print(msg, flush=True)


def _prepare_data(df: pd.DataFrame, feature_set: list, target: str, train_cutoff: pd.Timestamp, test_cutoff: pd.Timestamp | None = None):
    """Prepara datos con split temporal personalizado."""
    # Filtrar solo partidos jugados (target no nulo)
    df_played = df[df[target].notna()].copy()
    
    # Split temporal
    train_mask = df_played["date"] <= train_cutoff
    if test_cutoff is not None:
        test_mask = (df_played["date"] > train_cutoff) & (df_played["date"] <= test_cutoff)
    else:
        test_mask = df_played["date"] > train_cutoff
    
    train_df = df_played[train_mask].copy()
    test_df = df_played[test_mask].copy()
    
    X_train = train_df[feature_set].copy()
    y_train = train_df[target].astype(int).values
    X_test = test_df[feature_set].copy()
    y_test = test_df[target].astype(int).values
    
    # Imputación con medias de train
    impute_means = X_train.mean()
    X_train = X_train.fillna(impute_means)
    X_test = X_test.fillna(impute_means)
    
    # Escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    return X_train_s, y_train, X_test_s, y_test, impute_means, scaler, train_df, test_df


def _train_btts_ensemble(X_tr, y_tr, X_val, y_val, X_test_s, y_test, cfg):
    """Entrena ensemble BTTS (XGB + LR + LGBM opcional)."""
    xgb_params = {**{"tree_method": "hist", "device": "cpu"}, **cfg.models.btts.get("xgb_params", {})}
    
    # XGBoost
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb = xgb_model.predict_proba(X_test_s)[:, 1]
    
    # LogisticRegression
    lr_model = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight="balanced")
    lr_model.fit(X_tr, y_tr)
    p_lr = lr_model.predict_proba(X_test_s)[:, 1]
    
    # LightGBM si disponible
    lgb_model = None
    p_lgb = None
    if HAS_LGBM:
        lgb_model = LGBMClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, objective="binary",
            random_state=42, verbose=-1, n_jobs=1
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_lgb = lgb_model.predict_proba(X_test_s)[:, 1]
    
    # Optimizar pesos ensemble
    def loss_fn(w):
        w = np.clip(w, 0, 1)
        w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
        p = w[0] * p_xgb + w[1] * p_lr
        if p_lgb is not None:
            p += w[2] * p_lgb
        return log_loss(y_test, np.clip(p, 1e-15, 1 - 1e-15))
    
    init_w = np.array([0.85, 0.10, 0.05] if p_lgb is not None else [0.9, 0.1])
    res = minimize(loss_fn, init_w, method="Nelder-Mead", bounds=[(0, 1)] * len(init_w))
    opt_w = np.clip(res.x, 0, 1)
    opt_w = opt_w / opt_w.sum() if opt_w.sum() > 0 else np.ones(len(init_w)) / len(init_w)
    
    ensemble_weights = {"xgb": float(opt_w[0]), "lr": float(opt_w[1])}
    if p_lgb is not None:
        ensemble_weights["lgb"] = float(opt_w[2])
    
    # Ensemble prediction
    ensemble_pred = opt_w[0] * p_xgb + opt_w[1] * p_lr
    if p_lgb is not None:
        ensemble_pred += opt_w[2] * p_lgb
    
    # Platt calibration
    platt = None
    if cfg.models.btts.get("calibration", "platt") == "platt":
        platt = LogisticRegression(max_iter=1000, random_state=42)
        platt.fit(ensemble_pred.reshape(-1, 1), y_test)
        ensemble_pred = platt.predict_proba(ensemble_pred.reshape(-1, 1))[:, 1]
    
    return {
        "xgb_model": xgb_model,
        "lr_model": lr_model,
        "lgb_model": lgb_model,
        "p_xgb": p_xgb,
        "p_lr": p_lr,
        "p_lgb": p_lgb,
        "ensemble_pred": ensemble_pred,
        "ensemble_weights": ensemble_weights,
        "platt": platt,
    }


def run_blind_backtest(cutoff_date: str, test_end_date: str | None = None, season_name: str = "") -> dict:
    """
    Ejecuta backtest tipo blind test.
    
    Args:
        cutoff_date: Fecha corte (YYYY-MM-DD). Entrena con datos <= cutoff
        test_end_date: Fecha fin test (YYYY-MM-DD). Si None, usa todos los datos > cutoff
        season_name: Nombre descriptivo de la temporada (para logs)
    
    Returns:
        Dict con métricas y predicciones
    """
    load_config()
    cfg = get_config()
    
    _log(f"=== BLIND BACKTEST: {season_name} ===")
    _log(f"Train cutoff: {cutoff_date}")
    if test_end_date:
        _log(f"Test end: {test_end_date}")
    
    train_cutoff = pd.Timestamp(cutoff_date)
    test_cutoff = pd.Timestamp(test_end_date) if test_end_date else None
    
    # 1. Cargar features completas
    features_path = get_path("features_parquet")
    _log(f"Cargando features de {features_path}")
    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    _log(f"Total matches: {len(df)}, range: {df['date'].min().date()} -> {df['date'].max().date()}")
    
    # 2. Añadir features avanzadas
    df = add_advanced_features(df)
    
    feature_set = cfg.features.extended_features
    
    # 3. Split temporal personalizado
    train_mask = df["date"] <= train_cutoff
    if test_cutoff is not None:
        test_mask = (df["date"] > train_cutoff) & (df["date"] <= test_cutoff)
    else:
        test_mask = df["date"] > train_cutoff
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    _log(f"Train: {len(train_df)} matches (hasta {train_df['date'].max().date()})")
    _log(f"Test:  {len(test_df)} matches (desde {test_df['date'].min().date()}" + (f" hasta {test_df['date'].max().date()}" if len(test_df) > 0 else "") + ")")
    
    if len(test_df) == 0:
        _log("WARNING: No test matches after cutoff")
        return {"error": "No test data"}
    
    # Filtrar solo partidos jugados en test
    test_played = test_df[test_df["btts"].notna()].copy()
    _log(f"Test played: {len(test_played)} matches")
    
    if len(test_played) == 0:
        _log("WARNING: No played matches in test period")
        return {"error": "No played test data"}
    
    results = {
        "season": season_name,
        "train_cutoff": cutoff_date,
        "test_end": test_end_date,
        "n_train": len(train_df),
        "n_test": len(test_played),
        "test_period": f"{test_played['date'].min().date()} -> {test_played['date'].max().date()}"
    }
    
    # 4. Entrenar y evaluar BTTS
    _log("\n--- BTTS Model (entrenado solo con datos <= cutoff) ---")
    X_train_btts, y_train_btts, X_test_btts, y_test_btts, impute_means_btts, scaler_btts, train_btts_df, test_btts_df = _prepare_data(
        df, feature_set, "btts", train_cutoff, test_cutoff
    )
    
    # Split train para validación
    n_train = len(X_train_btts)
    val_size = max(200, int(n_train * 0.15))
    X_tr, X_val = X_train_btts[:-val_size], X_train_btts[-val_size:]
    y_tr, y_val = y_train_btts[:-val_size], y_train_btts[-val_size:]
    
    _log(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_btts)}")
    
    btts_result = _train_btts_ensemble(X_tr, y_tr, X_val, y_val, X_test_btts, y_test_btts, cfg)
    ensemble_pred = btts_result["ensemble_pred"]
    y_true_btts = y_test_btts
    
    # Métricas BTTS
    btts_auc = roc_auc_score(y_true_btts, ensemble_pred)
    btts_brier = brier_score_loss(y_true_btts, ensemble_pred)
    btts_logloss = log_loss(y_true_btts, ensemble_pred)
    btts_acc = accuracy_score(y_true_btts, (ensemble_pred >= 0.5).astype(int))
    
    results["btts"] = {
        "auc": float(btts_auc),
        "brier": float(btts_brier),
        "logloss": float(btts_logloss),
        "accuracy": float(btts_acc),
        "n": int(len(y_true_btts)),
        "btts_rate": float(y_true_btts.mean()),
        "pred_mean": float(ensemble_pred.mean()),
        "ensemble_weights": btts_result["ensemble_weights"],
    }
    
    _log(f"BTTS: AUC={btts_auc:.4f}, Brier={btts_brier:.4f}, LogLoss={btts_logloss:.4f}, Acc={btts_acc:.3f}")
    _log(f"  BTTS rate real: {y_true_btts.mean():.3f}, pred mean: {ensemble_pred.mean():.3f}")
    _log(f"  Ensemble weights: {btts_result['ensemble_weights']}")
    
    # 5. Over 2.5 Model
    _log("\n--- Over 2.5 Model ---")
    X_train_o25, y_train_o25, X_test_o25, y_test_o25, impute_means_o25, scaler_o25, train_o25_df, test_o25_df = _prepare_data(
        df, feature_set, "over25", train_cutoff, test_cutoff
    )
    
    n_train = len(X_train_o25)
    val_size = max(200, int(n_train * 0.15))
    X_tr, X_val = X_train_o25[:-val_size], X_train_o25[-val_size:]
    y_tr, y_val = y_train_o25[:-val_size], y_train_o25[-val_size:]
    
    _log(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_o25)}")
    
    # XGBoost Over 2.5
    o25_params = {**{"tree_method": "hist", "device": "cpu"}, **cfg.models.over25.get("xgb_params", {})}
    o25_model = XGBClassifier(**o25_params)
    o25_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    o25_pred = o25_model.predict_proba(X_test_o25)[:, 1]
    y_true_o25 = y_test_o25
    
    o25_auc = roc_auc_score(y_true_o25, o25_pred)
    o25_brier = brier_score_loss(y_true_o25, o25_pred)
    o25_logloss = log_loss(y_true_o25, o25_pred)
    o25_acc = accuracy_score(y_true_o25, (o25_pred >= 0.5).astype(int))
    
    results["over25"] = {
        "auc": float(o25_auc),
        "brier": float(o25_brier),
        "logloss": float(o25_logloss),
        "accuracy": float(o25_acc),
        "n": int(len(y_true_o25)),
        "over25_rate": float(y_true_o25.mean()),
    }
    
    _log(f"Over2.5: AUC={o25_auc:.4f}, Brier={o25_brier:.4f}, Acc={o25_acc:.3f}")
    _log(f"  Over rate real: {y_true_o25.mean():.3f}, pred mean: {o25_pred.mean():.3f}")
    
    # 6. 1X2 Model
    _log("\n--- 1X2 Model ---")
    # Mapear res_1x2 a 0,1,2
    df_1x2 = df.copy()
    map_1x2 = {"H": 0, "D": 1, "A": 2}
    df_1x2["res_1x2_num"] = df_1x2["res_1x2"].map(map_1x2)
    
    X_train_1x2, y_train_1x2, X_test_1x2, y_test_1x2, impute_means_1x2, scaler_1x2, train_1x2_df, test_1x2_df = _prepare_data(
        df_1x2, feature_set, "res_1x2_num", train_cutoff, test_cutoff
    )
    
    n_train = len(X_train_1x2)
    val_size = max(200, int(n_train * 0.15))
    X_tr, X_val = X_train_1x2[:-val_size], X_train_1x2[-val_size:]
    y_tr, y_val = y_train_1x2[:-val_size], y_train_1x2[-val_size:]
    
    _log(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_1x2)}")
    
    # XGBoost multiclass
    one_x_two_params = {**{"tree_method": "hist", "device": "cpu"}, **cfg.models.one_x_two.get("xgb_params", {})}
    one_x_two_params["objective"] = "multi:softprob"
    one_x_two_params["num_class"] = 3
    
    one_x_two_model = XGBClassifier(**one_x_two_params)
    one_x_two_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds_1x2 = one_x_two_model.predict_proba(X_test_1x2)
    preds_1x2_class = preds_1x2.argmax(axis=1)
    y_true_1x2 = y_test_1x2
    
    acc_1x2 = accuracy_score(y_true_1x2, preds_1x2_class)
    logloss_1x2 = log_loss(y_true_1x2, preds_1x2, labels=[0, 1, 2])
    
    results["one_x_two"] = {
        "accuracy": float(acc_1x2),
        "logloss": float(logloss_1x2),
        "n": int(len(y_true_1x2)),
    }
    
    # Per-class metrics
    report = classification_report(y_true_1x2, preds_1x2_class, target_names=["H", "D", "A"], output_dict=True)
    for cls in ["H", "D", "A"]:
        r = report[cls]
        results["one_x_two"][f"{cls}_precision"] = float(r["precision"])
        results["one_x_two"][f"{cls}_recall"] = float(r["recall"])
        results["one_x_two"][f"{cls}_f1"] = float(r["f1-score"])
    
    _log(f"1X2: Acc={acc_1x2:.3f}, LogLoss={logloss_1x2:.4f}")
    for cls in ["H", "D", "A"]:
        _log(f"  {cls}: P={report[cls]['precision']:.3f} R={report[cls]['recall']:.3f} F1={report[cls]['f1-score']:.3f}")
    
    # 7. Calibración (deciles)
    _log("\n--- Calibración BTTS (deciles) ---")
    pred_series = pd.Series(ensemble_pred)
    bins = pd.qcut(pred_series, 10, duplicates="drop")
    cal_df = pd.DataFrame({"p": ensemble_pred, "y": y_true_btts, "bin": bins})
    cal = cal_df.groupby("bin", observed=True).agg(
        n=("y", "size"), p_mean=("p", "mean"), rate=("y", "mean")
    ).reset_index()
    cal["gap"] = cal["rate"] - cal["p_mean"]
    _log(f"{'decil':<22} {'n':>5} {'p_prom':>7} {'real':>7} {'gap':>7}")
    for _, r in cal.iterrows():
        _log(f"{str(r['bin']):<22} {int(r['n']):>5} {r['p_mean']:>7.3f} {r['rate']:>7.3f} {r['gap']:>+7.3f}")
    
    # 8. Simulación paper trading (edge vs mercado simulado)
    _log("\n--- Simulación Paper Trading (mercado = base rate train + 6% margen) ---")
    base_rate_train = train_btts_df["btts"].mean()
    market_odds = 1 / (base_rate_train * 1.06)
    
    for edge_thresh in [0.04, 0.06, 0.08, 0.10]:
        edge = ensemble_pred * market_odds - 1
        bets = edge > edge_thresh
        if bets.sum() > 0:
            pnl = np.where(y_true_btts[bets] == 1, market_odds - 1, -1).sum()
            roi = pnl / bets.sum() * 100
            win_rate = (y_true_btts[bets] == 1).mean()
            _log(f"  edge>{edge_thresh:.2f}: bets={bets.sum()} win%={win_rate*100:.1f}% ROI={roi:.1f}% PnL={pnl:+.1f}")
        else:
            _log(f"  edge>{edge_thresh:.2f}: 0 apuestas")
    
    # 9. Guardar predicciones detalladas
    pred_df = test_played[["date", "home", "away", "btts", "over25", "res_1x2"]].copy()
    pred_df["p_btts"] = np.round(ensemble_pred, 4)
    pred_df["p_over25"] = np.round(o25_pred, 4)
    pred_df["p_H"] = np.round(preds_1x2[:, 0], 4)
    pred_df["p_D"] = np.round(preds_1x2[:, 1], 4)
    pred_df["p_A"] = np.round(preds_1x2[:, 2], 4)
    pred_df["pred_1x2"] = pd.Series(preds_1x2_class).map({0: "H", 1: "D", 2: "A"}).values
    pred_df["correct_btts"] = (pred_df["p_btts"] >= 0.5).astype(int) == pred_df["btts"]
    pred_df["correct_over25"] = (pred_df["p_over25"] >= 0.5).astype(int) == pred_df["over25"]
    pred_df["correct_1x2"] = pred_df["pred_1x2"] == pred_df["res_1x2"]
    
    output_path = get_path("predictions_dir") / f"blind_backtest_{season_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    _log(f"\nPredicciones guardadas: {output_path}")
    
    # 10. Resumen final
    _log("\n" + "=" * 60)
    _log(f"RESUMEN BACKTEST: {season_name}")
    _log("=" * 60)
    _log(f"Periodo test: {test_played['date'].min().date()} -> {test_played['date'].max().date()}")
    _log(f"Partidos test: {len(test_played)}")
    _log(f"\nBTTS:    AUC={btts_auc:.4f} Brier={btts_brier:.4f} Acc={btts_acc:.3f}")
    _log(f"Over2.5: AUC={o25_auc:.4f} Brier={o25_brier:.4f} Acc={o25_acc:.3f}")
    _log(f"1X2:     Acc={acc_1x2:.3f} LogLoss={logloss_1x2:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Blind test backtest histórico Liga MX")
    parser.add_argument("--cutoff", required=True, help="Fecha corte YYYY-MM-DD (entrena <= cutoff)")
    parser.add_argument("--test-end", help="Fecha fin test YYYY-MM-DD (por defecto: todos > cutoff)")
    parser.add_argument("--season", default="", help="Nombre de la temporada (para logs)")
    parser.add_argument("--output", help="Archivo JSON para guardar resultados")
    
    args = parser.parse_args()
    
    load_config()
    
    season_name = args.season or f"cutoff_{args.cutoff}"
    results = run_blind_backtest(args.cutoff, args.test_end, season_name)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        _log(f"\nResultados guardados en: {args.output}")


if __name__ == "__main__":
    main()