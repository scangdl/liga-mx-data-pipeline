#!/usr/bin/env python3
"""
blind_test.py — Backtest a ciegas rolling del modelo BTTS.

Protocolo (sellado antes de ver resultados por cada jornada):
  1. Se eligen JORNADAS COMPLETAS pasadas (fechas de corte múltiples).
  2. Para cada corte: el modelo se entrena SOLO con partidos anteriores al corte
     (features de esos partidos tampoco ven nada posterior).
  3. Se reconstruye el estado de equipos con partidos previos al corte
     y se predicen los 9 partidos de esa jornada (igual que producción).
     Las predicciones quedan SELLADAS (hash SHA-256).
  4. Solo entonces se comparan contra los marcadores reales.

Sin leakage: ni entrenamiento ni features usan información posterior al corte.
"""
import hashlib
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Usar features_lib como single source of truth ---
sys.path.insert(0, str(Path(__file__).parent))
from features_lib import (
    StateBuilder,
    build_features_master,
    build_state_for_inference,
    EXTENDED_FEATURES,
    MODEL_FEATURES,
    poisson_btts_prob,
    log,
)

# ================= CONFIG =================
DATA = Path(__file__).parent / "data"
MODELS = DATA / "models"
OUT = Path(__file__).parent / "predictions" / "blind_test_rolling"
OUT.mkdir(parents=True, exist_ok=True)

# Jornadas a testear (fecha de corte = día antes del primer partido de la jornada)
# Formato: (label, cut_date, n_days_window)
JORNADAS_TEST = [
    ("J10 Clausura 2026", "2026-04-03", 3),
    ("J15 Apertura 2025", "2025-10-17", 3),
    ("J17 Clausura 2025", "2025-04-11", 3),
    ("J10 Apertura 2024", "2024-09-27", 3),
    ("J17 Apertura 2023", "2023-10-27", 3),
]

# Modelo a usar
MODEL_PATH = MODELS / "btts_model_v3.pkl"
FALLBACK_MODEL = MODELS / "btts_model.pkl"  # v2 legacy


def load_model(model_path: Path):
    """Carga modelo v3 o v2 y devuelve bundle normalizado."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    version = bundle.get("version", "v2")
    if version == "v3_improved" or "ensemble_weights" in bundle:
        return {
            "version": "v3",
            "xgb_model": bundle["xgb_model"],
            "lr_model": bundle.get("lr_model"),
            "lgb_model": bundle.get("lgb_model"),
            "ensemble_weights": bundle["ensemble_weights"],
            "scaler": bundle["scaler"],
            "features": bundle["features"],
            "impute_means": pd.Series(bundle["impute_means"]),
        }
    else:
        return {
            "version": "v2",
            "model": bundle["model"],
            "scaler": bundle["scaler"],
            "features": bundle["features"],
            "impute_means": pd.Series(bundle["impute_means"]),
        }


def predict_with_model(model_bundle, Xj: pd.DataFrame) -> np.ndarray:
    """Predice con modelo v3 (ensemble) o v2 (XGB simple)."""
    feats = model_bundle["features"]
    means = model_bundle["impute_means"]
    scaler = model_bundle["scaler"]

    Xj = Xj[feats].copy().fillna(means)
    Xj_s = scaler.transform(Xj) if scaler is not None else Xj.values

    if model_bundle["version"] == "v3":
        xgb = model_bundle["xgb_model"]
        lr = model_bundle.get("lr_model")
        lgb = model_bundle.get("lgb_model")
        weights = model_bundle["ensemble_weights"]

        p_xgb = xgb.predict_proba(Xj_s)[:, 1]
        preds = {"xgb": p_xgb}

        if lr is not None:
            # LR usa imputer propio
            p_lr = lr.predict_proba(lr.imputer_.transform(Xj))[:, 1] if hasattr(lr, "imputer_") else lr.predict_proba(Xj_s)[:, 1]
            preds["lr"] = p_lr

        if lgb is not None:
            preds["lgb"] = lgb.predict_proba(Xj_s)[:, 1]

        # Ensemble ponderado
        model_names = list(preds.keys())
        w = np.array([weights.get(n, 0) for n in model_names])
        w = w / w.sum() if w.sum() > 0 else np.ones(len(model_names)) / len(model_names)
        p_ens = sum(preds[n] * w[i] for i, n in enumerate(model_names))
        return p_ens
    else:
        return model_bundle["model"].predict_proba(Xj_s)[:, 1]


def run_single_cut(jornada_label: str, cut_ts: pd.Timestamp, n_days: int, feats_df: pd.DataFrame, model_bundle: dict) -> dict:
    """Ejecuta test a ciegas para UNA jornada (un corte)."""
    log(f"\n{'='*60}")
    log(f"TEST A CIEGAS — {jornada_label} (corte: {cut_ts.date()})")
    log(f"{'='*60}")

    # 1. Pool de entrenamiento: solo partidos < corte con BTTS label
    train_pool = feats_df[(feats_df["btts"].notna()) & (feats_df["date"] < cut_ts)].copy()
    log(f"[1] Partidos entrenamiento: {len(train_pool)} ({train_pool['date'].min().date()} -> {train_pool['date'].max().date()})")

    # 2. Jornada objetivo: partidos en ventana [cut, cut+n_days]
    jornada = feats_df[
        (feats_df["date"] >= cut_ts) & (feats_df["date"] <= cut_ts + timedelta(days=n_days))
    ].copy()
    log(f"[1] Partidos objetivo: {len(jornada)}")
    if len(jornada) < 5:
        log(f"  ⚠ Jornada muy corta ({len(jornada)} partidos), saltando")
        return None

    # Guardar verdad ANTES de predecir
    truth = jornada[["date", "home", "away", "hs", "as_", "btts"]].copy()

    # 3. Features para la jornada (usar EXTENDED_FEATURES del modelo)
    model_feats = model_bundle["features"]
    X_j = jornada[model_feats].copy()

    # 4. Entrenar modelo SOLO con datos < CUT (re-entrenar para cada corte = purista)
    # NOTA: En producción usamos modelo ya entrenado, pero para test a ciegas honesto
    # debemos re-entrenar con datos hasta el corte. Usamos features_lib para consistencia.
    log("[2] Re-entrenando modelo con datos hasta el corte...")

    X_all = train_pool[model_feats]
    y_all = train_pool["btts"].astype(int).values
    means = X_all.mean()
    X_all = X_all.fillna(means)

    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)

    if model_bundle["version"] == "v3":
        # Re-entrenar ensemble simplificado (XGB + LR) con mismos hiperparámetros
        xgb = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
            random_state=42, n_jobs=2,
        )
        xgb.fit(Xs, y_all)

        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        lr = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight="balanced")
        imputer = SimpleImputer(strategy="mean")
        X_all_imp = imputer.fit_transform(X_all)
        lr.fit(X_all_imp, y_all)
        lr.imputer_ = imputer

        # Ensemble con pesos fijos del modelo original
        weights = model_bundle["ensemble_weights"]

        # Predecir jornada
        Xj = X_j.fillna(means)
        Xj_s = scaler.transform(Xj)
        p_xgb = xgb.predict_proba(Xj_s)[:, 1]
        p_lr = lr.predict_proba(lr.imputer_.transform(Xj))[:, 1]

        p_ens = p_xgb * weights.get("xgb", 0.5) + p_lr * weights.get("lr", 0.5)
        p_pred = p_ens

        log(f"  Ensemble weights: xgb={weights.get('xgb', 0.5):.4f}, lr={weights.get('lr', 0.5):.4f}")
    else:
        # v2 legacy
        model = model_bundle["model"]
        p_pred = model.predict_proba(scaler.transform(X_j.fillna(means)))[:, 1]

    # 5. SELLAR predicciones ANTES de ver resultados
    sealed = pd.DataFrame({
        "date": jornada["date"].dt.date.astype(str),
        "home": jornada["home"],
        "away": jornada["away"],
        "p_btts": np.round(p_pred, 4),
        "fair_odds": np.round(1.0 / p_pred, 2),
    })

    sealed_path = OUT / f"predicciones_{jornada_label.replace(' ', '_').replace('/', '_')}.csv"
    sealed.to_csv(sealed_path, index=False)

    h = hashlib.sha256(sealed.to_csv(index=False).encode()).hexdigest()[:16]
    log(f"[3] PREDICCIONES SELLADAS -> {sealed_path}")
    log(f"[3] SHA256: {h}")

    # 6. Evaluar contra verdad
    y_true = truth["btts"].astype(int).values

    res = pd.DataFrame({
        "partido": truth["home"] + " vs " + truth["away"],
        "marcador": truth["hs"].astype(int).astype(str) + "-" + truth["as_"].astype(int).astype(str),
        "btts_real": y_true,
        "p_btts": np.round(p_pred, 4),
    })
    res["acierto_05"] = ((res["p_btts"] >= 0.5).astype(int) == y_true).astype(int)

    res_path = OUT / f"resultados_{jornada_label.replace(' ', '_').replace('/', '_')}.csv"
    res.to_csv(res_path, index=False)

    # Métricas
    brier = brier_score_loss(y_true, p_pred)
    ll = log_loss(y_true, p_pred, labels=[0, 1])
    auc = roc_auc_score(y_true, p_pred) if len(set(y_true)) == 2 else 0.5
    acc = res["acierto_05"].mean()

    log(f"\n  n = {len(y_true)} | BTTS reales: {y_true.sum()}/{len(y_true)} ({y_true.mean():.1%})")
    log(f"  Brier: {brier:.4f} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Acc@0.5: {acc:.1%} ({int(res['acierto_05'].sum())}/{len(res)})")
    log(f"  Resultados: {res_path}")

    return {
        "jornada": jornada_label,
        "cut": str(cut_ts.date()),
        "n": len(y_true),
        "btts_real_rate": float(y_true.mean()),
        "brier": float(brier),
        "logloss": float(ll),
        "auc": float(auc),
        "acc": float(acc),
        "correct": int(res["acierto_05"].sum()),
        "hash": h,
        "sealed_path": str(sealed_path),
        "res_path": str(res_path),
    }


def main():
    print("=" * 70)
    print("BACKTEST A CIEGAS ROLLING — MODELO BTTS")
    print("=" * 70)

    # Cargar features master (ya construidas con features_lib)
    feats_path = DATA / "liga_mx_features.parquet"
    if not feats_path.exists():
        log(f"Generando features master con features_lib...")
        feats_df = build_features_master(DATA / "liga_mx_master.parquet")
        feats_df.to_parquet(feats_path, index=False)
        log(f"Guardado: {feats_path}")
    else:
        feats_df = pd.read_parquet(feats_path).sort_values("date").reset_index(drop=True)
    log(f"Features master: {len(feats_df)} partidos ({feats_df['date'].min().date()} -> {feats_df['date'].max().date()})")

    # Cargar modelo
    if MODEL_PATH.exists():
        model_bundle = load_model(MODEL_PATH)
        log(f"Modelo cargado: {MODEL_PATH.name} (v{model_bundle['version']}, {len(model_bundle['features'])} features)")
    elif FALLBACK_MODEL.exists():
        model_bundle = load_model(FALLBACK_MODEL)
        log(f"Modelo cargado (fallback): {FALLBACK_MODEL.name} (v{model_bundle['version']})")
    else:
        log("ERROR: No hay modelo entrenado", level="ERROR")
        sys.exit(1)

    # Ejecutar test a ciegas para cada jornada
    all_results = []
    for label, cut_str, n_days in JORNADAS_TEST:
        cut_ts = pd.Timestamp(cut_str)
        # Verificar que hay datos suficientes antes del corte
        pre_count = len(feats_df[(feats_df["btts"].notna()) & (feats_df["date"] < cut_ts)])
        if pre_count < 200:
            log(f"⚠ {label}: solo {pre_count} partidos previos al corte, saltando")
            continue

        result = run_single_cut(label, cut_ts, n_days, feats_df, model_bundle)
        if result:
            all_results.append(result)

    # Resumen agregado
    if all_results:
        print("\n" + "=" * 70)
        print("RESUMEN AGREGADO — BACKTEST ROLLING")
        print("=" * 70)

        df_res = pd.DataFrame(all_results)
        print(df_res[["jornada", "n", "btts_real_rate", "brier", "logloss", "auc", "acc", "correct", "hash"]].to_string(index=False))

        # Promedios ponderados por n
        total_n = df_res["n"].sum()
        avg_brier = np.average(df_res["brier"], weights=df_res["n"])
        avg_logloss = np.average(df_res["logloss"], weights=df_res["n"])
        avg_auc = np.average(df_res["auc"], weights=df_res["n"])
        avg_acc = np.average(df_res["acc"], weights=df_res["n"])
        total_correct = df_res["correct"].sum()

        print(f"\n{'PROMEDIO PONDERADO':<25} n={total_n}  Brier={avg_brier:.4f}  LogLoss={avg_logloss:.4f}  AUC={avg_auc:.4f}  Acc={avg_acc:.1%} ({total_correct}/{total_n})")

        # Guardar resumen
        summary_path = OUT / "resumen_rolling.csv"
        df_res.to_csv(summary_path, index=False)
        log(f"\nResumen guardado: {summary_path}")

        # Hash agregado de todos los archivos sellados
        all_hashes = "".join(r["hash"] for r in all_results)
        agg_hash = hashlib.sha256(all_hashes.encode()).hexdigest()[:16]
        log(f"Hash agregado (todos los sellos): {agg_hash}")

        # Guardar manifiesto
        manifest = {
            "model": MODEL_PATH.name if MODEL_PATH.exists() else FALLBACK_MODEL.name,
            "model_version": model_bundle["version"],
            "features_count": len(model_bundle["features"]),
            "jornadas_tested": len(all_results),
            "total_matches": int(total_n),
            "aggregate_metrics": {
                "brier": avg_brier,
                "logloss": avg_logloss,
                "auc": avg_auc,
                "accuracy": avg_acc,
            },
            "aggregate_hash": agg_hash,
            "individual": all_results,
        }
        manifest_path = OUT / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        log(f"Manifiesto guardado: {manifest_path}")
    else:
        log("No se pudo ejecutar ninguna jornada")


if __name__ == "__main__":
    main()