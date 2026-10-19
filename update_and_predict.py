#!/usr/bin/env python3
"""
update_and_predict.py — Actualiza el master con la temporada en curso desde
football-data.co.uk y genera predicciones BTTS de la proxima jornada.

Salidas:
  - data/liga_mx_master.parquet  (actualizado)
  - data/liga_mx_features.parquet (regeneradas)
  - predictions/btts_jornada.csv (partidos futuros + prob)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
PRED = FUTBOL / "predictions"
sys.path.insert(0, str(FUTBOL))
from team_names import canon_series  # noqa: E402

PRED.mkdir(exist_ok=True)
TMP = Path("/home/petyt/.hermes/cache/scratch/mex_now.csv")


def log(msg):
    print(msg, flush=True)


def download_current():
    url = "https://football-data.co.uk/new/MEX.csv"
    subprocess.run([
        "curl", "-sL", "-o", str(TMP),
        "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
        url,
    ], check=True)
    df = pd.read_csv(TMP, encoding="utf-8-sig", low_memory=False)
    df = df.rename(columns={"ï»¿Country": "Country"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["home"] = canon_series(df["Home"])
    df["away"] = canon_series(df["Away"])
    df = df.rename(columns={"HG": "hs", "AG": "as_", "Res": "res_1x2",
                            "PSCH": "odd_h_psc", "PSCD": "odd_d_psc", "PSCA": "odd_a_psc",
                            "MaxCH": "odd_h_max", "MaxCD": "odd_d_max", "MaxCA": "odd_a_max",
                            "AvgCH": "odd_h_avg", "AvgCD": "odd_d_avg", "AvgCA": "odd_a_avg",
                            "BFECH": "odd_h_bfe", "BFECD": "odd_d_bfe", "BFECA": "odd_a_bfe",
                            "B365CH": "odd_h_b365", "B365CD": "odd_d_b365", "B365CA": "odd_a_b365"})
    odd_cols = [c for c in df.columns if c.startswith("odd_")]
    df = df[["date", "Season", "home", "away", "hs", "as_", "res_1x2"] + odd_cols]
    log(f"Descargado: {len(df)} filas, {df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"  Sin jugar (hs nulo): {df['hs'].isna().sum()}")
    return df


def main():
    # 1. master actual (con features) para conocer el max de fecha jugada
    old = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date")
    old_played = old[old["hs"].notna()]
    log(f"Master actual: {len(old)} filas, ultimo jugado: {old_played['date'].max().date()}")

    # 2. descargar datos frescos
    fresh = download_current()
    fresh_played = fresh[fresh["hs"].notna()]
    fresh_future = fresh[fresh["hs"].isna()]

    # 3. nuevos partidos jugados (no en el master)
    key_old = set(zip(old_played["date"], old_played["home"], old_played["away"]))
    key_new = list(zip(fresh_played["date"], fresh_played["home"], fresh_played["away"]))
    is_new = [k not in key_old for k in key_new]
    new_matches = fresh_played[np.array(is_new)]
    log(f"Nuevos partidos jugados: {len(new_matches)}")

    # 4. reconstruir master: base kaggle + football-data nuevo completo
    #    (simple: master viejo + partidos nuevos, manteniendo columnas del master base)
    base_cols = ["date", "Season", "home", "away", "hs", "as_", "res_1x2",
                 "odd_h_psc", "odd_d_psc", "odd_a_psc", "odd_h_max", "odd_d_max", "odd_a_max",
                 "odd_h_avg", "odd_d_avg", "odd_a_avg", "odd_h_bfe", "odd_d_bfe", "odd_a_bfe",
                 "odd_h_b365", "odd_d_b365", "odd_a_b365"]
    new_base = new_matches[base_cols].copy() if len(new_matches) else pd.DataFrame(columns=base_cols)
    old_base = old[[c for c in base_cols if c in old.columns]]
    # kaggle pre-2018 no tiene Season/odds psc: rellenar
    master = pd.concat([old_base, new_base], ignore_index=True, sort=False)
    # concat con frame vacio puede degradar date a object: re-castear
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values("date").drop_duplicates(subset=["date", "home", "away"], keep="last")
    played_mask = master["hs"].notna() & master["as_"].notna()
    master["btts"] = np.where(played_mask, ((master["hs"] > 0) & (master["as_"] > 0)).astype(float), np.nan)
    master["over25"] = np.where(played_mask, ((master["hs"] + master["as_"]) > 2.5).astype(float), np.nan)
    master["torneo"] = master["date"].map(lambda d: "Apertura" if d.month >= 7 else "Clausura")
    master["anio_torneo"] = master["date"].dt.year
    master.to_parquet(DATA / "liga_mx_master.parquet", index=False)
    master.to_csv(DATA / "liga_mx_master.csv", index=False, encoding="utf-8-sig")
    log(f"Master actualizado: {len(master)} filas (agregados {len(new_matches)})")

    # 5. re-ejecutar make_features.py sobre el master actualizado
    log("\nRegenerando features...")
    r = subprocess.run([sys.executable, str(FUTBOL / "make_features.py")],
                       capture_output=True, text=True)
    log(r.stdout[-1500:] if r.returncode == 0 else f"ERROR:\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        sys.exit(1)

    # 6. re-entrenar poisson_btts.py (regenera lambdas y guarda)
    log("\nRegenerando lambdas...")
    r = subprocess.run([sys.executable, str(FUTBOL / "poisson_btts.py")],
                       capture_output=True, text=True)
    log(r.stdout[-1200:] if r.returncode == 0 else f"ERROR:\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        sys.exit(1)

    # 7. predecir la proxima jornada
    feats = pd.read_parquet(DATA / "liga_mx_features.parquet").sort_values("date")
    future = feats[feats["hs"].isna()].copy()
    log(f"\nPartidos futuros con features: {len(future)}")
    if not len(future):
        # football-data no publica fixtures futuros de Liga MX -> predecir la
        # jornada desde los fixtures de ESPN (predict_jornada.py, modelo v2)
        log("Sin fixtures en football-data; refrescando fixtures ESPN...")
        r = subprocess.run([sys.executable, str(FUTBOL / "fetch_fixtures_espn.py")],
                           capture_output=True, text=True)
        log(r.stdout[-600:] if r.returncode == 0 else f"ERROR fixtures ESPN:\n{r.stdout}\n{r.stderr}")
        if r.returncode != 0:
            sys.exit(1)
        log("Prediciendo jornada desde ESPN...")
        r = subprocess.run([sys.executable, str(FUTBOL / "predict_jornada.py")],
                           capture_output=True, text=True)
        log(r.stdout[-2500:] if r.returncode == 0 else f"ERROR predict_jornada:\n{r.stdout}\n{r.stderr}")
        return

    import pickle
    # entrenar modelo final con TODOS los datos jugados
    FEATS = ["elo_diff", "elo_h", "elo_a", "rest_h", "rest_a", "rest_diff",
             "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
             "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
             "gf_home_as_home", "ga_home_as_home", "gf_away_as_away", "ga_away_as_away",
             "imp_h", "imp_d", "imp_a", "lam_h", "lam_a"]
    train = feats[feats["btts"].notna()].copy()
    X_all = train[FEATS]
    y_all = train["btts"].astype(int).values
    means = X_all.mean()
    X_all = X_all.fillna(means)

    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                          random_state=42)
    model.fit(X_all, y_all)
    with open(DATA / "btts_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATS, "means": means.to_dict()}, f)
    log(f"Modelo final entrenado con {len(train)} partidos")

    Xf = future[FEATS].fillna(means)
    future["p_btts"] = model.predict_proba(Xf)[:, 1]
    out = future[["date", "home", "away", "odd_h", "odd_d", "odd_a", "p_btts"]].copy()
    out = out.sort_values("p_btts", ascending=False)
    out.to_csv(PRED / "btts_jornada.csv", index=False, encoding="utf-8-sig")
    log(f"\nPredicciones guardadas: {PRED / 'btts_jornada.csv'}")
    log("\nPRÓXIMA JORNADA (p_btts ordenado):")
    log(out.to_string(index=False))


if __name__ == "__main__":
    main()
