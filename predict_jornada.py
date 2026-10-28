#!/usr/bin/env python3
"""
predict_jornada.py — Predicciones BTTS para los fixtures futuros de ESPN.

Compatible con el modelo de producción v2 (33 features base + extendidas: odds+tabla+H2H+SOS+
congestion, scaler + impute_means). Usa features_lib.StateBuilder para reconstruir
el estado actual de cada equipo desde el master jugado (Elo final, forma, descanso,
EWMA ataque/defensa por rol, tabla del torneo en curso, H2H, SOS y congestión)
replicando la misma lógica pre-partido de make_features.py, y aplica btts_model.pkl.

Odds 1X2: si la Odds API trae h2h para el partido (data/odds_api_jornada.csv),
se usan como imp_* normalizadas; si no, se imputan con las medias de train
(consistente con el entrenamiento, donde ~37% de filas también fueron
imputadas). NOTA: el modelo v2 fue entrenado con imp_* de closing odds de
football-data; usar las de la Odds API es la mejor aproximación disponible.

Salida: predictions/btts_jornada.csv (p_btts, cuota justa, p_poisson, lambdas)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
PRED = FUTBOL / "predictions"
PRED.mkdir(exist_ok=True)
sys.path.insert(0, str(FUTBOL))

from team_names import canon_series  # noqa: E402
from features_lib import StateBuilder, build_state_for_inference, log, poisson_btts_prob, american_to_decimal  # noqa: E402


def log(msg):
    print(msg, flush=True)


def main():
    # 1. Cargar modelo v2
    with open(DATA / "btts_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    FEATS = bundle["features"]
    means = pd.Series(bundle["impute_means"])
    log(f"Modelo: {bundle.get('version')} ({len(FEATS)} features, scaler={'si' if scaler else 'no'})")

    # 2. Reconstruir estado final desde master jugado usando features_lib
    log("Reconstruyendo estado desde master jugado...")
    state = build_state_for_inference(DATA / "liga_mx_master.parquet")
    log(f"Estado reconstruido: {len(state['elo'])} equipos, torneo actual: {state.get('current_tor_id')}")

    # 3. Cargar odds 1X2 de la Odds API (para imp_*)
    h2h_odds = {}
    api_csv = DATA / "odds_api_jornada.csv"
    if api_csv.exists():
        try:
            ou_df = pd.read_csv(api_csv)
            for _, r_ou in ou_df.iterrows():
                try:
                    d_ou = pd.to_datetime(r_ou["commence"], utc=True).tz_convert(
                        "America/Mexico_City").tz_localize(None).normalize()
                except Exception:
                    continue
                h2h_odds[(d_ou, str(r_ou["home"]), str(r_ou["away"]))] = {
                    "odd_h": r_ou.get("h2h_h_med"), "odd_d": r_ou.get("h2h_d_med"),
                    "odd_a": r_ou.get("h2h_a_med"), "over": r_ou.get("over25_med")}
        except Exception as e:
            log(f"  (odds API csv ilegible: {e})")

    master_teams = set(pd.read_parquet(DATA / "liga_mx_master.parquet")["home"].unique())

    def canon_team_espn(name):
        try:
            c = canon_series(pd.Series([str(name)]))[0]
            if c in master_teams:
                return c
            stripped = str(c).strip().lower()
            for suf in (" fc", " uanl", " unam"):
                if stripped.endswith(suf):
                    c2 = canon_series(pd.Series([stripped[: -len(suf)]]))[0]
                    if c2 in master_teams:
                        return c2
            return c
        except Exception:
            return str(name)

    # 3. Cargar fixtures ESPN
    with open(DATA / "espn_fixtures.json") as f:
        fixtures = json.load(f)
    log(f"Fixtures: {len(fixtures)}")

    # Reconstruir StateBuilder desde estado guardado para inference
    master_all = pd.read_parquet(DATA / "liga_mx_master.parquet").sort_values("date")
    sb = StateBuilder.from_state(master_all, state)

    # Extraer estado reconstruido para funciones helper compatibles
    elo = state["elo"]
    last_match = {k: v for k, v in state["last_match"].items()}
    hist = {k: list(v) for k, v in state["hist"].items()}
    ewma = {k: {r: list(v) for r, v in v.items()} for k, v in state["ewma"].items()}
    team_state = {k: {"opp_elo": list(v["opp_elo"]), "dates": list(v["dates"])}
                  for k, v in state["team_state"].items()}
    tabla_torneo = {k: {tk: list(tv) for tk, tv in v.items()} for k, v in state["tabla_torneo"].items()}
    h2h = {k: list(v) for k, v in state["h2h"].items()}
    lgf_h = float(state["lgf_h"]) if isinstance(state["lgf_h"], (int, float)) else float(np.mean(state["lgf_h"]))
    lgf_a = float(state["lgf_a"]) if isinstance(state["lgf_a"], (int, float)) else float(np.mean(state["lgf_a"]))
    current_tor_id = state.get("current_tor_id")

    # Helpers compatibles con código original
    def form_features(team):
        h = hist.get(team, [])
        n = len(h)
        return {
            "gf10": np.mean([x[0] for x in h[-10:]]) if n else np.nan,
            "ga10": np.mean([x[1] for x in h[-10:]]) if n else np.nan,
            "btts_rate10": np.mean([x[2] for x in h[-10:]]) if n else np.nan,
            "ppg10": np.mean([x[3] for x in h[-10:]]) if n else np.nan,
        }

    def lam_features(home, away):
        sh = ewma.get(home, {}).get(1)
        sa = ewma.get(away, {}).get(0)
        if not sh or not sa or sh[0] is None or sa[0] is None:
            return np.nan, np.nan
        ah, dh = sh
        aa, da = sa
        lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
        lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
        return lam_h, lam_a

    def table_pos(team):
        th = tabla_torneo.get(team)
        if not th or th[2] == 0:
            return (np.nan, np.nan)
        pos = 1 + sum(1 for v in tabla_torneo.values() if v[0] > th[0])
        return (pos, th[0] / th[2])

    def h2h_features(h, a):
        key = tuple(sorted([h, a]))
        prev = h2h.get(key, [])
        if not prev:
            return (np.nan, np.nan, np.nan, 0)
        h_home = h == key[0]
        gf_side = [x[0] if h_home else x[1] for x in prev[-6:]]
        ga_side = [x[1] if h_home else x[0] for x in prev[-6:]]
        return (float(np.mean([x[2] for x in prev[-6:]])),
                float(np.mean(gf_side)), float(np.mean(ga_side)), len(prev[-6:]))

    def sos_cong(team, d):
        st = team_state.get(team)
        sos = float(np.mean(st["opp_elo"][-5:])) if st and st.get("opp_elo") else np.nan
        cong = sum(1 for x in st["dates"][-5:] if (d - x).days <= 14) if st and st.get("dates") else 0
        return sos, cong

    # 3. Fixtures ESPN -> features -> p_btts
    with open(DATA / "espn_fixtures.json") as f:
        fixtures = json.load(f)
    log(f"Fixtures: {len(fixtures)}")

    rows = []
    for fx in fixtures:
        if fx.get("state") != "pre" or not fx.get("home"):
            continue
        home = canon_series(pd.Series([fx["home"]]))[0]
        away = canon_series(pd.Series([fx["away"]]))[0]
        d_local = pd.to_datetime(fx["date"], utc=True).tz_convert(
            "America/Mexico_City").tz_localize(None).normalize()

        fh = form_features(home)
        fa = form_features(away)
        lam_h, lam_a = lam_features(home, away)
        pos_h, ppg_h = table_pos(home)
        pos_a, ppg_a = table_pos(away)
        h2h_btts, h2h_gf, h2h_ga, h2h_n = h2h_features(home, away)
        sos_h, cong_h = sos_cong(home, d_local)
        sos_a, cong_a = sos_cong(away, d_local)

        hhome = [y for y in hist.get(home, []) if y[4] == 1][-6:]
        aaway = [y for y in hist.get(away, []) if y[4] == 0][-6:]

        # odds 1X2 (Odds API) -> imp_* normalizadas; fallback DraftKings; sino NaN
        odd_h = odd_d = odd_a = np.nan
        for (d_ou, h_ou, a_ou), oo in h2h_odds.items():
            if d_ou == d_local and canon_team_espn(h_ou) == home and canon_team_espn(a_ou) == away:
                vals = [oo["odd_h"], oo["odd_d"], oo["odd_a"]]
                if all(pd.notna(v) and float(v) > 1.01 for v in vals):
                    ih, idd, ia = 1 / float(oo["odd_h"]), 1 / float(oo["odd_d"]), 1 / float(oo["odd_a"])
                    s = ih + idd + ia
                    odd_h, odd_d, odd_a = ih / s, idd / s, ia / s
                break
        if np.isnan(odd_h):
            dh = american_to_decimal(fx.get("ml_home"))
            dd = american_to_decimal(fx.get("draw_ml"))
            da = american_to_decimal(fx.get("ml_away"))
            vals = [dh, dd, da]
            if all(v is not None for v in vals):
                ih, idd, ia = 1 / dh, 1 / dd, 1 / da
                s = ih + idd + ia
                odd_h, odd_d, odd_a = ih / s, idd / s, ia / s

        rows.append({
            "date": d_local,
            "home": home, "away": away,
            "espn_home": fx["home"], "espn_away": fx["away"],
            "elo_h": elo.get(home, 1500.0), "elo_a": elo.get(away, 1500.0),
            "rest_h": (d_local - last_match[home]).days if home in last_match else np.nan,
            "rest_a": (d_local - last_match[away]).days if away in last_match else np.nan,
            "gf10_h": fh["gf10"], "ga10_h": fa["ga10"],
            "btts_rate10_h": fh["btts_rate10"], "ppg10_h": fh["ppg10"],
            "gf10_a": fa["gf10"], "ga10_a": fa["ga10"],
            "btts_rate10_a": fa["btts_rate10"], "ppg10_a": fa["ppg10"],
            "gf_home_as_home": np.mean([x[0] for x in hhome]) if hhome else np.nan,
            "ga_home_as_home": np.mean([x[1] for x in hhome]) if hhome else np.nan,
            "gf_away_as_away": np.mean([x[0] for x in aaway]) if aaway else np.nan,
            "ga_away_as_away": np.mean([x[1] for x in aaway]) if aaway else np.nan,
            "imp_h": odd_h, "imp_d": odd_d, "imp_a": odd_a,
            "pos_table_h": pos_h, "pos_table_a": pos_a,
            "ppg_table_h": ppg_h, "ppg_table_a": ppg_a,
            "h2h_btts_rate": h2h_btts, "h2h_gf": h2h_gf, "h2h_ga": h2h_ga, "h2h_n": h2h_n,
            "sos_h": sos_h, "sos_a": sos_a,
            "cong_h": cong_h, "cong_a": cong_a,
            "lam_h": lam_h, "lam_a": lam_a,
            "ou_line": fx.get("line"),
        })

    X = pd.DataFrame(rows)
    X["elo_diff"] = X["elo_h"] - X["elo_a"]
    X["rest_diff"] = X["rest_h"] - X["rest_a"]
    X["p_btts_poisson"] = np.where(
        X["lam_h"].notna() & X["lam_a"].notna(),
        (1 - poisson.pmf(0, X["lam_h"].fillna(0))) * (1 - poisson.pmf(0, X["lam_a"].fillna(0))),
        np.nan,
    )

    Xf = X[FEATS].copy()
    n_imputed = int(Xf.isna().any(axis=1).sum())
    Xf = Xf.fillna(means)
    Xs = scaler.transform(Xf) if scaler is not None else Xf.values
    X["p_btts"] = model.predict_proba(Xs)[:, 1]
    X["fair_odds"] = 1 / X["p_btts"]
    log(f"imp_* imputadas con medias de train en {n_imputed}/{len(X)} fixtures")

    # Salida
    out = X[["date", "home", "away", "p_btts", "fair_odds", "p_btts_poisson",
             "lam_h", "lam_a", "elo_h", "elo_a", "rest_h", "rest_a", "ou_line",
             "pos_table_h", "pos_table_a"]].copy()
    out = out.sort_values(["date", "p_btts"], ascending=[True, False])
    out.to_csv(PRED / "btts_jornada.csv", index=False, encoding="utf-8-sig")
    log(f"\nGuardado: {PRED / 'btts_jornada.csv'}")

    log("\n" + "=" * 88)
    log("PREDICCIONES BTTS — proxima jornada")
    log("=" * 88)
    with pd.option_context("display.width", 200):
        log(out.head(18).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("\nNota: fair_odds = 1/p (cuota justa, sin margen). p_btts_poisson es el")
    log("cruce Poisson puro como referencia; p_btts es el modelo XGB v2 final.")


if __name__ == "__main__":
    main()