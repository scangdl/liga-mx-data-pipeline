#!/usr/bin/env python3
"""
predict_jornada.py — Predicciones BTTS para los fixtures futuros de ESPN.

Compatible con el modelo de produccion v2 (33 features: odds+tabla+H2H+SOS+
congestion, scaler + impute_means). Reconstruye el estado actual de cada
equipo desde el master jugado (Elo final, forma, descanso, EWMA ataque/defensa
por rol, tabla del torneo en curso, H2H, SOS y congestión) replicando la
misma lógica pre-partido de make_features.py, y aplica btts_model.pkl.

Odds 1X2: si la Odds API trae h2h para el partido (data/odds_api_jornada.csv),
se usan como imp_* normalizadas; si no, se imputan con las medias de train
(consistente con el entrenamiento, donde ~37% de filas también fueron
imputadas). NOTA: el modelo v2 fue entrenado con imp_* de closing odds de
football-data; usar las de la Odds API es la mejor aproximacion disponible.

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


def log(msg):
    print(msg, flush=True)


def american_to_decimal(ml):
    if ml is None:
        return None
    try:
        ml = int(ml)
    except (ValueError, TypeError):
        return None
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1


# ============================================================
# 1. Estado final por equipo desde el master jugado
#    (misma logica secuencial de make_features.py, solo partidos jugados)
# ============================================================
master = pd.read_parquet(DATA / "liga_mx_master.parquet").sort_values("date").reset_index(drop=True)
played = master[master["hs"].notna() & master["as_"].notna()].reset_index(drop=True)
log(f"Master jugado: {len(played)} partidos, ultimo: {played['date'].max().date()}")

K, HOME_ADV = 32.0, 80.0
ALPHA = 0.93
elo = {}
last_match = {}
hist = {}            # team -> [(gf, ga, btts, pts, role), ...]
ewma = {}            # team -> {1: [atk, def], 0: [atk, def]}
team_state = {}       # team -> {"opp_elo": [...], "dates": [...]}  (SOS/congestion)
tabla = {}            # (anio, torneo) -> {team: [pts, gf, ga]}
tabla_id = None
h2h = {}              # (teamA, teamB) -> [(gf_A, ga_A, btts, date), ...]
league_gf_home, league_gf_away = [], []

for _, r in played.iterrows():
    h, a = r["home"], r["away"]
    hs, as_ = float(r["hs"]), float(r["as_"])
    eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
    exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV - ea) / 400.0)))
    s = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
    margen = abs(hs - as_)
    mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else 1.75 + (margen - 3) * 0.125)
    delta = K * mult * (s - exp_h)
    elo[h] = eh + delta
    elo[a] = ea - delta

    last_match[h] = last_match[a] = r["date"]

    ptos_h = 3.0 if hs > as_ else (1.0 if hs == as_ else 0.0)
    ptos_a = 3.0 - ptos_h if ptos_h in (0.0, 3.0) else 1.0
    btts = 1.0 if (hs > 0 and as_ > 0) else 0.0
    hist.setdefault(h, []).append((hs, as_, btts, ptos_h, 1))
    hist.setdefault(a, []).append((as_, hs, btts, ptos_a, 0))

    for team, role, gf, ga in ((h, 1, hs, as_), (a, 0, as_, hs)):
        st = ewma.setdefault(team, {}).setdefault(role, [None, None])
        st[0] = gf if st[0] is None else ALPHA * st[0] + (1 - ALPHA) * gf
        st[1] = ga if st[1] is None else ALPHA * st[1] + (1 - ALPHA) * ga

    # --- tabla por torneo (anio_torneo, torneo) ---
    tor_id = (r["anio_torneo"], r["torneo"])
    if tor_id != tabla_id:
        tabla = {}
        tabla_id = tor_id
    th = tabla.get(h, [0, 0, 0])
    ta = tabla.get(a, [0, 0, 0])
    th[0] += ptos_h; th[1] += hs; th[2] += as_
    ta[0] += ptos_a; ta[1] += as_; ta[2] += hs
    tabla[h] = th
    tabla[a] = ta

    # --- H2H (perspectiva key[0]) ---
    key = tuple(sorted([h, a]))
    k0_gf = hs if h == key[0] else as_
    k1_gf = as_ if h == key[0] else hs
    h2h.setdefault(key, []).append((k0_gf, k1_gf, btts, r["date"]))

    # --- SOS y congestión ---
    for team, pts, opp in [
        (h, ptos_h, elo.get(a, 1500.0)),
        (a, ptos_a, elo.get(h, 1500.0)),
    ]:
        st = team_state.setdefault(team, {"opp_elo": [], "dates": []})
        st["opp_elo"].append(opp)
        st["dates"].append(r["date"])

    league_gf_home.append(hs)
    league_gf_away.append(as_)

lgf_h = float(np.mean(league_gf_home[-800:]))
lgf_a = float(np.mean(league_gf_away[-800:]))
log(f"Medias de liga (ultimos 800): local {lgf_h:.3f}, visita {lgf_a:.3f}")
log(f"Elo top 5: {sorted(elo.items(), key=lambda x: -x[1])[:5]}")


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
    if not sh or not sa:
        return np.nan, np.nan
    ah, dh = sh
    aa, da = sa
    lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
    lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
    return lam_h, lam_a


def table_pos(team):
    th = tabla.get(team)
    if not th or th[2] == 0:
        return (np.nan, np.nan)
    pos = 1 + sum(1 for v in tabla.values() if v[0] > th[0])
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


# ============================================================
# 2. Odds 1X2 de la Odds API (para imp_*)
# ============================================================
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

master_teams = set(master["home"].unique()) | set(master["away"].unique())


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


# ============================================================
# 3. Fixtures ESPN -> features -> p_btts
# ============================================================
with open(DATA / "espn_fixtures.json") as f:
    fixtures = json.load(f)
log(f"Fixtures: {len(fixtures)}")

with open(DATA / "btts_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
scaler = bundle["scaler"]
FEATS = bundle["features"]
means = pd.Series(bundle["impute_means"])
log(f"Modelo: {bundle.get('version')} ({len(FEATS)} features, scaler={'si' if scaler else 'no'})")

rows = []
for fx in fixtures:
    if fx.get("state") != "pre" or not fx.get("home"):
        continue
    home = canon_series(pd.Series([fx["home"]]))[0]
    away = canon_series(pd.Series([fx["away"]]))[0]
    d_local = pd.to_datetime(fx["date"], utc=True).tz_convert("America/Mexico_City").tz_localize(None).normalize()

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
        "gf10_h": fh["gf10"], "ga10_h": fh["ga10"],
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

# ============================================================
# 4. Salida
# ============================================================
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
