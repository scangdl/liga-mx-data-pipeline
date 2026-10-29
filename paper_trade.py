#!/usr/bin/env python3
"""
paper_trade.py — Paper trading Liga MX (Over 2.5 + 1X2).

Flujo semanal:
  place   : lee fixtures ESPN, predice con modelos, compara contra cuotas,
            registra apuestas con valor (edge > umbral) en CSV con stake Kelly.
  settle  : descarga resultados (update), liquida cada pendiente con el
            marcador real, actualiza bankroll en JSON.

Mercados soportados:
  - over25: Over 2.5 goles (binario)
  - 1x2:    Resultado final H/D/A (multiclass)

Reglas:
  - Bankroll inicial 1000 (u). Stake = kelly_frac * edge / (odds-1), cap 5%.
  - Solo apuesta si edge > EDGE_MIN (default 4%).
  - Sin cuota real -> no apuesta.
  - Max 10 apuestas abiertas concurrentes por mercado.
"""
import argparse
import json
import pickle
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
from features_lib import StateBuilder, build_state_for_inference, log, american_to_decimal, poisson_btts_prob  # noqa: E402
from improve_btts_v2 import add_advanced_features

# ============================================================
# CONFIG
# ============================================================
BANKROLL0 = 1000.0
KELLY_FRAC = 0.25
EDGE_MIN = 0.04
STAKE_CAP = 0.05
MAX_OPEN_BETS = 10  # per market
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"

MARKETS = {
    "over25": {"model_file": "over25_model.pkl", "target": "over25", "line": 2.5},
    "1x2":    {"model_file": "model_1x2.pkl",  "target": "res_1x2", "line": None},
}

# ============================================================
# UTILS
# ============================================================
def log(msg):
    print(msg, flush=True)

def kelly_stake(p_model: float, odds: float, bankroll: float,
                frac: float = KELLY_FRAC, cap: float = STAKE_CAP) -> float:
    """Kelly fraccional con cap % bankroll."""
    if odds <= 1.0 or p_model <= 0 or p_model >= 1:
        return 0.0
    b = odds - 1
    edge = p_model * odds - 1
    if edge <= 0:
        return 0.0
    kelly = frac * edge / b
    return min(max(kelly, 0.0), cap) * bankroll

def american_to_decimal(ml):
    if ml is None:
        return None
    try:
        ml = int(ml)
    except (ValueError, TypeError):
        return None
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1

def canon_team_espn(name, master_teams):
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
# ESTADO GLOBAL (cargado una vez)
# ============================================================
state = build_state_for_inference(DATA / "liga_mx_master.parquet")
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

# ============================================================
# HELPERS FEATURES
# ============================================================
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
    lam_h = (ah / lgf_h) * (sa[1] / lgf_h) * lgf_h
    lam_a = (aa / lgf_a) * (sh[1] / lgf_a) * lgf_a
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

# ============================================================
# FIXTURE ROW (features base para ambos mercados)
# ============================================================
def fixture_row(home, away, d_local):
    def last10(team):
        h = hist.get(team, [])
        if not h:
            return (np.nan, np.nan, np.nan, np.nan)
        arr = h[-10:]
        return (np.mean([x[0] for x in arr]), np.mean([x[1] for x in arr]),
                np.mean([x[2] for x in arr]), np.mean([x[3] for x in arr]))

    gh = last10(home)
    ga = last10(away)
    hhome = [y for y in hist.get(home, []) if y[4] == 1][-6:]
    aaway = [y for y in hist.get(away, []) if y[4] == 0][-6:]

    sh = ewma.get(home, {}).get(1)
    sa = ewma.get(away, {}).get(0)
    lam_h = lam_a = np.nan
    if sh and sa:
        lam_h = (sh[0] / lgf_h) * (sa[1] / lgf_h) * lgf_h
        lam_a = (sa[0] / lgf_a) * (sh[1] / lgf_a) * lgf_a

    # Posición en tabla
    pos_h, ppg_h = table_pos(home)
    pos_a, ppg_a = table_pos(away)

    # H2H
    h2h_btts, h2h_gf, h2h_ga, h2h_n = h2h_features(home, away)

    # SOS / congestión
    sos_h, cong_h = sos_cong(home, d_local)
    sos_a, cong_a = sos_cong(away, d_local)

    return {
        "elo_h": elo.get(home, 1500.0), "elo_a": elo.get(away, 1500.0),
        "rest_h": (d_local - last_match[home]).days if home in last_match else np.nan,
        "rest_a": (d_local - last_match[away]).days if away in last_match else np.nan,
        "gf10_h": gh[0], "ga10_h": gh[1], "btts_rate10_h": gh[2], "ppg10_h": gh[3],
        "gf10_a": ga[0], "ga10_a": ga[1], "btts_rate10_a": ga[2], "ppg10_a": ga[3],
        "gf_home_as_home": np.mean([x[0] for x in hhome]) if hhome else np.nan,
        "ga_home_as_home": np.mean([x[1] for x in hhome]) if hhome else np.nan,
        "gf_away_as_away": np.mean([x[0] for x in aaway]) if aaway else np.nan,
        "ga_away_as_away": np.mean([x[1] for x in aaway]) if aaway else np.nan,
        "lam_h": lam_h, "lam_a": lam_a,
        "imp_h": np.nan, "imp_d": np.nan, "imp_a": np.nan,
        # Extended features
        "pos_table_h": pos_h, "pos_table_a": pos_a,
        "ppg_table_h": ppg_h, "ppg_table_a": ppg_a,
        "h2h_btts_rate": h2h_btts, "h2h_gf": h2h_gf, "h2h_ga": h2h_ga, "h2h_n": h2h_n,
        "sos_h": sos_h, "sos_a": sos_a,
        "cong_h": cong_h, "cong_a": cong_a,
    }

# ============================================================
# LOAD MODEL
# ============================================================
def load_model(market: str):
    """Carga modelo y retorna (model, FEATS, means, bundle)."""
    model_file = MARKETS[market]["model_file"]
    with open(DATA / model_file, "rb") as f:
        bundle = pickle.load(f)
    # Soportar tanto 'means' como 'impute_means'
    means_key = "means" if "means" in bundle else "impute_means"
    return bundle["model"], bundle["features"], pd.Series(bundle[means_key]), bundle

# ============================================================
# ODDS HELPERS
# ============================================================
def load_odds_api():
    """Carga odds multi-casa de la Odds API."""
    ou_api = {}
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
                ou_api[(d_ou, str(r_ou["home"]), str(r_ou["away"]))] = r_ou
        except Exception as e:
            log(f"  (odds API csv ilegible: {e})")
    return ou_api

def get_odds_for_fixture(ou_api, fx, home, away, d_local, market):
    """Extrae cuota decimal para el mercado del fixture."""
    # 1) Odds API multi-casa
    for (d_ou, h_ou, a_ou), r_ou in ou_api.items():
        if d_ou == d_local and canon_team_espn(h_ou) == home and canon_team_espn(a_ou) == away:
            if market == "over25":
                try:
                    med = float(r_ou.get("over25_med"))
                    ln = float(r_ou.get("ou_line")) if pd.notna(r_ou.get("ou_line")) else 2.5
                    if med > 1.01:
                        return med, ln
                except (ValueError, TypeError):
                    pass
            elif market == "1x2":
                try:
                    h_odd = float(r_ou.get("h2h_h_med"))
                    d_odd = float(r_ou.get("h2h_d_med"))
                    a_odd = float(r_ou.get("h2h_a_med"))
                    if h_odd > 1.01 and d_odd > 1.01 and a_odd > 1.01:
                        return (h_odd, d_odd, a_odd), None
                except (ValueError, TypeError):
                    pass
            break

    # 2) Fallback ESPN DraftKings
    if market == "over25":
        dec = american_to_decimal(fx.get("over_ml"))
        line = fx.get("line")
        return dec, line
    elif market == "1x2":
        dh = american_to_decimal(fx.get("ml_home"))
        dd = american_to_decimal(fx.get("draw_ml"))
        da = american_to_decimal(fx.get("ml_away"))
        if all(v is not None for v in (dh, dd, da)):
            return (dh, dd, da), None
    return None, None

# ============================================================
# PLACE: mercado genérico
# ============================================================
def cmd_place(market: str = "over25"):
    mcfg = MARKETS[market]
    log(f"\n{'='*60}")
    log(f"PLACE {market.upper()}")
    log(f"{'='*60}")

    model, FEATS, means, bundle = load_model(market)
    ou_api = load_odds_api()

    trades_file = PRED / f"paper_trades_{market}.csv"
    if trades_file.exists():
        trades = pd.read_csv(trades_file)
        if len(trades) == 0:
            trades = pd.DataFrame()
            done_keys = set()
            nobet_keys = set()
        else:
            done_dates = pd.to_datetime(trades["date"], format="mixed").dt.normalize()
            done_keys = set()
            nobet_keys = set()
            for d, h, a, s in zip(done_dates, trades["home"], trades["away"], trades["status"]):
                if s == "NO_BET":
                    nobet_keys.add((d, h, a))
                else:
                    done_keys.add((d, h, a))
    else:
        trades = pd.DataFrame()
        done_keys = set()
        nobet_keys = set()

    with open(DATA / "espn_fixtures.json") as f:
        fixtures = json.load(f)

    rows = []
    for fx in fixtures:
        if fx.get("state") != "pre":
            continue
        home = canon_series(pd.Series([fx["home"]]))[0]
        away = canon_series(pd.Series([fx["away"]]))[0]
        d_local = pd.to_datetime(fx["date"], utc=True).tz_convert(
            "America/Mexico_City").tz_localize(None).normalize()
        key = (d_local, home, away)
        if key in done_keys:
            continue

        feats = fixture_row(home, away, d_local)
        feats["elo_diff"] = feats["elo_h"] - feats["elo_a"]
        feats["rest_diff"] = feats["rest_h"] - feats["rest_a"]
        # Convert to DataFrame and add advanced features
        Xf = pd.DataFrame([feats])
        Xf = add_advanced_features(Xf)
        Xf = Xf[FEATS].fillna(means)

        if market == "over25":
            # Binary
            p_model = float(model.predict_proba(Xf)[:, 1][0])
            dec, line = get_odds_for_fixture(ou_api, fx, home, away, d_local, market)
            if dec is None or not line or float(line) != 2.5:
                if key in nobet_keys:
                    continue
                rows.append(make_row(d_local, home, away, p_model, None, None, 0.0, "NO_BET"))
                continue
            p_mkt = 1.0 / dec
            edge = p_model - p_mkt
            bet_key = "over25"
            odds_dict = {"over": dec}

        elif market == "1x2":
            # Multiclass: 3 outcomes
            p = model.predict_proba(Xf)[0]  # [p_H, p_D, p_A]
            odds_tuple, _ = get_odds_for_fixture(ou_api, fx, home, away, d_local, market)
            if odds_tuple is None:
                if key in nobet_keys:
                    continue
                rows.append(make_row(
                    date=d_local, home=home, away=away,
                    p_model=None, p_H=None, p_D=None, p_A=None,
                    dec_odds=None, odds_dict=None, edge=None, stake=0.0,
                    status="NO_BET", market="1x2", bet_key=None
                ))
                continue
            h_odd, d_odd, a_odd = odds_tuple
            # Implied probs
            ih, idd, ia = 1/h_odd, 1/d_odd, 1/a_odd
            s = ih + idd + ia
            p_mkt_H, p_mkt_D, p_mkt_A = ih/s, idd/s, ia/s
            # Edge por outcome
            edges = {"H": p[0] - p_mkt_H, "D": p[1] - p_mkt_D, "A": p[2] - p_mkt_A}
            best_outcome = max(edges, key=edges.get)
            best_edge = edges[best_outcome]
            best_odds = {"H": h_odd, "D": d_odd, "A": a_odd}[best_outcome]
            p_model_best = p[{"H":0, "D":1, "A":2}[best_outcome]]
            p_mkt_best = {"H": p_mkt_H, "D": p_mkt_D, "A": p_mkt_A}[best_outcome]
            edge = best_edge
            dec = best_odds
            bet_key = f"1x2_{best_outcome}"
            odds_dict = {"H": h_odd, "D": d_odd, "A": a_odd}
            p_model = p_model_best  # para stake Kelly

        else:
            continue

        # Decidir apuesta
        if edge > EDGE_MIN:
            stake = kelly_stake(p_model, dec, bankroll())
            if stake > 0:
                # Risk management: limit concurrent open bets
                if len(trades) > 0 and "status" in trades.columns:
                    open_bets = len(trades[(trades["status"] == "OPEN") & (trades["market"] == market)])
                else:
                    open_bets = 0
                if open_bets >= MAX_OPEN_BETS:
                    status = "SKIP_MAX"
                    stake = 0.0
                else:
                    status = "OPEN"
            else:
                stake, status = 0.0, "SKIP"
        else:
            stake, status = 0.0, "SKIP"

        rows.append(make_row(
            d_local, home, away,
            p_model=round(p_model, 4) if market=="over25" else None,
            p_H=round(p[0], 4) if market=="1x2" else None,
            p_D=round(p[1], 4) if market=="1x2" else None,
            p_A=round(p[2], 4) if market=="1x2" else None,
            dec_odds=round(dec, 3) if market=="over25" else None,
            odds_dict=odds_dict,
            edge=round(edge, 4),
            stake=round(stake, 2),
            status=status,
            market=market,
            bet_key=bet_key,
        ))

    new = pd.DataFrame(rows)
    all_trades = pd.concat([trades, new], ignore_index=True) if len(new) else trades
    all_trades.to_csv(trades_file, index=False, encoding="utf-8-sig")

    if len(new):
        open_cnt = (new["status"] == "OPEN").sum()
        log(f"Registradas {len(new)} entradas ({open_cnt} apuestas, "
            f"{(new['status'] == 'SKIP').sum()} skip, "
            f"{(new['status'] == 'NO_BET').sum()} sin cuota)")
        show = new[new["status"] != "NO_BET"]
        if len(show):
            log("\n" + show.to_string(index=False))
    else:
        log("Sin entradas nuevas")

def make_row(date, home, away, p_model=None, p_H=None, p_D=None, p_A=None,
             dec_odds=None, odds_dict=None, edge=None, stake=0.0,
             status="NO_BET", market="over25", bet_key=None):
    row = {
        "date": date, "home": home, "away": away,
        "market": market, "bet_key": bet_key,
        "p_model": p_model, "p_H": p_H, "p_D": p_D, "p_A": p_A,
        "dec_odds": dec_odds, "odds_dict": json.dumps(odds_dict) if odds_dict else None,
        "edge": edge, "stake": stake, "status": status,
        "result": None, "pnl": None, "close_odds": None, "clv_pct": None,
    }
    return row

# ============================================================
# SETTLE: mercado genérico
# ============================================================
def cmd_settle(market: str = "over25"):
    mcfg = MARKETS[market]
    log(f"\n{'='*60}")
    log(f"SETTLE {market.upper()}")
    log(f"{'='*60}")

    trades_file = PRED / f"paper_trades_{market}.csv"
    if not trades_file.exists():
        log("No hay trades.")
        return

    # Actualizar master
    r = subprocess.run([sys.executable, str(FUTBOL / "update_and_predict.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"AVISO: fallo update ({r.stderr[-200:]}) — liquidando con master actual")

    master = pd.read_parquet(DATA / "liga_mx_master.parquet")
    master["key_date"] = pd.to_datetime(master["date"]).dt.normalize()
    keymap = {}
    for r in master.itertuples():
        for off in (-1, 0, 1):
            keymap[(r.key_date + pd.Timedelta(days=off), r.home, r.away)] = (r.hs, r.as_, r.res_1x2)
            keymap[(r.key_date, r.home, r.away)] = (r.hs, r.as_, r.res_1x2)

    trades = pd.read_csv(trades_file)
    trades["date"] = pd.to_datetime(trades["date"], format="mixed").dt.normalize()
    if "result" in trades.columns:
        trades["result"] = trades["result"].astype(object)

    # Historial odds para CLV
    hist_f = DATA / "odds_history.csv"
    hist = None
    if hist_f.exists():
        try:
            hist = pd.read_csv(hist_f)
            hist["fetched_at"] = pd.to_datetime(hist["fetched_at"], format="mixed", utc=True)
            hist["commence"] = pd.to_datetime(hist["commence"], utc=True)
            hist["norm_home"] = canon_series(hist["home"].astype(str).str.strip().str.lower())
            hist["norm_away"] = canon_series(hist["away"].astype(str).str.strip().str.lower())
        except Exception as e:
            log(f"  (historial odds ilegible: {e})")

    def closing_line(trade_row):
        if hist is None:
            return None
        th = canon_series(pd.Series([str(trade_row["home"])]))[0]
        ta = canon_series(pd.Series([str(trade_row["away"])]))[0]
        sub = hist[(hist["norm_home"] == th) & (hist["norm_away"] == ta)]
        if sub.empty:
            return None
        kickoff = sub["commence"].max()
        pre = sub[sub["fetched_at"] < kickoff].sort_values("fetched_at")
        if pre.empty:
            pre = sub.sort_values("fetched_at")
        if pre.empty:
            return None
        if market == "over25":
            return float(pre["over25_med"].iloc[-1])
        elif market == "1x2":
            # Return dict with 3 closing odds
            h2h = float(pre["h2h_h_med"].iloc[-1])
            d2h = float(pre["h2h_d_med"].iloc[-1])
            a2h = float(pre["h2h_a_med"].iloc[-1])
            return {"H": h2h, "D": d2h, "A": a2h}
        return None

    settled = 0
    for i, t in trades.iterrows():
        if t["status"] != "OPEN":
            continue
        k = (t["date"], t["home"], t["away"])
        if k not in keymap:
            continue
        hs, as_, res = keymap[k]

        if market == "over25":
            total = float(hs) + float(as_)
            win = total > 2.5
            pnl = t["stake"] * (t["dec_odds"] - 1) if win else -t["stake"]
            cl = closing_line(t)
            if cl is not None and cl > 0:
                trades.loc[i, "close_odds"] = round(cl, 3)
                trades.loc[i, "clv_pct"] = round((float(t["dec_odds"]) / cl - 1) * 100, 2)

        elif market == "1x2":
            bet_outcome = t["bet_key"].split("_")[1]  # H/D/A
            win = (res == bet_outcome)
            odds_dict = json.loads(t["odds_dict"]) if pd.notna(t.get("odds_dict")) else {}
            bet_odds = odds_dict.get(bet_outcome, 0)
            pnl = t["stake"] * (bet_odds - 1) if win else -t["stake"]
            cl = closing_line(t)
            if cl and isinstance(cl, dict):
                bet_cl = cl.get(bet_outcome)
                if bet_cl and bet_cl > 0:
                    trades.loc[i, "close_odds"] = round(bet_cl, 3)
                    trades.loc[i, "clv_pct"] = round((float(bet_odds) / bet_cl - 1) * 100, 2)

        trades.loc[i, "result"] = "WIN" if win else "LOSE"
        trades.loc[i, "pnl"] = round(pnl, 2)
        trades.loc[i, "status"] = "SETTLED"
        settled += 1

    br = bankroll()
    state_f = PRED / f"paper_state_{market}.json"
    if settled:
        br = BANKROLL0 + float(trades.loc[trades["status"] == "SETTLED", "pnl"].sum())
        with open(state_f, "w") as f:
            json.dump({"bankroll": round(br, 2)}, f)

    trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    s = trades[trades["status"] == "SETTLED"]
    log(f"Liquidadas: {settled} | Bankroll: {br:.2f} "
        f"(base {BANKROLL0}, PnL {br - BANKROLL0:+.2f})")
    if len(s):
        wins = int((s["result"] == "WIN").sum())
        losses = int((s["result"] == "LOSE").sum())
        log(f"  Record: {wins}-{losses} | "
            f"Yield: {s['pnl'].sum() / s['stake'].sum() * 100:.1f}% sobre {len(s)} apuestas")
    if "clv_pct" in trades.columns:
        sc = trades[trades["clv_pct"].notna()]
        if len(sc):
            beat = (sc["clv_pct"] > 0).sum()
            log(f"  CLV: {len(sc)} trades | CLV promedio {sc['clv_pct'].mean():+.2f}% | "
                f"mediana {sc['clv_pct'].median():+.2f}% | {beat}/{len(sc)} batieron closing ({beat/len(sc)*100:.0f}%)")

def bankroll():
    # Para simplificar, usar bankroll global (suma de ambos mercados)
    # En producción podrías separar por mercado
    total = BANKROLL0
    for m in ["over25", "1x2"]:
        state_f = PRED / f"paper_state_{m}.json"
        if state_f.exists():
            with open(state_f) as f:
                total += json.load(f).get("bankroll", 0) - BANKROLL0
    return total

# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["place", "settle", "place_all", "settle_all"])
    ap.add_argument("--market", choices=["over25", "1x2", "all"], default="over25")
    args = ap.parse_args()
    PRED.mkdir(exist_ok=True)

    if args.cmd == "place":
        cmd_place(args.market)
    elif args.cmd == "place_all":
        cmd_place("over25")
        cmd_place("1x2")
    elif args.cmd == "settle":
        cmd_settle(args.market)
    elif args.cmd == "settle_all":
        cmd_settle("over25")
        cmd_settle("1x2")

if __name__ == "__main__":
    main()