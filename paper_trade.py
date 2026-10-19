#!/usr/bin/env python3
"""
paper_trade.py — Paper trading Over 2.5 Liga MX.

Flujo semanal:
  place   : lee fixtures ESPN, predice con over25_model.pkl, compara contra
            cuota justa derivada de la linea O/U de DraftKings (p_dk = 1/odds
            normalizado), registra apuestas con valor (edge > umbral) en
            predictions/paper_trades.csv con stake kelly fraccional.
  settle  : descarga resultados (update), liquida cada pendiente con el
            marcador real, actualiza bankroll en predictions/paper_state.json.

Reglas:
  - Bankroll inicial 1000 (u). Stake = kelly_frac * edge / (odds-1), cap 5%.
  - Solo apuesta si edge > EDGE_MIN (default 4%).
  - Sin cuota Under/Over real -> no apuesta (nada inventado).
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

BANKROLL0 = 1000.0
KELLY_FRAC = 0.25
EDGE_MIN = 0.04
STAKE_CAP = 0.05
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def log(msg):
    print(msg, flush=True)


def american_to_decimal(ml):
    """ML americano (int o str) -> decimal. None -> None."""
    if ml is None:
        return None
    try:
        ml = int(ml)
    except (ValueError, TypeError):
        return None
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1


# ============================================================
# Features de un fixture (identico a predict_jornada.py)
# ============================================================
def build_state():
    master = pd.read_parquet(DATA / "liga_mx_master.parquet").sort_values("date").reset_index(drop=True)
    played = master[master["hs"].notna() & master["as_"].notna()].reset_index(drop=True)

    K, HOME_ADV = 32.0, 80.0
    elo, last_match, hist, ewma = {}, {}, {}, {}
    lgf_h, lgf_a = [], []
    ALPHA = 0.93

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
        lgf_h.append(hs)
        lgf_a.append(as_)

    return {
        "elo": elo, "last": last_match, "hist": hist, "ewma": ewma,
        "lgf_h": float(np.mean(lgf_h[-800:])), "lgf_a": float(np.mean(lgf_a[-800:])),
    }


def fixture_row(st, home, away, d_local):
    def last10(team):
        h = st["hist"].get(team, [])
        if not h:
            return (np.nan, np.nan, np.nan, np.nan)
        arr = h[-10:]
        return (np.mean([x[0] for x in arr]), np.mean([x[1] for x in arr]),
                np.mean([x[2] for x in arr]), np.mean([x[3] for x in arr]))

    gh = last10(home)
    ga = last10(away)
    hhome = [y for y in st["hist"].get(home, []) if y[4] == 1][-6:]
    aaway = [y for y in st["hist"].get(away, []) if y[4] == 0][-6:]

    sh = st["ewma"].get(home, {}).get(1)
    sa = st["ewma"].get(away, {}).get(0)
    lam_h = lam_a = np.nan
    if sh and sa:
        lgf_h, lgf_a = st["lgf_h"], st["lgf_a"]
        lam_h = (sh[0] / lgf_h) * (sa[1] / lgf_h) * lgf_h
        lam_a = (sa[0] / lgf_a) * (sh[1] / lgf_a) * lgf_a

    return {
        "elo_h": st["elo"].get(home, 1500.0), "elo_a": st["elo"].get(away, 1500.0),
        "rest_h": (d_local - st["last"][home]).days if home in st["last"] else np.nan,
        "rest_a": (d_local - st["last"][away]).days if away in st["last"] else np.nan,
        "gf10_h": gh[0], "ga10_h": gh[1], "btts_rate10_h": gh[2], "ppg10_h": gh[3],
        "gf10_a": ga[0], "ga10_a": ga[1], "btts_rate10_a": ga[2], "ppg10_a": ga[3],
        "gf_home_as_home": np.mean([x[0] for x in hhome]) if hhome else np.nan,
        "ga_home_as_home": np.mean([x[1] for x in hhome]) if hhome else np.nan,
        "gf_away_as_away": np.mean([x[0] for x in aaway]) if aaway else np.nan,
        "ga_away_as_away": np.mean([x[1] for x in aaway]) if aaway else np.nan,
        "lam_h": lam_h, "lam_a": lam_a,
        "imp_h": np.nan, "imp_d": np.nan, "imp_a": np.nan,
    }


# ============================================================
# PLACE: fixtures ESPN -> apuestas
# ============================================================
def cmd_place():
    with open(DATA / "espn_fixtures.json") as f:
        fixtures = json.load(f)
    with open(DATA / "over25_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, FEATS, means = bundle["model"], bundle["features"], pd.Series(bundle["means"])

    trades_file = PRED / "paper_trades.csv"
    if trades_file.exists():
        trades = pd.read_csv(trades_file)
        done_dates = pd.to_datetime(trades["date"], format="mixed").dt.normalize()
        # solo bloquean re-evaluacion los fixtures evaluados CON cuota
        # (SKIP/OPEN/SETTLED); los NO_BET (sin cuota en su momento) deben
        # poder re-evaluarse cuando aparezcan cuotas, sin duplicarse
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

    st = build_state()
    # equipos canonicos del master (para match de nombres ESPN/Odds API)
    master_teams = set(pd.read_parquet(DATA / "liga_mx_master.parquet")["home"].unique())
    # odds O/U 2.5 multi-casa de la Odds API (mediana de 14-19 books) si existen
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

    def canon_team_espn(name):
        # nombres de la Odds API (ESPN): normalizar y si no esta en aliases,
        # limpiar sufijos 'fc' y volver a intentar
        try:
            c = canon_series(pd.Series([str(name)]))[0]
            if c in set(master_teams):
                return c
            # reintentar sin sufijos FC/UANL/UNAM (ej: 'atlante fc' -> 'atlante')
            stripped = str(c).strip().lower()
            for suf in (" fc", " uanl", " unam"):
                if stripped.endswith(suf):
                    c2 = canon_series(pd.Series([stripped[: -len(suf)]]))[0]
                    if c2 in set(master_teams):
                        return c2
            return c
        except Exception:
            return str(name)

    rows = []
    for fx in fixtures:
        if fx.get("state") != "pre":
            continue
        home = canon_series(pd.Series([fx["home"]]))[0]
        away = canon_series(pd.Series([fx["away"]]))[0]
        d_local = pd.to_datetime(fx["date"], utc=True).tz_convert("America/Mexico_City").tz_localize(None).normalize()
        key = (d_local, home, away)
        if key in done_keys:
            continue

        feats = fixture_row(st, home, away, d_local)
        feats["elo_diff"] = feats["elo_h"] - feats["elo_a"]
        feats["rest_diff"] = feats["rest_h"] - feats["rest_a"]
        Xf = pd.DataFrame([feats])[FEATS].fillna(means)
        p_model = float(model.predict_proba(Xf)[:, 1][0])

        # cuota O2.5: 1) Odds API multi-casa, 2) fallback DraftKings via ESPN
        dec = None
        line = None
        for (d_ou, h_ou, a_ou), r_ou in ou_api.items():
            if d_ou == d_local and canon_team_espn(h_ou) == home and canon_team_espn(a_ou) == away:
                try:
                    med = float(r_ou["over25_med"])
                    ln = float(r_ou["ou_line"]) if pd.notna(r_ou["ou_line"]) else 2.5
                    if med > 1.01:
                        dec, line = med, ln
                except (ValueError, TypeError):
                    pass
                break
        if dec is None:
            o = fx  # fallback viejo: JSON de ESPN con over_ml de DraftKings
            dec = american_to_decimal(o.get("over_ml"))
            line = o.get("line")
        if dec is None or not line or float(line) != 2.5:
            if key in nobet_keys:
                continue  # ya esta registrada sin cuota; no duplicar
            rows.append({"date": d_local, "home": home, "away": away,
                         "p_model": round(p_model, 4), "line": line,
                         "dec_odds": None, "edge": None, "stake": 0.0,
                         "status": "NO_BET", "result": None, "pnl": None})
            continue
        p_mkt = 1.0 / dec  # sin normalizar (margen pequeño en O/U 2.5)
        edge = p_model - p_mkt
        if edge > EDGE_MIN:
            kelly = KELLY_FRAC * (p_model * dec - 1) / (dec - 1)
            stake = min(max(kelly, 0.0), STAKE_CAP) * bankroll()
            status = "OPEN"
        else:
            stake, status = 0.0, "SKIP"
        rows.append({"date": d_local, "home": home, "away": away,
                     "p_model": round(p_model, 4), "line": 2.5,
                     "dec_odds": round(dec, 3), "edge": round(edge, 4) if edge is not None else None,
                     "stake": round(stake, 2), "status": status, "result": None, "pnl": None})

    new = pd.DataFrame(rows)
    all_trades = pd.concat([trades, new], ignore_index=True) if len(new) else trades
    all_trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    if len(new):
        log(f"Registradas {len(new)} entradas ({(new['status'] == 'OPEN').sum()} apuestas, "
            f"{(new['status'] == 'SKIP').sum()} skip, {(new['status'] == 'NO_BET').sum()} sin cuota)")
        show = new[new["status"] != "NO_BET"]
        if len(show):
            log("\n" + show.to_string(index=False))
    else:
        log("Sin entradas nuevas (todos los fixtures ya evaluados o sin cuota)")


def bankroll():
    state_f = PRED / "paper_state.json"
    if state_f.exists():
        with open(state_f) as f:
            return json.load(f).get("bankroll", BANKROLL0)
    return BANKROLL0


# ============================================================
# SETTLE: liquida apuestas abiertas
# ============================================================
def cmd_settle():
    trades_file = PRED / "paper_trades.csv"
    if not trades_file.exists():
        log("No hay trades.")
        return
    # actualizar master con resultados ya jugados
    r = subprocess.run([sys.executable, str(FUTBOL / "update_and_predict.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"AVISO: fallo update ({r.stderr[-200:]}) — liquidando con master actual")

    master = pd.read_parquet(DATA / "liga_mx_master.parquet")
    master["key_date"] = pd.to_datetime(master["date"]).dt.normalize()
    # football-data registra fecha UK y ESPN fecha local (CDMX, UTC-6):
    # el mismo partido puede caer a un dia distinto en cada fuente.
    keymap = {}
    for r in master.itertuples():
        for off in (-1, 0, 1):
            keymap[(r.key_date + pd.Timedelta(days=off), r.home, r.away)] = (r.hs, r.as_)
            keymap[(r.key_date, r.home, r.away)] = (r.hs, r.as_)

    trades = pd.read_csv(trades_file)
    trades["date"] = pd.to_datetime(trades["date"], format="mixed").dt.normalize()

    # ---- closing line + CLV: ultima fila pre-kickoff del historial de cuotas ----
    hist_f = DATA / "odds_history.csv"
    hist = None
    if hist_f.exists():
        try:
            hist = pd.read_csv(hist_f)
            hist["fetched_at"] = pd.to_datetime(hist["fetched_at"], format="mixed")
            hist["commence"] = pd.to_datetime(hist["commence"], utc=True)
            # normalizar nombres Odds API -> canonicos del master (mismo mapa de team_names)
            hist["norm_home"] = canon_series(hist["home"].astype(str).str.strip().str.lower())
            hist["norm_away"] = canon_series(hist["away"].astype(str).str.strip().str.lower())
        except Exception as e:
            log(f"  (historial odds ilegible: {e})")

    def closing_line(trade_row):
        """Over mediano mas cercano ANTES del kickoff para este partido (None si no hay)."""
        if hist is None:
            return None
        th = canon_series(pd.Series([str(trade_row["home"])]))[0]
        ta = canon_series(pd.Series([str(trade_row["away"])]))[0]
        sub = hist[(hist["norm_home"] == th) & (hist["norm_away"] == ta)]
        if sub.empty:
            return None
        kickoff = sub["commence"].min()
        pre = sub[sub["commence"] == kickoff].sort_values("fetched_at")
        if pre.empty:
            return None
        return float(pre["over25_med"].iloc[-1])

    settled = 0
    for i, t in trades.iterrows():
        if t["status"] != "OPEN":
            continue
        k = (t["date"], t["home"], t["away"])
        if k not in keymap:
            continue
        hs, as_ = keymap[k]
        total = float(hs) + float(as_)
        win = total > 2.5
        pnl = t["stake"] * (t["dec_odds"] - 1) if win else -t["stake"]
        trades.loc[i, "result"] = "WIN" if win else "LOSE"
        trades.loc[i, "pnl"] = round(pnl, 2)
        trades.loc[i, "status"] = "SETTLED"
        settled += 1
        # CLV: closing over vs cuota de entrada del trade
        cl = closing_line(t)
        if cl is not None and cl > 0:
            trades.loc[i, "close_odds"] = round(cl, 3)
            # CLV% = (cuota entrada / closing) - 1  -> positivo = batimos la closing line
            trades.loc[i, "clv_pct"] = round((float(t["dec_odds"]) / cl - 1) * 100, 2)

    br = bankroll()
    state_f = PRED / "paper_state.json"
    if settled:
        # bankroll = base + suma de TODOS los pnl liquidados (simple y auditable)
        br = BANKROLL0 + float(trades.loc[trades["status"] == "SETTLED", "pnl"].sum())
        with open(state_f, "w") as f:
            json.dump({"bankroll": round(br, 2)}, f)

    trades.to_csv(trades_file, index=False, encoding="utf-8-sig")
    s = trades[trades["status"] == "SETTLED"]
    log(f"Liquidadas: {settled} | Bankroll: {br:.2f} "
        f"(base {BANKROLL0}, PnL acumulado {br - BANKROLL0:+.2f})")
    if len(s):
        log(f"  Record: {int((s['result'] == 'WIN').sum())}-{int((s['result'] == 'LOSE').sum())} "
            f"| Yield: {s['pnl'].sum() / s['stake'].sum() * 100:.1f}% sobre {len(s)} apuestas")
    # ---- reporte CLV: la metrica honesta de edge ----
    if "clv_pct" in trades.columns:
        sc = trades[trades["clv_pct"].notna()]
        if len(sc):
            beat = (sc["clv_pct"] > 0).sum()
            log(f"  CLV: {len(sc)} trades con closing line | "
                f"CLV promedio {sc['clv_pct'].mean():+.2f}% | mediana {sc['clv_pct'].median():+.2f}% | "
                f"{beat}/{len(sc)} batieron la closing ({beat/len(sc)*100:.0f}%)")
            log("  (CLV>0 sistematico = edge real; el acierto y el yield son varianza)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["place", "settle"])
    args = ap.parse_args()
    PRED.mkdir(exist_ok=True)
    if args.cmd == "place":
        cmd_place()
    else:
        cmd_settle()


if __name__ == "__main__":
    main()
