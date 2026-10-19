#!/usr/bin/env python3
"""
make_features.py — Features pre-partido (sin leakage) sobre el master.

Regla de oro: todo feature usa solo información disponible ANTES del partido.
  - Elo: actualizado secuencialmente partido a partido
  - Forma/destanso: ventanas de partidos previos
  - Odds: cuotas de cierre (disponibles pre-partido)

Salida: /home/petyt/futbolmx/data/liga_mx_features.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUTBOL = Path("/home/petyt/futbolmx")
sys.path.insert(0, str(FUTBOL))

OUT = FUTBOL / "data"
MASTER = OUT / "liga_mx_master.parquet"
UNIFIED_OLD = Path("/home/petyt/liga_mx_data/liga_mx_unified_2018_2025.parquet")


def log(msg):
    print(msg, flush=True)


df = pd.read_parquet(MASTER).sort_values("date").reset_index(drop=True)
log(f"Master: {len(df)} partidos")

# ============================================================
# 1. ELO (secuencial, ventaja local + multiplicador por goles)
# ============================================================
K = 32.0
HOME_ADV = 80.0  # ventaja local en puntos Elo
elo = {}
elo_h, elo_a, elo_res = [], [], []
for _, r in df.iterrows():
    h, a = r["home"], r["away"]
    eh = elo.get(h, 1500.0)
    ea = elo.get(a, 1500.0)
    elo_h.append(eh)
    elo_a.append(ea)
    # resultado esperado local
    exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV - ea) / 400.0)))
    if pd.notna(r["hs"]) and pd.notna(r["as_"]):
        if r["hs"] > r["as_"]:
            s = 1.0
        elif r["hs"] < r["as_"]:
            s = 0.0
        else:
            s = 0.5
        margen = abs(r["hs"] - r["as_"])
        mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else 1.75 + (margen - 3) * 0.125)
        delta = K * mult * (s - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta
    elo_res.append(None)
df["elo_h"] = elo_h
df["elo_a"] = elo_a
df["elo_diff"] = df["elo_h"] - df["elo_a"]
log(f"Elo calculado. Ejemplo ultimo: {sorted(elo.items(), key=lambda x: -x[1])[:3]}")

# ============================================================
# 2. DESCANSO (dias desde partido previo) y PARTIDOS PREVIOS
# ============================================================
last_match = {}
rest_h, rest_a, mp_h, mp_a = [], [], [], []
for _, r in df.iterrows():
    h, a = r["home"], r["away"]
    rh = (r["date"] - last_match[h]).days if h in last_match else np.nan
    ra = (r["date"] - last_match[a]).days if a in last_match else np.nan
    rest_h.append(rh)
    rest_a.append(ra)
    mp_h.append(1 if h in last_match else 0)
    mp_a.append(1 if a in last_match else 0)
    last_match[h] = r["date"]
    last_match[a] = r["date"]
df["rest_h"] = rest_h
df["rest_a"] = rest_a
df["rest_diff"] = df["rest_h"] - df["rest_a"]
df["has_prev_h"] = mp_h
df["has_prev_a"] = mp_a
log("Descanso calculado.")

# ============================================================
# 3. FORMA: media movil de goles a favor/contra (ultimos N, propia y global)
# ============================================================
hist = {t: [] for t in pd.concat([df["home"], df["away"]]).unique()}


def roll_stats(team, n):
    h = hist[team]
    if not h:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    arr = np.array(h[-n:]) if len(h) >= 1 else None
    gf = arr[:, 0].mean() if arr is not None else np.nan
    ga = arr[:, 1].mean() if arr is not None else np.nan
    bs = arr[:, 2].mean() if arr is not None else np.nan  # btts del partido
    pts = arr[:, 3].mean() if arr is not None else np.nan  # puntos: 3/1/0
    nplayed = len(arr) if arr is not None else 0
    return (gf, ga, bs, pts, nplayed)


rows_form = []
for _, r in df.iterrows():
    h, a = r["home"], r["away"]
    fh = roll_stats(h, 10)
    fa = roll_stats(a, 10)
    # forma como LOCAL del home y como VISITA del away
    hh = [x for x in hist[h] if x[4] == 1]  # partidos previos como local
    aa = [x for x in hist[a] if x[4] == 0]  # partidos previos como visita
    hhome = np.array(hh[-6:]) if hh else None
    aaway = np.array(aa[-6:]) if aa else None
    rows_form.append({
        "gf10_h": fh[0], "ga10_h": fh[1], "btts_rate10_h": fh[2], "ppg10_h": fh[3],
        "gf10_a": fa[0], "ga10_a": fa[1], "btts_rate10_a": fa[2], "ppg10_a": fa[3],
        "gf_home_as_home": hhome[:, 0].mean() if hhome is not None and len(hhome) else np.nan,
        "ga_home_as_home": hhome[:, 1].mean() if hhome is not None and len(hhome) else np.nan,
        "gf_away_as_away": aaway[:, 0].mean() if aaway is not None and len(aaway) else np.nan,
        "ga_away_as_away": aaway[:, 1].mean() if aaway is not None and len(aaway) else np.nan,
    })
    # registrar partido actual (con rol: 1 local, 0 visita)
    if pd.notna(r["hs"]) and pd.notna(r["as_"]):
        ptos_h = 3.0 if r["hs"] > r["as_"] else (1.0 if r["hs"] == r["as_"] else 0.0)
        btts_val = 1.0 if (r["hs"] > 0 and r["as_"] > 0) else 0.0
        hist[h].append((r["hs"], r["as_"], btts_val, ptos_h, 1))
        hist[a].append((r["as_"], r["hs"], btts_val, 3.0 - ptos_h if ptos_h in (0, 3) else 1.0, 0))
form = pd.DataFrame(rows_form)
df = pd.concat([df, form], axis=1)
log("Forma calculada.")

# ============================================================
# 4. ODDS -> probabilidades implcitas 1X2 (normalizadas, sin margen)
# ============================================================
def pick_odds(row, pref):
    for c in pref:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return np.nan


pref_h = ["odd_h_psc", "odd_h_avg", "odd_h_kag", "odd_h_max", "odd_h_b365"]
pref_d = ["odd_d_psc", "odd_d_avg", "odd_d_kag", "odd_d_max", "odd_d_b365"]
pref_a = ["odd_a_psc", "odd_a_avg", "odd_a_kag", "odd_a_max", "odd_a_b365"]
oh, od, oa = [], [], []
for _, r in df.iterrows():
    oh.append(pick_odds(r, pref_h))
    od.append(pick_odds(r, pref_d))
    oa.append(pick_odds(r, pref_a))
df["odd_h"] = oh
df["odd_d"] = od
df["odd_a"] = oa
inv_sum = (1 / df["odd_h"] + 1 / df["odd_d"] + 1 / df["odd_a"])
df["imp_h"] = (1 / df["odd_h"]) / inv_sum
df["imp_d"] = (1 / df["odd_d"]) / inv_sum
df["imp_a"] = (1 / df["odd_a"]) / inv_sum
df["has_odds"] = df["odd_h"].notna().astype(int)
log(f"Odds disponibles: {df['has_odds'].sum()}/{len(df)}")

# ============================================================
# 5. ALTITUD de sede (mapeo del parquet unificado viejo)
# ============================================================
try:
    old = pd.read_parquet(UNIFIED_OLD)
    stadium_alt = old.dropna(subset=["stadium"]).drop_duplicates("stadium")[["stadium", "altitude", "city"]]
    stadium_alt = stadium_alt.set_index("stadium")["altitude"].to_dict()
    # venue del ESPN esta en df
    df["altitude"] = df["venue"].map(stadium_alt)
    log(f"Altitud mapeada: {df['altitude'].notna().sum()}/{len(df)} partidos (venues: {df['venue'].nunique()})")
    log(f"  Ejemplos: {dict(list(stadium_alt.items())[:3])}")
except Exception as e:
    log(f"Altitud: no disponible ({e})")
    df["altitude"] = np.nan

# ============================================================
# 5b. FEATURES NUEVAS — solo con informacion PRE-partido
#     (estado calculado partido a partido, orden cronologico)
# ============================================================
log("== Features nuevas: tabla, calendario, H2H, congestion ==")
df = df.sort_values("date").reset_index(drop=True)

# --- estado por equipo para tabla y calendario ---
team_state = {}   # team -> {"pts":, "gf":, "ga":, "played":, "opp_elo_sum":, "n":, "dates": []}
h2h = {}          # (teamA, teamB) -> [(gf_A, ga_A, btts, date), ...]

rows_new = []
for _, r in df.iterrows():
    h, a = r["home"], r["away"]
    st_h = team_state.get(h)
    st_a = team_state.get(a)

    # ---- posicion en tabla pre-partido (por torneo actual) ----
    # se reconstruye la tabla acumulando puntos del torneo en curso (anio_torneo + torneo)
    tor_id = (r.get("anio_torneo"), r.get("torneo"))
    if "tabla_torneo" not in dir() or tabla_torneo_id != tor_id:
        tabla_torneo = {}      # team -> [pts, gf, ga]
        tabla_torneo_id = tor_id
    th = tabla_torneo.get(h, [0, 0, 0])
    ta = tabla_torneo.get(a, [0, 0, 0])
    n_teams = len(tabla_torneo) if len(tabla_torneo) >= 12 else 18  # aprox si arrancamos a media temporada
    pos_h = 1 + sum(1 for v in tabla_torneo.values() if v[0] > th[0])
    pos_a = 1 + sum(1 for v in tabla_torneo.values() if v[0] > ta[0])
    ppg_h_table = th[0] / max(1, th[2]) if th[2] > 0 else np.nan
    ppg_a_table = ta[0] / max(1, ta[2]) if ta[2] > 0 else np.nan

    # ---- H2H historico (ultimos 6, cualquier sede) ----
    key = tuple(sorted([h, a]))
    prev = h2h.get(key, [])
    h_home = h == key[0]
    if len(prev) >= 1:
        gf_side = [x[0] if h_home else x[1] for x in prev[-6:]]
        ga_side = [x[1] if h_home else x[0] for x in prev[-6:]]
        h2h_btts = float(np.mean([x[2] for x in prev[-6:]]))
        h2h_gf = float(np.mean(gf_side))
        h2h_ga = float(np.mean(ga_side))
        h2h_n = len(prev[-6:])
    else:
        h2h_btts = h2h_gf = h2h_ga = np.nan
        h2h_n = 0

    # ---- fuerza del calendario reciente (Elo promedio de rivales ultimos 5) ----
    def sos(st):
        if st and st.get("opp_elo", []):
            return float(np.mean(st["opp_elo"][-5:]))
        return np.nan
    sos_h = sos(st_h)
    sos_a = sos(st_a)

    # ---- congestion: partidos en los ultimos 14 dias ----
    def congest(st, d):
        if st and st.get("dates"):
            return sum(1 for x in st["dates"][-5:] if (d - x).days <= 14)
        return 0
    cong_h = congest(st_h, r["date"])
    cong_a = congest(st_a, r["date"])

    rows_new.append({
        "pos_table_h": pos_h, "pos_table_a": pos_a,
        "ppg_table_h": ppg_h_table, "ppg_table_a": ppg_a_table,
        "h2h_btts_rate": h2h_btts, "h2h_gf": h2h_gf, "h2h_ga": h2h_ga, "h2h_n": h2h_n,
        "sos_h": sos_h, "sos_a": sos_a,
        "cong_h": cong_h, "cong_a": cong_a,
    })

    # ---- actualizar estados CON el resultado (para el siguiente partido) ----
    if pd.notna(r["hs"]) and pd.notna(r["as_"]):
        hs, as_ = float(r["hs"]), float(r["as_"])
        pts_h = 3.0 if hs > as_ else (1.0 if hs == as_ else 0.0)
        # tabla del torneo
        th[0] += pts_h; th[1] += hs; th[2] += as_
        ta[0] += 3.0 - pts_h; ta[1] += as_; ta[2] += hs
        tabla_torneo[h] = th; tabla_torneo[a] = ta
        # h2h (perspectiva key[0])
        k0_gf = hs if h == key[0] else as_
        k1_gf = as_ if h == key[0] else hs
        h2h.setdefault(key, []).append((k0_gf, k1_gf, 1.0 if hs > 0 and as_ > 0 else 0.0, r["date"]))
        # estados de equipo
        for team, gf, ga, pts, opp in [
            (h, hs, as_, pts_h, r.get("elo_a")),
            (a, as_, hs, 3.0 - pts_h if pts_h in (0, 3) else 1.0, r.get("elo_h")),
        ]:
            st = team_state.setdefault(team, {"opp_elo": [], "dates": []})
            st["opp_elo"].append(opp if pd.notna(opp) else np.nan)
            st["dates"].append(r["date"])

new_feats = pd.DataFrame(rows_new)
df = pd.concat([df, new_feats], axis=1)
log(f"  Features nuevas agregadas: {list(new_feats.columns)}")
log(f"  h2h con historia previa: {(new_feats['h2h_n'] > 0).sum()}/{len(new_feats)} partidos")

# ============================================================
# 6. GUARDAR
# ============================================================
out = OUT / "liga_mx_features.parquet"
df.to_parquet(out, index=False)
log(f"\nGuardado: {out} ({len(df)} filas, {len(df.columns)} cols)")
log("\nColumnas: " + ", ".join(df.columns))
