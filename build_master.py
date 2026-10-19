#!/usr/bin/env python3
"""
build_master.py — Consolida las fuentes de Liga MX en un master limpio.

Arreglos:
  - BOM UTF-8 (utf-8-sig), fechas dayfirst, ESPN UTC -> America/Mexico_City
  - Nombres canonicos via team_names.py
  - ESPN: JSON anidado -> tabla plana (venue, attendance, marcador)
  - xG FootyStats descartado (corrupto: snapshot duplicado)

Union: Kaggle mirror (2012 – jun/2018) + football-data.co.uk (jul/2018 – 2026)
Enriquecimiento con ESPN por (fecha ±1 dia, home, away).

Salida: /home/petyt/futbolmx/data/liga_mx_master.{parquet,csv}
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUTBOL = Path("/home/petyt/futbolmx")
sys.path.insert(0, str(FUTBOL))
from team_names import UNKNOWN, canon_series  # noqa: E402

OUT = FUTBOL / "data"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. KAGGLE MIRROR — 2012 a 2024 (resultados + odds 1X2 + forma)
# ============================================================
log("== Cargando Kaggle mirror ==")
kag = pd.read_csv(FUTBOL / "ligamx_data/kaggle_ligamx_raw.csv", low_memory=False)
kag["date"] = pd.to_datetime(kag["MatchDate"], errors="coerce")
kag["home"] = canon_series(kag["HomeTeam"])
kag["away"] = canon_series(kag["AwayTeam"])
kag = kag.rename(columns={
    "FTHome": "hs", "FTAway": "as_", "FTResult": "res_1x2",
    "OddHome": "odd_h_kag", "OddDraw": "odd_d_kag", "OddAway": "odd_a_kag",
    "MaxHome": "max_h_kag", "MaxDraw": "max_d_kag", "MaxAway": "max_a_kag",
    "Form3Home": "form3_h", "Form5Home": "form5_h",
    "Form3Away": "form3_a", "Form5Away": "form5_a",
})[["date", "home", "away", "hs", "as_", "res_1x2",
    "odd_h_kag", "odd_d_kag", "odd_a_kag", "max_h_kag", "max_d_kag", "max_a_kag",
    "form3_h", "form5_h", "form3_a", "form5_a"]]
log(f"  Kaggle: {len(kag)} filas, {kag['date'].min().date()} -> {kag['date'].max().date()}")

# ============================================================
# 2. FOOTBALL-DATA — jul/2018 a 2026 (odds cierre 1X2)
# ============================================================
log("== Cargando football-data.co.uk ==")
fd = pd.read_csv(FUTBOL / "liga_mx_odds_2018_2025.csv", encoding="utf-8-sig", low_memory=False)
fd = fd.rename(columns={"ï»¿Country": "Country"})
fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce")
fd["home"] = canon_series(fd["Home"])
fd["away"] = canon_series(fd["Away"])
fd = fd.rename(columns={
    "HG": "hs", "AG": "as_", "Res": "res_1x2",
    "PSCH": "odd_h_psc", "PSCD": "odd_d_psc", "PSCA": "odd_a_psc",
    "MaxCH": "odd_h_max", "MaxCD": "odd_d_max", "MaxCA": "odd_a_max",
    "AvgCH": "odd_h_avg", "AvgCD": "odd_d_avg", "AvgCA": "odd_a_avg",
    "BFECH": "odd_h_bfe", "BFECD": "odd_d_bfe", "BFECA": "odd_a_bfe",
    "B365CH": "odd_h_b365", "B365CD": "odd_d_b365", "B365CA": "odd_a_b365",
})
odd_cols = [c for c in fd.columns if c.startswith("odd_")]
fd = fd[["date", "Season", "home", "away", "hs", "as_", "res_1x2"] + odd_cols]
log(f"  football-data: {len(fd)} filas, {fd['date'].min().date()} -> {fd['date'].max().date()}")
log(f"  Partidos sin jugar (hs nulo): {fd['hs'].isna().sum()}")

# ============================================================
# 3. UNION KAGGLE + FOOTBALL-DATA
# ============================================================
log("== Uniendo Kaggle (pre-jul/2018) + football-data ==")
cut = fd["date"].min()
kag_only = kag[kag["date"] < cut]
log(f"  Kaggle pre-{cut.date()}: {len(kag_only)} filas")
base = pd.concat([kag_only, fd], ignore_index=True, sort=False)
base = base.sort_values("date").reset_index(drop=True)
base = base.drop_duplicates(subset=["date", "home", "away"], keep="last")
log(f"  Base unida: {len(base)} partidos, {base['date'].min().date()} -> {base['date'].max().date()}")
n_played = base["hs"].notna().sum()
log(f"  Con marcador: {n_played}")

# ============================================================
# 4. ESPN — enriquecimiento (venue, attendance, verificacion)
# ============================================================
log("== Parseando ESPN ==")
espn = pd.read_parquet(FUTBOL / "liga_mx_data/espn_schedules.parquet")
rows = []
for _, r in espn.iterrows():
    try:
        comp = r["competitions"]
        if isinstance(comp, str):
            comp = ast.literal_eval(comp)
        if isinstance(comp, dict):
            comp = [comp]
        c = comp[0]
        comps = c.get("competitors", [])
        home = next((x for x in comps if x.get("homeAway") == "home"), None)
        away = next((x for x in comps if x.get("homeAway") == "away"), None)
        rows.append({
            "date_utc": r["date"],
            "home": canon_series(pd.Series([home["team"]["displayName"]]))[0] if home else None,
            "away": canon_series(pd.Series([away["team"]["displayName"]]))[0] if away else None,
            "hs_espn": home.get("score", {}).get("value") if home else None,
            "as_espn": away.get("score", {}).get("value") if away else None,
            "venue": c.get("venue", {}).get("fullName"),
            "attendance": c.get("attendance"),
            "season_type": r.get("season_type_name"),
            "season_year": r.get("season_year"),
            "completed": c.get("status", {}).get("type", {}).get("completed"),
        })
    except Exception:
        continue
es = pd.DataFrame(rows)
es["date"] = pd.to_datetime(es["date_utc"], utc=True, errors="coerce")
es["date"] = es["date"].dt.tz_convert("America/Mexico_City").dt.tz_localize(None).dt.normalize()
es = es.dropna(subset=["date", "home", "away"])
es = es.drop_duplicates(subset=["date", "home", "away"], keep="last")
es = es[(es["date"] >= base["date"].min()) & (es["date"] <= base["date"].max() + pd.Timedelta(days=7))]
log(f"  ESPN parseado: {len(es)} filas, {es['date'].min().date()} -> {es['date'].max().date()}")

# Merge con tolerancia ±1 dia
log("== Enriquecimiento con ESPN (±1 dia) ==")
es_small = es[["date", "home", "away", "venue", "attendance", "season_type", "hs_espn", "as_espn"]]
es_small = es_small.rename(columns={"date": "date_espn"})
es_idx = es_small.copy()
for delta in (-1, 0, 1):
    es_idx[f"date_{delta}"] = es_idx["date_espn"] + pd.Timedelta(days=delta)

matched = es_idx.merge(base[["date", "home", "away"]].assign(_row=base.index),
                       left_on=["date_0", "home", "away"],
                       right_on=["date", "home", "away"], how="inner")
# tolerancia: intento exacto primero, luego ±1
for delta in (-1, 1):
    extra = es_idx.merge(
        base[["date", "home", "away"]].assign(_row=base.index),
        left_on=[f"date_{delta}", "home", "away"],
        right_on=["date", "home", "away"], how="inner")
    extra = extra[~extra["_row"].isin(matched["_row"])]
    matched = pd.concat([matched, extra], ignore_index=True)
log(f"  Matches ESPN: {len(matched)} partidos enriquecidos")

es_cols = matched.set_index("_row")[["venue", "attendance", "season_type", "hs_espn", "as_espn"]]
base = base.join(es_cols)

# Verificacion de marcadores
both = base.dropna(subset=["hs", "hs_espn"])
agree = (both["hs"] == both["hs_espn"]).mean()
log(f"  Marcadores concuerdan en {agree*100:.1f}% de {len(both)} partidos con ambas fuentes")

# ============================================================
# 5. VARIABLES OBJETIVO
# ============================================================
log("== Variables objetivo ==")
played = base["hs"].notna() & base["as_"].notna()
base["btts"] = np.where(played, ((base["hs"] > 0) & (base["as_"] > 0)).astype(int), np.nan)
base["over25"] = np.where(played, ((base["hs"] + base["as_"]) > 2.5).astype(int), np.nan)
log(f"  BTTS rate: {base.loc[played, 'btts'].mean():.3f}  | Over2.5 rate: {base.loc[played, 'over25'].mean():.3f}")

# ============================================================
# 6. TORNEO (Apertura/Clausura) y temporada
# ============================================================
def get_torneo(d):
    return "Apertura" if d.month >= 7 else "Clausura"


base["torneo"] = base["date"].map(get_torneo)
# Convencion Liga MX: Apertura 2011 = jul-dic 2011; Clausura 2012 = ene-may 2012
base["anio_torneo"] = base["date"].dt.year

# ============================================================
# 7. GUARDAR
# ============================================================
log("== Guardando ==")
base = base.sort_values("date").reset_index(drop=True)
parquet_path = OUT / "liga_mx_master.parquet"
csv_path = OUT / "liga_mx_master.csv"
base.to_parquet(parquet_path, index=False)
base.to_csv(csv_path, index=False, encoding="utf-8-sig")
log(f"  {parquet_path} ({len(base)} filas, {len(base.columns)} cols)")
log(f"  {csv_path}")

if UNKNOWN:
    log(f"  ADVERTENCIA equipos sin alias: {sorted(UNKNOWN)}")

# Resumen por año
log("\n== Partidos por temporada ==")
per_season = base.groupby(["anio_torneo", "torneo"]).size()
log(per_season.to_string())
n_played = int((base["hs"].notna() & base["as_"].notna()).sum())
log(f"\nTOTAL: {len(base)} | con marcador: {n_played} | BTTS: {int((base['btts'] == 1).sum())} | sin jugar: {int(base['hs'].isna().sum())}")
