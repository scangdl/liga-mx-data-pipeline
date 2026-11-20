"""Master data builder — consolidates sources into clean master dataset."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from liga_mx.config import get_config, get_path
from liga_mx.data.team_names import canon_series, UNKNOWN

log = structlog.get_logger(__name__)


def _log(msg: str) -> None:
    print(msg, flush=True)


def build_master(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build master dataset from sources.

    Steps:
    1. Load Kaggle mirror (2012 - jun/2018) + football-data.co.uk (jul/2018 - 2026)
    2. Union with date cutoff
    3. Enrich with ESPN (venue, attendance, score verification)
    4. Add target variables (btts, over25)
    5. Add tournament info (Apertura/Clausura)
    6. Save to parquet and CSV

    Returns:
        Master DataFrame with all matches
    """
    cfg = get_config()
    FUTBOL = cfg.paths.project_root

    OUT = get_path("data_dir")
    OUT.mkdir(parents=True, exist_ok=True)

    _log("=== Building Liga MX Master ===")

    # ============================================================
    # 1. KAGGLE MIRROR — 2012 a 2024
    # ============================================================
    _log("Loading Kaggle mirror...")
    kag_path = get_path("kaggle_raw")
    if not kag_path.exists():
        raise FileNotFoundError(f"Kaggle data not found at {kag_path}. Run data ingestion first.")

    kag = pd.read_csv(kag_path, low_memory=False)
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
    _log(f"  Kaggle: {len(kag)} rows, {kag['date'].min().date()} -> {kag['date'].max().date()}")

    # ============================================================
    # 2. FOOTBALL-DATA — jul/2018 a 2026
    # ============================================================
    _log("Loading football-data.co.uk...")
    fd_path = get_path("football_data_odds")
    if not fd_path.exists():
        _log("  Football-data file not found, downloading...")
        from liga_mx.data.sources import download_football_data
        download_football_data(fd_path)

    fd = pd.read_csv(fd_path, encoding="utf-8-sig", low_memory=False)
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
    _log(f"  football-data: {len(fd)} rows, {fd['date'].min().date()} -> {fd['date'].max().date()}")
    _log(f"  Partidos sin jugar (hs nulo): {fd['hs'].isna().sum()}")

    # ============================================================
    # 3. UNION KAGGLE + FOOTBALL-DATA
    # ============================================================
    _log("Uniting Kaggle (pre-jul/2018) + football-data...")
    cut = fd["date"].min()
    kag_only = kag[kag["date"] < cut]
    _log(f"  Kaggle pre-{cut.date()}: {len(kag_only)} rows")
    base = pd.concat([kag_only, fd], ignore_index=True, sort=False)
    base = base.sort_values("date").reset_index(drop=True)
    base = base.drop_duplicates(subset=["date", "home", "away"], keep="last")
    _log(f"  Base united: {len(base)} matches, {base['date'].min().date()} -> {base['date'].max().date()}")
    n_played = base["hs"].notna().sum()
    _log(f"  Con marcador: {n_played}")

    # ============================================================
    # 4. ESPN — enrichment
    # ============================================================
    _log("Parsing ESPN...")
    espn_path = get_path("espn_schedules")
    if not espn_path.exists():
        _log("  ESPN schedules not found, skipping enrichment")
        es = pd.DataFrame()
    else:
        espn = pd.read_parquet(espn_path)
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
        _log(f"  ESPN parsed: {len(es)} rows, {es['date'].min().date()} -> {es['date'].max().date()}")

        # Merge with ±1 day tolerance
        _log("Enriching with ESPN (±1 day)...")
        es_small = es[["date", "home", "away", "venue", "attendance", "season_type", "hs_espn", "as_espn"]]
        es_small = es_small.rename(columns={"date": "date_espn"})

        matched = pd.DataFrame()
        for delta in (-1, 0, 1):
            es_idx = es_small.copy()
            es_idx[f"date_{delta}"] = es_idx["date_espn"] + pd.Timedelta(days=delta)
            m = es_idx.merge(
                base[["date", "home", "away"]].assign(_row=base.index),
                left_on=[f"date_{delta}", "home", "away"],
                right_on=["date", "home", "away"],
                how="inner"
            )
            m = m[~m["_row"].isin(matched["_row"])] if not matched.empty else m
            matched = pd.concat([matched, m], ignore_index=True)

        _log(f"  ESPN matches: {len(matched)} enriched")
        es_cols = matched.set_index("_row")[["venue", "attendance", "season_type", "hs_espn", "as_espn"]]
        base = base.join(es_cols)

        # Score verification
        both = base.dropna(subset=["hs", "hs_espn"])
        agree = (both["hs"] == both["hs_espn"]).mean()
        _log(f"  Scores agree: {agree*100:.1f}% of {len(both)} with both sources")

    # ============================================================
    # 5. TARGET VARIABLES
    # ============================================================
    _log("Adding target variables...")
    played = base["hs"].notna() & base["as_"].notna()
    base["btts"] = np.where(played, ((base["hs"] > 0) & (base["as_"] > 0)).astype(int), np.nan)
    base["over25"] = np.where(played, ((base["hs"] + base["as_"]) > 2.5).astype(int), np.nan)
    _log(f"  BTTS rate: {base.loc[played, 'btts'].mean():.3f} | Over2.5 rate: {base.loc[played, 'over25'].mean():.3f}")

    # ============================================================
    # 6. TOURNAMENT INFO
    # ============================================================
    def get_torneo(d: pd.Timestamp) -> str:
        return "Apertura" if d.month >= 7 else "Clausura"

    base["torneo"] = base["date"].map(get_torneo)
    base["anio_torneo"] = base["date"].dt.year

    # ============================================================
    # 7. SAVE
    # ============================================================
    _log("Saving master...")
    base = base.sort_values("date").reset_index(drop=True)
    parquet_path = get_path("master_parquet")
    csv_path = get_path("master_csv")
    base.to_parquet(parquet_path, index=False)
    base.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _log(f"  {parquet_path} ({len(base)} rows, {len(base.columns)} cols)")
    _log(f"  {csv_path}")

    if UNKNOWN:
        _log(f"  WARNING unknown teams: {sorted(UNKNOWN)}")

    # Summary by season
    _log("\n=== Matches by season ===")
    per_season = base.groupby(["anio_torneo", "torneo"]).size()
    _log(per_season.to_string())
    n_played = int((base["hs"].notna() & base["as_"].notna()).sum())
    _log(f"\nTOTAL: {len(base)} | played: {n_played} | BTTS: {int((base['btts'] == 1).sum())} | unplayed: {int(base['hs'].isna().sum())}")

    return base


if __name__ == "__main__":
    build_master()