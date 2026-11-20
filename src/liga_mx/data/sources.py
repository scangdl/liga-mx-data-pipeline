"""Data source fetchers for Liga MX Pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog

from liga_mx.config import get_config, get_env
from liga_mx.data.team_names import canon_series

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Football-Data.co.uk
# ------------------------------------------------------------------
def download_football_data(output_path: Path | None = None) -> pd.DataFrame:
    """Download current season data from football-data.co.uk."""
    cfg = get_config()
    url = cfg.data_sources.football_data.base_url + cfg.data_sources.football_data.current_season_path
    ua = cfg.data_sources.football_data.user_agent

    if output_path is None:
        output_path = get_path("football_data_odds")

    _log(f"Downloading football-data from {url}")
    subprocess.run(
        [
            "curl", "-sL", "-o", str(output_path),
            "-H", f"User-Agent: {ua}",
            url,
        ],
        check=True,
    )

    df = pd.read_csv(output_path, encoding=cfg.data_sources.football_data.encoding, low_memory=False)
    df = df.rename(columns={"ï»¿Country": "Country"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["home"] = canon_series(df["Home"])
    df["away"] = canon_series(df["Away"])
    df = df.rename(columns={
        "HG": "hs", "AG": "as_", "Res": "res_1x2",
        "PSCH": "odd_h_psc", "PSCD": "odd_d_psc", "PSCA": "odd_a_psc",
        "MaxCH": "odd_h_max", "MaxCD": "odd_d_max", "MaxCA": "odd_a_max",
        "AvgCH": "odd_h_avg", "AvgCD": "odd_d_avg", "AvgCA": "odd_a_avg",
        "BFECH": "odd_h_bfe", "BFECD": "odd_d_bfe", "BFECA": "odd_a_bfe",
        "B365CH": "odd_h_b365", "B365CD": "odd_d_b365", "B365CA": "odd_a_b365",
    })
    odd_cols = [c for c in df.columns if c.startswith("odd_")]
    df = df[["date", "Season", "home", "away", "hs", "as_", "res_1x2"] + odd_cols]
    _log(f"Downloaded: {len(df)} rows, {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


# ------------------------------------------------------------------
# ESPN Fixtures
# ------------------------------------------------------------------
def fetch_espn_fixtures(lookahead_days: int | None = None, output_path: Path | None = None) -> list[dict]:
    """Fetch upcoming fixtures from ESPN public API."""
    cfg = get_config()
    if lookahead_days is None:
        lookahead_days = cfg.data_sources.espn.lookahead_days

    base = cfg.data_sources.espn.base_url
    ua = cfg.data_sources.espn.user_agent
    timeout = cfg.data_sources.espn.timeout_seconds

    if output_path is None:
        output_path = get_path("espn_fixtures")

    today = datetime.now(timezone.utc).date()
    out = []

    for i in range(lookahead_days + 1):
        day = today + timedelta(days=i)
        ds = day.strftime("%Y%m%d")
        url = f"{base}?dates={ds}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            _log(f"  {ds} error: {e}")
            continue

        for ev in d.get("events", []):
            c = ev["competitions"][0]
            teams = {}
            for t in c.get("competitors", []):
                teams[t["homeAway"]] = t["team"]["displayName"]

            odds = [x for x in (c.get("odds") or []) if x]
            o = odds[0] if odds else {}
            prov = (o.get("provider") or {}).get("name")

            tot = o.get("total") or {}
            over = tot.get("over") or {}
            under = tot.get("under") or {}
            over_ml = (over.get("close") or {}).get("odds") or (over.get("open") or {}).get("odds")
            under_ml = (under.get("close") or {}).get("odds") or (under.get("open") or {}).get("odds")
            draw_ml = (o.get("drawOdds") or {}).get("moneyLine")

            out.append({
                "id": ev["id"],
                "date": ev["date"],
                "home": teams.get("home"),
                "away": teams.get("away"),
                "state": c["status"]["type"]["state"],
                "line": o.get("overUnder"),
                "over_ml": over_ml,
                "under_ml": under_ml,
                "draw_ml": draw_ml,
                "provider": prov,
            })

    # Deduplicate by event id
    seen = set()
    dedup = []
    for fx in out:
        if fx["id"] in seen:
            continue
        seen.add(fx["id"])
        dedup.append(fx)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dedup, f, ensure_ascii=False, indent=1)

    _log(f"Saved {len(dedup)} fixtures to {output_path}")
    return dedup


# ------------------------------------------------------------------
# Odds API (the-odds-api.com)
# ------------------------------------------------------------------
def _median(prices: list[float]) -> float:
    s = sorted(prices)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def fetch_odds_api(output_path: Path | None = None, history_path: Path | None = None) -> pd.DataFrame:
    """Fetch odds from the-odds-api.com for Liga MX."""
    cfg = get_config()
    api_key = get_env("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not set in .env")

    base = cfg.data_sources.odds_api.base_url
    sport = cfg.data_sources.odds_api.sport
    regions = cfg.data_sources.odds_api.regions
    markets = cfg.data_sources.odds_api.markets
    odds_format = cfg.data_sources.odds_api.odds_format
    timeout = cfg.data_sources.odds_api.timeout_seconds

    if output_path is None:
        output_path = get_path("odds_api_jornada")
    if history_path is None:
        history_path = get_path("odds_history")

    # 1) Fetch events + odds
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    url = f"{base}/sports/{sport}/odds"
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": "futbolmx/1.0"})

    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        events = json.loads(r.read().decode())
        headers = dict(r.headers)

    _log(f"Odds API: status={status}, remaining={headers.get('x-requests-remaining')}")

    if status != 200:
        raise RuntimeError(f"Odds API failed: {json.dumps(events)[:200]}")

    # Save snapshot
    snapshot = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": events}
    snapshot_path = get_path("odds_api_snapshot")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)

    # 2) Flatten to rows
    rows = []
    for ev in events:
        h, a = ev["home_team"], ev["away_team"]
        ou_o, ou_u, h2h_h, h2h_d, h2h_a = [], [], [], [], []
        line = None

        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m["key"] == "totals":
                    for o in m["outcomes"]:
                        if o.get("point") == 2.5:
                            line = 2.5
                            if o["name"] == "Over":
                                ou_o.append(o["price"])
                            else:
                                ou_u.append(o["price"])
                elif m["key"] == "h2h":
                    for o in m["outcomes"]:
                        if o["name"] == h:
                            h2h_h.append(o["price"])
                        elif o["name"] == a:
                            h2h_a.append(o["price"])
                        elif o["name"].lower() == "draw":
                            h2h_d.append(o["price"])

        rows.append({
            "commence": ev["commence_time"],
            "home": h,
            "away": a,
            "n_books_ou": len(ou_o),
            "ou_line": line,
            "over25_med": round(_median(ou_o), 3) if ou_o else "",
            "under25_med": round(_median(ou_u), 3) if ou_u else "",
            "ou_margin": round(1 / _median(ou_o) + 1 / _median(ou_u) - 1, 4) if ou_o and ou_u else "",
            "h2h_h_med": round(_median(h2h_h), 3) if h2h_h else "",
            "h2h_d_med": round(_median(h2h_d), 3) if h2h_d else "",
            "h2h_a_med": round(_median(h2h_a), 3) if h2h_a else "",
        })

    # 3) Save CSV for jornada
    out_csv = output_path
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4) Append to history for CLV
    hist_fields = [
        "fetched_at", "commence", "home", "away", "ou_line",
        "over25_med", "under25_med", "h2h_h_med", "h2h_d_med", "h2h_a_med"
    ]
    new_file = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hist_fields)
        if new_file:
            w.writeheader()
        for r in rows:
            if r.get("ou_line") and r.get("over25_med"):
                w.writerow({k: r.get(k, "") for k in hist_fields} | {"fetched_at": snapshot["fetched_at"]})

    _log(f"Saved {len(rows)} odds to {out_csv}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def get_path(name: str) -> Path:
    """Get resolved path from config."""
    from liga_mx.config import get_config
    return get_config().paths.resolve(name)


__all__ = ["download_football_data", "fetch_espn_fixtures", "fetch_odds_api"]