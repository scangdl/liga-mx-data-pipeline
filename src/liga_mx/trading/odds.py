"""Odds fetching utilities for paper trading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from liga_mx.config import get_config, get_path
from liga_mx.data.sources import fetch_odds_api


class OddsFetcher:
    """Fetches and manages odds from multiple sources."""

    def __init__(self):
        self.cfg = get_config()

    def fetch_jornada_odds(self) -> pd.DataFrame:
        """Fetch odds for upcoming jornada from Odds API."""
        return fetch_odds_api()

    def load_jornada_odds(self) -> pd.DataFrame | None:
        """Load cached odds API jornada CSV."""
        path = get_path("odds_api_jornada")
        if path.exists():
            return pd.read_csv(path)
        return None

    def load_odds_history(self) -> pd.DataFrame | None:
        """Load odds history for CLV calculation."""
        path = get_path("odds_history")
        if path.exists():
            return pd.read_csv(path)
        return None

    def get_closing_line(self, home: str, away: str, market: str) -> Any:
        """Get closing line for a specific fixture and market."""
        hist = self.load_odds_history()
        if hist is None:
            return None

        # Normalize team names
        from liga_mx.data.team_names import canon_series
        hist["norm_home"] = canon_series(hist["home"].astype(str).str.strip().str.lower())
        hist["norm_away"] = canon_series(hist["away"].astype(str).str.strip().str.lower())
        th = canon_series(pd.Series([home]))[0].lower()
        ta = canon_series(pd.Series([away]))[0].lower()

        sub = hist[(hist["norm_home"] == th) & (hist["norm_away"] == ta)]
        if sub.empty:
            return None

        sub["commence"] = pd.to_datetime(sub["commence"], utc=True)
        sub["fetched_at"] = pd.to_datetime(sub["fetched_at"], format="mixed", utc=True)

        kickoff = sub["commence"].max()
        pre = sub[sub["fetched_at"] < kickoff].sort_values("fetched_at")
        if pre.empty:
            pre = sub.sort_values("fetched_at")
        if pre.empty:
            return None

        if market == "over25":
            return float(pre["over25_med"].iloc[-1])
        elif market == "1x2":
            return {
                "H": float(pre["h2h_h_med"].iloc[-1]),
                "D": float(pre["h2h_d_med"].iloc[-1]),
                "A": float(pre["h2h_a_med"].iloc[-1]),
            }
        return None


__all__ = ["OddsFetcher"]