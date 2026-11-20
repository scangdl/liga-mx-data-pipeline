"""Paper Trading Engine for Liga MX."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from liga_mx.config import get_config, get_path
from liga_mx.features import StateBuilder, build_state_for_inference, poisson_btts_prob, _american_to_decimal
from liga_mx.models import BTTSModel, Over25Model, OneXTwoModel

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Kelly Stake Calculator
# ------------------------------------------------------------------
class KellyStakeCalculator:
    """Kelly Criterion stake calculator with fractional Kelly and cap."""

    def __init__(self, kelly_fraction: float = 0.25, stake_cap: float = 0.05):
        self.kelly_fraction = kelly_fraction
        self.stake_cap = stake_cap

    def calculate(self, p_model: float, odds: float, bankroll: float) -> float:
        """
        Calculate Kelly stake.

        Args:
            p_model: Model's predicted probability
            odds: Decimal odds
            bankroll: Current bankroll

        Returns:
            Stake amount
        """
        if odds <= 1.0 or p_model <= 0 or p_model >= 1:
            return 0.0

        b = odds - 1
        edge = p_model * odds - 1
        if edge <= 0:
            return 0.0

        kelly = self.kelly_fraction * edge / b
        stake = min(max(kelly, 0.0), self.stake_cap) * bankroll
        return round(stake, 2)


# ------------------------------------------------------------------
# Paper Trader
# ------------------------------------------------------------------
class PaperTrader:
    """
    Paper trading engine for Liga MX.

    Markets:
    - over25: Over 2.5 goals (binary)
    - 1x2: Match result H/D/A (multiclass)

    Features:
    - Kelly fractional staking with cap
    - Minimum edge threshold
    - Max concurrent open bets per market
    - Closing line value (CLV) tracking
    - SHA-256 blind test protocol support
    """

    MARKETS = {
        "over25": {"model_class": Over25Model, "target": "over25", "line": 2.5},
        "1x2": {"model_class": OneXTwoModel, "target": "res_1x2", "line": None},
    }

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        edge_min: float = 0.04,
        stake_cap: float = 0.05,
        max_open_bets: int = 10,
    ):
        self.initial_bankroll = initial_bankroll
        self.kelly = KellyStakeCalculator(kelly_fraction, stake_cap)
        self.edge_min = edge_min
        self.max_open_bets = max_open_bets

        # Load global state (shared across markets)
        self.state = build_state_for_inference(get_path("master_parquet"))
        self._load_models()

        # Master teams for name canonicalization
        self.master_teams = set(
            pd.read_parquet(get_path("master_parquet"))["home"].unique()
        )

    def _load_models(self) -> None:
        """Load all models."""
        self.models = {}
        model_path_map = {
            "over25": "over25_model",
            "1x2": "model_1x2",
        }
        for market, cfg in self.MARKETS.items():
            model_path = get_path(model_path_map[market])
            if model_path.exists():
                self.models[market] = cfg["model_class"].load(model_path)
                log.info(f"Loaded {market} model from {model_path}")
            else:
                log.warning(f"Model not found: {model_path}")

    def _canon_team_espn(self, name: str) -> str:
        """Canonicalize ESPN team name to master format."""
        from liga_mx.data.team_names import canon_series
        try:
            c = canon_series(pd.Series([str(name)]))[0]
            if c in self.master_teams:
                return c
            stripped = str(c).strip().lower()
            for suf in (" fc", " uanl", " unam"):
                if stripped.endswith(suf):
                    c2 = canon_series(pd.Series([stripped[: -len(suf)]]))[0]
                    if c2 in self.master_teams:
                        return c2
            return c
        except Exception:
            return str(name)

    def _load_odds_api(self) -> dict:
        """Load Odds API jornada CSV."""
        odds_path = get_path("odds_api_jornada")
        ou_api = {}
        if odds_path.exists():
            try:
                ou_df = pd.read_csv(odds_path)
                for _, r_ou in ou_df.iterrows():
                    try:
                        d_ou = pd.to_datetime(r_ou["commence"], utc=True).tz_convert(
                            "America/Mexico_City"
                        ).tz_localize(None).normalize()
                    except Exception:
                        continue
                    ou_api[(d_ou, str(r_ou["home"]), str(r_ou["away"]))] = r_ou
            except Exception as e:
                log.warning(f"Odds API CSV unreadable: {e}")
        return ou_api

    def _get_odds_for_fixture(
        self, ou_api: dict, fx: dict, home: str, away: str, d_local: pd.Timestamp, market: str
    ) -> tuple[Any, Any]:
        """Extract odds for a fixture and market."""
        # 1) Odds API multi-book
        for (d_ou, h_ou, a_ou), r_ou in ou_api.items():
            if d_ou == d_local and self._canon_team_espn(h_ou) == home and self._canon_team_espn(a_ou) == away:
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
        from liga_mx.features import _american_to_decimal
        if market == "over25":
            dec = _american_to_decimal(fx.get("over_ml"))
            line = fx.get("line")
            return dec, line
        elif market == "1x2":
            dh = _american_to_decimal(fx.get("ml_home"))
            dd = _american_to_decimal(fx.get("draw_ml"))
            da = _american_to_decimal(fx.get("ml_away"))
            if all(v is not None for v in (dh, dd, da)):
                return (dh, dd, da), None
        return None, None

    def _fixture_features(self, home: str, away: str, d_local: pd.Timestamp) -> dict:
        """Build feature dict for a fixture using StateBuilder."""
        sb = StateBuilder.from_state(
            pd.read_parquet(get_path("master_parquet")).sort_values("date"),
            self.state
        )
        # We need to process up to the fixture date to get correct state
        # For simplicity, use the pre-built state snapshot
        # In production, you'd reconstruct StateBuilder up to d_local

        # Extract from state snapshot
        elo = self.state["elo"]
        last_match = {k: v for k, v in self.state["last_match"].items()}
        hist = {k: list(v) for k, v in self.state["hist"].items()}
        ewma = {k: {r: list(v) for r, v in v.items()} for k, v in self.state["ewma"].items()}
        team_state = {k: {"opp_elo": list(v["opp_elo"]), "dates": list(v["dates"])} for k, v in self.state["team_state"].items()}
        tabla_torneo = {k: {tk: list(tv) for tk, tv in v.items()} for k, v in self.state["tabla_torneo"].items()}
        h2h = {k: list(v) for k, v in self.state["h2h"].items()}
        lgf_h = float(self.state["lgf_h"]) if isinstance(self.state["lgf_h"], (int, float)) else float(np.mean(self.state["lgf_h"]))
        lgf_a = float(self.state["lgf_a"]) if isinstance(self.state["lgf_a"], (int, float)) else float(np.mean(self.state["lgf_a"]))
        current_tor_id = self.state.get("current_tor_id")

        def _form_features(team: str) -> dict:
            h = hist.get(team, [])
            n = len(h)
            return {
                "gf10": np.mean([x[0] for x in h[-10:]]) if n else np.nan,
                "ga10": np.mean([x[1] for x in h[-10:]]) if n else np.nan,
                "btts_rate10": np.mean([x[2] for x in h[-10:]]) if n else np.nan,
                "ppg10": np.mean([x[3] for x in h[-10:]]) if n else np.nan,
            }

        def _lam_features(h: str, a: str) -> tuple[float, float]:
            sh = ewma.get(h, {}).get(1)
            sa = ewma.get(a, {}).get(0)
            if not sh or not sa or sh[0] is None or sa[0] is None:
                return np.nan, np.nan
            ah, dh = sh
            aa, da = sa
            lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
            lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
            return lam_h, lam_a

        def _table_pos(team: str) -> tuple[float, float]:
            th = tabla_torneo.get(team)
            if not th or th[2] == 0:
                return np.nan, np.nan
            pos = 1 + sum(1 for v in tabla_torneo.values() if v[0] > th[0])
            return float(pos), th[0] / th[2]

        def _h2h_features(h: str, a: str) -> tuple[float, float, float, int]:
            key = tuple(sorted([h, a]))
            prev = h2h.get(key, [])
            if not prev:
                return np.nan, np.nan, np.nan, 0
            h_home = h == key[0]
            btts_vals = [x[2] for x in prev[-6:]]
            gf_side = [x[0] if h_home else x[1] for x in prev[-6:]]
            ga_side = [x[1] if h_home else x[0] for x in prev[-6:]]
            return (
                float(np.mean(btts_vals)),
                float(np.mean(gf_side)),
                float(np.mean(ga_side)),
                len(prev[-6:]),
            )

        def _sos_cong(team: str, d: pd.Timestamp) -> tuple[float, int]:
            st = team_state.get(team)
            sos = float(np.mean(st["opp_elo"][-5:])) if st and st.get("opp_elo") else np.nan
            cong = sum(1 for x in st["dates"][-5:] if (d - x).days <= 14) if st and st.get("dates") else 0
            return sos, cong

        gh = _form_features(home)
        ga = _form_features(away)
        lam_h, lam_a = _lam_features(home, away)
        pos_h, ppg_h = _table_pos(home)
        pos_a, ppg_a = _table_pos(away)
        h2h_btts, h2h_gf, h2h_ga, h2h_n = _h2h_features(home, away)
        sos_h, cong_h = _sos_cong(home, d_local)
        sos_a, cong_a = _sos_cong(away, d_local)

        hhome = [y for y in hist.get(home, []) if y[4] == 1][-6:]
        aaway = [y for y in hist.get(away, []) if y[4] == 0][-6:]

        return {
            "elo_h": elo.get(home, 1500.0),
            "elo_a": elo.get(away, 1500.0),
            "rest_h": (d_local - last_match[home]).days if home in last_match else np.nan,
            "rest_a": (d_local - last_match[away]).days if away in last_match else np.nan,
            "gf10_h": gh["gf10"],
            "ga10_h": gh["ga10"],
            "btts_rate10_h": gh["btts_rate10"],
            "ppg10_h": gh["ppg10"],
            "gf10_a": ga["gf10"],
            "ga10_a": ga["ga10"],
            "btts_rate10_a": ga["btts_rate10"],
            "ppg10_a": ga["ppg10"],
            "gf_home_as_home": np.mean([x[0] for x in hhome]) if hhome else np.nan,
            "ga_home_as_home": np.mean([x[1] for x in hhome]) if hhome else np.nan,
            "gf_away_as_away": np.mean([x[0] for x in aaway]) if aaway else np.nan,
            "ga_away_as_away": np.mean([x[1] for x in aaway]) if aaway else np.nan,
            "lam_h": lam_h,
            "lam_a": lam_a,
            "imp_h": np.nan,
            "imp_d": np.nan,
            "imp_a": np.nan,
            "pos_table_h": pos_h,
            "pos_table_a": pos_a,
            "ppg_table_h": ppg_h,
            "ppg_table_a": ppg_a,
            "h2h_btts_rate": h2h_btts,
            "h2h_gf": h2h_gf,
            "h2h_ga": h2h_ga,
            "h2h_n": float(h2h_n),
            "sos_h": sos_h,
            "sos_a": sos_a,
            "cong_h": float(cong_h),
            "cong_a": float(cong_a),
        }

    def place(self, market: str = "over25") -> pd.DataFrame:
        """Place bets for a market."""
        mcfg = self.MARKETS[market]
        log.info(f"Placing {market.upper()} bets")

        model = self.models.get(market)
        if model is None:
            raise RuntimeError(f"No model loaded for {market}")

        ou_api = self._load_odds_api()
        trades_file = get_path(f"paper_trades_{market}")
        predictions_file = get_path(f"paper_state_{market}")

        # Load existing trades
        if trades_file.exists():
            trades = pd.read_csv(trades_file)
        else:
            trades = pd.DataFrame()

        # Load fixtures
        with open(get_path("espn_fixtures")) as f:
            fixtures = json.load(f)

        rows = []
        for fx in fixtures:
            if fx.get("state") != "pre":
                continue

            home = self._canon_team_espn(fx["home"])
            away = self._canon_team_espn(fx["away"])
            d_local = pd.to_datetime(fx["date"], utc=True).tz_convert(
                "America/Mexico_City"
            ).tz_localize(None).normalize()

            key = (d_local, home, away)

            # Check if already processed
            if len(trades) > 0:
                done = trades[
                    (trades["date"] == str(d_local.date())) &
                    (trades["home"] == home) &
                    (trades["away"] == away)
                ]
                if len(done) > 0:
                    continue

            # Build features
            feats = self._fixture_features(home, away, d_local)
            feats["elo_diff"] = feats["elo_h"] - feats["elo_a"]
            feats["rest_diff"] = feats["rest_h"] - feats["rest_a"]

            # Add advanced features
            from liga_mx.features import add_advanced_features
            Xf = pd.DataFrame([feats])
            Xf = add_advanced_features(Xf)

            # Get model features
            FEATS = model.bundle.features
            means = pd.Series(model.bundle.impute_means)
            Xf = Xf[FEATS].fillna(means)
            Xs = model.bundle.scaler.transform(Xf) if model.bundle.scaler is not None else Xf.values

            # Predict
            if market == "over25":
                p_model = float(model.predict_proba(Xf)[0])
                dec, line = self._get_odds_for_fixture(ou_api, fx, home, away, d_local, market)
                if dec is None or not line or float(line) != 2.5:
                    rows.append(self._make_row(d_local, home, away, p_model, None, None, 0.0, "NO_BET", market))
                    continue

                p_mkt = 1.0 / dec
                edge = p_model - p_mkt
                bet_key = "over25"
                odds_dict = {"over": dec}

            elif market == "1x2":
                p = model.predict_proba(Xf)[0]  # [p_H, p_D, p_A]
                odds_tuple, _ = self._get_odds_for_fixture(ou_api, fx, home, away, d_local, market)
                if odds_tuple is None:
                    rows.append(self._make_row(
                        d_local, home, away, None, None, None, None,
                        None, None, None, 0.0, "NO_BET", market, None
                    ))
                    continue

                h_odd, d_odd, a_odd = odds_tuple
                ih, idd, ia = 1/h_odd, 1/d_odd, 1/a_odd
                s = ih + idd + ia
                p_mkt_H, p_mkt_D, p_mkt_A = ih/s, idd/s, ia/s

                edges = {"H": p[0] - p_mkt_H, "D": p[1] - p_mkt_D, "A": p[2] - p_mkt_A}
                best_outcome = max(edges, key=edges.get)
                best_edge = edges[best_outcome]
                best_odds = {"H": h_odd, "D": d_odd, "A": a_odd}[best_outcome]
                p_model_best = p[{"H":0, "D":1, "A":2}[best_outcome]]

                edge = best_edge
                dec = best_odds
                bet_key = f"1x2_{best_outcome}"
                odds_dict = {"H": h_odd, "D": d_odd, "A": a_odd}
                p_model = p_model_best

            else:
                continue

            # Decide bet
            if edge > self.edge_min:
                stake = self.kelly.calculate(p_model, dec, self._current_bankroll())
                if stake > 0:
                    # Check max open bets
                    open_bets = len(trades[
                        (trades.get("status", "") == "OPEN") &
                        (trades.get("market", "") == market)
                    ]) if len(trades) > 0 else 0

                    if open_bets >= self.max_open_bets:
                        status = "SKIP_MAX"
                        stake = 0.0
                    else:
                        status = "OPEN"
                else:
                    stake, status = 0.0, "SKIP"
            else:
                stake, status = 0.0, "SKIP"

            rows.append(self._make_row(
                d_local, home, away,
                p_model=round(p_model, 4) if market == "over25" else None,
                p_H=round(p[0], 4) if market == "1x2" else None,
                p_D=round(p[1], 4) if market == "1x2" else None,
                p_A=round(p[2], 4) if market == "1x2" else None,
                dec_odds=round(dec, 3) if market == "over25" else None,
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
            log.info(f"Recorded {len(new)} entries ({open_cnt} bets)")
            show = new[new["status"] != "NO_BET"]
            if len(show):
                print(show.to_string(index=False))
        else:
            log.info("No new entries")

        return new

    def settle(self, market: str = "over25") -> pd.DataFrame:
        """Settle open bets for a market."""
        mcfg = self.MARKETS[market]
        log.info(f"Settling {market.upper()} bets")

        trades_file = get_path(f"paper_trades_{market}")
        if not trades_file.exists():
            log.info("No trades file")
            return pd.DataFrame()

        # Update master with latest results
        from liga_mx.data.sources import download_football_data
        try:
            download_football_data()
        except Exception as e:
            log.warning(f"Update failed, using current master: {e}")

        master = pd.read_parquet(get_path("master_parquet"))
        master["key_date"] = pd.to_datetime(master["date"]).dt.normalize()

        # Build keymap with ±1 day tolerance
        keymap = {}
        for r in master.itertuples():
            for off in (-1, 0, 1):
                keymap[(r.key_date + pd.Timedelta(days=off), r.home, r.away)] = (r.hs, r.as_, r.res_1x2)

        trades = pd.read_csv(trades_file)
        trades["date"] = pd.to_datetime(trades["date"], format="mixed").dt.normalize()

        # Load odds history for CLV
        hist_path = get_path("odds_history")
        hist = None
        if hist_path.exists():
            try:
                hist = pd.read_csv(hist_path)
                hist["fetched_at"] = pd.to_datetime(hist["fetched_at"], format="mixed", utc=True)
                hist["commence"] = pd.to_datetime(hist["commence"], utc=True)
                from liga_mx.data.team_names import canon_series
                hist["norm_home"] = canon_series(hist["home"].astype(str).str.strip().str.lower())
                hist["norm_away"] = canon_series(hist["away"].astype(str).str.strip().str.lower())
            except Exception as e:
                log.warning(f"Odds history unreadable: {e}")

        def _closing_line(trade_row: pd.Series) -> Any:
            if hist is None:
                return None
            th = trade_row["home"]
            ta = trade_row["away"]
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
                return {
                    "H": float(pre["h2h_h_med"].iloc[-1]),
                    "D": float(pre["h2h_d_med"].iloc[-1]),
                    "A": float(pre["h2h_a_med"].iloc[-1]),
                }
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

                cl = _closing_line(t)
                if cl is not None and cl > 0:
                    trades.loc[i, "close_odds"] = round(cl, 3)
                    trades.loc[i, "clv_pct"] = round((float(t["dec_odds"]) / cl - 1) * 100, 2)

            elif market == "1x2":
                bet_outcome = t["bet_key"].split("_")[1]
                win = (res == bet_outcome)
                odds_dict = json.loads(t["odds_dict"]) if pd.notna(t.get("odds_dict")) else {}
                bet_odds = odds_dict.get(bet_outcome, 0)
                pnl = t["stake"] * (bet_odds - 1) if win else -t["stake"]

                cl = _closing_line(t)
                if cl and isinstance(cl, dict):
                    bet_cl = cl.get(bet_outcome)
                    if bet_cl and bet_cl > 0:
                        trades.loc[i, "close_odds"] = round(bet_cl, 3)
                        trades.loc[i, "clv_pct"] = round((float(bet_odds) / bet_cl - 1) * 100, 2)

            trades.loc[i, "result"] = "WIN" if win else "LOSE"
            trades.loc[i, "pnl"] = round(pnl, 2)
            trades.loc[i, "status"] = "SETTLED"
            settled += 1

        # Update bankroll
        br = self._update_bankroll(market, trades)

        trades.to_csv(trades_file, index=False, encoding="utf-8-sig")

        s = trades[trades["status"] == "SETTLED"]
        log.info(f"Settled: {settled} | Bankroll: {br:.2f} (base {self.initial_bankroll}, PnL {br - self.initial_bankroll:+.2f})")

        if len(s):
            wins = int((s["result"] == "WIN").sum())
            losses = int((s["result"] == "LOSE").sum())
            log.info(f"  Record: {wins}-{losses} | Yield: {s['pnl'].sum() / s['stake'].sum() * 100:.1f}% over {len(s)} bets")

        if "clv_pct" in trades.columns:
            sc = trades[trades["clv_pct"].notna()]
            if len(sc):
                beat = (sc["clv_pct"] > 0).sum()
                log.info(f"  CLV: {len(sc)} trades | avg {sc['clv_pct'].mean():+.2f}% | median {sc['clv_pct'].median():+.2f}% | {beat}/{len(sc)} beat closing ({beat/len(sc)*100:.0f}%)")

        return trades

    def place_all(self) -> None:
        """Place bets for all markets."""
        self.place("over25")
        self.place("1x2")

    def settle_all(self) -> None:
        """Settle bets for all markets."""
        self.settle("over25")
        self.settle("1x2")

    def _current_bankroll(self) -> float:
        """Calculate current total bankroll across markets."""
        total = self.initial_bankroll
        for m in ["over25", "1x2"]:
            state_file = get_path(f"paper_state_{m}")
            if state_file.exists():
                with open(state_file) as f:
                    total += json.load(f).get("bankroll", 0) - self.initial_bankroll
        return total

    def _update_bankroll(self, market: str, trades: pd.DataFrame) -> float:
        """Update bankroll state file."""
        state_file = get_path(f"paper_state_{market}")
        br = self.initial_bankroll + float(trades.loc[trades["status"] == "SETTLED", "pnl"].sum())
        with open(state_file, "w") as f:
            json.dump({"bankroll": round(br, 2)}, f)
        return br

    def _make_row(
        self,
        date: pd.Timestamp,
        home: str,
        away: str,
        p_model: float | None = None,
        p_H: float | None = None,
        p_D: float | None = None,
        p_A: float | None = None,
        dec_odds: float | None = None,
        odds_dict: dict | None = None,
        edge: float | None = None,
        stake: float = 0.0,
        status: str = "NO_BET",
        market: str = "over25",
        bet_key: str | None = None,
    ) -> dict:
        return {
            "date": date,
            "home": home,
            "away": away,
            "market": market,
            "bet_key": bet_key,
            "p_model": p_model,
            "p_H": p_H,
            "p_D": p_D,
            "p_A": p_A,
            "dec_odds": dec_odds,
            "odds_dict": json.dumps(odds_dict) if odds_dict else None,
            "edge": edge,
            "stake": stake,
            "status": status,
            "result": None,
            "pnl": None,
            "close_odds": None,
            "clv_pct": None,
        }


__all__ = ["PaperTrader", "KellyStakeCalculator"]