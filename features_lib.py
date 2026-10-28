#!/usr/bin/env python3
"""
features_lib.py — Librería unificada de features pre-partido (sin leakage).

Single source of truth para:
  - make_features.py (entrenamiento batch)
  - predict_jornada.py (inference jornada)
  - paper_trade.py (place/settle)

Regla de oro: todo feature usa solo información disponible ANTES del partido.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy.stats import poisson


# ============================================================
# CONSTANTES GLOBALES (iguales en train e inference)
# ============================================================
K = 32.0
HOME_ADV = 80.0
ALPHA_EWMA = 0.93
LGF_WINDOW = 800  # últimos N partidos para medias de liga

# Features que espera el modelo v2 (orden fijo)
MODEL_FEATURES = [
    "elo_diff", "elo_h", "elo_a",
    "rest_h", "rest_a", "rest_diff",
    "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
    "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
    "gf_home_as_home", "ga_home_as_home",
    "gf_away_as_away", "ga_away_as_away",
    "imp_h", "imp_d", "imp_a",
]

# Features extendidas (make_features + predict_jornada v2)
EXTENDED_FEATURES = MODEL_FEATURES + [
    "pos_table_h", "pos_table_a",
    "ppg_table_h", "ppg_table_a",
    "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
    "sos_h", "sos_a",
    "cong_h", "cong_a",
    "lam_h", "lam_a",
]


# ============================================================
# UTILIDADES BASE
# ============================================================

def log(msg):
    print(msg, flush=True)


def american_to_decimal(ml) -> Optional[float]:
    """ML americano -> decimal. None -> None."""
    if ml is None:
        return None
    try:
        ml = int(ml)
    except (ValueError, TypeError):
        return None
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1


def pick_odds(row: pd.Series, prefs: List[str]) -> float:
    """Primera cuota no-NaN en orden de preferencia."""
    for c in prefs:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return np.nan


def normalize_1x2(odd_h: float, odd_d: float, odd_a: float) -> Tuple[float, float, float]:
    """Normaliza 3 cuotas 1X2 a probabilidades implícitas sin margen."""
    ih, idd, ia = 1 / odd_h, 1 / odd_d, 1 / odd_a
    s = ih + idd + ia
    return ih / s, idd / s, ia / s


# ============================================================
# STATE BUILDER — construye estado secuencial partido a partido
# ============================================================
class StateBuilder:
    """
    Construye estado de equipos procesando partidos EN ORDEN CRONOLÓGICO.
    Único punto de verdad para: Elo, forma, EWMA, tabla, H2H, SOS, congestión, lambdas.
    """

    def __init__(self, master: pd.DataFrame):
        """
        master: DataFrame con columnas mínimas:
          date, home, away, hs, as_, res_1x2, season, torneo, anio_torneo,
          odd_h, odd_d, odd_a, venue (opcional para altitud)
        """
        self.master = master.sort_values("date").reset_index(drop=True)

        # Estado mutable
        self.elo: Dict[str, float] = {}
        self.last_match: Dict[str, pd.Timestamp] = {}
        self.hist: Dict[str, List[Tuple]] = {}  # team -> [(gf, ga, btts, pts, role)]
        self.ewma: Dict[str, Dict[int, List[float]]] = {}  # team -> {role: [atk, def]}
        self.team_state: Dict[str, Dict[str, List]] = {}  # team -> {opp_elo: [], dates: []}
        self.tabla_torneo: Dict[Tuple, Dict[str, List]] = {}  # tor_id -> {team: [pts, gf, ga]}
        self.h2h: Dict[Tuple, List] = {}  # (teamA, teamB) -> [(gf_A, ga_A, btts, date)]
        self.lgf_home: List[float] = []
        self.lgf_away: List[float] = []

        # Para tracking
        self.current_tor_id: Optional[Tuple] = None

    # ---- helpers internos ----
    def _get_elo(self, team: str) -> float:
        return self.elo.get(team, 1500.0)

    def _update_elo(self, home: str, away: str, hs: float, as_: float):
        eh, ea = self._get_elo(home), self._get_elo(away)
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV - ea) / 400.0)))
        if hs > as_:
            s = 1.0
        elif hs < as_:
            s = 0.0
        else:
            s = 0.5
        margen = abs(hs - as_)
        mult = 1.0 if margen <= 1 else (1.5 if margen == 2 else 1.75 + (margen - 3) * 0.125)
        delta = K * mult * (s - exp_h)
        self.elo[home] = eh + delta
        self.elo[away] = ea - delta

    def _roll_stats(self, team: str, n: int) -> Tuple[float, float, float, float, float]:
        h = self.hist.get(team, [])
        if not h or len(h) < 1:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        arr = np.array(h[-n:])
        gf = float(arr[:, 0].mean())
        ga = float(arr[:, 1].mean())
        bs = float(arr[:, 2].mean())
        pts = float(arr[:, 3].mean())
        return (gf, ga, bs, pts, float(len(arr)))

    def _ewma_get(self, team: str, role: int) -> Tuple[Optional[float], Optional[float]]:
        st = self.ewma.get(team, {}).get(role)
        if not st or st[0] is None:
            return None, None
        return float(st[0]), float(st[1])

    def _sos(self, team: str) -> float:
        st = self.team_state.get(team)
        if st and st.get("opp_elo"):
            vals = [v for v in st["opp_elo"][-5:] if pd.notna(v)]
            return float(np.mean(vals)) if vals else np.nan
        return np.nan

    def _congestion(self, team: str, d: pd.Timestamp) -> int:
        st = self.team_state.get(team)
        if st and st.get("dates"):
            return sum(1 for x in st["dates"][-5:] if (d - x).days <= 14)
        return 0

    def _lam_features(self, home: str, away: str) -> Tuple[float, float]:
        sh = self._ewma_get(home, 1)
        sa = self._ewma_get(away, 0)
        if not sh or not sa or sh[0] is None or sa[0] is None:
            return np.nan, np.nan
        ah, dh = sh
        aa, da = sa
        lgf_h = float(np.mean(self.lgf_home[-LGF_WINDOW:])) if self.lgf_home else 1.4
        lgf_a = float(np.mean(self.lgf_away[-LGF_WINDOW:])) if self.lgf_away else 1.15
        lam_h = (ah / lgf_h) * (da / lgf_h) * lgf_h
        lam_a = (aa / lgf_a) * (dh / lgf_a) * lgf_a
        return lam_h, lam_a

    def _table_pos(self, tor_id: Tuple, team: str) -> Tuple[float, float]:
        tabla = self.tabla_torneo.get(tor_id, {})
        th = tabla.get(team, [0, 0, 0])
        if th[2] == 0:
            return np.nan, np.nan
        pos = 1 + sum(1 for v in tabla.values() if v[0] > th[0])
        ppg = th[0] / max(1, th[2])
        return float(pos), ppg

    def _h2h_features(self, home: str, away: str) -> Tuple[float, float, float, int]:
        key = tuple(sorted([home, away]))
        prev = self.h2h.get(key, [])
        if not prev:
            return np.nan, np.nan, np.nan, 0
        h_home = (home == key[0])
        btts_vals = [x[2] for x in prev[-6:]]
        gf_side = [x[0] if h_home else x[1] for x in prev[-6:]]
        ga_side = [x[1] if h_home else x[0] for x in prev[-6:]]
        return (
            float(np.mean(btts_vals)),
            float(np.mean(gf_side)),
            float(np.mean(ga_side)),
            len(prev[-6:]),
        )

    def _form_feats(self, team: str) -> Dict[str, float]:
        n10 = self._roll_stats(team, 10)
        hh = [x for x in self.hist.get(team, []) if x[4] == 1]
        aa = [x for x in self.hist.get(team, []) if x[4] == 0]
        gf_hh = float(np.mean([x[0] for x in hh[-6:]])) if hh else np.nan
        ga_hh = float(np.mean([x[1] for x in hh[-6:]])) if hh else np.nan
        gf_aa = float(np.mean([x[0] for x in aa[-6:]])) if aa else np.nan
        ga_aa = float(np.mean([x[1] for x in aa[-6:]])) if aa else np.nan
        return {
            "gf10": n10[0], "ga10": n10[1], "btts_rate10": n10[2], "ppg10": n10[3],
            "gf_home_as_home": gf_hh, "ga_home_as_home": ga_hh,
            "gf_away_as_away": gf_aa, "ga_away_as_away": ga_aa,
        }

    # ---- API PRINCIPAL ----
    def process_match(self, row: pd.Series) -> Dict[str, float]:
        """
        Procesa UN partido (en orden cronológico) y devuelve features pre-partido.
        Actualiza estado interno para el siguiente partido.
        """
        h = str(row["home"])
        a = str(row["away"])
        d = pd.Timestamp(row["date"])
        hs = row["hs"] if pd.notna(row["hs"]) else np.nan
        as_ = row["as_"] if pd.notna(row["as_"]) else np.nan
        played = pd.notna(hs) and pd.notna(as_)

        # --- Elo pre-partido ---
        eh = self._get_elo(h)
        ea = self._get_elo(a)
        elo_diff = eh - ea

        # --- Descanso ---
        rh = (d - self.last_match[h]).days if h in self.last_match else np.nan
        ra = (d - self.last_match[a]).days if a in self.last_match else np.nan
        rest_diff = rh - ra if pd.notna(rh) and pd.notna(ra) else np.nan

        # --- Forma (10 últimos) ---
        ff_h = self._form_feats(h)
        ff_a = self._form_feats(a)

        # --- EWMA / Lambdas ---
        lam_h, lam_a = self._lam_features(h, a)

        # --- Tabla del torneo ---
        tor_id = (row.get("anio_torneo"), row.get("torneo"))
        pos_h, ppg_h = self._table_pos(tor_id, h)
        pos_a, ppg_a = self._table_pos(tor_id, a)

        # --- H2H ---
        h2h_btts, h2h_gf, h2h_ga, h2h_n = self._h2h_features(h, a)

        # --- SOS y congestión ---
        sos_h = self._sos(h)
        sos_a = self._sos(a)
        cong_h = self._congestion(h, d)
        cong_a = self._congestion(a, d)

        # --- Odds implícitas (si existen en row) ---
        imp_h = imp_d = imp_a = np.nan
        if all(c in row.index for c in ("odd_h", "odd_d", "odd_a")):
            if pd.notna(row["odd_h"]) and pd.notna(row["odd_d"]) and pd.notna(row["odd_a"]):
                imp_h, imp_d, imp_a = normalize_1x2(
                    float(row["odd_h"]), float(row["odd_d"]), float(row["odd_a"])
                )

        # ==== CONSTRUIR FEATURES PRE-PARTIDO ====
        feats = {
            "elo_h": eh, "elo_a": ea, "elo_diff": elo_diff,
            "rest_h": rh, "rest_a": ra, "rest_diff": rest_diff,
            "gf10_h": ff_h["gf10"], "ga10_h": ff_h["ga10"],
            "btts_rate10_h": ff_h["btts_rate10"], "ppg10_h": ff_h["ppg10"],
            "gf10_a": ff_a["gf10"], "ga10_a": ff_a["ga10"],
            "btts_rate10_a": ff_a["btts_rate10"], "ppg10_a": ff_a["ppg10"],
            "gf_home_as_home": ff_h["gf_home_as_home"],
            "ga_home_as_home": ff_h["ga_home_as_home"],
            "gf_away_as_away": ff_a["gf_away_as_away"],
            "ga_away_as_away": ff_a["ga_away_as_away"],
            "imp_h": imp_h, "imp_d": imp_d, "imp_a": imp_a,
            # extendidas
            "pos_table_h": pos_h, "pos_table_a": pos_a,
            "ppg_table_h": ppg_h, "ppg_table_a": ppg_a,
            "h2h_btts_rate": h2h_btts, "h2h_gf": h2h_gf,
            "h2h_ga": h2h_ga, "h2h_n": float(h2h_n),
            "sos_h": sos_h, "sos_a": sos_a,
            "cong_h": float(cong_h), "cong_a": float(cong_a),
            "lam_h": lam_h, "lam_a": lam_a,
        }

        # ==== ACTUALIZAR ESTADO (solo si partido jugado) ====
        if played:
            hs_f, as_f = float(hs), float(as_)

            # Elo
            self._update_elo(h, a, hs_f, as_f)

            # Descanso / último partido
            self.last_match[h] = self.last_match[a] = d

            # Puntos y BTTS
            pts_h = 3.0 if hs_f > as_f else (1.0 if hs_f == as_f else 0.0)
            pts_a = 3.0 - pts_h if pts_h in (0.0, 3.0) else 1.0
            btts_val = 1.0 if (hs_f > 0 and as_f > 0) else 0.0

            # Historial (para forma)
            self.hist.setdefault(h, []).append((hs_f, as_f, btts_val, pts_h, 1))
            self.hist.setdefault(a, []).append((as_f, hs_f, btts_val, pts_a, 0))

            # EWMA
            for team, role, gf, ga in ((h, 1, hs_f, as_f), (a, 0, as_f, hs_f)):
                st = self.ewma.setdefault(team, {}).setdefault(role, [None, None])
                st[0] = gf if st[0] is None else ALPHA_EWMA * st[0] + (1 - ALPHA_EWMA) * gf
                st[1] = ga if st[1] is None else ALPHA_EWMA * st[1] + (1 - ALPHA_EWMA) * ga

            # Medias de liga
            self.lgf_home.append(hs_f)
            self.lgf_away.append(as_f)

            # Tabla del torneo
            tor_id = (row.get("anio_torneo"), row.get("torneo"))
            if not hasattr(self, "_current_tor_id") or self._current_tor_id != tor_id:
                self.tabla_torneo[tor_id] = {}
                self._current_tor_id = tor_id
            tabla = self.tabla_torneo[tor_id]
            th = tabla.get(h, [0, 0, 0])
            ta = tabla.get(a, [0, 0, 0])
            th[0] += pts_h; th[1] += hs_f; th[2] += as_f
            ta[0] += pts_a; ta[1] += as_f; ta[2] += hs_f
            tabla[h] = th
            tabla[a] = ta

            # H2H
            key = tuple(sorted([h, a]))
            k0_gf = hs_f if h == tuple(sorted([h, a]))[0] else as_f
            k1_gf = as_f if h == tuple(sorted([h, a]))[0] else hs_f
            self.h2h.setdefault(key, []).append((k0_gf, k1_gf, btts_val, row["date"]))

            # Estados para SOS/congestión
            for team, gf, ga, pts, opp_elo in [
                (h, hs_f, as_f, pts_h, ea),
                (a, as_f, hs_f, pts_a, eh),
            ]:
                st = self.team_state.setdefault(team, {"opp_elo": [], "dates": []})
                st["opp_elo"].append(opp_elo if pd.notna(opp_elo) else np.nan)
                st["dates"].append(d)

        return feats

    def build_all(self) -> pd.DataFrame:
        """Procesa todo el master en orden y devuelve DataFrame con features."""
        rows = []
        for _, row in self.master.iterrows():
            feats = self.process_match(row)
            rows.append(feats)
        return pd.DataFrame(rows, index=self.master.index)

    # ---- utilidades para inference (predict_jornada / paper_trade) ----
    def get_current_state(self) -> Dict:
        """Devuelve snapshot del estado actual para inference."""
        lgf_h = float(np.mean(self.lgf_home[-LGF_WINDOW:])) if self.lgf_home else 1.4
        lgf_a = float(np.mean(self.lgf_away[-LGF_WINDOW:])) if self.lgf_away else 1.15
        return {
            "elo": self.elo.copy(),
            "last_match": {k: v for k, v in self.last_match.items()},
            "hist": {k: list(v) for k, v in self.hist.items()},
            "ewma": {k: {r: list(v) for r, v in v.items()} for k, v in self.ewma.items()},
            "team_state": {k: {"opp_elo": list(v["opp_elo"]), "dates": list(v["dates"])}
                           for k, v in self.team_state.items()},
            "tabla_torneo": {k: {tk: list(tv) for tk, tv in v.items()}
                             for k, v in self.tabla_torneo.items()},
            "h2h": {k: list(v) for k, v in self.h2h.items()},
            "lgf_h": lgf_h, "lgf_a": lgf_a,
            "current_tor_id": getattr(self, "_current_tor_id", None),
        }

    @classmethod
    def from_state(cls, master: pd.DataFrame, state: Dict) -> "StateBuilder":
        """Reconstruye StateBuilder desde snapshot (para inference)."""
        sb = cls(master.iloc[0:0])  # master vacío, solo para estructura
        sb.elo = state["elo"]
        sb.last_match = {k: pd.Timestamp(v) for k, v in state["last_match"].items()}
        sb.hist = {k: list(v) for k, v in state["hist"].items()}
        sb.ewma = {k: {r: list(v) for r, v in v.items()} for k, v in state["ewma"].items()}
        sb.team_state = {k: {"opp_elo": list(v["opp_elo"]), "dates": [pd.Timestamp(x) for x in v["dates"]]}
                         for k, v in state["team_state"].items()}
        sb.tabla_torneo = {k: {tk: list(tv) for tk, tv in v.items()}
                           for k, v in state["tabla_torneo"].items()}
        sb.h2h = {k: list(v) for k, v in state["h2h"].items()}
        sb.lgf_home = list(state["lgf_h"] * np.ones(LGF_WINDOW)) if isinstance(state["lgf_h"], float) else list(state["lgf_h"])
        sb.lgf_away = list(state["lgf_a"] * np.ones(LGF_WINDOW)) if isinstance(state["lgf_a"], float) else list(state["lgf_a"])
        sb._current_tor_id = state.get("current_tor_id")
        return sb


# ============================================================
# FEATURE ENGINEERING DE ALTO NIVEL
# ============================================================
def build_features_master(master_path: Path) -> pd.DataFrame:
    """Pipeline completo: carga master, construye features, guarda parquet."""
    master = pd.read_parquet(master_path).sort_values("date").reset_index(drop=True)
    sb = StateBuilder(master)
    feats_df = sb.build_all()
    out = pd.concat([master.reset_index(drop=True), feats_df], axis=1)
    return out


def build_state_for_inference(master_path: Path) -> Dict:
    """Construye estado final del master para inference (predict_jornada / paper_trade)."""
    master = pd.read_parquet(master_path).sort_values("date").reset_index(drop=True)
    sb = StateBuilder(master)
    sb.build_all()  # procesa todo para construir estado
    return sb.get_current_state()


# ============================================================
# POISSON BTTS (feature de referencia)
# ============================================================
def poisson_btts_prob(lam_h: float, lam_a: float) -> float:
    """P(BTTS) = (1 - P(0|lam_h)) * (1 - P(0|lam_a)) bajo Poisson independiente."""
    if pd.isna(lam_h) or pd.isna(lam_a):
        return np.nan
    return (1 - poisson.pmf(0, lam_h)) * (1 - poisson.pmf(0, lam_a))


# ============================================================
# EXPORTS
# ============================================================
__all__ = [
    "StateBuilder",
    "build_features_master",
    "build_state_for_inference",
    "poisson_btts_prob",
    "american_to_decimal",
    "pick_odds",
    "normalize_1x2",
    "log",
    "MODEL_FEATURES",
    "EXTENDED_FEATURES",
    "K", "HOME_ADV", "ALPHA_EWMA", "LGF_WINDOW",
]