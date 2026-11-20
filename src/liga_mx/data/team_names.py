"""team_names.py — normalización y nombres canónicos de equipos Liga MX."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

UNKNOWN: set[str] = set()

ALIASES: dict[str, str] = {
    "club america": "America",
    "america": "America",
    "atlante": "Atlante",
    "atlante fc": "Atlante",
    "atl san luis": "San Luis",
    "atletico san luis": "San Luis",
    "atletico de san luis": "San Luis",
    "atlas": "Atlas",
    "chiapas": "Chiapas",
    "club leon": "Leon",
    "leon": "Leon",
    "club tijuana": "Tijuana",
    "tijuana": "Tijuana",
    "cruz azul": "Cruz Azul",
    "dorados de sinaloa": "Dorados",
    "guadalajara chivas": "Chivas",
    "guadalajara": "Chivas",
    "cd guadalajara": "Chivas",
    "juarez": "Juarez",
    "fc juarez": "Juarez",
    "leones negros": "Leones Negros",
    "lobos buap": "Lobos BUAP",
    "mazatlan fc": "Mazatlan",
    "mazatlan": "Mazatlan",
    "monarcas": "Monarcas",
    "club atletico morelia": "Monarcas",
    "monterrey": "Monterrey",
    "necaxa": "Necaxa",
    "pachuca": "Pachuca",
    "puebla": "Puebla",
    "queretaro": "Queretaro",
    "santos laguna": "Santos",
    "santos": "Santos",
    "toluca": "Toluca",
    "u a n l tigres": "Tigres",
    "tigres uanl": "Tigres",
    "u n a m pumas": "Pumas",
    "unam pumas": "Pumas",
    "pumas unam": "Pumas",
    "tigres": "Tigres",
    "pumas": "Pumas",
    "veracruz": "Veracruz",
}


def norm(s: str | None) -> str | None:
    """Normalize string: lowercase, strip accents, collapse spaces."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon(s: str | None) -> str | None:
    """Map normalized string to canonical team name."""
    n = norm(s)
    if n is None:
        return None
    if n in ALIASES:
        return ALIASES[n]
    UNKNOWN.add(n)
    return n


def canon_series(ser: pd.Series) -> pd.Series:
    """Apply canon to a pandas Series."""
    return ser.map(canon)


__all__ = ["norm", "canon", "canon_series", "UNKNOWN", "ALIASES"]