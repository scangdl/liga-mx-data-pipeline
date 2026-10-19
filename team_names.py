"""team_names.py — normalización y nombres canónicos de equipos Liga MX."""
import re
import unicodedata
import pandas as pd

UNKNOWN = set()

ALIASES = {
    "club america": "America", "america": "America",
    "atlante": "Atlante", "atlante fc": "Atlante",
    "atl san luis": "San Luis", "atletico san luis": "San Luis",
    "atletico de san luis": "San Luis",
    "atlas": "Atlas", "chiapas": "Chiapas",
    "club leon": "Leon", "leon": "Leon",
    "club tijuana": "Tijuana", "tijuana": "Tijuana",
    "cruz azul": "Cruz Azul",
    "dorados de sinaloa": "Dorados",
    "guadalajara chivas": "Chivas", "guadalajara": "Chivas",
    "cd guadalajara": "Chivas",
    "juarez": "Juarez", "fc juarez": "Juarez",
    "leones negros": "Leones Negros",
    "lobos buap": "Lobos BUAP",
    "mazatlan fc": "Mazatlan", "mazatlan": "Mazatlan",
    "monarcas": "Monarcas", "club atletico morelia": "Monarcas",
    "monterrey": "Monterrey", "necaxa": "Necaxa",
    "pachuca": "Pachuca", "puebla": "Puebla",
    "queretaro": "Queretaro",
    "santos laguna": "Santos", "santos": "Santos",
    "toluca": "Toluca",
    "u a n l tigres": "Tigres", "tigres uanl": "Tigres",
    "u n a m pumas": "Pumas", "unam pumas": "Pumas",
    "pumas unam": "Pumas",
    "tigres": "Tigres", "pumas": "Pumas", "atlante": "Atlante",
    "veracruz": "Veracruz",
}


def norm(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon(s):
    n = norm(s)
    if n is None:
        return None
    if n in ALIASES:
        return ALIASES[n]
    UNKNOWN.add(n)
    return n


def canon_series(ser):
    return ser.map(canon)
