#!/usr/bin/env python3
"""test_odds_api.py — prueba la Odds API: key valida, deportes MX, mercados y cuota restante."""
import json
import urllib.request

KEY = "6d99e9b6c069a0f6de71b5eee957b864"
BASE = "https://api.the-odds-api.com/v4"


def get(path, params=None):
    qs = ""
    if params:
        qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(BASE + path + qs, headers={"User-Agent": "futbolmx/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), dict(e.headers)


# 1) lista de deportes
st, body, hdrs = get("/sports", {"apiKey": KEY})
print("GET /sports ->", st)
print("cuota restante:", hdrs.get("x-requests-remaining"), "| usadas:", hdrs.get("x-requests-used"))
if st != 200:
    print(body[:300])
    raise SystemExit(1)

sports = json.loads(body)
mex = [s for s in sports if "mexico" in s.get("key", "") or "mexico" in s.get("title", "").lower()]
print("deportes Mexico:", [(s["key"], s["title"]) for s in mex])
print("total deportes:", len(sports))

# 2) si hay liga mx, ver eventos proximos y mercados
if not mex:
    raise SystemExit("sin ligas de Mexico en el plan")
key_liga = mex[0]["key"]
st, body, hdrs = get(f"/sports/{key_liga}/odds", {
    "apiKey": KEY, "regions": "us,eu", "markets": "h2h,totals", "oddsFormat": "decimal",
})
print()
print(f"GET odds {key_liga} ->", st)
print("cuota restante:", hdrs.get("x-requests-remaining"), "| usadas:", hdrs.get("x-requests-used"))
if st == 200:
    events = json.loads(body)
    print("eventos:", len(events))
    for ev in events[:3]:
        print(" -", ev["home_team"], "vs", ev["away_team"], "|", ev["commence_time"])
        for bm in ev.get("bookmakers", [])[:2]:
            mkts = {m["key"]: m for m in bm.get("markets", [])}
            if "totals" in mkts:
                for o in mkts["totals"]["outcomes"]:
                    print(f"    {bm['title']}: O/U {o.get('point')} -> {o['name']} {o['price']}")
else:
    print(body[:300])
