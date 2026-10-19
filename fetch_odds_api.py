#!/usr/bin/env python3
"""
fetch_odds_api.py — Odds API (the-odds-api.com) para Liga MX.

Guarda odds 1X2 (h2h) y O/U 2.5 (totals) de las casas disponibles,
por evento. Salida: data/odds_api_snapshot.json + CSV plano.

Uso:
    fetch_odds_api.py            # jornada proxima (eventos activos)
"""
import csv
import json
import time
import urllib.request
from pathlib import Path

FUTBOL = Path("/home/petyt/futbolmx")
DATA = FUTBOL / "data"
KEY_FILE = FUTBOL / "data" / ".odds_api_key"
SPORT = "soccer_mexico_ligamx"
BASE = "https://api.the-odds-api.com/v4"

key = KEY_FILE.read_text().strip()
if " " in key:
    key = key.split()[-1]  # por si el archivo trae "odds api <key>"


def get(path, params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"User-Agent": "futbolmx/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.status, json.loads(r.read().decode()), dict(r.headers)


def median(prices):
    s = sorted(prices)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# 1) eventos + odds h2h y totals
st, events, hdrs = get(f"/sports/{SPORT}/odds", {
    "apiKey": key, "regions": "us,eu,uk",
    "markets": "h2h,totals", "oddsFormat": "decimal",
})
print("odds:", st, "| cuota restante:", hdrs.get("x-requests-remaining"))
if st != 200:
    raise SystemExit("fallo odds: " + json.dumps(events)[:200])

snapshot = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": events}
(DATA / "odds_api_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=1))

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
        "home": h, "away": a,
        "n_books_ou": len(ou_o),
        "ou_line": line,
        "over25_med": round(median(ou_o), 3) if ou_o else "",
        "under25_med": round(median(ou_u), 3) if ou_u else "",
        "ou_margin": round(1 / median(ou_o) + 1 / median(ou_u) - 1, 4) if ou_o and ou_u else "",
        "h2h_h_med": round(median(h2h_h), 3) if h2h_h else "",
        "h2h_d_med": round(median(h2h_d), 3) if h2h_d else "",
        "h2h_a_med": round(median(h2h_a), 3) if h2h_a else "",
    })

out_csv = DATA / "odds_api_jornada.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- historial para CLV: append con timestamp de la corrida ----
hist_csv = DATA / "odds_history.csv"
hist_fields = ["fetched_at", "commence", "home", "away", "ou_line", "over25_med", "under25_med", "h2h_h_med", "h2h_d_med", "h2h_a_med"]
new_file = not hist_csv.exists()
with open(hist_csv, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=hist_fields)
    if new_file:
        w.writeheader()
    for r in rows:
        if r.get("ou_line") and r.get("over25_med"):
            w.writerow({k: r.get(k, "") for k in hist_fields} | {"fetched_at": snapshot["fetched_at"]})

print(f"guardado: {out_csv} ({len(rows)} partidos)")
for r in rows:
    print(f"  {r['home']} vs {r['away']}: O/U {r['ou_line']} med over {r['over25_med']} "
          f"| under {r['under25_med']} | books {r['n_books_ou']} | margen {r['ou_margin']}")
