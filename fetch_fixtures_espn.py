#!/usr/bin/env python3
"""
fetch_fixtures_espn.py — Refresca data/espn_fixtures.json desde la API publica
de ESPN (site.web.api.espn.com), ventana de 21 dias hacia adelante.

Formato de salida (igual al consumido por predict_jornada.py / paper_trade.py):
  [{"id", "date" (UTC iso "2026-10-10T01:00Z"), "home", "away", "state",
    "line", "over_ml", "under_ml", "draw_ml", "provider"}, ...]

Los ML de DraftKings suelen venir null en la API publica -> paper_trade.py usa
como fuente primaria la Odds API (odds_api_jornada.csv) y solo cae a estos.
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone

FUTBOL = "/home/petyt/futbolmx"
OUT = FUTBOL + "/data/espn_fixtures.json"
BASE = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def get(day):
    url = BASE + "?dates=" + day
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def main():
    today = datetime.now(timezone.utc).date()
    out = []
    for i in range(0, 22):
        day = today + timedelta(days=i)
        ds = day.strftime("%Y%m%d")
        try:
            d = get(ds)
        except Exception as e:
            print("  %s error: %s" % (ds, e))
            continue
        for ev in d.get("events", []):
            c = ev["competitions"][0]
            teams = {}
            for t in c.get("competitors", []):
                teams[t["homeAway"]] = t["team"]["displayName"]
            odds = [x for x in (c.get("odds") or []) if x]
            o = odds[0] if odds else {}
            prov = (o.get("provider") or {}).get("name")
            # ESPN anida las cuotas: total.over.close.odds, total.under.close.odds
            # y drawOdds.moneyLine (no existen planas en la raiz del objeto odds)
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
    seen = set()
    dedup = []
    for fx in out:
        if fx["id"] in seen:
            continue
        seen.add(fx["id"])
        dedup.append(fx)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dedup, f, ensure_ascii=False, indent=1)
    print("guardado: %s (%d fixtures)" % (OUT, len(dedup)))
    for fx in dedup:
        print("  %s %4s %s vs %s" % (fx["date"], fx["state"], fx["home"], fx["away"]))


if __name__ == "__main__":
    main()
