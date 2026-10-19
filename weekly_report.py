#!/usr/bin/env python3
"""
weekly_report.py — Resumen del paper trading Liga MX para el cron semanal.

Ejecuta settle (via update implicito leyendo el master ya actualizado por
paper_trade settle) y luego imprime un reporte:
  - apuestas liquidadas de la semana
  - record y PnL acumulado
  - apuestas abiertas de la proxima jornada
No toca nada: settle lo maneja paper_trade.py.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

FUTBOL = Path("/home/petyt/futbolmx")
PRED = FUTBOL / "predictions"


def log(msg):
    print(msg, flush=True)


# 1. Liquidar (paper_trade settle ya actualiza el master primero)
log("== LIQUIDACION ==")
r = subprocess.run([sys.executable, str(FUTBOL / "paper_trade.py"), "settle"],
                    capture_output=True, text=True, cwd=str(FUTBOL))
log(r.stdout.strip() or r.stderr.strip())

# 2. Reporte del estado
t = pd.read_csv(PRED / "paper_trades.csv")
t["date"] = pd.to_datetime(t["date"])

s = t[t["status"] == "SETTLED"].copy()
n_win = int((s["result"] == "WIN").sum())
n_lose = int((s["result"] == "LOSE").sum())
pnl = float(s["pnl"].sum()) if len(s) else 0.0
staked = float(s["stake"].sum()) if len(s) else 0.0
bank = 1000.0 + pnl

log("\n== ESTADO DEL PAPER TRADING ==")
log(f"Bankroll: {bank:.2f} (base 1000, PnL {pnl:+.2f})")
if len(s):
    if staked:
        log(f"Record: {n_win}-{n_lose} | Volumen: {staked:.2f} u | "
            f"ROI: {pnl / staked * 100:+.1f}%")
    else:
        log("Record: sin volumen")
    log("\nUltimas liquidadas:")
    last = s.sort_values("date").tail(8)
    log(last[["date", "home", "away", "p_model", "dec_odds", "stake", "result", "pnl"]]
        .to_string(index=False))

op = t[t["status"] == "OPEN"].copy()
if len(op):
    log("\n== APUESTAS ABIERTAS ==")
    log(op[["date", "home", "away", "p_model", "dec_odds", "edge", "stake"]]
        .to_string(index=False))
else:
    log("\n== APUESTAS ABIERTAS: ninguna ==")

# 3. Calibracion rapida (solo si hay liquidadas)
if len(s):
    log("\n== CALIBRACION (liquidadas) ==")
    try:
        bins = pd.cut(s["p_model"], bins=[0, 0.45, 0.55, 0.65, 1.0])
        cal = s.groupby(bins, observed=True).agg(n=("result", "size"),
                                                  p_media=("p_model", "mean"),
                                                  win_rate=("result", lambda x: (x == "WIN").mean()))
        log(cal.round(3).to_string())
    except Exception as e:
        log(f"(calibracion no disponible: {e})")
