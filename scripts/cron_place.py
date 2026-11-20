#!/usr/bin/env python3
"""Cron wrapper: Place paper trading bets (Monday 10:00)."""

from liga_mx.trading import PaperTrader
from liga_mx.config import load_config

if __name__ == "__main__":
    load_config()
    print("=== CRON: Place Bets ===")
    trader = PaperTrader()
    trader.place_all()