#!/usr/bin/env python3
"""Cron wrapper: Settle paper trading bets (Monday 15:00)."""

from liga_mx.trading import PaperTrader
from liga_mx.config import load_config

if __name__ == "__main__":
    load_config()
    print("=== CRON: Settle Bets ===")
    trader = PaperTrader()
    trader.settle_all()