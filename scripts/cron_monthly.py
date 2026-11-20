#!/usr/bin/env python3
"""Cron wrapper: Monthly backtest (1st of month 06:00)."""

from liga_mx.monitoring import MonthlyBacktest
from liga_mx.config import load_config

if __name__ == "__main__":
    load_config()
    print("=== CRON: Monthly Backtest ===")
    backtest = MonthlyBacktest()
    backtest.run()