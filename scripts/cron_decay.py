#!/usr/bin/env python3
"""Cron wrapper: Weekly decay detection."""

from liga_mx.monitoring import DecayDetector
from liga_mx.config import load_config

if __name__ == "__main__":
    load_config()
    print("=== CRON: Decay Detection ===")
    detector = DecayDetector()
    detector.run()