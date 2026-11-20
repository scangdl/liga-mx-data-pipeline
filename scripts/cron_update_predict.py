#!/usr/bin/env python3
"""Cron wrapper: Update master, regenerate features, retrain models, predict jornada."""

from liga_mx.cli.commands import update, train, predict
from liga_mx.config import load_config
import typer

if __name__ == "__main__":
    load_config()
    print("=== CRON: Update & Predict ===")
    typer.run(update)
    typer.run(train)
    typer.run(predict)