#!/usr/bin/env python3
"""Setup script for Liga MX Pipeline."""

from setuptools import setup

if __name__ == "__main__":
    setup(
        entry_points={
            "console_scripts": [
                "liga_mx_update = liga_mx.cli.commands:update_cmd",
                "liga_mx_predict = liga_mx.cli.commands:predict_cmd",
                "liga_mx_train = liga_mx.cli.commands:train_cmd",
                "liga_mx_trade = liga_mx.cli.commands:trade_cmd",
                "liga_mx_decay = liga_mx.cli.commands:decay_cmd",
                "liga_mx_backtest = liga_mx.cli.commands:backtest_cmd",
                "liga_mx_fetch = liga_mx.cli.commands:fetch_cmd",
            ]
        }
    )
