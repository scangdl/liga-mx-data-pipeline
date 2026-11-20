"""Liga MX Pipeline - CLI module."""

from .commands import (
    update,
    predict,
    train,
    trade,
    decay,
    backtest,
    fetch,
)

__all__ = [
    "update",
    "predict",
    "train",
    "trade",
    "decay",
    "backtest",
    "fetch",
]