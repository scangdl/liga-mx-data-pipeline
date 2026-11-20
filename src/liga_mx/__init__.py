"""Liga MX Pipeline — Data, Features, Models, Trading, Monitoring."""

__version__ = "0.3.0"
__author__ = "Armando Sánchez Cendejas"

from .config import get_config, load_config
from .data import team_names, sources, master
from .features import StateBuilder, build_features_master, add_advanced_features
from .models import BTTSModel, Over25Model, OneXTwoModel
from .trading import PaperTrader, KellyStakeCalculator
from .monitoring import DecayDetector, MonthlyBacktest

__all__ = [
    "get_config",
    "load_config",
    "team_names",
    "sources",
    "master",
    "StateBuilder",
    "build_features_master",
    "add_advanced_features",
    "BTTSModel",
    "Over25Model",
    "OneXTwoModel",
    "PaperTrader",
    "KellyStakeCalculator",
    "DecayDetector",
    "MonthlyBacktest",
]