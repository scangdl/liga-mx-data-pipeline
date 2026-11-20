"""Configuration loader for Liga MX Pipeline.

Loads config.yaml and merges with .env environment variables.
Provides typed access to all configuration sections.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ------------------------------------------------------------------
# Pydantic models for type-safe config access
# ------------------------------------------------------------------
class PathsConfig(BaseModel):
    project_root: Path
    data_dir: Path
    models_dir: Path
    predictions_dir: Path
    scripts_dir: Path
    cache_dir: Path
    logs_dir: Path

    # Input files
    kaggle_raw: str
    football_data_odds: str
    espn_schedules: str
    espn_fixtures: str
    odds_api_snapshot: str
    odds_api_jornada: str
    odds_history: str

    # Output files
    master_parquet: str
    master_csv: str
    features_parquet: str

    # Model files
    btts_model: str
    btts_model_v3: str
    over25_model: str
    model_1x2: str

    # Prediction files
    btts_jornada: str
    paper_trades_over25: str
    paper_trades_1x2: str
    paper_state_over25: str
    paper_state_1x2: str
    monthly_backtest_history: str
    decay_snapshots: str

    def resolve(self, name: str) -> Path:
        """Resolve a file path relative to the appropriate directory."""
        attr = getattr(self, name)
        if name in (
            "kaggle_raw", "football_data_odds", "espn_schedules", "espn_fixtures",
            "odds_api_snapshot", "odds_api_jornada", "odds_history",
            "master_parquet", "master_csv", "features_parquet",
        ):
            return self.data_dir / attr
        if name in ("btts_model", "btts_model_v3", "over25_model", "model_1x2"):
            return self.models_dir / attr
        if name in (
            "btts_jornada", "paper_trades_over25", "paper_trades_1x2",
            "paper_state_over25", "paper_state_1x2",
            "monthly_backtest_history", "decay_snapshots",
        ):
            return self.predictions_dir / attr
        if name in ("data_dir", "models_dir", "predictions_dir", "scripts_dir", "cache_dir", "logs_dir"):
            return attr  # Already resolved to full Path in load_config
        raise ValueError(f"Unknown path key: {name}")


class FootballDataConfig(BaseModel):
    base_url: str
    current_season_path: str
    encoding: str
    user_agent: str


class ESPNConfig(BaseModel):
    base_url: str
    lookahead_days: int
    user_agent: str
    timeout_seconds: int


class OddsAPIConfig(BaseModel):
    base_url: str
    sport: str
    regions: str
    markets: str
    odds_format: str
    timeout_seconds: int


class SportmonksConfig(BaseModel):
    base_url: str


class DataSourcesConfig(BaseModel):
    football_data: FootballDataConfig
    espn: ESPNConfig
    odds_api: OddsAPIConfig
    sportmonks: SportmonksConfig


class FeaturesConfig(BaseModel):
    elo_k: float
    elo_home_advantage: float
    ewma_alpha: float
    lgf_window: int
    form_window: int
    h2h_window: int
    sos_window: int
    congestion_window_days: int
    congestion_lookback: int
    btts_threshold: int
    over25_line: float
    base_features: list[str]
    extended_features: list[str]
    advanced_features: list[str]


class ModelConfig(BaseModel):
    train_cut_date: str

    btts: dict
    over25: dict
    one_x_two: dict


class TradingConfig(BaseModel):
    initial_bankroll: float
    kelly_fraction: float
    edge_min: float
    stake_cap: float
    max_open_bets: int
    markets: dict


class MonitoringConfig(BaseModel):
    decay_detector: dict
    monthly_backtest: dict


class LoggingConfig(BaseModel):
    level: str
    format: str
    file: str
    max_bytes: int
    backup_count: int


class TelegramConfig(BaseModel):
    bot_token: str = ""
    chat_id: str = ""


class AppConfig(BaseModel):
    paths: PathsConfig
    data_sources: DataSourcesConfig
    features: FeaturesConfig
    models: ModelConfig
    trading: TradingConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig
    telegram: TelegramConfig = TelegramConfig()


# ------------------------------------------------------------------
# Config singleton
# ------------------------------------------------------------------
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from config.yaml and .env."""
    global _config

    if _config is not None:
        return _config

    # Load .env first (so env vars can override config.yaml)
    load_dotenv()

    # Determine config file path
    if config_path is None:
        # Try project root / config.yaml, then cwd / config.yaml
        for base in [Path.cwd(), Path(__file__).parents[3]]:
            candidate = base / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        
        # Also check the source directory explicitly
        if config_path is None:
            source_base = Path("/home/scan/liga-mx-data-pipeline")
            candidate = source_base / "config.yaml"
            if candidate.exists():
                config_path = candidate

    if config_path is None:
        raise FileNotFoundError("config.yaml not found. Copy config.yaml.example or set path explicitly.")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Resolve paths relative to project_root
    project_root = Path(raw["paths"]["project_root"])
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root

    # Update paths with project_root
    paths_dict = raw["paths"]
    paths_dict["project_root"] = project_root
    paths_dict["data_dir"] = project_root / paths_dict["data_dir"]
    paths_dict["models_dir"] = project_root / paths_dict["models_dir"]
    paths_dict["predictions_dir"] = project_root / paths_dict["predictions_dir"]
    paths_dict["scripts_dir"] = project_root / paths_dict["scripts_dir"]
    paths_dict["cache_dir"] = project_root / paths_dict["cache_dir"]
    paths_dict["logs_dir"] = project_root / paths_dict["logs_dir"]

    # Ensure directories exist
    for key in ["data_dir", "models_dir", "predictions_dir", "scripts_dir", "cache_dir", "logs_dir"]:
        paths_dict[key].mkdir(parents=True, exist_ok=True)

    _config = AppConfig(**raw)
    return _config


def get_config() -> AppConfig:
    """Get cached config, loading if necessary."""
    if _config is None:
        return load_config()
    return _config


def get_path(name: str) -> Path:
    """Convenience: resolve a configured path by name."""
    return get_config().paths.resolve(name)


def get_env(key: str, default: Any = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.getenv(key, default)