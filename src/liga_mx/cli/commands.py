"""CLI Commands for Liga MX Pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from liga_mx.config import get_config, load_config
from liga_mx.data.master import build_master
from liga_mx.data.sources import download_football_data, fetch_espn_fixtures, fetch_odds_api
from liga_mx.features import build_features_master, build_state_for_inference
from liga_mx.models import BTTSModel, Over25Model, OneXTwoModel
from liga_mx.trading import PaperTrader
from liga_mx.monitoring import DecayDetector, MonthlyBacktest

app = typer.Typer(help="Liga MX Pipeline — Data, Models, Trading, Monitoring")
console = Console()


def _setup_config(config_file: Path | None = None):
    """Load config, optionally from custom path."""
    if config_file:
        load_config(config_file)
    else:
        load_config()


@app.command()
def update(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
    force: bool = typer.Option(False, "--force", "-f", help="Force rebuild master"),
):
    """Update master data and regenerate features + models."""
    _setup_config(config)
    console.print("[bold blue]=== Liga MX Pipeline Update ===[/bold blue]")

    # 1. Download fresh football-data
    console.print("\n[1/5] Downloading football-data.co.uk...")
    download_football_data()

    # 2. Fetch ESPN fixtures
    console.print("\n[2/5] Fetching ESPN fixtures...")
    fetch_espn_fixtures()

    # 3. Fetch Odds API
    console.print("\n[3/5] Fetching Odds API...")
    try:
        fetch_odds_api()
    except Exception as e:
        console.print(f"[yellow]Odds API fetch failed (continuing): {e}[/yellow]")

    # 4. Rebuild master
    console.print("\n[4/5] Building master dataset...")
    build_master(force_rebuild=force)

    # 5. Regenerate features
    console.print("\n[5/5] Regenerating features...")
    features_df = build_features_master(get_config().paths.data_dir / "liga_mx_master.parquet")
    features_path = get_config().paths.data_dir / "liga_mx_features.parquet"
    features_df.to_parquet(features_path, index=False)
    console.print(f"\n[green]✓ Update complete! Features: {features_path}[/green]")


@app.command()
def predict(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
    market: str = typer.Option("btts", "--market", "-m", help="Market: btts, over25, 1x2"),
    output: Path = typer.Option(None, "--output", "-o", help="Output CSV path"),
):
    """Generate predictions for upcoming fixtures."""
    _setup_config(config)
    console.print(f"[bold blue]=== Predicting {market.upper()} ===[/bold blue]")

    if market == "btts":
        model = BTTSModel.load(get_config().paths.models_dir / "btts_model_v3.pkl")
    elif market == "over25":
        model = Over25Model.load(get_config().paths.models_dir / "over25_model.pkl")
    elif market == "1x2":
        model = OneXTwoModel.load(get_config().paths.models_dir / "model_1x2.pkl")
    else:
        console.print(f"[red]Unknown market: {market}[/red]")
        raise typer.Exit(1)

    # Build inference state
    state = build_state_for_inference(get_config().paths.data_dir / "liga_mx_master.parquet")

    # Load fixtures
    from liga_mx.data.sources import fetch_espn_fixtures
    fixtures = fetch_espn_fixtures()

    # TODO: Implement prediction logic per market
    console.print("[yellow]Prediction CLI not fully implemented yet[/yellow]")


@app.command()
def train(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
    model: str = typer.Option("btts", "--model", "-m", help="Model to train: btts, over25, 1x2, all"),
    features_path: Path = typer.Option(None, "--features", "-f", help="Features parquet path"),
):
    """Train models."""
    _setup_config(config)
    console.print(f"[bold blue]=== Training {model.upper()} ===[/bold blue]")

    if features_path is None:
        features_path = get_config().paths.data_dir / "liga_mx_features.parquet"

    if model in ("btts", "all"):
        console.print("\n[bold]Training BTTS model...[/bold]")
        btts = BTTSModel()
        btts.train(features_path)
        btts.save()

    if model in ("over25", "all"):
        console.print("\n[bold]Training Over 2.5 model...[/bold]")
        over25 = Over25Model()
        over25.train(features_path)
        over25.save()

    if model in ("1x2", "all"):
        console.print("\n[bold]Training 1X2 model...[/bold]")
        one_x_two = OneXTwoModel()
        one_x_two.train(features_path)
        one_x_two.save()

    console.print("\n[green]✓ Training complete![/green]")


@app.command()
def trade(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
    action: str = typer.Argument(..., help="Action: place, settle, place_all, settle_all"),
    market: str = typer.Option("over25", "--market", "-m", help="Market: over25, 1x2, all"),
):
    """Paper trading: place or settle bets."""
    _setup_config(config)
    console.print(f"[bold blue]=== Paper Trading: {action} {market} ===[/bold blue]")

    trader = PaperTrader()

    if action == "place":
        if market == "all":
            trader.place_all()
        else:
            trader.place(market)
    elif action == "settle":
        if market == "all":
            trader.settle_all()
        else:
            trader.settle(market)
    elif action == "place_all":
        trader.place_all()
    elif action == "settle_all":
        trader.settle_all()
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def decay(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
):
    """Run weekly decay detection."""
    _setup_config(config)
    console.print("[bold blue]=== Decay Detection ===[/bold blue]")

    detector = DecayDetector()
    result = detector.run()

    if result.get("alerts"):
        console.print(f"\n[red]⚠️ {len(result['alerts'])} alert(s) triggered![/red]")
        for alert in result["alerts"]:
            console.print(f"  {alert['msg']}")
    else:
        console.print("\n[green]✅ No alerts - model stable[/green]")


@app.command()
def backtest(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
):
    """Run monthly backtest."""
    _setup_config(config)
    console.print("[bold blue]=== Monthly Backtest ===[/bold blue]")

    backtest = MonthlyBacktest()
    result = backtest.run()

    if result.get("metrics"):
        m = result["metrics"]
        console.print(f"\n[bold]Period:[/bold] {m['period_start']} to {m['period_end']}")
        console.print(f"[bold]Matches:[/bold] {m['n_matches']}")
        console.print(f"[bold]BTTS Rate:[/bold] {m['btts_rate']:.1%}")
        console.print(f"[bold]Brier:[/bold] {m['brier']:.4f} (base: {m['base_brier']:.4f}, improvement: {m['improvement_pct']:+.1f}%)")
        console.print(f"[bold]AUC:[/bold] {m['auc']:.4f}")
        console.print(f"[bold]Acc@0.5:[/bold] {m['acc_05']:.1%}")


@app.command()
def fetch(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
    source: str = typer.Argument(..., help="Source: football-data, espn, odds-api, all"),
):
    """Fetch data from external sources."""
    _setup_config(config)
    console.print(f"[bold blue]=== Fetching {source} ===[/bold blue]")

    if source in ("football-data", "all"):
        console.print("\n[bold]Downloading football-data.co.uk...[/bold]")
        download_football_data()

    if source in ("espn", "all"):
        console.print("\n[bold]Fetching ESPN fixtures...[/bold]")
        fetch_espn_fixtures()

    if source in ("odds-api", "all"):
        console.print("\n[bold]Fetching Odds API...[/bold]")
        try:
            fetch_odds_api()
        except Exception as e:
            console.print(f"[red]Odds API failed: {e}[/red]")
            raise typer.Exit(1)

    console.print("\n[green]✓ Fetch complete![/green]")


@app.command()
def init(
    config: Path = typer.Option(None, "--config", "-c", help="Custom config.yaml path"),
):
    """Initialize project: create directories, download initial data."""
    _setup_config(config)
    console.print("[bold blue]=== Initializing Liga MX Pipeline ===[/bold blue]")

    cfg = get_config()

    # Create directories
    for key in ["data_dir", "models_dir", "predictions_dir", "scripts_dir", "cache_dir", "logs_dir"]:
        path = getattr(cfg.paths, key)
        path.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {path}")

    # Download initial football-data
    console.print("\n[bold]Downloading initial football-data...[/bold]")
    download_football_data()

    console.print("\n[green]✓ Initialization complete![/green]")
    console.print("\nNext steps:")
    console.print("  1. Copy .env.example to .env and add your API keys")
    console.print("  2. Run: liga-mx-update")
    console.print("  3. Run: liga-mx-train --model all")
    console.print("  4. Run: liga-mx-trade place_all")


if __name__ == "__main__":
    app()