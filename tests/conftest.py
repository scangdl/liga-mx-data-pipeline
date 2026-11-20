"""pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture(scope="session")
def sample_master():
    """Sample master DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame([
        {"date": "2024-07-13", "home": "America", "away": "Tigres", "hs": 2, "as_": 1, "res_1x2": "H", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-14", "home": "Chivas", "away": "Cruz Azul", "hs": 1, "as_": 1, "res_1x2": "D", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-20", "home": "Tigres", "away": "America", "hs": 0, "as_": 2, "res_1x2": "A", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-21", "home": "Cruz Azul", "away": "Chivas", "hs": 3, "as_": 0, "res_1x2": "H", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-27", "home": "America", "away": "Chivas", "hs": None, "as_": None, "res_1x2": None, "torneo": "Apertura", "anio_torneo": 2024},
    ])