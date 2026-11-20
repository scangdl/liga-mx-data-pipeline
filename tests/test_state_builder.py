"""Tests for StateBuilder."""

from __future__ import annotations

import pandas as pd
import pytest

from liga_mx.features.state_builder import StateBuilder


@pytest.fixture
def sample_master():
    """Create a small sample master DataFrame for testing."""
    data = [
        {"date": "2024-07-13", "home": "America", "away": "Tigres", "hs": 2, "as_": 1, "res_1x2": "H", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-14", "home": "Chivas", "away": "Cruz Azul", "hs": 1, "as_": 1, "res_1x2": "D", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-20", "home": "Tigres", "away": "America", "hs": 0, "as_": 2, "res_1x2": "A", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-21", "home": "Cruz Azul", "away": "Chivas", "hs": 3, "as_": 0, "res_1x2": "H", "torneo": "Apertura", "anio_torneo": 2024},
        {"date": "2024-07-27", "home": "America", "away": "Chivas", "hs": None, "as_": None, "res_1x2": None, "torneo": "Apertura", "anio_torneo": 2024},
    ]
    return pd.DataFrame(data)


def test_state_builder_initialization(sample_master):
    """Test StateBuilder initializes correctly."""
    sb = StateBuilder(sample_master)
    assert len(sb.master) == 5
    assert sb.elo == {}
    assert sb.last_match == {}


def test_elo_updates(sample_master):
    """Test Elo ratings update correctly after matches."""
    sb = StateBuilder(sample_master)

    # Process first match
    row = sample_master.iloc[0]
    feats = sb.process_match(row)

    # Check initial Elo
    assert feats["elo_h"] == 1500.0
    assert feats["elo_a"] == 1500.0

    # Check Elo updated after match
    assert sb.elo["America"] > 1500.0  # Won
    assert sb.elo["Tigres"] < 1500.0   # Lost


def test_form_features_accumulate(sample_master):
    """Test form features accumulate over matches."""
    sb = StateBuilder(sample_master)

    # Process first two matches
    sb.process_match(sample_master.iloc[0])  # America vs Tigres
    sb.process_match(sample_master.iloc[1])  # Chivas vs Cruz Azul
    feats = sb.process_match(sample_master.iloc[2])  # Tigres vs America (rematch)

    # Tigres should have 1 match history now
    assert "Tigres" in sb.hist
    assert len(sb.hist["Tigres"]) == 1


def test_rest_days_calculation(sample_master):
    """Test rest days calculated correctly."""
    sb = StateBuilder(sample_master)

    # First match - no previous match
    feats1 = sb.process_match(sample_master.iloc[0])
    assert pd.isna(feats1["rest_h"])
    assert pd.isna(feats1["rest_a"])

    # Second match (different teams) - no rest
    feats2 = sb.process_match(sample_master.iloc[1])
    assert pd.isna(feats2["rest_h"])
    assert pd.isna(feats2["rest_a"])

    # Third match: Tigres vs America, 6 days after first match
    feats3 = sb.process_match(sample_master.iloc[2])
    assert feats3["rest_h"] == 6  # Tigres played 2024-07-13, now 2024-07-20
    assert feats3["rest_a"] == 6  # America played 2024-07-13, now 2024-07-20


def test_tournament_table(sample_master):
    """Test tournament table updates correctly."""
    sb = StateBuilder(sample_master)

    sb.process_match(sample_master.iloc[0])  # America 2-1 Tigres
    sb.process_match(sample_master.iloc[1])  # Chivas 1-1 Cruz Azul

    # Check table for Apertura 2024
    tor_id = (2024, "Apertura")
    assert tor_id in sb.tabla_torneo

    # America: 3 pts, 2 GF, 1 GA
    america_stats = sb.tabla_torneo[tor_id]["America"]
    assert america_stats[0] == 3  # pts
    assert america_stats[1] == 2  # gf
    assert america_stats[2] == 1  # ga

    # Tigres: 0 pts, 1 GF, 2 GA
    tigres_stats = sb.tabla_torneo[tor_id]["Tigres"]
    assert tigres_stats[0] == 0
    assert tigres_stats[1] == 1
    assert tigres_stats[2] == 2

    # Chivas: 1 pt, 1 GF, 1 GA
    chivas_stats = sb.tabla_torneo[tor_id]["Chivas"]
    assert chivas_stats[0] == 1


def test_h2h_tracking(sample_master):
    """Test head-to-head history tracking."""
    sb = StateBuilder(sample_master)

    sb.process_match(sample_master.iloc[0])  # America vs Tigres
    sb.process_match(sample_master.iloc[2])  # Tigres vs America (return fixture)

    key = tuple(sorted(["America", "Tigres"]))
    assert key in sb.h2h
    assert len(sb.h2h[key]) == 2


def test_state_persistence_roundtrip(sample_master):
    """Test saving and loading state for inference."""
    sb = StateBuilder(sample_master)
    sb.build_all()

    state = sb.get_current_state()
    assert "elo" in state
    assert "last_match" in state
    assert "hist" in state

    # Reconstruct from state
    sb2 = StateBuilder.from_state(sample_master, state)
    assert sb2.elo == state["elo"]
    assert sb2.last_match == state["last_match"]


def test_features_output_structure(sample_master):
    """Test features DataFrame has expected columns."""
    sb = StateBuilder(sample_master)
    feats_df = sb.build_all()

    expected_cols = [
        "elo_h", "elo_a", "elo_diff",
        "rest_h", "rest_a", "rest_diff",
        "gf10_h", "ga10_h", "btts_rate10_h", "ppg10_h",
        "gf10_a", "ga10_a", "btts_rate10_a", "ppg10_a",
        "gf_home_as_home", "ga_home_as_home",
        "gf_away_as_away", "ga_away_as_away",
        "imp_h", "imp_d", "imp_a",
        "pos_table_h", "pos_table_a",
        "ppg_table_h", "ppg_table_a",
        "h2h_btts_rate", "h2h_gf", "h2h_ga", "h2h_n",
        "sos_h", "sos_a",
        "cong_h", "cong_a",
        "lam_h", "lam_a",
    ]

    for col in expected_cols:
        assert col in feats_df.columns, f"Missing column: {col}"

    assert len(feats_df) == len(sample_master)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])