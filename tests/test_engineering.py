"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from liga_mx.features.engineering import add_advanced_features


@pytest.fixture
def sample_features():
    """Create sample DataFrame with base features."""
    return pd.DataFrame({
        "elo_h": [1600, 1550],
        "elo_a": [1500, 1450],
        "elo_diff": [100, 100],
        "rest_h": [7, 3],
        "rest_a": [7, 10],
        "gf10_h": [1.8, 1.2],
        "ga10_h": [1.1, 1.5],
        "btts_rate10_h": [0.6, 0.4],
        "ppg10_h": [2.1, 1.3],
        "gf10_a": [1.2, 1.8],
        "ga10_a": [1.5, 1.1],
        "btts_rate10_a": [0.4, 0.6],
        "ppg10_a": [1.3, 2.1],
        "gf_home_as_home": [2.0, 1.0],
        "ga_home_as_home": [0.8, 1.2],
        "gf_away_as_away": [1.0, 2.2],
        "ga_away_as_away": [1.2, 0.9],
        "imp_h": [0.45, 0.55],
        "imp_d": [0.30, 0.25],
        "imp_a": [0.25, 0.20],
        "pos_table_h": [2, 5],
        "pos_table_a": [10, 3],
        "ppg_table_h": [2.2, 1.5],
        "ppg_table_a": [1.1, 1.8],
        "h2h_btts_rate": [0.5, 0.3],
        "h2h_gf": [1.5, 1.0],
        "h2h_ga": [1.0, 1.5],
        "h2h_n": [4, 2],
        "sos_h": [1520, 1480],
        "sos_a": [1480, 1520],
        "cong_h": [1, 2],
        "cong_a": [0, 1],
        "lam_h": [1.6, 1.2],
        "lam_a": [1.1, 1.5],
    })


def test_add_advanced_features_creates_expected_columns(sample_features):
    """Test that advanced features adds all expected columns."""
    result = add_advanced_features(sample_features)

    expected_new = [
        "elo_diff_sq", "elo_sum", "elo_ratio",
        "rest_min", "rest_max", "rest_short_h", "rest_short_a", "both_short_rest",
        "gf10_diff", "ga10_diff", "btts_rate10_diff", "ppg10_diff",
        "att_h", "def_h", "att_a", "def_a",
        "att_sum", "def_sum", "att_vs_def_h", "att_vs_def_a",
        "home_form", "away_form", "home_form_diff",
        "lam_sum", "lam_ratio", "lam_diff", "poisson_btts",
        "imp_draw", "imp_h_minus_a", "imp_entropy",
        "pos_diff", "ppg_diff", "pos_product",
        "h2h_gf_sum", "h2h_gf_diff", "has_h2h",
        "sos_diff", "cong_diff", "cong_sum", "high_cong",
        "total_attack", "total_defense", "attack_def_balance",
    ]

    for col in expected_new:
        assert col in result.columns, f"Missing column: {col}"


def test_elo_interactions(sample_features):
    """Test Elo interaction features."""
    result = add_advanced_features(sample_features)

    # elo_diff_sq = elo_diff^2
    assert (result["elo_diff_sq"] == result["elo_diff"] ** 2).all()

    # elo_sum = elo_h + elo_a
    assert (result["elo_sum"] == result["elo_h"] + result["elo_a"]).all()

    # elo_ratio = elo_h / elo_a
    assert (result["elo_ratio"] == result["elo_h"] / (result["elo_a"] + 1e-6)).all()


def test_rest_interactions(sample_features):
    """Test rest interaction features."""
    result = add_advanced_features(sample_features)

    # rest_min = min(rest_h, rest_a)
    assert (result["rest_min"] == result[["rest_h", "rest_a"]].min(axis=1)).all()

    # rest_short_h = (rest_h <= 3)
    assert (result["rest_short_h"] == (result["rest_h"] <= 3).astype(float)).all()

    # both_short_rest = rest_short_h * rest_short_a
    assert (result["both_short_rest"] == result["rest_short_h"] * result["rest_short_a"]).all()


def test_form_differentials(sample_features):
    """Test form differential features."""
    result = add_advanced_features(sample_features)

    assert (result["gf10_diff"] == result["gf10_h"] - result["gf10_a"]).all()
    assert (result["ga10_diff"] == result["ga10_h"] - result["ga10_a"]).all()
    assert (result["btts_rate10_diff"] == result["btts_rate10_h"] - result["btts_rate10_a"]).all()
    assert (result["ppg10_diff"] == result["ppg10_h"] - result["ppg10_a"]).all()


def test_poisson_btts(sample_features):
    """Test Poisson BTTS feature."""
    from scipy.stats import poisson
    result = add_advanced_features(sample_features)

    expected = (1 - poisson.pmf(0, result["lam_h"].fillna(0))) * (1 - poisson.pmf(0, result["lam_a"].fillna(0)))
    assert (result["poisson_btts"] - expected).abs().max() < 1e-10


def test_imp_entropy(sample_features):
    """Test implied odds entropy calculation."""
    result = add_advanced_features(sample_features)

    # entropy = -sum(p * log(p))
    expected = -(
        result["imp_h"] * result["imp_h"].apply(lambda x: np.log(x + 1e-10)) +
        result["imp_d"] * result["imp_d"].apply(lambda x: np.log(x + 1e-10)) +
        result["imp_a"] * result["imp_a"].apply(lambda x: np.log(x + 1e-10))
    )
    # Use numpy for vectorized
    import numpy as np
    expected = -(
        result["imp_h"] * np.log(result["imp_h"] + 1e-10) +
        result["imp_d"] * np.log(result["imp_d"] + 1e-10) +
        result["imp_a"] * np.log(result["imp_a"] + 1e-10)
    )
    assert (result["imp_entropy"] - expected).abs().max() < 1e-10


def test_preserves_original_rows(sample_features):
    """Test that advanced features preserves all original rows."""
    result = add_advanced_features(sample_features)
    assert len(result) == len(sample_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])