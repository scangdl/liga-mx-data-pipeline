"""Advanced feature engineering for Liga MX models.

Adds engineered features on top of base features from StateBuilder.
All features are computed without leakage (only pre-match info).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to DataFrame with base features. No leakage."""
    df = df.copy()

    # Elo interactions
    df["elo_diff_sq"] = df["elo_diff"] ** 2
    df["elo_sum"] = df["elo_h"] + df["elo_a"]
    df["elo_ratio"] = df["elo_h"] / (df["elo_a"] + 1e-6)

    # Rest interactions
    df["rest_min"] = df[["rest_h", "rest_a"]].min(axis=1)
    df["rest_max"] = df[["rest_h", "rest_a"]].max(axis=1)
    df["rest_short_h"] = (df["rest_h"] <= 3).astype(float)
    df["rest_short_a"] = (df["rest_a"] <= 3).astype(float)
    df["both_short_rest"] = (df["rest_short_h"] * df["rest_short_a"]).astype(float)

    # Form differentials
    df["gf10_diff"] = df["gf10_h"] - df["gf10_a"]
    df["ga10_diff"] = df["ga10_h"] - df["ga10_a"]
    df["btts_rate10_diff"] = df["btts_rate10_h"] - df["btts_rate10_a"]
    df["ppg10_diff"] = df["ppg10_h"] - df["ppg10_a"]

    # Home/away attack/defense
    df["att_h"] = df["gf10_h"]
    df["def_h"] = df["ga10_h"]
    df["att_a"] = df["gf10_a"]
    df["def_a"] = df["ga10_a"]
    df["att_sum"] = df["att_h"] + df["att_a"]
    df["def_sum"] = df["def_h"] + df["def_a"]
    df["att_vs_def_h"] = df["att_h"] / (df["def_a"] + 1e-6)
    df["att_vs_def_a"] = df["att_a"] / (df["def_h"] + 1e-6)

    # Home advantage specific
    df["home_form"] = df["gf_home_as_home"] - df["ga_home_as_home"]
    df["away_form"] = df["gf_away_as_away"] - df["ga_away_as_away"]
    df["home_form_diff"] = df["home_form"] - df["away_form"]

    # Lambda interactions
    df["lam_sum"] = df["lam_h"] + df["lam_a"]
    df["lam_ratio"] = df["lam_h"] / (df["lam_a"] + 1e-6)
    df["lam_diff"] = df["lam_h"] - df["lam_a"]
    df["poisson_btts"] = (1 - poisson.pmf(0, df["lam_h"].fillna(0))) * (
        1 - poisson.pmf(0, df["lam_a"].fillna(0))
    )

    # Odds / market features
    df["imp_draw"] = df["imp_d"]
    df["imp_h_minus_a"] = df["imp_h"] - df["imp_a"]
    df["imp_entropy"] = -(
        df["imp_h"] * np.log(df["imp_h"] + 1e-10)
        + df["imp_d"] * np.log(df["imp_d"] + 1e-10)
        + df["imp_a"] * np.log(df["imp_a"] + 1e-10)
    )

    # Table position interactions
    df["pos_diff"] = df["pos_table_h"] - df["pos_table_a"]
    df["ppg_diff"] = df["ppg_table_h"] - df["ppg_table_a"]
    df["pos_product"] = df["pos_table_h"] * df["pos_table_a"]

    # H2H features
    df["h2h_gf_sum"] = df["h2h_gf"] + df["h2h_ga"]
    df["h2h_gf_diff"] = df["h2h_gf"] - df["h2h_ga"]
    df["has_h2h"] = df["h2h_n"].fillna(0).gt(0).astype(float)

    # SOS / congestion interactions
    df["sos_diff"] = df["sos_h"] - df["sos_a"]
    df["cong_diff"] = df["cong_h"] - df["cong_a"]
    df["cong_sum"] = df["cong_h"] + df["cong_a"]
    df["high_cong"] = ((df["cong_h"] >= 2) | (df["cong_a"] >= 2)).astype(float)

    # Combined attack/defense strength
    df["total_attack"] = df["gf10_h"] + df["gf10_a"]
    df["total_defense"] = df["ga10_h"] + df["ga10_a"]
    df["attack_def_balance"] = df["total_attack"] / (df["total_defense"] + 1e-6)

    return df


__all__ = ["add_advanced_features"]