#!/usr/bin/env python3
"""debug_fixture_feats.py — compara los features de un fixture contra su rango historico."""
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/petyt/futbolmx")
import paper_trade as pt  # noqa: E402  (reusa build_state/fixture_row)

DATA = pt.DATA

b = pickle.load(open(DATA / "over25_model.pkl", "rb"))
FEATS = b["features"]
means = pd.Series(b["means"])

hist = pd.read_parquet(DATA / "liga_mx_features.parquet")
home, away = "Atlante", "Monterrey"
d = pd.Timestamp("2026-09-25")

st = pt.build_state()
f = pt.fixture_row(st, home, away, d)
f["elo_diff"] = f["elo_h"] - f["elo_a"]
f["rest_diff"] = f["rest_h"] - f["rest_a"]

print(f"{'feature':<20} {'fixture':>10} {'hist_p50':>10} {'hist_p05':>10} {'hist_p95':>10} {'z':>7}")
for k in FEATS:
    v = f.get(k, np.nan)
    hv = hist[k].dropna()
    p50, p05, p95 = hv.median(), hv.quantile(0.05), hv.quantile(0.95)
    sd = hv.std()
    z = (v - hv.mean()) / sd if sd and pd.notna(v) else np.nan
    flag = "  <-- OUT" if (pd.notna(v) and (v < p05 - 3 * sd or v > p95 + 3 * sd)) else ""
    print(f"{k:<20} {v:>10.3f} {p50:>10.3f} {p05:>10.3f} {p95:>10.3f} {z:>7.2f}{flag}")

Xf = pd.DataFrame([f])[FEATS].fillna(means)
p = b["model"].predict_proba(Xf)[:, 1][0]
print(f"\np_model Over2.5 = {p:.4f}")
