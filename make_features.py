#!/usr/bin/env python3
"""
make_features.py — Features pre-partido (sin leakage) sobre el master.

Regla de oro: todo feature usa solo información disponible ANTES del partido.
  - Elo: actualizado secuencialmente partido a partido
  - Forma/descanso: ventanas de partidos previos
  - Odds: cuotas de cierre (disponibles pre-partido)

Salida: <data_dir>/liga_mx_features.parquet

USA features_lib.StateBuilder como single source of truth + config.yaml paths.
"""
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent))

from src.liga_mx.config import get_config, get_path
from features_lib import StateBuilder, log

OUT = get_path("features_parquet")
MASTER = get_path("master_parquet")


def main():
    log("Cargando master...")
    df = pd.read_parquet(MASTER).sort_values("date").reset_index(drop=True)
    log(f"Master: {len(df)} partidos")

    # ============================================================
    # 1-5. Features base via StateBuilder (Elo, descanso, forma, odds, etc.)
    # ============================================================
    sb = StateBuilder(df)
    feats_df = sb.build_all()

    # Concatenar features base
    df = pd.concat([df.reset_index(drop=True), feats_df], axis=1)

    # ============================================================
    # Odds -> probabilidades implícitas 1X2 (normalizadas, sin margen)
    # ============================================================
    def pick_odds(row, pref):
        for c in pref:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return np.nan

    pref_h = ["odd_h_psc", "odd_h_avg", "odd_h_kag", "odd_h_max", "odd_h_b365"]
    pref_d = ["odd_d_psc", "odd_d_avg", "odd_d_kag", "odd_d_max", "odd_d_b365"]
    pref_a = ["odd_a_psc", "odd_a_avg", "odd_a_kag", "odd_a_max", "odd_a_b365"]
    oh, od, oa = [], [], []
    for _, r in df.iterrows():
        oh.append(pick_odds(r, pref_h))
        od.append(pick_odds(r, pref_d))
        oa.append(pick_odds(r, pref_a))
    df["odd_h"] = oh
    df["odd_d"] = od
    df["odd_a"] = oa
    inv_sum = (1 / df["odd_h"] + 1 / df["odd_d"] + 1 / df["odd_a"])
    df["imp_h"] = (1 / df["odd_h"]) / inv_sum
    df["imp_d"] = (1 / df["odd_d"]) / inv_sum
    df["imp_a"] = (1 / df["odd_a"]) / inv_sum
    df["has_odds"] = df["odd_h"].notna().astype(int)
    log(f"Odds disponibles: {df['has_odds'].sum()}/{len(df)}")

    # ============================================================
    # GUARDAR
    # ============================================================
    out = OUT
    df.to_parquet(out, index=False)
    log(f"\nGuardado: {out} ({len(df)} filas, {len(df.columns)} cols)")
    log("\nColumnas: " + ", ".join(df.columns))


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    main()