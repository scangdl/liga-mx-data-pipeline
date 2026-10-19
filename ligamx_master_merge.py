#!/usr/bin/env python3
"""
Pipeline completo consolidado - Liga MX Historical Data con BTTS
===============================================================

FUENTES:
1. football-data.co.uk (gratis, 1X2 + O/U 2.5, SIN BTTS) - 2012-2026
2. Kaggle/HuggingFace mirror (mismo dato, + Elo ratings) - 2012-2024
3. FootyStats (gratis, CON BTTS odds, descarga manual) - 2012-2027
4. Sportmonks API (trial 14 días, CON BTTS probs + odds reales) - actual
5. TheStatsAPI (trial 7 días, $50/mes) - alternativa API

ARCHIVOS GENERADOS EN ~/ligamx_data/:
- kaggle_ligamx_raw.csv          # 4080 partidos, 2012-2024, sin BTTS
- footystats_ligamx_combined.csv # (tras descarga manual) CON BTTS
- sportmonks_ligamx_btts.csv     # (tras API) BTTS probs + odds
- ligamx_master.csv              # Merge final con todas las fuentes
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "ligamx_data"


def inspect_kaggle():
    """Inspecciona datos Kaggle ya descargados"""
    df = pd.read_csv(DATA_DIR / "kaggle_ligamx_raw.csv")
    print("=" * 60)
    print("KAGGLE / HUGGINGFACE DATA")
    print("=" * 60)
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Temporadas: {df['MatchDate'].str[:4].nunique()} ({df['MatchDate'].str[:4].min()}-{df['MatchDate'].str[:4].max()})")
    print(f"Equipos únicos: {pd.concat([df['HomeTeam'], df['AwayTeam']]).nunique()}")
    print("\nColumnas clave:")
    for c in ['Division','MatchDate','HomeTeam','AwayTeam','FTHome','FTAway','FTResult',
              'OddHome','OddDraw','OddAway','Over25','Under25','MaxHome','MaxDraw','MaxAway']:
        if c in df.columns:
            non_null = df[c].notna().sum()
            print(f"  {c}: {non_null}/{len(df)} ({non_null/len(df)*100:.0f}%)")
    print("\n⚠️  NO tiene columnas BTTS")


def merge_all_sources():
    """Une todas las fuentes disponibles en un master"""
    print("\n" + "=" * 60)
    print("CREANDO MASTER MERGE")
    print("=" * 60)
    
    # Base: Kaggle (más completo histórico)
    master = pd.read_csv(DATA_DIR / "kaggle_ligamx_raw.csv")
    master['match_key'] = master['MatchDate'] + '_' + master['HomeTeam'] + '_' + master['AwayTeam']
    print(f"Base Kaggle: {len(master)} filas")
    
    # Intentar mergear FootyStats si existe
    fs_file = DATA_DIR / "footystats_ligamx_combined.csv"
    if fs_file.exists():
        fs = pd.read_csv(fs_file)
        print(f"FootyStats: {len(fs)} filas, columnas: {len(fs.columns)}")
        # Crear key similar
        if 'Date' in fs.columns and 'Home' in fs.columns and 'Away' in fs.columns:
            fs['match_key'] = fs['Date'] + '_' + fs['Home'] + '_' + fs['Away']
            # Merge en BTTS columns
            btts_cols = [c for c in fs.columns if 'btts' in c.lower() or 'both' in c.lower()]
            if btts_cols:
                merge_cols = ['match_key'] + btts_cols
                master = master.merge(fs[merge_cols], on='match_key', how='left', suffixes=('', '_fs'))
                print(f"  ✅ Mergeadas columnas BTTS: {btts_cols}")
            else:
                print("  ⚠️  No columnas BTTS en FootyStats")
    
    # Intentar mergear Sportmonks si existe
    sm_file = DATA_DIR / "sportmonks_ligamx_btts.csv"
    if sm_file.exists():
        sm = pd.read_csv(sm_file)
        print(f"Sportmonks: {len(sm)} filas")
        # Merge por fixture_id o key similar
    
    # Guardar master
    output = DATA_DIR / "ligamx_master.csv"
    master.to_csv(output, index=False)
    print(f"\n✅ Master guardado: {output} ({len(master)} filas, {len(master.columns)} cols)")
    
    # Resumen BTTS
    btts_final = [c for c in master.columns if 'btts' in c.lower() or 'both' in c.lower()]
    print(f"Columnas BTTS finales: {btts_final}")
    
    return master


def show_next_steps():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    PRÓXIMOS PASOS                            ║
╠══════════════════════════════════════════════════════════════╣
║ 1. DESCARGA FOOTYSTATS (tiene BTTS odds históricos):         ║
║    python ligamx_btts_pipeline.py --footystats-manual        ║
║    → Abre URLs en navegador, descarga 15 CSVs               ║
║    → python ligamx_btts_pipeline.py --footystats-combine    ║
║                                                              ║
║ 2. SPORTMONKS API (trial 14 días, BTTS probs + odds):       ║
║    → Regístrate: https://www.sportmonks.com/                ║
║    → python ligamx_btts_pipeline.py --sportmonks TU_TOKEN   ║
║                                                              ║
║ 3. CREAR MASTER FINAL:                                       ║
║    python ligamx_master_merge.py                             ║
║                                                              ║
║ 4. ANÁLISIS BTTS:                                            ║
║    python -c "import pandas as pd; df=pd.read_csv            ║
║    ('~/ligamx_data/ligamx_master.csv');                      ║
║    print(df['btts_yes_odds'].describe())"                    ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    inspect_kaggle()
    merge_all_sources()
    show_next_steps()