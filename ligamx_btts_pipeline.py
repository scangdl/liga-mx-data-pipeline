#!/usr/bin/env python3
"""
Pipeline completo para obtener datos históricos Liga MX con BTTS
Combina: Kaggle/HuggingFace (gratis, sin BTTS) + FootyStats (gratis, con BTTS, descarga manual) + Sportmonks API (trial 14 días, con BTTS)
"""

import os
import pandas as pd
import requests
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATA_DIR = Path("/home/scan/ligamx_data")
DATA_DIR.mkdir(exist_ok=True)

# FootyStats competition IDs por temporada
FOOTYSTATS_SEASONS = {
    '2026-2027': 17099,
    '2025-2026': 15234,
    '2024-2025': 12136,
    '2023-2024': 9525,
    '2022-2023': 7425,
    '2021-2022': 6038,
    '2020-2021': 4507,
    '2019-2020': 2270,
    '2018-2019': 1582,
    '2017-2018': 307,
    '2016-2017': 308,
    '2015-2016': 309,
    '2014-2015': 310,
    '2013-2014': 311,
    '2012-2013': 312,
}

# ============================================================
# FUENTE 1: KAGGLE / HUGGINGFACE (gratis, completo, SIN BTTS)
# ============================================================
def load_kaggle_ligamx():
    """Carga datos Liga MX desde HuggingFace mirror de Kaggle"""
    print("📥 Cargando datos de Kaggle/HuggingFace...")
    
    file_path = Path.home() / ".cache/huggingface/hub/datasets--jason1966--adamgbor_club-football-match-data-2000-2025/snapshots/ba2f8ca76796ed7a7b80a6052577dfa14d0000aa/Matches.csv"
    
    if not file_path.exists():
        print("❌ No encontrado. Ejecuta primero:")
        print("   pip install --break-system-packages huggingface_hub")
        print("   python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='jason1966/adamgbor_club-football-match-data-2000-2025', filename='Matches.csv', repo_type='dataset')\"")
        return None
    
    df = pd.read_csv(file_path, low_memory=False)
    ligamx = df[df['Division'] == 'MEX'].copy()
    
    # Limpiar columnas vacías
    ligamx = ligamx.dropna(axis=1, how='all')
    
    output = DATA_DIR / "kaggle_ligamx_raw.csv"
    ligamx.to_csv(output, index=False)
    print(f"✅ Guardado: {output} ({len(ligamx)} partidos, 2012-2024)")
    return ligamx


# ============================================================
# FUENTE 2: FOOTYSTATS (gratis, CON BTTS, requiere descarga manual o browser)
# ============================================================
def download_footystats_manual():
    """Instrucciones para descarga manual de FootyStats"""
    print("\n📋 DESCARGA MANUAL FOOTYSTATS (tiene columnas BTTS):")
    print("   1. Abre: https://footystats.org/mexico/liga-mx/datasets")
    print("   2. Click en 'Match CSV' para cada temporada")
    print("   3. Guarda en: ", DATA_DIR / "footystats/")
    print("\n   URLs directas por temporada:")
    for season, comp_id in FOOTYSTATS_SEASONS.items():
        url = f"https://footystats.org/c-dl.php?type=matches&comp={comp_id}"
        print(f"   {season}: {url}")
    
    (DATA_DIR / "footystats").mkdir(exist_ok=True)
    
    # Verificar qué archivos ya existen
    existing = list((DATA_DIR / "footystats").glob("*_matches.csv"))
    print(f"\n   Ya descargados: {len(existing)}/15 temporadas")
    for f in sorted(existing):
        print(f"     - {f.name}")


def load_footystats_combined():
    """Combina todos los CSVs de FootyStats descargados"""
    footystats_dir = DATA_DIR / "footystats"
    files = sorted(footystats_dir.glob("*_matches.csv"))
    
    if not files:
        print("❌ No hay archivos FootyStats. Ejecuta download_footystats_manual() primero")
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_season'] = f.stem.replace('_matches', '')
            dfs.append(df)
            print(f"  ✅ {f.name}: {len(df)} filas, cols: {len(df.columns)}")
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    output = DATA_DIR / "footystats_ligamx_combined.csv"
    combined.to_csv(output, index=False)
    print(f"\n✅ Combinado: {output} ({len(combined)} partidos)")
    
    # Verificar columnas BTTS
    btts_cols = [c for c in combined.columns if 'btts' in c.lower() or 'both' in c.lower()]
    print(f"   Columnas BTTS encontradas: {btts_cols}")
    
    return combined


# ============================================================
# FUENTE 3: SPORTMONKS API (trial 14 días, CON BTTS + odds reales)
# ============================================================
class SportmonksClient:
    """Cliente simple para Sportmonks API v3"""
    
    BASE_URL = "https://api.sportmonks.com/v3/football"
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get(self, endpoint: str, params: dict = None):
        url = f"{self.BASE_URL}/{endpoint}"
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    
    def find_liga_mx_id(self):
        """Busca el league_id de Liga MX"""
        data = self.get("leagues", {"filter[country]": "Mexico"})
        for league in data.get('data', []):
            if 'liga mx' in league.get('name', '').lower():
                return league['id']
        return None
    
    def get_fixtures_with_btts(self, league_id: int, season_id: int = None):
        """Obtiene fixtures con predicciones BTTS y odds"""
        params = {
            "filter[league]": league_id,
            "include": "predictions,odds,participants",
            "per_page": 500
        }
        if season_id:
            params["filter[season]"] = season_id
        
        data = self.get("fixtures", params)
        return data.get('data', [])
    
    def extract_btts_data(self, fixtures):
        """Extrae datos BTTS relevantes de la respuesta"""
        rows = []
        for fx in fixtures:
            row = {
                'fixture_id': fx.get('id'),
                'date': fx.get('starting_at'),
                'home_team': fx.get('participants', [{}])[0].get('name') if fx.get('participants') else None,
                'away_team': fx.get('participants', [{}])[1].get('name') if len(fx.get('participants', [])) > 1 else None,
                'home_score': fx.get('scores', [{}])[0].get('score') if fx.get('scores') else None,
                'away_score': fx.get('scores', [{}])[1].get('score') if len(fx.get('scores', [])) > 1 else None,
            }
            
            # Predicciones BTTS
            predictions = fx.get('predictions', [])
            for pred in predictions:
                if pred.get('type') == 'btts':
                    row['btts_yes_prob'] = pred.get('predictions', {}).get('yes')
                    row['btts_no_prob'] = pred.get('predictions', {}).get('no')
            
            # Odds (tomar las de Bet365 si están)
            odds = fx.get('odds', [])
            for odd in odds:
                if odd.get('bookmaker', {}).get('name') == 'Bet365':
                    for market in odd.get('markets', []):
                        if market.get('name') == 'Both Teams To Score':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == 'Yes':
                                    row['btts_yes_odds'] = outcome.get('odds')
                                elif outcome.get('name') == 'No':
                                    row['btts_no_odds'] = outcome.get('odds')
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def demo_sportmonks(token: str = None):
    """Demo de uso Sportmonks (requiere token válido)"""
    if not token:
        print("\n🔑 SPORTMONKS API - Requiere registro gratis (14 días trial):")
        print("   1. Regístrate en: https://www.sportmonks.com/")
        print("   2. Obtén tu token en: https://my.sportmonks.com/")
        print("   3. Ejecuta: python ligamx_btts_pipeline.py --sportmonks TU_TOKEN")
        return
    
    print("\n🔄 Conectando a Sportmonks API...")
    client = SportmonksClient(token)
    
    try:
        league_id = client.find_liga_mx_id()
        if not league_id:
            print("❌ No se encontró Liga MX")
            return
        
        print(f"✅ Liga MX ID: {league_id}")
        
        fixtures = client.get_fixtures_with_btts(league_id)
        print(f"📊 Obtenidos {len(fixtures)} fixtures")
        
        df = client.extract_btts_data(fixtures)
        output = DATA_DIR / "sportmonks_ligamx_btts.csv"
        df.to_csv(output, index=False)
        print(f"✅ Guardado: {output}")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Liga MX BTTS Data Pipeline")
    parser.add_argument("--kaggle", action="store_true", help="Cargar datos Kaggle/HuggingFace")
    parser.add_argument("--footystats-manual", action="store_true", help="Mostrar instrucciones descarga FootyStats")
    parser.add_argument("--footystats-combine", action="store_true", help="Combinar CSVs FootyStats descargados")
    parser.add_argument("--sportmonks", type=str, help="Token Sportmonks para API")
    parser.add_argument("--all", action="store_true", help="Ejecutar todo lo posible")
    
    args = parser.parse_args()
    
    if args.all or args.kaggle:
        load_kaggle_ligamx()
    
    if args.all or args.footystats_manual:
        download_footystats_manual()
    
    if args.all or args.footystats_combine:
        load_footystats_combined()
    
    if args.sportmonks:
        demo_sportmonks(args.sportmonks)
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 EJEMPLOS:")
        print("  python ligamx_btts_pipeline.py --kaggle")
        print("  python ligamx_btts_pipeline.py --footystats-manual")
        print("  python ligamx_btts_pipeline.py --footystats-combine")
        print("  python ligamx_btts_pipeline.py --sportmonks TU_TOKEN")
        print("  python ligamx_btts_pipeline.py --all")


if __name__ == "__main__":
    main()