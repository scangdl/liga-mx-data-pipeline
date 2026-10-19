#!/home/petyt/venv/bin/python
# Analisis INFORMAL de los juegos del fin de semana (16-18 oct) contra cuotas frescas
# NO toca los archivos sellados: salida puramente a consola
import json, sys
import pandas as pd

# 1) cuotas del snapshot: buscar mercado totals 2.5 (over) por partido
d = json.load(open('data/odds_api_snapshot.json'))
overs = {}
for e in d['events']:
    hora = e['commence_time'][:10]
    home, away = e['home_team'], e['away_team']
    best_over = None
    for b in e.get('bookmakers', []):
        for m in b.get('markets', []):
            if m['key'] == 'totals':
                for o in m['outcomes']:
                    if o['name'] == 'Over' and o['point'] == 2.5:
                        p = o['price']
                        if best_over is None or p > best_over:
                            best_over = p
    if best_over:
        overs[(home, away, hora)] = best_over

print(f"Juegos con cuota Over 2.5 en el snapshot: {len(overs)}")
for k, v in sorted(overs.items(), key=lambda x: x[0][2]):
    print(f"  {k[2]} | {k[0]} vs {k[1]}: Over 2.5 @ {v}")

# 2) probabilidades del modelo para esos partidos (correr el modelo en memoria)
sys.path.insert(0, '.')
import predict_jornada as pj
# ver si tiene funcion reutilizable
funcs = [f for f in dir(pj) if not f.startswith('_')]
print("\nFunciones disponibles en predict_jornada:", funcs[:10])
