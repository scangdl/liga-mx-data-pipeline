# ⚽ Liga MX Data Pipeline — Predicción y Paper Trading

Pipeline de análisis de datos para la Liga MX (fútbol mexicano): ingesta de datos
públicos, ingeniería de características, modelos predictivos y un sistema de
paper trading con protocolo de test a ciegas.

## Arquitectura

```
build_master.py        -> consolidado maestro (5,499 partidos, 2018-2026)
fetch_fixtures_espn.py -> fixtures próximos vía API pública de ESPN
fetch_odds_api.py       -> snapshot de cuotas del mercado (Odds API)
make_features.py        -> features: forma reciente, Elo, descanso, tendencia goleadora
poisson_btts.py        -> modelo Poisson BTTS (ambos equipos anotan)
over25_model.pkl        -> clasificador Over/Under 2.5 goles (21 features)
predict_jornada.py      -> predicciones por jornada
paper_trade.py          -> paper trading: stake Kelly fraccional, sellado con SHA-256
eval_walkforward.py     -> evaluación walk-forward (sin fuga de datos)
eval_calibracion_backtest.py -> curvas de calibración y Brier por temporalidad
```

## Principios del proyecto

- **Nada inventado:** cada número proviene de un archivo auditable. Fuentes:
  football-data.co.uk (histórico 2018-2025) y APIs públicas (ESPN, Odds API).
- **Test a ciegas:** las predicciones de cada jornada se sellan con SHA-256
  ANTES de los partidos y se liquidan con marcadores reales — imposible ajustar.
- **Honestidad de backtesting:** evaluación walk-forward; la calibración del
  modelo se audita antes de arriesgar capital (aunque sea simulado).

## Uso

```bash
python build_master.py          # reconstruye el maestro
python update_and_predict.py    # fixtures + predicción de la jornada
python paper_trade.py place     # registra apuestas con valor (edge > umbral)
python paper_trade.py settle    # liquida con marcadores reales
```

## Estado

- Modelos v2 en producción de paper trading (71 trades registrados)
- Backtest histórico y curva de calibración en progreso
- Roadmap: umbral de edge adaptativo por calibración, tracking de CLV

---
Autor: Armando Sánchez Cendejas — Zapopan, Jalisco.
Datos públicos deportivos; sin datos personales de terceros.
