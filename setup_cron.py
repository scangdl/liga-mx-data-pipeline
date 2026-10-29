#!/usr/bin/env python3
"""
setup_cron.py — Configura cron jobs automáticos para el pipeline Liga MX.

Ejecuta: python3 setup_cron.py
Genera crontab y opcionalmente lo instala.
"""
import subprocess
import sys
from pathlib import Path

FUTBOL = Path("/home/petyt/futbolmx")
PY = "/usr/bin/python3"  # system python que ve el venv

CRON_JOBS = f"""
# === LIGA MX PIPELINE (auto-generado) ===
# Jueves 11:00 — actualizar master + predecir jornada (sellado)
0 11 * * 4 {PY} {FUTBOL}/update_and_predict.py >> {FUTBOL}/predictions/cron_update.log 2>&1

# Lunes 06:00 — decay detector (alerta si modelo degrada)
0 6 * * 1 {PY} {FUTBOL}/decay_detector.py >> {FUTBOL}/predictions/cron_decay.log 2>&1

# Lunes 10:00 — place apuestas (Over 2.5 + 1X2)
0 10 * * 1 {PY} {FUTBOL}/paper_trade.py place_all >> {FUTBOL}/predictions/cron_place.log 2>&1

# Lunes 15:00 — settle apuestas (resultados del finde)
0 15 * * 1 {PY} {FUTBOL}/paper_trade.py settle_all >> {FUTBOL}/predictions/cron_settle.log 2>&1

# 1ro de cada mes 06:00 — backtest mensual completo
0 6 1 * * {PY} {FUTBOL}/monthly_backtest.py >> {FUTBOL}/predictions/cron_monthly.log 2>&1

# === FIN LIGA MX ===
"""

def main():
    print("=" * 60)
    print("CONFIGURACIÓN CRON JOBS — Liga MX Pipeline")
    print("=" * 60)
    print(CRON_JOBS)

    # Verificar crontab actual
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    current = result.stdout if result.returncode == 0 else ""

    if "LIGA MX PIPELINE" in current:
        print("⚠️  Ya existe configuración Liga MX en crontab")
        print("\nCrontab actual:")
        print(current)
        # No interactivo: sobrescribir automáticamente si se ejecuta sin TTY
        import sys
        if not sys.stdin.isatty():
            print("Modo no interactivo: sobrescribiendo...")
            resp = 's'
        else:
            resp = input("\n¿Sobrescribir? (s/N): ").strip().lower()
        if resp != 's':
            print("Cancelado.")
            return

    # Instalar
    new_cron = current.strip() + "\n" + CRON_JOBS if current.strip() else CRON_JOBS
    proc = subprocess.run(["crontab", "-"], input=new_cron, text=True, capture_output=True)
    if proc.returncode == 0:
        print("✅ Crontab instalado correctamente")
    else:
        print(f"❌ Error instalando crontab: {proc.stderr}")
        sys.exit(1)

    # Verificar
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    print("\nCrontab activo:")
    print(result.stdout)


if __name__ == "__main__":
    main()