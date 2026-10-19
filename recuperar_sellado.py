#!/usr/bin/env python3
# Recuperacion del btts_jornada.csv sellado (sobrescrito accidentalmente)
import pandas as pd
import hashlib, datetime

# 1) Guardar la version fresca con otro nombre
fresca = pd.read_csv('/home/petyt/futbolmx/predictions/btts_jornada.csv')
fresca.to_csv('/home/petyt/futbolmx/predictions/btts_jornada_fresco_2026-10-17.csv', index=False)
print("version fresca guardada:", len(fresca), "filas")

# 2) RECONSTRUIR el archivo sellado (23 renglones verbatim del registro de sesion)
sellado = pd.DataFrame([
    ("2026-10-16","Necaxa","Atlas",0.681470),
    ("2026-10-16","Tijuana","Puebla",0.549550),
    ("2026-10-16","Atlante","Pumas",0.548059),
    ("2026-10-17","Toluca","San Luis",0.679891),
    ("2026-10-17","Cruz Azul","Juarez",0.602964),
    ("2026-10-17","Leon","America",0.558196),
    ("2026-10-17","Chivas","Tigres",0.509231),
    ("2026-10-17","Santos","Queretaro",0.430994),
    ("2026-10-18","Monterrey","Pachuca",0.544409),
    ("2026-10-20","Chivas","Necaxa",0.648909),
    ("2026-10-20","San Luis","Queretaro",0.645400),
    ("2026-10-20","Tigres","Leon",0.626484),
    ("2026-10-20","Juarez","Atlante",0.580934),
    ("2026-10-21","Pachuca","Cruz Azul",0.701917),
    ("2026-10-21","Atlas","America",0.665374),
    ("2026-10-21","Toluca","Tijuana",0.658268),
    ("2026-10-21","Puebla","Monterrey",0.591171),
    ("2026-10-21","Santos","Pumas",0.484457),
    ("2026-10-23","Necaxa","Juarez",0.627758),
    ("2026-10-23","Atlante","San Luis",0.582203),
    ("2026-10-24","Leon","Toluca",0.620335),
    ("2026-10-24","Pumas","Tigres",0.619445),
    ("2026-10-24","Monterrey","Chivas",0.518140),
], columns=["date","home","away","p_btts"])
sellado.to_csv('/home/petyt/futbolmx/predictions/btts_jornada.csv', index=False)

sha = hashlib.sha256(open('/home/petyt/futbolmx/predictions/btts_jornada.csv','rb').read()).hexdigest()

nota = f"""NOTA DE RECUPERACION - {datetime.date.today()}
El btts_jornada.csv sellado del jueves 15-oct (11:03) fue sobreescrito accidentalmente
el 17-oct durante el analisis informal (import de predict_jornada ejecuto el pipeline completo).
RECUPERACION: reconstruido con los 23 renglones verbatim del registro de sesion (impresion
del archivo sellado hecha ANTES del sobrescribir). Columnas: date, home, away, p_btts.
El registro real de apuestas (paper_trades.csv) NUNCA fue tocado - intacto.
La version regenerada (37 fixtures, master al 12-oct) quedo en btts_jornada_fresco_2026-10-17.csv
SHA-256 del archivo reconstruido: {sha}
LECCION: convertir predict_jornada.py en modulo importable sin efectos secundarios
        (guardar la escritura de archivos bajo if __name__ == '__main__').
"""
open('/home/petyt/futbolmx/predictions/NOTE_recuperacion_btts_jornada.txt','w').write(nota)
print("sellado reconstruido:", len(sellado), "filas | sha256:", sha[:16], "...")
print("nota escrita")
