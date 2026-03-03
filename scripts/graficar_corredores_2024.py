#!/usr/bin/env python3
"""
Script para graficar top corredores migratorios.
"""
import argparse
from pathlib import Path
import pandas as pd

# Intentamos usar la función del módulo auxiliar si es importable
USE_HELPER = True
try:
    from funciones_auxiliares import graficar_corredores_principales
except Exception:
    USE_HELPER = False

import matplotlib.pyplot as plt
import numpy as np


def runner(infile: str, año: int = 2024, top: int = 30, outfile: str = 'resultados/corredores_2024_top30.png', palette: str = 'viridis'):
    df = pd.read_csv(infile, dtype=str)

    # Normalizamos tipos
    if 'migrantes' in df.columns:
        df['migrantes'] = pd.to_numeric(df['migrantes'], errors='coerce').fillna(0)
    if 'año' in df.columns:
        df['año'] = df['año'].astype(int)

    if USE_HELPER:
        try:
            # Usamos la función del módulo (maneja stock o agrega desde migrantes)
            graficar_corredores_principales(df, año=año, top_n=top, palette=palette, out_path=outfile)
            print(f'Gráfico generado por helper y guardado en: {outfile}')
            return
        except Exception as e:
            print('No se pudo usar la función auxiliar, caemos al modo local. Error:', e)

    # Modo local (misma lógica pero autocontenida)
    req = {'origen_ES', 'destino_ES', 'año', 'migrantes'}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Faltan columnas en el CSV para construir la visualización: {req - set(df.columns)}")

    df_a = df[df['año'] == int(año)].copy()
    df_a['corredor_origen'] = df_a['origen_ES'].astype(str)
    df_a['corredor_destino'] = df_a['destino_ES'].astype(str)
    df_a['corredor'] = df_a['corredor_origen'] + ' → ' + df_a['corredor_destino']

    agg = df_a.groupby(['corredor', 'corredor_origen', 'corredor_destino'], as_index=False)['migrantes'].sum()
    agg = agg.rename(columns={'migrantes': 'stock'})
    df_vis = agg.sort_values('stock', ascending=False).head(int(top)).reset_index(drop=True)

    labels = [f"{row['corredor_origen'][:20]} → {row['corredor_destino'][:20]}" for _, row in df_vis.iterrows()]
    cmap = plt.get_cmap(palette)
    colors = cmap(np.linspace(0.3, 0.9, len(df_vis)))

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(df_vis))))
    ax.barh(range(len(df_vis)), df_vis['stock'].astype(float), color=colors)
    ax.set_yticks(range(len(df_vis)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Stock de migrantes', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {len(df_vis)} Corredores Migratorios por Stock Absoluto ({año})', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6):.1f}M'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Gráfico guardado en: {outfile}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graficar top corredores migratorios para un año')
    parser.add_argument('--año', type=int, default=2024)
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--palette', type=str, default='viridis', help='Nombre de la paleta matplotlib (ej: viridis, plasma)')
    parser.add_argument('--in', dest='infile', default='fuentes-de-datos/migraciones.csv')
    parser.add_argument('--out', dest='outfile', default='resultados/corredores_2024_top30.png')
    args = parser.parse_args()

    runner(args.infile, año=args.año, top=args.top, outfile=args.outfile, palette=args.palette)
