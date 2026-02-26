import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import textwrap
from pathlib import Path

from collections import Counter
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from tqdm.notebook import tqdm





def crear_tabla_de_referencias(df_migraciones, df_m49) -> None:
    
    df_nombres_paises = (
        df_migraciones.groupby(['origen_ES'], as_index=False)
        .agg({'cod_orig':'first'})
    ).rename(columns={'origen_ES':'País'})
    
    df_paises_alfa = (
        df_nombres_paises.merge(
            df_m49,
            left_on=['cod_orig'],
            right_on=['cod_m49'],
            how='left',
        )
    )
    
    df_paises_alfa = df_paises_alfa[
        [
            'cod_m49',
            'iso2_m49',
            'iso3_m49',
            'País',
            'subregion_ES',
            'region_ES',
            'menos_desarrollado',
            'sin_litoral',
        ]
    ].rename(columns=
             {
                 'cod_m49':'Cód.', 
                 'iso2_m49':'Alfa-2',
                 'iso3_m49':'Alfa-3',
                 'subregion_ES': 'Subregión',
                 'region_ES': 'Región',
                 'menos_desarrollado': 'Menos-desarrollado',
                 'sin_litoral': 'Sin-litoral',
             }
    )
    df_paises_alfa = (
        df_paises_alfa.sort_values(
            ['Alfa-2', 'Alfa-3'], 
            ascending=[True, True]
        ).reset_index(drop=True)
    )
    
    # Exportamos la tabla como imagen
    fig, eje = plt.subplots(figsize=(10, 43))
    eje.axis('tight')
    eje.axis('off')
    tabla = (
        eje.table(
            cellText=df_paises_alfa.values, 
            colLabels=df_paises_alfa.columns, 
            cellLoc='center', 
            loc='center'
        )
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    # Las columnas se ajustan al valor más ancho
    tabla.auto_set_column_width(col=list(range(len(df_paises_alfa.columns))))

    # Encabezadao y filas impares con color
    for fila, col in tabla.get_celld().items():
        if fila[0] == 0:
            col.set_facecolor('#eb5959')        
        elif fila[0] % 2 != 0:
            col.set_facecolor('#94ebda')
        else:
            col.set_facecolor('white')
    
    fig.tight_layout(pad=5.0)
    plt.savefig('recursos/tabla_paises.png', bbox_inches='tight', dpi=150)
    plt.close()





def graficar_concentracion_emigracion(
    df_cant_destinos: pd.DataFrame,
    paises_que_explican: set[str],
    medianas: list[int | float], 
    promedios: list[int | float], 
    medianas_restantes: list[int | float], 
    promedios_restantes: list[int | float]    
) -> tuple[Figure, Axes]:    
    # Preparamos los datos para visualizar
    df_vis = pd.DataFrame(
        {
            'año': np.concatenate(
                [
                    df_cant_destinos.año.unique(),
                    df_cant_destinos.año.unique(),                
                    df_cant_destinos.año.unique(),
                    df_cant_destinos.año.unique()
                ]
            ),
            'valor': np.concatenate(
                [
                    medianas, promedios, medianas_restantes, promedios_restantes
                ]
            ),
            'medida':( 
                (['Mediana 90%'] * len(medianas)) +
                (['Promedio 90%'] * len(promedios)) +
                (['Promedio 10% restante'] * len(promedios_restantes)) +
                (['Mediana 10% restante'] * len(medianas_restantes))
            )
        }
    )
    
    # COnfiguración del gráfico
    color_ejes = '#121411'
    color_nombre_eje = '#1c1f1b'
    color_borde_refs = '#ffffff'
    tam_titulo = 10
    tam_nombre_eje = 9
    tam_marcas_ejes = 8
    tam_ref = 8
    colores = {
        'Mediana 90%': '#ff2643',
        'Promedio 90%': '#00b5b5',
        'Mediana 10% restante': '#b01a2e',
        'Promedio 10% restante': '#005e5e',
    }    
    fig, eje = plt.subplots(figsize=(9, 4))
    eje.minorticks_on()
    eje.grid(True, which="minor", axis="y", alpha=0.3)
    for medida, df_g in df_vis.groupby("medida"):
        linea, grosor = ('--', 1.1) if medida.startswith('P') else ('-', 1)
        alfa = 0.8 if medida.endswith('e') else 1     
        eje.plot(
            df_g['año'],
            df_g['valor'],
            linewidth=grosor,
            label=medida,       
            linestyle=linea,
            color=colores[medida],
            alpha=alfa,
        )
        eje.scatter(df_g['año'], df_g['valor'], s=5, color=colores[medida], alpha=alfa)
        
    eje.set_xlabel('Año', fontsize=tam_nombre_eje)
    eje.set_ylabel('Número de destinos', fontsize=tam_nombre_eje)
    eje.tick_params(axis="both", labelsize=tam_marcas_ejes)
    eje.spines['right'].set_visible(False)
    eje.spines['top'].set_visible(False)
    eje.spines['left'].set_color(color_ejes)
    eje.spines['bottom'].set_color(color_ejes)
    eje.tick_params(axis='x', colors=color_ejes)
    eje.tick_params(axis='y', colors=color_ejes)
    eje.xaxis.label.set_color(color_ejes)
    eje.yaxis.label.set_color(color_ejes)
    
    objetos, etiquetas = eje.get_legend_handles_labels()
    orden = [
        'Mediana 90%',
        'Promedio 90%',
        'Mediana 10% restante',
        'Promedio 10% restante',
    ]
    id_orden = [etiquetas.index(etq) for etq in orden]
    eje.legend(
        [objetos[i] for i in id_orden],
        [etiquetas[i] for i in id_orden],
        title=None, 
        fontsize=tam_ref, 
        framealpha=1,
        edgecolor=color_borde_refs,
    )
    titulo = 'Concentración de la emigración: número de destinos '
    titulo += 'que explican el 90% principal y el 10% restante (1990-2024)'
    eje.set_title(titulo, fontsize=tam_titulo, pad=13)
    
    descripcion = 'La figura muestra cuántos destinos necesitó un país para explicar el 90% '
    descripcion += ' y el 10% de su emigración en cada año.'
    descripcion += f'\n{len(paises_que_explican)} países explican, como origen o destino, el '
    descripcion += ' 90% del total de los datos migratorios durante el período.'
    fig.text(0.1, 0.01, descripcion, ha="left", va="top", fontsize=9, color=color_ejes)
    
    plt.tight_layout()
    plt.savefig('resultados/concentración_emigración.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig, eje





def graficar_comunidades_ppales(
    grafo_pobla_ext: nx.DiGraph,
    eje: Axes,
    año: str,
    **args,
) -> None:
    
    # Colores del mapa
    agua = args.get('agua','#fafafa')
    tierra = args.get('tierra', '#b1b1b1')
    fronteras = args.get('fronteras', '#f3f3f3')
    continentes = args.get('continentes', '#90877f')   
    # Colores del grafo
    color_nodos = args.get('color_nodos', '#b1b1b1')
    color_etq_pais = args.get('color_etq_pais', '#003cff')
    
    # Vsualización 
    eje.set_facecolor(agua) 
    # eje.axis('off')

    # Hacemos zoom en la región de interés
    posiciones_ingresadas = nx.get_node_attributes(grafo_pobla_ext, 'pos')
    lons = [pos[0] for pos in posiciones_ingresadas.values()]
    lats = [pos[1] for pos in posiciones_ingresadas.values()]
    
    # Márgenes que permiten regular el zoom sobre el mapa
    margen_lat = 5
    margen_lon = 4    
    
    # Extensión del mapa
    eje.set_extent([
        min(lons) - margen_lon,
        max(lons) + margen_lon,
        min(lats) - margen_lat,
        max(lats) + margen_lat
    ], crs=ccrs.PlateCarree())
    
    # Elementos del mapa
    eje.add_feature(cfeature.LAND, facecolor=tierra)
    eje.add_feature(cfeature.OCEAN, facecolor=agua)
    eje.add_feature(cfeature.COASTLINE, edgecolor=continentes, linewidth=0.8)
    eje.add_feature(cfeature.BORDERS, edgecolor=fronteras, linewidth=0.6)

    # Diccionario donde se guardan las coordenadas con referencia al mapa
    pos_nodos = {}
    for nodo in grafo_pobla_ext.nodes():
        lon, lat = posiciones_ingresadas[nodo]
        pos_nodos[nodo] = (lon, lat)
    
    # NODOS
    nx.draw_networkx_nodes(
        grafo_pobla_ext,
        pos_nodos,
        node_size=1,
        ax=eje,
        node_color=tierra
    )
    
    # # ETIQUETAS
    etiquetas = nx.get_node_attributes(grafo_pobla_ext, 'etiqueta')
    for nodo, (x, y) in pos_nodos.items():
        eje.text(
            x,
            y,
            etiquetas[nodo],
            fontsize=9.5,
            fontweight='bold',
            fontstyle='italic',
            color=color_etq_pais,
            ha='center',
            va='center',
            # linespacing=interlineado,
            alpha=.6
        )
        
    # Colores y texto de las referencias
    tex_ref = f'Origen de la principal comunidad de inmigrantes en {año}.'
    
    ref_pais = mpatches.Patch(color=color_etq_pais, label=tex_ref)
    
    ref1 = eje.legend(
        # title='REFERENCIAS',
        handles=[ref_pais],
        loc='lower left',
        fontsize=12,
        frameon=False,
        facecolor='black',
        framealpha=0.2,
        edgecolor='black',
    )
    # Color del texto de los ítems
    for text in ref1.get_texts():
        text.set_color('black')
    # Color del título
    ref1.get_title().set_color('white')
    eje.add_artist(ref1)




def cargar_nombres_es(
    path: str = 'fuentes-de-datos/nombres_es.csv', 
    key: str = 'iso3'
    ) -> dict:
    """Carga el listado de nombres normalizado 
    devuelve dict {key: name_es}.
    key puede ser 'iso3' o 'cod_m49'.
    """
    p = Path(path)
    m = pd.read_csv(p, dtype=str)
    if key not in m.columns:
        raise KeyError(f"La clave '{key}' no está en el listado")
    m[key] = m[key].astype(str)
    m['name_es'] = m.get('name_es', '').fillna('').astype(str)
    return dict(zip(m[key], m['name_es']))