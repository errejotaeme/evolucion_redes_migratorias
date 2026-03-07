import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import dipot
import itertools
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import seaborn.objects as so
import textwrap
from pathlib import Path


from collections import Counter
from IPython.display import display
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
    año: int,
    **args,
) -> None:
    
    # Colores del mapa
    agua = args.get('agua','#ececec')
    tierra = args.get('tierra', '#cfcfcf')
    fronteras = args.get('fronteras', '#f3f3f3')
    continentes = args.get('continentes', '#a6a6a6')   
    # Colores del grafo
    color_nodos = args.get('color_nodos', '#b1b1b1')
    colores_por_region = {
        'Asia': '#b51b1e',
        'África': '#197c45',
        'Europa': '#0f00b0',
        'Américas': '#212121',
        'Oceanía': '#c67d00',        
    }
    
    # Vsualización 
    eje.set_facecolor(agua) 

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
    eje.add_feature(cfeature.COASTLINE, edgecolor=continentes, linewidth=0.8, zorder=1)
    eje.add_feature(cfeature.BORDERS, edgecolor=fronteras, linewidth=0.6, zorder=1)

    # Diccionario donde se guardan las coordenadas con referencia al mapa
    pos_nodos = {}
    for nodo in grafo_pobla_ext.nodes():
        lon, lat = posiciones_ingresadas[nodo]
        pos_nodos[nodo] = (lon, lat)
    
    # NODOS
    nx.draw_networkx_nodes(
        grafo_pobla_ext,
        pos_nodos,
        node_size=.5,
        ax=eje,
        node_color=tierra
    )
    
    # # ETIQUETAS
    etiquetas = nx.get_node_attributes(grafo_pobla_ext, 'etiqueta')
    region = nx.get_node_attributes(grafo_pobla_ext, 'region_inmig')
    for nodo, (x, y) in pos_nodos.items():
        eje.text(
            x,
            y,
            etiquetas[nodo],
            fontsize=11,
            fontweight='bold',
            fontstyle='italic',
            color=colores_por_region.get(region[nodo], 'none'),
            ha='center',
            va='center',
            alpha=.7
        )
    
    for comu, pais in grafo_pobla_ext.edges():
        nx.draw_networkx_edges(
            grafo_pobla_ext,
            pos_nodos,
            edgelist=[(comu, pais)],
            width=.5,
            edge_color=[colores_por_region.get(region[pais], 'none')],
            alpha=.6,
            connectionstyle='arc3,rad=0.2',            
            arrowstyle='->',
            arrows=True,
            ax=eje,
        )
        
    # Colores y texto de las referencias
    if año == 2024:
        del colores_por_region['Oceanía']
        lista_ref = []
        for region, color in colores_por_region.items():
            tex_ref = f'Comunidad inmigrante con origen en {region}.'
            ref = mpatches.Patch(color=color, label=tex_ref)
            lista_ref.append(ref)
    
        ref1 = eje.legend(
            # title='REFERENCIAS',
            handles=lista_ref,
            loc='lower left',
            fontsize=14,
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
    path: str | None = None,
    key: str = 'iso3'
) -> dict:
    """Carga el listado de nombres normalizado.
    Devuelve {clave: name_es}. clave puede ser 'iso3' o 'cod_m49'.
    Si se elige cod_m49, las claves se convierten a int.
    """
    if path is None:
        p = Path(__file__).resolve().parent / 'fuentes_de_datos' / 'nombres_es.csv'
    else:
        p = Path(path)
    
    m = pd.read_csv(p, dtype=str)
    if key not in m.columns:
        raise KeyError(f"La clave '{key}' no está en el listado. Columnas disponibles: {list(m.columns)}")
    if key == 'cod_m49':
        m[key] = pd.to_numeric(m[key], errors='coerce').dropna().astype(int)
    m['name_es'] = m.get('name_es', '').fillna('').astype(str)
    return dict(zip(m[key], m['name_es']))




def graficar_distribucion_pobla_emig_inmig(
    dicc_datos_por_año: dict[int, pd.DataFrame] 
) -> None:
    
    # Definimos el dominio
    lados = 10
    angulos = np.linspace(0, 2*np.pi, lados, endpoint=False)
    vertices_dominio = [(np.cos(theta), np.sin(theta)) for theta in angulos]
    
    dpi = 300
   
    años = [1990, 2024]
    
    lista_magnitudes_objetivo = [
        ('poblacion', 'Marrón'), ('emigrantes', 'Verde'), ('inmigrantes', 'Roja')
    ]

    for magnitud_objetivo, paleta in lista_magnitudes_objetivo:
        for año in años:  
        
            df = dicc_datos_por_año[año].copy()
            
            columnas_de_interes = [
                'alfa2',
                magnitud_objetivo,
                f'pct_aporte_{magnitud_objetivo}',
                'lon', 
                'lat', 
            ]            
            # Recortamos a las columnas de interes y ordenamos
            df = df[columnas_de_interes]
            df = df.sort_values([columnas_de_interes[2]], ascending=False).reset_index(drop=True)
            # Separamos los paises que explican 90%
            df[f'pct_aporte_{magnitud_objetivo}_acum'] = df[f'pct_aporte_{magnitud_objetivo}'].cumsum()
            df = df[df[f'pct_aporte_{magnitud_objetivo}_acum'] < 91]
    
            # Datos para la construcción del gráfico
            nombres_de_celdas = df.alfa2.values
        
            # Datos para la construcción del diagrama
            valores_magnitud_objetivo = df[magnitud_objetivo].values
            x, y = df.lon.values, df.lat.values    
        
            # Normalizamos las coordenadas a [-1, 1]
            x = (x + 180) / 180 - 1
            y = (y + 90) / 90 - 1
            coordenadas_de_sitios = np.column_stack((x, y))

                   
            pct_barra = 0.0
            formato_barra: str = "{desc}: [{bar}] {percentage:3.0f}% | {elapsed}"
            texto_barra: str = f'Construyendo diagrama: {magnitud_objetivo}-{año}'       
    
            diagrama = dipot.DiagramaDePotencia(
                vertices_dominio,
                coordenadas_de_sitios,
                nombres_de_celdas,
                valores_magnitud_objetivo,   
            )
            
            with tqdm(total=100, desc=texto_barra, bar_format=formato_barra) as barra_pct:    
                diagrama.construir_diagrama(        
                    .13, # alfa_base
                    500, # max_iteraciones
                    5, # umbral_estancamiento
                    1e-5, # error_rel_max_permitido,
                    False, # imprimir progreso
                    barra_pct
                )
        
            
            config_grafico = {
                # Título
                'margen_titulo': 7,
                # Nombres de celdas
                'alfa_nombres_de_celdas': .8, 
                'factor_aumento': 4,
                # Celdas
                'tam_min_nombre_celda': 8,
                'grosor_borde_celdas': 1,
                'alfa_borde_celdas': .7,
                'paleta_celdas': paleta,
            }    
                    
            vis = diagrama._graficar_diagrama(    
                10, # Ancho
                10, # Alto
                dpi, # DPI
                '', # Título
                '', # Nota al pie
                0, # Núm. de caracteres por línea
                f'resultados/diagrama_potencia_{magnitud_objetivo}_{año}', # Ruta de salida
                None, # eje
                **config_grafico,
            )

            display(vis)




  


def graficar_corredores_principales(
    df_migraciones: pd.DataFrame,
    año: int = 2024,
    top_n: int = 30,
    palette: str = 'viridis',
    out_path: str | None = None,
    show=False,
    xlim: tuple | None = None,
) -> tuple:
    """
    Construye el Top N corredores a partir del DF de migraciones y grafica. 
    Parámetros:
      - df_migraciones: DF con columnas 'cod_orig','cod_des','año','migrantes'
      - año: año a filtrar
      - top_n: cantidad de corredores a mostrar
      - palette: paleta de matplotlib (ej. 'viridis', 'plasma')
      - out_path: ruta de salida para la imagen
      - show: si True, muestra el gráfico al finalizar
      - xlim: tupla (min, max) para el límite del eje x. Sino, se ajusta automáticamente.
    """
    if out_path is None:
        out_path = f'resultados/corredores_{año}_top{top_n}.png'

    dicc_nombres_es = cargar_nombres_es(key='cod_m49')  # claves: int

    df = df_migraciones.copy()
    if 'migrantes' in df.columns:
        df['migrantes'] = pd.to_numeric(df['migrantes'], errors='coerce').fillna(0)
    if 'año' in df.columns:
        df['año'] = df['año'].astype(int)

    # Numérico para que coincida con las claves int del diccionario
    df['cod_orig'] = pd.to_numeric(df['cod_orig'], errors='coerce')
    df['cod_des']  = pd.to_numeric(df['cod_des'],  errors='coerce')

    df_a = df[df['año'] == int(año)].copy()

    df_a['origen_nombre_sp'] = df_a['cod_orig'].map(dicc_nombres_es)
    df_a['destino_nombre_sp'] = df_a['cod_des'].map(dicc_nombres_es)
    df_a['corredor'] = df_a['origen_nombre_sp'] + ' → ' + df_a['destino_nombre_sp']

    agg = df_a.groupby(
        ['corredor', 'origen_nombre_sp', 'destino_nombre_sp'], as_index=False
    )['migrantes'].sum().rename(columns={'migrantes': 'stock'})

    top_vis = agg.sort_values('stock', ascending=False).head(int(top_n)).reset_index(drop=True)

    labels = [
        f"{row['origen_nombre_sp'][:20]} → {row['destino_nombre_sp'][:20]}"
        for _, row in top_vis.iterrows()
    ]

    cmap   = plt.get_cmap(palette)
    colors = cmap(np.linspace(0.3, 0.9, len(top_vis)))

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(top_vis))))
    ax.barh(range(len(top_vis)), top_vis['stock'].astype(float), color=colors)
    ax.set_yticks(range(len(top_vis)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Stock de migrantes', fontsize=11, fontweight='bold')
    # No title by request; show the año discretely in the lower-left corner
    ax.text(
        0.01,
        0.02,
        str(año),
        transform=ax.transAxes,
        fontsize=10,
        fontweight='semibold',
        verticalalignment='bottom',
        horizontalalignment='left',
        color='0.15'
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6):.1f}M'))
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    if xlim is not None:
        ax.set_xlim(*xlim)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return fig, ax





def graficar_red_corredores(
    df_migraciones: pd.DataFrame,
    df_coordenadas: pd.DataFrame,
    año: int = 2024,
    threshold_pct: float = 1.0,
    palette: str = 'YlOrRd',
    tipo_linea: str = 'recta',
    dpi: int = 300,
    excluir_otros: bool = True,
    out_path: str | None = None,
    show: bool = False,
) -> tuple:
    """
    Grafica la red global de corredores migratorios para un año dado.

    Parámetros:
      - df_migraciones : DF con columnas cod_orig, iso3_orig, cod_des, iso3_des, año, migrantes
      - df_coordenadas : DF con columnas iso3_coord, lat, lon
      - año            : año a visualizar
      - threshold_pct  : porcentaje superior de aristas a mostrar por peso (ej: 1 → top 1%)
      - palette        : paleta matplotlib para nodos
      - tipo_linea     : 'recta' o 'geodesica'
      - ancho_px       : ancho de salida en píxeles (default 6000)
      - excluir_otros  : excluir el nodo ZZZ ('Otros / origen desconocido')
      - out_path       : ruta de salida; si es None se genera automáticamente
      - show           : mostrar en notebook
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if out_path is None:
        out_path = f'resultados/red_corredores_{año}_top{int(threshold_pct)}pct.png'

    dicc_nombres = cargar_nombres_es(key='iso3')

    df = df_migraciones.copy()
    df['migrantes'] = pd.to_numeric(df['migrantes'], errors='coerce').fillna(0)
    df['año'] = df['año'].astype(int)
    df_a = df[df['año'] == año].copy()

    if excluir_otros:
        df_a = df_a[(df_a['iso3_orig'] != 'ZZZ') & (df_a['iso3_des'] != 'ZZZ')]

    coords = df_coordenadas.set_index('iso3_coord')[['lat', 'lon']].to_dict('index')

    # --- Construir grafo ---
    G = nx.DiGraph()

    paises = pd.concat([
        df_a[['iso3_orig']].rename(columns={'iso3_orig': 'iso3'}),
        df_a[['iso3_des']].rename(columns={'iso3_des':  'iso3'}),
    ]).drop_duplicates('iso3')

    for _, row in paises.iterrows():
        iso = row['iso3']
        c = coords.get(iso, {})
        if c:
            G.add_node(iso, nombre=dicc_nombres.get(iso, iso), lat=c['lat'], lon=c['lon'])

    for _, row in df_a.iterrows():
        u, v, w = row['iso3_orig'], row['iso3_des'], row['migrantes']
        if w > 0 and u in G.nodes and v in G.nodes:
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)

    # --- Aplicar threshold ---
    all_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    umbral = np.percentile(all_weights, 100 - threshold_pct)
    G_vis = nx.DiGraph()
    G_vis.add_nodes_from(G.nodes(data=True))
    G_vis.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] >= umbral])

    # --- Figura ---
    ancho_in = 40
    alto_in  = ancho_in * (9 / 16)

    fig = plt.figure(figsize=(ancho_in, alto_in), dpi=dpi, facecolor='white')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('white')

    ax.add_feature(cfeature.LAND,      facecolor='#f2f2f2', alpha=1.0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#ddeeff', alpha=1.0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#888888')
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor='#aaaaaa')
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle='--', color='gray')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'color': '#555555'}
    gl.ylabel_style = {'size': 14, 'color': '#555555'}

    # --- Aristas ---
    transform_linea = ccrs.Geodetic() if tipo_linea == 'geodesica' else ccrs.PlateCarree()

    pesos_vis = np.array([d['weight'] for _, _, d in G_vis.edges(data=True)])
    log_min = np.log10(pesos_vis.min() + 1)
    log_max = np.log10(pesos_vis.max() + 1)

    LW_MIN, LW_MAX       = 0.2, 7.0
    ALPHA_MIN, ALPHA_MAX = 0.04, 0.65

    for u, v, data in G_vis.edges(data=True):
        lon_u = G_vis.nodes[u].get('lon')
        lat_u = G_vis.nodes[u].get('lat')
        lon_v = G_vis.nodes[v].get('lon')
        lat_v = G_vis.nodes[v].get('lat')
        if None in (lon_u, lat_u, lon_v, lat_v):
            continue

        norm     = (np.log10(data['weight'] + 1) - log_min) / (log_max - log_min) if log_max > log_min else 0.5
        norm_exp = norm ** 0.4

        ax.plot(
            [lon_u, lon_v], [lat_u, lat_v],
            color='#cc2200',
            linewidth=LW_MIN + (LW_MAX - LW_MIN) * norm_exp,
            alpha=ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * norm_exp,
            transform=transform_linea,
            zorder=2,
            solid_capstyle='round',
        )

    # --- Nodos ---
    grados = dict(G_vis.degree())
    lons, lats, sizes, colores_nodo = [], [], [], []

    for node_id, attrs in G_vis.nodes(data=True):
        if 'lon' not in attrs or 'lat' not in attrs:
            continue
        lons.append(attrs['lon'])
        lats.append(attrs['lat'])
        grado = grados.get(node_id, 0)
        sizes.append(max(120, grado ** 1.6 * 8))
        colores_nodo.append(grado)

    scatter = ax.scatter(
        lons, lats,
        s=sizes,
        c=colores_nodo,
        cmap=palette,
        alpha=0.9,
        edgecolors='#333333',
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # --- Etiquetas top 20 por grado ---
    top20 = sorted(
        [(n, a) for n, a in G_vis.nodes(data=True) if 'lon' in a and 'lat' in a],
        key=lambda x: grados.get(x[0], 0),
        reverse=True
    )[:20]

    for node_id, attrs in top20:
        ax.annotate(
            attrs.get('nombre', node_id)[:22],
            (attrs['lon'], attrs['lat']),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=16,
            fontweight='bold',
            color='#111111',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, linewidth=0),
            zorder=6,
            transform=ccrs.PlateCarree(),
        )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.018, pad=0.02, shrink=0.55)
    cbar.set_label('Grado (conexiones)', rotation=270, labelpad=22, fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    ax.set_title(
        f'Red Global de Corredores Migratorios ({año})  —  Top {threshold_pct:.1f}% por stock absoluto\n'
        f'Umbral: {umbral:,.0f} migrantes  |  '
        f'{G_vis.number_of_nodes()} países  |  {G_vis.number_of_edges():,} aristas',
        fontsize=22,
        fontweight='bold',
        color='#111111',
        pad=18,
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    plt.close()
    print(f'Red guardada en: {out_path}')
    return fig, ax



    
def graficar_migraciones_africa(
        
    migras_hacia_africa: pd.DataFrame,
    migras_desde_africa: pd.DataFrame,
) -> Figure:   


    fig1 = plt.figure(figsize=(9, 6), dpi=300)  

    # Hacia
    vis1 = (
        so.Plot(migras_hacia_africa, 'año', 'migrantes', color='region_orig_ES')
        .add(so.Dot(pointsize=3.5))
        .add(so.Line(linewidth=1.7))
        .scale(color='colorblind')
            .layout(size=(9, 6))
        .label(
            title='Migraciones hacia África desde otros continentes',
            x='Año',
            y='Migrantes',
            color='Continente origen'
        )
    )    
    vis1.on(fig1).plot()
    fig1.tight_layout()
    fig1.savefig('resultados/migras_hacia_Africa.png', bbox_inches='tight', dpi=300)

    # Desde
    fig2 = plt.figure(figsize=(9, 6), dpi=300)
    vis2 = (
        so.Plot(migras_desde_africa, 'año', 'migrantes', color='region_des_ES')
        .add(so.Dot(pointsize=3.5))
        .add(so.Line(linewidth=1.7))
        .scale(color='colorblind')
            .layout(size=(9, 6))
        .label(
            title='Migraciones desde África hacia otros continentes',
            x='Año',
            y='Migrantes',
            color='Continente destino'
        )
    )    
    vis2.on(fig2).plot()
    fig2.tight_layout()
    fig2.savefig('resultados/migras_desde_Africa.png', bbox_inches='tight', dpi=300)



# Función para convertir valores
def convertir_valor(valor: int, redondeo:int = 2) -> str:
    if valor < 1e3:
        return f'{valor}'
    elif valor < 1e6:
        prefijo = 'k'
        denominador = 1e3
    else:
        prefijo = 'M'
        denominador = 1e6
    conversion = valor / denominador
    valor_redondeado = round(conversion, redondeo)
    # si el decimal es 0 lo descarto
    if valor_redondeado.is_integer():
        return f'{int(valor_redondeado)}{prefijo}'
    else:
        return f'{valor_redondeado}{prefijo}'
    





    
def graficar_bloques_africa(
    paises_sin_bloque: set[str],
    dicc_bloques_asignados: dict[int, set[str]],
    dicc_cohesion_por_bloque:  dict[int, float],
    df_coords: pd.DataFrame,
) -> Figure: 

    paises_africa = [
        pais 
        for bloques in dicc_bloques_asignados.values() 
        for pais in bloques
    ]
    
    # Creamos la red que permite graficar sobre el mapa
    red_bloques = nx.Graph()
    for bloque, cohesion  in zip(dicc_bloques_asignados.values(), dicc_cohesion_por_bloque.values()):
        for tupla_miembros in itertools.combinations(bloque, 2):
            # Nombres de países
            m1 = tupla_miembros[0]
            m2 = tupla_miembros[1]
            
            # Posiciones de los países
            lon_m1 = df_coords.loc[df_coords.iso3_coord == m1, 'lon'].iloc[0]
            lat_m1 = df_coords.loc[df_coords.iso3_coord == m1, 'lat'].iloc[0]        
            lon_m2 = df_coords.loc[df_coords.iso3_coord == m2, 'lon'].iloc[0]
            lat_m2 = df_coords.loc[df_coords.iso3_coord == m2, 'lat'].iloc[0]        

            # Agregamos el vínculo
            red_bloques.add_edge(m1, m2, weight=cohesion)

            # Si el nodo origen aún no fue ingresado
            if not bool(red_bloques.nodes[m1]):
                red_bloques.add_node(m1, pos=(lon_m1, lat_m1))
            if not bool(red_bloques.nodes[m2]):  
                red_bloques.add_node(m2, pos=(lon_m2, lat_m2))

    
    # Configuración de la visualización
    # Tamaño del gráfico
    tam_figura = (20,17)  
    # Márgenes que permiten regular el zoom sobre el mapa
    margen_lon = 7
    margen_lat = 7
    # Colores del mapa
    agua = '#ffffff'
    tierra = '#c3c3ca'
    fronteras = '#cdced5'
    continentes = '#525255'
    # Colores del grafo
    color_titulo = '#0c0c0c'
    color_nodos = '#ba0000'
    color_aris = '#000000'#5eebc1'
    color_etq_pais = '#000000'
    # Paleta de colores para los nodos
    paleta = [
        '#9d3c43',
        '#d9d697',
        '#98eaa4',
        '#f98c78',
        '#b172ff',
        '#06c9d3',
        '#377054',
        '#f1d438',
        '#f8474a',
        '#3e38ff',
        '#ff1a98',
    ]
    
    
    # Tamaños de fuentes
    tam_tex_etq = 12 # nodos
    tam_tex_titulo = 17 # título gráfico
    tam_tex_ref = 17 # referencias
    
    # Vsualización de la red
    fig = plt.figure(figsize=tam_figura)
    
    # Configuración de colores
    fig.set_facecolor(agua) # Color predominante (fondo)
    eje = fig.add_subplot(1,1,1)
    eje.set_facecolor(agua)  # Fondo interior donde se dibuja el grafo
    eje.axis('off') # Remueve el marco y el color blanco dentro del grafo
    
    # Hago zoom en la región de interés
    posiciones_ingresadas = nx.get_node_attributes(red_bloques, 'pos')
    lons = [pos[0] for pos in posiciones_ingresadas.values()]
    lats = [pos[1] for pos in posiciones_ingresadas.values()]
    
    
    # Proyección
    eje = plt.axes(projection=ccrs.PlateCarree())
    
    # Extensión del mapa
    eje.set_extent(
        [
            min(lons) - margen_lon,
            max(lons) + margen_lon,
            min(lats) - margen_lat,
            max(lats) + margen_lat,
        ],
        crs=ccrs.PlateCarree()
    )
    
    # Elementos del mapa
    eje.add_feature(cfeature.LAND, facecolor=tierra)
    eje.add_feature(cfeature.OCEAN, facecolor=agua)
    eje.add_feature(cfeature.COASTLINE, edgecolor=continentes, linewidth=0.8)
    eje.add_feature(cfeature.BORDERS, edgecolor=fronteras, linewidth=0.6)
    forma_paises = shpreader.natural_earth(
        resolution='110m',
        category='cultural',
        name='admin_0_countries'
    )
    lector = shpreader.Reader(forma_paises)
    poligonos_paises = {}
    for p in lector.records():
        iso = p.attributes['ISO_A3']
        nombre = p.attributes['NAME']    
        if nombre == 'Somaliland':
            iso = 'SOMI'    
        poligonos_paises[iso] = p.geometry
        
    
    # Geometrías de países
    for i, bloque in enumerate(dicc_bloques_asignados.values()):
        for pais in bloque:
            if poligonos_paises.get(pais, None):
                eje.add_geometries(
                    [poligonos_paises[pais]],
                    crs=ccrs.PlateCarree(),
                    edgecolor='black',
                    linewidth=.6,
                    facecolor=paleta[i],
                    alpha=.7,
                    hatch='//' if pais in paises_sin_bloque else None
                )
            if pais == 'SOM':                   
                eje.add_geometries(
                    [poligonos_paises['SOMI']],
                    crs=ccrs.PlateCarree(),
                    edgecolor='black',
                    linewidth=.6,
                    facecolor=paleta[i],
                    alpha=.7, 
                    hatch='//' if pais in paises_sin_bloque else None
                )
    
    for i, bloque in enumerate(dicc_bloques_asignados.values()):
        for pais in bloque:
            if pais in ['MUS', 'SYC', 'COM', 'STP', 'CPV']:
                tamaño_nodo = 900
            else:
                tamaño_nodo = 0
            nx.draw_networkx_nodes(
                red_bloques,            
                posiciones_ingresadas,
                nodelist=[pais],
                node_size=tamaño_nodo,
                node_color=paleta[i],
                node_shape='o',
                alpha=.7,
                ax=eje,
            )
    
    
    # ETIQUETAS    
    for nodo in red_bloques.nodes():
        if nodo in paises_sin_bloque:
            fuente = 'normal'
        else: 
            fuente = 'black'            
        nx.draw_networkx_labels(
            red_bloques,
            posiciones_ingresadas,
            labels={nodo: nodo},
            font_color=color_etq_pais,
            font_weight=fuente,
            alpha=1,
            font_size=tam_tex_etq,
            ax=eje,
        )


    
    # REFERENCIAS
    lista_ref = []
    for i, cohesion in dicc_cohesion_por_bloque.items():
        lista_ref.append(
            mpatches.Patch(
                color=to_rgba(paleta[i], alpha=.7),
                label=f'Cohesión: {round(cohesion, 3)}'
            )
        )
    
    ref1 = eje.legend(
        # title='REFERENCIAS',
        handles=lista_ref,
        loc='lower left',
        fontsize=tam_tex_ref,
        frameon=True,
        facecolor='none',
        framealpha=0,
    )
    
    for text in ref1.get_texts():
        text.set_color('black')
    eje.add_artist(ref1)
    
    
    eje.set_title(
        '',
        fontsize=tam_tex_titulo,
        color=color_titulo,
    )
    
    fig.tight_layout()
    plt.savefig("resultados/bloques_africa.png", bbox_inches='tight', dpi=300) 
    plt.close()
    
    return fig
    
    
