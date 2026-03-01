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







def graficar_distribucion_pobla_emig_inmig(
    dicc_datos_por_año: dict[int, pd.DataFrame] 
) -> Figure:
    
    # Definimos el dominio
    lados = 10
    angulos = np.linspace(0, 2*np.pi, lados, endpoint=False)
    vertices_dominio = [(np.cos(theta), np.sin(theta)) for theta in angulos]
    
    dpi = 300
    fig, ejes = plt.subplots(
        4, 
        3, 
        figsize=(20, 21),
        dpi=dpi,
        gridspec_kw={'height_ratios': [0.1, 1, 1, 1]},
    )
    ejes = ejes.ravel()
    
    indices_magnitud = [i for i in range(len(ejes)) if i % 3 == 0 and i > 1]
    indices_años = [1, 2]
    indices_diag = [i for i in range(len(ejes)) if i % 3 != 0 and i > 3]
    iter_indices = iter(indices_diag)
    
    años = [1990, 2024]
    
    lista_magnitudes_objetivo = [
        ('poblacion', 'Marrón'), ('emigrantes', 'Verde'), ('inmigrantes', 'Roja')
    ]
    
    i = 0
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
                'alfa_nombres_de_celdas': .9, 
                'factor_aumento': 4,
                # Celdas
                'tam_min_nombre_celda': 5,
                'grosor_borde_celdas': .6,
                'alfa_borde_celdas': .7,
                'paleta_celdas': paleta,
            }   
    
                    
            diagrama._graficar_diagrama(    
                6, # Ancho
                6, # Alto
                dpi, # DPI
                '', # Título
                '', # Nota al pie
                150, # Núm. de caracteres por línea
                None, # Ruta de salida
                ejes[next(iter_indices)], # eje
                **config_grafico,
            )
            
            i += 1
    
    for (magnitud, _), indice in zip(lista_magnitudes_objetivo, indices_magnitud):
        if indice == 3:
            texto = 'POBLACIÓN'
        else:
            texto = magnitud.upper()
        ejes[indice].text(
            .5,
            .5,
            texto,
            ha='center',
            fontsize=30,
            weight='bold',
        )
        ejes[indice].axis('off')
    
    ejes[0].set_visible(False)
    
    for año, indice in zip(años, indices_años):
        ejes[indice].text(
            .5,
            .5,
            str(año),
            ha='center',
            fontsize=30,
            weight='bold',
        )
        ejes[indice].axis('off')
    
    ruta_salida = 'resultados/distribucion_poblacion_emigracion_inmigracion_1990-2024.png'
    plt.tight_layout(pad=4)
    plt.savefig(ruta_salida, bbox_inches='tight', dpi=dpi)
    plt.close()

    return fig




    
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
    



def graficar_desglose_ZZ_africa(tabla) -> Figure:

    fig, eje = plt.subplots(figsize=(8, 8), dpi=300)
    
    mat_color = eje.imshow(
        tabla,
        aspect='auto',
        cmap='RdPu',
    )

    eje.set_xticks(range(len(tabla.columns)))
    eje.set_xticklabels(tabla.columns)    
    eje.set_yticks(range(len(tabla.index)))
    eje.set_yticklabels(tabla.index)    
    eje.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True)

    
    for i in range(len(tabla.index)):
        for j in range(len(tabla.columns)):
            if i == 1 and j == 5:
                color_tex = 'yellow'
            else:
                color_tex = 'black'
                
            valor = tabla.iloc[i, j]
            eje.text(
                j, i, 
                convertir_valor(valor),
                ha='center', 
                va='center', 
                color=color_tex,
                fontsize=7,
            )
    
    barra_color = fig.colorbar(
        mat_color, ax=eje, fraction=0.1, aspect=100, pad=0.02,  
    )
    barra_color.set_label('Inmigrantes con origen desconocido')    
    plt.tight_layout()
    plt.savefig('resultados/desglose_ZZ_africa.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return fig




    
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
    
    
