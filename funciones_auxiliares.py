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

from collections import Counter
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from tqdm.notebook import tqdm



# FUNCIONES AUXILIARES PARA GRAFICAR (primero las únicas y después las que se reutilizan)

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







def graficar_resultados(
    df_resultados: pd.DataFrame,
    titulo: str = 'T',
    subtitulo: str = 'S',
    descripcion: str = 'D',
    caracteres_por_linea: int = 50,
    nombre_salida: str = 'A',
    tam_fig: int = 21,
    dpi: int = 300,
) -> Figure:

    # Configuración de tamaños
    tam_tex_ref = .5 * tam_fig        
    tam_titulo = tam_fig
    tam_subtitulo = .93 * tam_fig
    tam_titulo_ejes = .8 * tam_fig
    tam_nombre_eje = .75 * tam_fig
    tam_marcas_ejes = .6 * tam_fig
    tam_ref = .8 * tam_fig

    # Configuración aspecto general
    color_ejes = '#121411'
    color_datos = '#234437' 
    
    # Configuración del gráfico de caja
    ancho_cajas = .0125 * tam_fig
    viso_caja = {'facecolor': color_datos, 'color': color_datos}
    viso_mediana = { 'color': '#ff0000', 'linewidth': .15 * tam_fig}
    viso_bigotes = {'color': color_datos, 'linewidth': 0.05 * tam_fig}
    viso_punta_bigotes = {'color': color_datos, 'linewidth': .13 * tam_fig}
    viso_atipicos = {
        'marker': 'o',
        'markerfacecolor': color_datos,
        'markeredgecolor': color_datos,
        'markersize': 0.1875 * tam_fig,
    }
    # COnfiguración puntos
    tam_punto_lineas = 0.1875 * tam_fig

    # COnfiguración líneas
    tam_tex_leyenda = .6 * tam_fig
    
    fig, (eje1, eje2, eje3) = plt.subplots(1, 3, figsize=(tam_fig, tam_fig // 3))
    
    # Cajas
    años = sorted(df_resultados['año'].unique())
    datos_por_año = (
        [df_resultados[df_resultados['año'] == año]['modularidad'] for año in años]
    )
    eje1.boxplot(
        datos_por_año,
        widths=ancho_cajas,
        tick_labels=años,
        patch_artist=True,
        boxprops=viso_caja,
        medianprops=viso_mediana,
        whiskerprops=viso_bigotes,
        flierprops=viso_atipicos,
        capprops=viso_punta_bigotes,
    )
    eje1.set_xlabel('Año', fontsize=tam_nombre_eje)
    eje1.set_ylabel('Modularidad', fontsize=tam_nombre_eje)      
    eje1.set_title('Modularidad por año', fontsize=tam_titulo_ejes)
    
    # Puntos
    eje2.scatter(
        df_resultados['num_comunidades'],
        df_resultados['modularidad'],
        alpha=0.3, 
        color=color_datos,
        edgecolor='none',
    )
    eje2.set_xlabel('Número de comunidades', fontsize=tam_nombre_eje)
    eje2.set_ylabel('Modularidad', fontsize=tam_nombre_eje)
    eje2.set_title('Modularidad y número de comunidades', fontsize=tam_titulo_ejes)
    
    # Líneas
    
    paleta = cm.get_cmap('Dark2', len(años))
    for i, año in enumerate(años):
        df_año = df_resultados[df_resultados['año'] == año]
        promedio_por_resol = df_año.groupby('resolucion')['modularidad'].mean()
        eje3.plot(
            promedio_por_resol.index, 
            promedio_por_resol.values, 
            marker='o',
            markersize=tam_punto_lineas,
            label=str(año),
            color=paleta(i)
        )
    eje3.set_xlabel('Resolución', fontsize=tam_nombre_eje)
    eje3.set_ylabel('Modularidad promedio', fontsize=tam_nombre_eje)
    eje3.set_title('Modularidad promedio y resolución', fontsize=tam_titulo_ejes)
    eje3.legend(fontsize=tam_tex_leyenda)    

    # Guías verticales de retícula 
    for eje in (eje2, eje3):
        eje.minorticks_on()
        eje.grid(True, which="minor", axis="x", alpha=0.3)
    # Bordes, nombres de ejes y marcas
    for eje in (eje1, eje2, eje3):
        for spine in eje.spines.values():
            spine.set_edgecolor(color_ejes)
            spine.set_linewidth(.04 * tam_fig)
        eje.tick_params(axis="both", labelsize=tam_marcas_ejes)
        eje.tick_params(axis='x', colors=color_ejes)
        eje.tick_params(axis='y', colors=color_ejes)
        eje.xaxis.label.set_color(color_ejes)
        eje.yaxis.label.set_color(color_ejes)        
    
    # Título, subtítulo y nota al pie
    fig.suptitle(titulo, fontsize=tam_titulo, y=0.97)    
    fig.text(
        0.5, 
        0.87,
        subtitulo,
        ha='center',
        fontsize=tam_subtitulo,
        color=color_ejes,
        fontstyle='italic'
    )    
    nota_al_pie = textwrap.fill(descripcion, width=caracteres_por_linea)
    fig.text(
        0.05, 
        0.01,
        nota_al_pie, 
        ha="left",
        va="top",
        fontsize=tam_ref,
        color=color_ejes
    )
    
    plt.tight_layout(pad=4) # Espcio intraejes
    plt.savefig(f'resultados/{nombre_salida}.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return fig



    




def graficar_distribucion_coocurrencia(
    años: list[int],
    dicc_cooc_por_año: dict[int, pd.DataFrame],
    titulo: str = 'T',
    subtitulo: str = 'S',
    descripcion:str = 'D',
    nombre_salida: str = 'A', # Nombre de la imagen a guardar (sin extensión)
    caracteres_por_linea:int = 115, # Ancho máximo de la nota al pie
    dpi: int= 300,
    ancho_tam_fig: int = 8, # Fija el tamaño de los demas elementos
) -> tuple[Figure, Axes]: 

    # Configuración del gráfico
    color_ejes = '#121411'
    color_datos = '#234437' 
    tam_titulo = 1.3125 * ancho_tam_fig
    tam_subtitulo = 1.25 * ancho_tam_fig
    tam_nombre_eje = 1.125 * ancho_tam_fig
    tam_marcas_ejes = ancho_tam_fig
    tam_ref = 1.125 * ancho_tam_fig     
   
    # Configuración del gráfico de caja
    ancho_cajas = .0125 * ancho_tam_fig
    viso_caja = {
        'facecolor': color_datos,
        'color': color_datos,
    }
    viso_mediana = {
        'color': '#ff0000',
        'linewidth': .25 * ancho_tam_fig,
    }
    viso_bigotes = {
        'color': color_datos,
        'linewidth': 0.125 * ancho_tam_fig   
    }
    viso_punta_bigotes = {
        'color': color_datos,
        'linewidth': .2125 * ancho_tam_fig,
    }
    viso_atipicos = {
        'marker': 'o',
        'markerfacecolor': color_datos,
        'markeredgecolor': color_datos,
        'markersize': 0.1875 * ancho_tam_fig,
    }
    
    # Extraemos los datos de la triangular superior de la matriz de coocurrencias
    # y los guardamos para visualizar
    datos_caja = []
    for año in años:    
        vec_cooc_año = (
            dicc_cooc_por_año[año].values[
                np.triu_indices_from(dicc_cooc_por_año[año].values, k=1)
            ]
        )
        datos_caja.append(vec_cooc_año)

    # Visualizamos
    fig = plt.figure(figsize=(ancho_tam_fig, 0.625 * ancho_tam_fig))
    plt.boxplot(
        datos_caja,
        widths=ancho_cajas,
        tick_labels=años,
        patch_artist=True,
        boxprops=viso_caja,
        medianprops=viso_mediana,
        whiskerprops=viso_bigotes,
        flierprops=viso_atipicos,
        capprops=viso_punta_bigotes,
    )

    eje = plt.gca()
    eje.set_xlabel('Año', fontsize=tam_nombre_eje)
    eje.set_ylabel('Coocurrencia', fontsize=tam_nombre_eje)
    eje.tick_params(axis="both", labelsize=tam_marcas_ejes)

    for spine in eje.spines.values():
        spine.set_edgecolor(color_ejes)
        spine.set_linewidth(.0625 * ancho_tam_fig)

    eje.tick_params(axis='x', colors=color_ejes)
    eje.tick_params(axis='y', colors=color_ejes)
    eje.xaxis.label.set_color(color_ejes)
    eje.yaxis.label.set_color(color_ejes)


    # Título, subtítulo y nota al pie
    eje.set_title(titulo, fontsize=tam_titulo, pad=3.375 * ancho_tam_fig)
    fig.text(
        0.5, 
        0.90,
        subtitulo,
        ha='center',
        fontsize=tam_subtitulo,
        color=color_ejes,
        fontstyle='italic'
    )   
    nota_al_pie = textwrap.fill(descripcion, width=caracteres_por_linea)
    fig.text(0.08, 0.01, nota_al_pie, ha="left", va="top", fontsize=tam_ref, color=color_ejes)

    plt.tight_layout()
    plt.savefig(f'resultados/{nombre_salida}.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    return fig, eje







# FUNCIONES ASOCIADAS AL GRÁFICO DE TRAYECTORIAS ######

def distribuir_posy_invariantes(dicc_invariantes, paises_invariantes) -> dict[str, float]:
    """
    Devuelve un diccionario cuyas claves son los códigos de países invariantes
    y los valores son decimales que indican su coordenada y correspondiente.
    """
    dicc_invar_separadas = obtener_claves_separadas(dicc_invariantes)
    pos_invariantes = {v:c for c,v in dicc_invar_separadas.items() if v in paises_invariantes}
    # Espaciamos de forma equivalente la coordenada y entre los países invariantes
    coord_y_invars = list(pos_invariantes.values())
    coord_y_invars_separados = (
        list(
            np.linspace(
                min(coord_y_invars),
                max(coord_y_invars),
                num=len(coord_y_invars)
            )
        )
    )
    posy_invariantes_separadas = {
        invar: coord_y for invar, coord_y 
        in zip(pos_invariantes.keys(),coord_y_invars_separados)
    }
    return posy_invariantes_separadas


def obtener_claves_separadas(dicc_invariantes) -> dict[int, str]:
    """
    Devuelve un diccionario donde cada clave es un entero. El orden numérico
    respeta el orden subregional mayoritario  establecido previamente.
    """
    res = {}
    pos = 0
    ultima_pos = -float('inf')
    # El orden numérico ya está asociado a un orden subregional
    for sdr in sorted(dicc_invariantes):
        # Obtenemos los miembros del invariante que corresponden a ese sdr
        miembros = dicc_invariantes[sdr]
        cant_miembros = len(miembros)
        ancho = cant_miembros if cant_miembros % 2 == 1 else cant_miembros + 1
        c = pos + ancho // 2
        if cant_miembros % 2 == 1:
            # La mitad de las cantidades impares es un entero
            intervalos = range(-cant_miembros//2, cant_miembros//2 + 1)
        else:
            intervalos = list(range(-cant_miembros//2, 0)) + list(range(1, cant_miembros//2 + 1))
        for miembro, itv in zip(miembros, intervalos):
            pos_y = c + itv
            if pos_y <= ultima_pos:
                pos_y = ultima_pos + 1
            res[pos_y] = miembro
            ultima_pos = pos_y
        pos += ancho
    return res



def encontrar_sdr(paises, años, dicc_invariantes, dicc_cooc_por_año) -> dict[str, list[int]]:
    """
    El resultado contiene la historia de los invariantes con los que cada país
    tuvo mayor promedio de coocurrencia a través de los años.
    """
    res = {}
    for pais in paises:
        lista_sdr_cercanos = []
        for año in años:
            # Obtenemos la mayor coocurrencia promedio con los invariantes
            # Un arreglo guarda los promedios y el otro las coordenadas_y asociadas al sdr
            promedios_cooc = np.array([], dtype=float)
            pos_y_sdr_invar = np.array([], dtype=int)
            for pos_y_invar in dicc_invariantes.keys():
                promedios_cooc = np.append(
                    promedios_cooc, 
                    promedio_cooc(pais, año, pos_y_invar, dicc_cooc_por_año, dicc_invariantes)
                )
                pos_y_sdr_invar = np.append(pos_y_sdr_invar, pos_y_invar)
            # Encontramos el segmento de referencia más cercano 
            # (con el que convergió o coocurrió más seguido)
            # y lo definimos como la coordenada_y donde "debería" estar
            indice_max = np.argmax(promedios_cooc)
            sdr_cercano_del_año = pos_y_sdr_invar[indice_max]
            lista_sdr_cercanos.append(sdr_cercano_del_año)
        res[pais] = lista_sdr_cercanos
    return res



def promedio_cooc(pais, año, pos_invariante, dicc_cooc_por_año, dicc_invariantes):
    """
    Devuelve el promedio de coocurrencia que el país tuvo con el invariante.
    """
    df_cooc_año = dicc_cooc_por_año[año]
    cooc_pais = df_cooc_año.loc[pais]
    miembros = list(dicc_invariantes[pos_invariante])
    if pais in miembros:
        promedio = 1.0
    else:
        promedio = cooc_pais[list(miembros)].mean()
    return promedio




def ordenar_posiciones_de_paises(pos_paises, dicc_invariantes) -> dict[str, int]:
    """
    Define coordenadas_y para los países no invariantes previniendo que sean colineales a los sdr. 
    """
    paises_ordenados_por_sdr = sorted(pos_paises.items(), key=lambda tupla_cv: tupla_cv[1])

    # Reindizo los elementos
    laux = []
    i=0
    for (pais,_) in paises_ordenados_por_sdr:
        laux.append((pais, i))
        i +=1
    paises_ordenados_por_sdr = laux

    # Obtengo los índices que corresponden a los sdr
    indices_sdr = set(dicc_invariantes.keys())
    # Calculo sus extremos
    id_sdr_min = min(indices_sdr)
    id_sdr_max = max(indices_sdr)
    # Calculo la cantidad de países que entran entre los sdr
    cant_paises_entre_sdr = (id_sdr_max - id_sdr_min) - (len(indices_sdr) - 1)

    # Obtengo la magnitud que permite obtener de la mitad de la lista
    # la cantidad justa de países que entran entre los sdr 
    magnitud_de_corte = (
        (len(paises_ordenados_por_sdr) - cant_paises_entre_sdr) // 2
    )
    
    # Creo las particiones
    particion_inferior = paises_ordenados_por_sdr[: magnitud_de_corte]
    particion_media = paises_ordenados_por_sdr[
        magnitud_de_corte: magnitud_de_corte + cant_paises_entre_sdr
        ]
    particion_superior = paises_ordenados_por_sdr[magnitud_de_corte + cant_paises_entre_sdr:]


    # Obtengo los indices de cada partición
    # Particion inferior
    indices_particion_inferior = [-i for i in range(len(particion_inferior))]
    indices_particion_inferior = indices_particion_inferior[::-1][:-1]

    # Particion media
    indices_particion_media = {i for i in range(0, id_sdr_max)}
    # Elimino los indices que corresponden a un sdr
    indices_particion_media = sorted(list(indices_particion_media - indices_sdr)) 

    # Partición superior
    indices_particion_superior = [(id_sdr_max + 1) + i for i in range(len(particion_superior))]
    indices_particion_superior

    pos_y_de_paises = (
        indices_particion_inferior + indices_particion_media + indices_particion_superior
    )

    paises_ordenados_por_sdr = [
        (pais, pos_y) for (pais,_), pos_y in zip(paises_ordenados_por_sdr, pos_y_de_paises)
    ]
    return {pais: pos_y for (pais, pos_y) in paises_ordenados_por_sdr}




def esta_en_gint(pais, lista_gints) -> tuple[bool, set[str] | None]:
    for gint in lista_gints:
        if pais in gint:
            return True, gint
    return False, None


def obtener_invariantes_en_gint(gint_del_pais, dicc_invariantes) -> list[float]:
    invars_en_gint = []
    for y_invar, invariante in dicc_invariantes.items():
        if invariante.issubset(gint_del_pais):
            invars_en_gint.append(y_invar)
    return invars_en_gint



def confirmar_salto(y_origen, y_actual, gint_del_pais, dicc_invariantes) -> float:
    invars_en_gint = obtener_invariantes_en_gint(gint_del_pais, dicc_invariantes)
    no_hay_invars = len(invars_en_gint) == 0
    solo_hay_un_invar = len(invars_en_gint) == 1
    if no_hay_invars:
        return y_origen
    elif solo_hay_un_invar:
        return invars_en_gint.pop()
    else:
        if y_actual in invars_en_gint:
            return y_actual
        else:
            nueva_pos_y = min(invars_en_gint, key=lambda y_invar: abs(y_invar - y_actual))
            return nueva_pos_y
        
        
def obtener_coordy_del_año_(
    pais,
    año,
    dicc_invariantes,
    gints,
    y_origen,
    y_actual, # Al principio: y_actual = y_origen
) -> float | int: # (y, {gint}):
    """
    Devuelve la coordenada 'y' en la que debe estar el país en ese año.
    """
    esta_en_origen: bool =  y_actual == y_origen
    if esta_en_origen:
        esta_en_un_gint, gint_del_pais = esta_en_gint(pais, gints)
        if not esta_en_un_gint:
            return y_origen
        else:
            pos_y_invariantes = obtener_invariantes_en_gint(gint_del_pais, dicc_invariantes)
            no_hay_invars = len(pos_y_invariantes) == 0
            solo_hay_un_invar = len(pos_y_invariantes) == 1
            if no_hay_invars:
                return y_origen
            elif solo_hay_un_invar:
                # Devuelvo la coordenada del unico invariante
                return pos_y_invariantes.pop() 
            else:
                # Devuelvo la coordenada del invariante más cercano
                return min(pos_y_invariantes, key=lambda y_invar: abs(y_invar - y_actual))
            
    else: # Estaba en un invar
        sigue_en_un_gint, gint_del_pais = esta_en_gint(pais, gints)
        if sigue_en_un_gint:
            y_nuevo_invar = confirmar_salto(y_origen, y_actual, gint_del_pais, dicc_invariantes) # Maneja los casos
            return y_nuevo_invar
        else:
            return y_origen
        
 





def obtener_trayectoria(
    pais, # El país al que se le identifica la trayectoria
    x_origen, # El punto de partida
    y_origen,
    invariantes, # Contiene los grupos y la coordenada del sdr
    gint_por_año,
    años,
    tam_fig,
) -> list[tuple[float, float]]: # [(x, y)]
    """
    Devuelve la trayectoria de un país en función de la coocurrencia con un invariante.    
    """
    trayectoria = []
    pos_x, pos_y = x_origen, y_origen
    # Registramos posición
    trayectoria.append([pos_x, pos_y])
    # Avanzamos
    pos_x += .125 * tam_fig
    # Registramos posición
    trayectoria.append([pos_x, pos_y])
    # Decisión
    for año in años:
        y_actual = pos_y
        
        gints = gint_por_año[año]
        pos_x_año = año
        coord_y_del_año = obtener_coordy_del_año_(
            pais, año, invariantes, gints, y_origen, y_actual
        )        
        # Avanzamos        
        pos_x += .125 * tam_fig if año != años[-1] else .1 * tam_fig
        # Registramos posición
        trayectoria.append([pos_x, coord_y_del_año])
        # Avanzamos
        pos_x += .125 * tam_fig if año != años[-1] else .1 * tam_fig
        # Registramos posición
        trayectoria.append([pos_x, coord_y_del_año])
        pos_y = coord_y_del_año
    
    # Avanzamos y registramos (para coincidir con los puntos al final del periodod)
    pos_x += .075 * tam_fig
    trayectoria.append([pos_x, trayectoria[-1][1]])
    
    return trayectoria



def encontrar_invariante(pais, trayectoria) -> list[tuple[float, float, str]]:
    """
    Devuelve el invariante donde se encuentra el país cuando pega un salto,
    y las coordenadas necesarias para graficarlo.
    """
    # Nos quedamos con los puntos que permiten identificar donde esta la línea en un año
    # Le restamos uno a la trayectoria porque el ultimo segmento es solo para ordenar los elementos
    indices_coords_relevantes = [
        i for i in range(len(trayectoria)-1) if (i != 0 and i % 2 == 0)
    ]
    
    indices_coords_siguientes = [
        i+1 for i in range(len(trayectoria)-1) if (i != 0 and i % 2 == 0)
    ]

    coords_relevantes = [trayectoria[i] for i in indices_coords_relevantes]
    coords_siguientes = [trayectoria[i] for i in indices_coords_siguientes]
    y_origen = trayectoria[0][1]
    inv_a_registrar = []
    for (x0, y0), (x1, _) in zip(coords_relevantes, coords_siguientes):
        if y0 != y_origen:
            denominador = 3 if x0 < 2021 else 5 # El ùltimo intervalo es más chiquito
            pos_x = x0 + ((x1 - x0) / denominador)
            inv_a_registrar.append((pos_x, y_origen - .27, f'{pais} ➜ #{y0}'))
    return inv_a_registrar




def obtener_colores_lineas(
    coords, 
    salto, 
    descenso, 
    constante,
    y_origen
) -> list[str]:    
    colores = []
    for i in range(len(coords) - 1):
        if coords[i + 1][1] > coords[i][1]: 
            colores.append(to_rgba(salto, alpha=1)) # Sube --> más intensa
        elif coords[i + 1][1] < coords[i][1]:
            colores.append(to_rgba(descenso, alpha=1)) # Baja --> más apagada
        else:
            if coords[i][1] != y_origen:
                colores.append(to_rgba(constante, alpha=0)) # Transparente --> converge en un invariante
            else:
                colores.append(to_rgba(constante, alpha=1)) # Normal
    return colores



def obtener_gint_de_invariante(invar, lista_gints) -> set[str] | None:
    for gint in lista_gints:
        if invar.issubset(gint):
            return gint
    return None




def dibujar_trayectorias_de_invariantes(
    eje,
    pos_y,
    gint_max,
    trayectoria,
    color_sdr
) -> None:
    """
    Graficar el grosor de los SDR en función de la cantidad de países
    con los que coocurre.
    """    
    años = []
    y_sup = []
    y_inf = []
    
    # Pre-datos
    años.append(trayectoria[0][0] - 1.25)    
    y_sup.append(pos_y)
    y_inf.append(pos_y)    
    # Datos
    for año, grosor in trayectoria:
        grosor = grosor / gint_max
        años.append(año)
        y_sup.append(pos_y + grosor / 2)
        y_inf.append(pos_y - grosor / 2) 
    # Post-datos
    años.append(trayectoria[-1][0] + .7)    
    y_sup.append(y_sup[-1])
    y_inf.append(y_inf[-1])    
    
    poligono_x = años + años[::-1]
    poligono_y = y_sup + y_inf[::-1]
    eje.fill_between(
        poligono_x,
        poligono_y,
        pos_y,
        color=color_sdr,
        alpha=1,
        zorder=9,
    )



def registrar_grupos_de_invariantes(
    dicc_invariantes,
    dicc_gint_por_año,
    eje,
    color_pto_grupo_de_invars,
    tam_pto_grupo_invars,
    tam_puente_entre_invars,
) -> None:
    """
    En cada año registra sobre los sdr un código que vincula entre sí
    a aquellos invariantes que coincidieron en un GINT.
    """
    invars_que_coinciden = {}
    for año, gints in dicc_gint_por_año.items():
        lista_invars_coincidentes = []
        for gint in gints:
            invars_en_gint = set()
            for pos_y, invar in dicc_invariantes.items():
                if invar.issubset(gint):
                    invars_en_gint |= {pos_y}
            if len(invars_en_gint) > 1:
                lista_invars_coincidentes.append(invars_en_gint) 
        invars_que_coinciden[año] = lista_invars_coincidentes
        
    # Dibujo los puentes que unen a los invariantes    
    for pos_x in invars_que_coinciden.keys():
        for grupo_de_invars in invars_que_coinciden[pos_x]: # año
            inicio_puente = min(grupo_de_invars)
            fin_puente = max(grupo_de_invars)
            separacion = np.log(fin_puente - inicio_puente) / 2
            # Dibujo el puente entre invariantes que coocurren
            eje.plot(
                [pos_x - separacion, pos_x - separacion],
                [inicio_puente + separacion, fin_puente - separacion],
                color_pto_grupo_de_invars,
                linewidth=tam_puente_entre_invars,               
                alpha=1,
                zorder=10,
            )
            # Dibujo los accesos desde cada invariante
            for pos_y in grupo_de_invars:
                if pos_y == inicio_puente:                
                    eje.plot(
                        [pos_x, pos_x - separacion],
                        [pos_y, pos_y + separacion],
                        color_pto_grupo_de_invars, 
                        linewidth=tam_puente_entre_invars,
                        alpha=1,
                        zorder=10,
                    )
                elif pos_y == fin_puente: 
                    eje.plot(
                        [pos_x, pos_x - separacion],
                        [pos_y, pos_y - separacion],
                        color_pto_grupo_de_invars,                        
                        linewidth=tam_puente_entre_invars,                        
                        alpha=1,
                        zorder=10,
                    )            
                else:
                    # Calculo cual está mas lejos, el inicio o el fin y lo tiro hacia ese lado
                    # si estan a la misma distancia lo mando derecho
                    cerca_de_inicio = abs(inicio_puente - pos_y) < abs(fin_puente - pos_y)
                    en_el_medio = abs(inicio_puente - pos_y) == abs(fin_puente - pos_y)
                    if cerca_de_inicio:
                        sep = separacion
                    elif en_el_medio:
                        sep = 0
                    else: 
                        sep = -separacion
                    eje.plot(
                        [pos_x, pos_x - separacion],
                        [pos_y, pos_y + sep],
                        color_pto_grupo_de_invars,                        
                        linewidth=tam_puente_entre_invars,                        
                        alpha=1,
                        zorder=10,
                    )
    for año, lista_de_invars in invars_que_coinciden.items():
        n = 0
        for grupo_de_invars in lista_de_invars:
            n += 1
            for pos_y_miembro in grupo_de_invars:
                eje.scatter(
                    año, 
                    pos_y_miembro,
                    s=tam_pto_grupo_invars,
                    color=color_pto_grupo_de_invars,                   
                    alpha=1,
                    zorder=10,
                )


def obtener_grupos_laterales_de_año(
    año,
    dicc_gint_por_año,
    dicc_invariantes
) ->  list[set[str]]:
    """
    Devuelve los grupos laterales del año.
    """
    lista_grupos_laterales = []
    for gint in dicc_gint_por_año[año]:
        gint_no_contiene_invar = True
        for invariante in dicc_invariantes.values():
            if invariante.issubset(gint):
                gint_no_contiene_invar = False
                break
        if gint_no_contiene_invar:
            lista_grupos_laterales.append(gint)
    
    return lista_grupos_laterales



def registrar_grupos_laterales(
    eje,
    dicc_invariantes,
    dicc_gint_por_año,
    pos_paises,
    color_grupo_lateral,
    tam_puente_lateral,
    tam_pto_puente_lateral,
) -> None:
    """
    """
    for año in dicc_gint_por_año.keys():
        lista_grupos_laterales = obtener_grupos_laterales_de_año(
            año, 
            dicc_gint_por_año, 
            dicc_invariantes
        )
        if len(lista_grupos_laterales) > 0:
            for grupo_lateral in lista_grupos_laterales:
                coords_y_de_miembros = [pos_paises[miembro] for miembro in grupo_lateral]
                # Dibujo los puentes que unen a los miembros del grupo lateral    
                inicio_puente = min(coords_y_de_miembros)
                fin_puente = max(coords_y_de_miembros)
                separacion = np.log(fin_puente - inicio_puente) / 3
                pos_x = año
                # Dibujo el puente entre invariantes que coocurren
                eje.plot(
                    [pos_x + separacion, pos_x + separacion],
                    [inicio_puente + separacion, fin_puente - separacion],                    
                    color_grupo_lateral,
                    linewidth=tam_puente_lateral,                  
                    alpha=1,
                    zorder=25,
                )
                # Dibujo los accesos desde cada invariante
                
                for pos_y in coords_y_de_miembros:
                    if pos_y == inicio_puente:
                        eje.plot(
                            [pos_x, pos_x + separacion],
                            [pos_y, pos_y + separacion],
                            color_grupo_lateral, 
                            linewidth=tam_puente_lateral,
                            alpha=1,
                            zorder=25,
                        )
                    elif pos_y == fin_puente: 
                        eje.plot(
                            [pos_x, pos_x + separacion],
                            [pos_y, pos_y - separacion],
                            color_grupo_lateral,                        
                            linewidth=tam_puente_lateral,
                            alpha=1,
                            zorder=25,
                        )            
                    else:
                        # Calculo cual está mas lejos, el inicio o el fin y lo tiro hacia ese lado
                        # si estan a la misma distancia lo mando derecho
                        cerca_de_inicio = abs(inicio_puente - pos_y) < abs(fin_puente - pos_y)
                        en_el_medio = abs(inicio_puente - pos_y) == abs(fin_puente - pos_y)
                        if cerca_de_inicio:
                            sep = separacion
                        elif en_el_medio:
                            sep = 0
                        else: 
                            sep = -separacion
                        eje.plot( # Puentes intermedios
                            [pos_x, pos_x + separacion],
                            [pos_y, pos_y + sep],
                            color_grupo_lateral,                        
                            linewidth=tam_puente_lateral,
                            alpha=1,
                            zorder=25,
                        )                    
                    eje.scatter(
                        año, 
                        pos_y,
                        s=tam_pto_puente_lateral,
                        color=color_grupo_lateral,
                        alpha=1,
                        zorder=25,
                    )


def crear_diagrama_de_trayectorias(
    lista_paises: list[str],
    dicc_invariantes: dict[int, set[str]],
    dicc_gint_por_año: dict[int, list[set[str]]],
    dicc_cooc_por_año: dict[int, pd.DataFrame],
    titulo='',
    descripcion='',
    nombre_salida='trayectorias',
    barra=None,
    **args
) -> tuple[Figure, Axes]:
    """
    """
    # Para seguir el progreso
    if barra is not None:        
        procesos_terminados = 0
        total_procesos = 5
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()


    # VALORES POR DEFECTO
    color_de_fondo = args.get('color_de_fondo', '#ecf0eb')
    color_sdr = args.get('color_sdr', '#34432f')
    color_linea_constante_paises = args.get('color_linea_constante_paises', '#9cb193')
    # color_salto_paises = args.get('color_salto_paises', '#ff5601')
    # color_descenso_paises = args.get('color_descenso_paises', '#0276ce')    
    # color_salto_paises = args.get('color_salto_paises', '#00bc71')
    # color_descenso_paises = args.get('color_descenso_paises', '#8d0073')    
    color_salto_paises = args.get('color_salto_paises', '#e85743')
    color_descenso_paises = args.get('color_descenso_paises', '#02cf92')
    color_tex_año = args.get('color_tex_año', '#171e15')
    color_ejes_año = args.get('color_ejes_año', '#171e15')
    color_inv_año = args.get('color_inv_año', '#7d9488')
    color_pto_grupo_de_invars = args.get('color_pto_grupo_de_invars', '#202a1d')
    color_grupo_lateral = args.get('color_grupo_lateral', '#5f7a56')
    dpi = args.get('dpi', 300)
        
    # CONFIGURACIÓN DEL GRÁFICO
    tam_fig = 20
    tam_titulo = 0.8 * tam_fig
    tam_tex_pais = 0.375 * tam_fig
    tam_tex_marcas_ejes = 0.7 * tam_fig
    tam_tex_sdr= 0.35 * tam_fig
    tam_tex_inv_año = 0.25 * tam_fig
    tam_punto_invar = tam_fig
    tam_punto_pais = 0.25 * tam_fig
    tam_eje = 0.02 * tam_fig
    tam_pto_grupo_invars = .7 * tam_fig
    tam_puente_entre_invars = 0.08 * tam_fig
    tam_ref = 0.6 * tam_fig
    tam_puente_lateral = 0.03 * tam_fig
    tam_pto_puente_lateral = 0.3 * tam_fig
    ancho_sdr = 0.06 * tam_fig
    ancho_trayec_pais = 0.045 * tam_fig
    ancho_pie = int(10 * tam_fig)
    margen_entre_paises = 0.01 * tam_fig
    margen_etq = 0.0125 * tam_fig # (0.25) Magnitud de ref. para desplazar las etiquetas de países
    # Lo márgenes de las etiquetas (códigos de países, referencias, etc.)
    # hay que ordenarlos a mano: usando solo la posición de los datos quedan un poco corridos

    fig, eje = plt.subplots(
        figsize=(tam_fig, tam_fig), 
        facecolor=color_de_fondo
    )
    
    if barra is not None:
        procesos_terminados += 1
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()

    # DATOS DE DIAGRAMACIÓN
    años = list(dicc_gint_por_año.keys())
    x_inicio_datos = min(años)
    x_fin_datos = max(años)
    periodo_pre_datos = 0.4375 * tam_fig # 8.75
    periodo_post_datos = 0.1875 * tam_fig # 3.75
    
    
    # DATOS DE INVARIANTES
    paises_invariantes = [
        pais for  invariante in dicc_invariantes.values() for pais in invariante
    ]
    pos_invariantes = distribuir_posy_invariantes(
        dicc_invariantes, paises_invariantes
    )    
    # Posición inicio de países invariantes
    etq_invar = list(pos_invariantes.keys())
    pos_y_invar = list(pos_invariantes.values())    
    pos_x_inicio_invar = [x_inicio_datos - periodo_pre_datos] * len(pos_y_invar)


    # Extremo inicial y final de los segmentos de referencia
    puntos_iniciales_a_sdr = [] # Segmento que va de los puntos de países invariantes al sdr
    ext_inicial_sdr = [] # Inicio del sdr
    for y_des, invars in dicc_invariantes.items():
        for invar in invars:
            puntos_iniciales_a_sdr.append((x_inicio_datos - periodo_pre_datos, pos_invariantes[invar]))
            ext_inicial_sdr.append((x_inicio_datos - 0.3625 * tam_fig, y_des)) # 7.25
    ext_final_sdr = [(x_fin_datos + periodo_post_datos, y) for (_,y) in ext_inicial_sdr]
    
    
    # DATOS DE PAÍSES
    dicc_sdr_cercanos = encontrar_sdr(
        lista_paises,
        años,
        dicc_invariantes,
        dicc_cooc_por_año
    )
    
    # Ubica cada país en la recta del sdr con el que más convergió a través de los años
    dicc_ubicaciones_y = { 
        pais: Counter(sdr_cercanos).most_common(1)[0][0] 
        for pais, sdr_cercanos in dicc_sdr_cercanos.items()
    }
    # Posición inicial de países
    pos_paises = {c:v for c,v in dicc_ubicaciones_y.items() if c not in paises_invariantes}
    pos_paises = ordenar_posiciones_de_paises(pos_paises, dicc_invariantes)
    etq_paises = list(pos_paises.keys())
    pos_y_paises = list(pos_paises.values())
    pos_x_inicio_pais = [x_inicio_datos - 0.3125 * tam_fig] * len(pos_y_paises) # 6.25 


    # Actualizamos la barra
    if barra is not None:
        procesos_terminados += 1
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()

    
    # ELEMENTOS VISUALES

    for año in años:
        plt.axvline(
            año,
            linestyle=':',
            color=color_ejes_año,
            linewidth=tam_eje,
            alpha=.3
        )
    
    # Segmentos de referencia
    for origen, destino in zip(ext_inicial_sdr, ext_final_sdr):
        eje.plot(
            [origen[0], destino[0]],
            [origen[1], destino[1]],
            alpha=1,
            color=color_sdr,
            linewidth=ancho_sdr,
            zorder=1,
        )

    # PAÍSES: TRAYECTORIAS        
    trayectorias_paises = {} # Para colorear las líneas
    y_finales_paises = [] # Para puntear las líneas al final
    etq_final_pais = [] #  Para nombrar las líneas cuando no terminan en un sdr
    for pais in lista_paises:
        if pais in paises_invariantes:
            continue
        x_origen = x_inicio_datos - 0.3125 * tam_fig # 6.25 
        y_origen = pos_paises[pais]    
        coords_pais = obtener_trayectoria(
            pais,
            x_origen,
            y_origen,
            dicc_invariantes,
            dicc_gint_por_año,
            años,
            tam_fig
        )   
        x, y = zip(*coords_pais)        
        colores = obtener_colores_lineas(
            coords_pais,     
            color_salto_paises,
            color_descenso_paises,
            color_linea_constante_paises,
            y_origen
        )
        puntos = np.array([x, y]).T.reshape(-1, 1, 2) # Cada fila es un punto (x,y)
        segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
        coleccion_lineas = LineCollection(
            segmentos,
            colors=colores,
            linewidths=ancho_trayec_pais,
            zorder=15
        )
        eje.add_collection(coleccion_lineas)
        final_de_trayectoria = y[-1]
        y_finales_paises.append(final_de_trayectoria)
        if final_de_trayectoria == y_origen:
            etq_final_pais.append(pais)
        trayectorias_paises[pais] = coords_pais    
    
    # Registramos los saltos y descensos de países
    for pais, trayectoria in trayectorias_paises.items():
        inv_a_registrar = encontrar_invariante(pais, trayectoria)
        for x, y, etq in inv_a_registrar:      
            eje.text(
                x,
                y,
                etq,
                fontsize=tam_tex_inv_año,
                color=color_inv_año,
                bbox=dict(facecolor=color_de_fondo, edgecolor='none', pad=0),
                zorder=30
            )


    # Actualizamos la barra
    if barra is not None:
        procesos_terminados += 1
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()


    # PAÍSES: NOMBRES Y PUNTOS    
    # Etiquetas iniciales de países
    for i, etq in enumerate(etq_paises):
        eje.text(
            pos_x_inicio_pais[i] - (margen_etq * 2.3),
            pos_y_paises[i] -(margen_etq * 1.1),
            etq,
            fontsize=tam_tex_pais,
            color=color_linea_constante_paises,
        )     
    # Puntos iniciales de países
    eje.scatter(
        pos_x_inicio_pais, 
        pos_y_paises, 
        s=tam_punto_pais,
        c=color_linea_constante_paises,
    ) 
    # Puntos finales de los países
    eje.scatter(
        [x_fin_datos + 0.135 * tam_fig] * len(y_finales_paises), 
        y_finales_paises, 
        s=tam_punto_pais,
        c=color_linea_constante_paises,
    )    
    # Etiquetas finales de países que no terminan sobre un sdr
    for pais in etq_final_pais:
        eje.text(
            x_fin_datos + 0.145 * tam_fig, 
            pos_paises[pais] - (margen_etq * 1.1),
            pais,
            fontsize=tam_tex_pais,
            color=color_linea_constante_paises,
        )
    
    # Actualizamos la barra
    if barra is not None:
        procesos_terminados += 1
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()

    
    # INVARIANTES: NOMBRES, PUNTOS Y SEGMENTOS    
    # Etiquetas de los invariantes
    for i, etq in enumerate(etq_invar):
        eje.text(
            pos_x_inicio_invar[i] - (margen_etq * 2.5),
            pos_y_invar[i] - (margen_etq * 1.1), 
            etq,
            fontsize=tam_tex_pais,
            color=color_sdr,
        ) 
    # Puntos inciales de los invariantes
    eje.scatter(
        pos_x_inicio_invar,
        pos_y_invar,
        s=tam_punto_invar,
        c=color_sdr,
    )
    # Líneas que convergen a los sdr que establecen los invariantes
    for origen, destino in zip(puntos_iniciales_a_sdr, ext_inicial_sdr):
        eje.plot(
            [origen[0], destino[0]],
            [origen[1], destino[1]],
            alpha=1,
            color=color_sdr,
            linewidth=ancho_sdr,            
        )

    # Puntos finales de los sdr 
    eje.scatter(  
        [x for (x,_) in ext_final_sdr],
        [y for (_,y) in ext_final_sdr], 
        s=tam_punto_invar,
        c=color_sdr,
    )
    # Agreamos el número de sdr
    for y_sdr in dicc_invariantes.keys():
        eje.text(
            x_fin_datos + 0.2 * tam_fig, #
            y_sdr - margen_etq,
            f'#{y_sdr}',
            fontsize=tam_tex_sdr,
            color=color_sdr,
        ) 

    # Registramos la trayectoria de cada invariante
    trayectorias_invars: dict[int, list[tuple[int, int]]] = {}
    gint_max = 0 # Para normalizar la trayectoria de los invariantes
    for pos_y, invar in dicc_invariantes.items():
        historial_uniones = []
        for año, lista_gints in dicc_gint_por_año.items():
            gint_de_invar = obtener_gint_de_invariante(invar, lista_gints)
            if gint_de_invar is not None:
                cantidad_de_miembros = len(gint_de_invar)
                historial_uniones.append((año, cantidad_de_miembros))
                if cantidad_de_miembros > gint_max:
                    gint_max = cantidad_de_miembros
            else:
                historial_uniones.append((año, 0))
        trayectorias_invars[pos_y] = historial_uniones

    for pos_y, trayectoria in trayectorias_invars.items():
        dibujar_trayectorias_de_invariantes(
            eje,
            pos_y,
            gint_max,
            trayectoria,
            color_sdr
        )

    # Registramos los grupos de invariantes que en algún año coincidieron en un gint
    registrar_grupos_de_invariantes(        
        dicc_invariantes,
        dicc_gint_por_año,
        eje,
        color_pto_grupo_de_invars,
        tam_pto_grupo_invars,
        tam_puente_entre_invars,
    )
    # Registramos los gints que no convergen con invariantes
    registrar_grupos_laterales(
        eje,
        dicc_invariantes,
        dicc_gint_por_año,
        pos_paises,
        color_grupo_lateral,
        tam_puente_lateral,
        tam_pto_puente_lateral,
    ) 
        
    # CONFIGURACIÓN FINAL DEL DIAGRAMA
    eje.spines['right'].set_visible(False)
    eje.spines['left'].set_visible(False)
    eje.spines['top'].set_visible(True)    
    eje.spines['top'].set_color(color_ejes_año)    
    eje.spines['top'].set_linewidth(tam_eje)
    eje.spines['top'].set_alpha(0.5)
    eje.spines['bottom'].set_color(color_ejes_año) 
    eje.spines['bottom'].set_linewidth(tam_eje)
    eje.spines['bottom'].set_alpha(0.5)
    eje.set_xlim(x_inicio_datos - .5 * tam_fig, x_fin_datos + .3 * tam_fig) # 10, 6
    eje.set_ylim(min(pos_y_paises) - .15 * tam_fig, max(pos_y_paises) + .15 * tam_fig) # 3, 3
    eje.grid(False)
    eje.set_xticks(años)
    eje.tick_params(
        labelleft=False,
        left=False,
        colors=color_ejes_año,
        labelsize=tam_tex_marcas_ejes,
        width=tam_eje, 
        length=tam_eje*10,
    )   
    for año in eje.get_xticklabels():
        año.set_alpha(0.5)
        año.set_fontweight('bold')    
    for linea_eje in eje.xaxis.get_major_ticks():
        linea_eje.tick1line.set_alpha(0.5)
        linea_eje.tick2line.set_alpha(0.5)
    
    eje.set_facecolor(color_de_fondo)
    eje.set_title(titulo, color=color_sdr, fontsize=tam_titulo, pad=tam_fig)

    descripcion = textwrap.fill(descripcion, width=ancho_pie)
    fig.text(
        0.003 * tam_fig, 
        -0.008 * tam_fig,
        descripcion,
        ha="left", 
        fontsize=tam_ref,
        color=color_tex_año
    )
    fig.tight_layout()
    fig.savefig(f'resultados/{nombre_salida}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'resultados/{nombre_salida}.pdf', bbox_inches='tight')
    plt.close()

    # Actualizamos la barra
    if barra is not None:
        procesos_terminados += 1
        barra.n = (procesos_terminados * 100) / total_procesos
        barra.refresh()
        
    return fig, eje

# Fin funciones asociadas al grafico de trayectorias




# FUNCIÓN DE VISUALIZACIÓN DE LA PROYECCIÓN AL PLANO

def graficar_coocurrencia_en_plano(
    dicc_invariantes: dict[int, set[str]],
    df_incrustacion: pd.DataFrame,
    paises: list[str],
    paises_sin_gint: set[str],
    titulo: str,
    subtitulo: str,
    nota_al_pie: str,    
    caracteres_por_linea = 100,
    **args
) -> tuple[Figure, Axes]:
    """
    """
    # VALORES POR DEFECTO
    color_etq_invariantes = args.get('color_etq_invariantes', '#512c1f') 
    color_vinculo_entre_miembros_invar = args.get('color_vinculo_entre_miembros_invar', '#d47251')
    color_etq_gints = args.get('color_etq_gints', '#006c78')
    color_etq_sin_gint = args.get('color_etq_sin_gint', '#ff0000')
    color_texto = args.get('color_texto', '#0d0d0d')
    color_marco_eje = args.get('color_marco_eje', '#afafaf')

    # CONFIGURACIÓN DE LOS ELEMENTOS
    tam_fig = 20
    alfa_vinculo_invariantes = .6
    alfa_vinculo_gints = .55
    alfa_etiquetas = .7
    alfa_densidad = .35
    tam_punto = .5 * tam_fig
    tam_titulo_gral = 1.3 * tam_fig
    tam_titulo_subtitulo = 1.1 * tam_fig
    tam_titulo_ejes = 1.35 * tam_fig
    tam_tex_etq = .5 * tam_fig
    tam_tex_ref = .5 * tam_fig
    tam_tex_pie = .9 * tam_fig
    ancho_ref = .05 * tam_fig
    alto_ref = .025 * tam_fig
    dpi= 300

    # Creamos conjuntos que utilizamos para la diagramación
    miembros_de_invariantes = {
        pais for invariante in dicc_invariantes.values() for pais in invariante
    }
    pares_invariantes = [
        invariante for invariante in dicc_invariantes.values()
        if len(invariante) == 2
    ]
    paises_en_gint_periodo = (set(paises) - miembros_de_invariantes) - paises_sin_gint

    fig, eje = plt.subplots(figsize=(tam_fig, tam_fig))
        
    # Obtenemos lass coordenadas de los paises incrustados 
    coordenadas_de_paises_incrustados = (
        df_incrustacion[['x', 'y']].values
    )
    
    #  Calculamos la densidad de los países y sus ubicaciones
    kde = KernelDensity(bandwidth=.5, kernel='gaussian')
    kde.fit(coordenadas_de_paises_incrustados)
        
    # Discretizamos el dominio que contiene a los datos (más el margen) en un retícula
    margen = 1.0
    x_reticula = np.linspace(
        df_incrustacion.x.min() - margen,
        df_incrustacion.x.max() + margen,
        300
    )
    y_reticula = np.linspace(
        df_incrustacion.y.min() - margen,
        df_incrustacion.y.max() + margen,
        300
    )
    X, Y = np.meshgrid(x_reticula, y_reticula)
    coords_reticula = np.vstack([X.ravel(), Y.ravel()]).T

    # Obtenemos la probabilidad, en cada punto de la retícula
    # que discretiza el dominio que contiene a las incrustaciones,
    # de encontrar un país (KDE ya aprendió su ubicación)
    log_densidad_estimada = kde.score_samples(coords_reticula)
    # Convertimos las densidades de probabilidades de escala log a normal 
    # y las reorganizamos para convertirlas en puntos en el plano
    densidad_en_reticula = np.exp(log_densidad_estimada).reshape(X.shape)
        
    eje.contourf(
        X,
        Y, 
        densidad_en_reticula,
        levels=10,
        cmap='binary',
        alpha=alfa_densidad
    ) 
        
    # Identificamos cada invariante y sus miembros
    for invariante in dicc_invariantes.values():
        coords_del_invariante = (
            df_incrustacion.loc[
                df_incrustacion.iso2.isin(invariante),
                ['x', 'y']
            ].values
        )    

        if len(coords_del_invariante) > 2:
            envolvente = ConvexHull(coords_del_invariante)
            # Rellenamos el área de la envolvente
            eje.fill(
                coords_del_invariante[envolvente.vertices, 0],
                coords_del_invariante[envolvente.vertices, 1],
                alpha=alfa_vinculo_invariantes,
                color=color_vinculo_entre_miembros_invar,
                linewidth=0,
                zorder = 10
            )
        else: # Línea que los conecta        
            eje.scatter(
                coords_del_invariante[:,0],
                coords_del_invariante[:,1],
                s=tam_punto,
                c=color_vinculo_entre_miembros_invar,
                alpha=alfa_vinculo_invariantes,
                edgecolor='none',
            )
            eje.plot(
                coords_del_invariante[:,0],
                coords_del_invariante[:,1],
                linewidth=.075 * tam_fig,
                alpha=alfa_vinculo_invariantes,
                color=color_vinculo_entre_miembros_invar,
                zorder = 15
            )

    # Etiquetas
    for _, fila in df_incrustacion.iterrows():
        if fila.iso2 in miembros_de_invariantes:
            color = color_etq_invariantes
        elif fila.iso2 in paises_sin_gint:
            color = color_etq_sin_gint
        else:
            color = color_etq_gints
        eje.text(
            fila.x,
            fila.y,
            fila.iso2,
            fontsize=tam_tex_etq,
            color=color,
            weight='bold',
            alpha=alfa_etiquetas,
            zorder=20
        )
        
    # TExto informativo
    info_grafico = (
        f'Número de GINV: {len(dicc_invariantes)}; países en GINV: {len(miembros_de_invariantes)}. '
        f'Países en GINT durante el período: {len(paises_en_gint_periodo)}. '
        f'Países sin GINT durante el período: {len(paises_sin_gint)}.'
    )
    ref_info_grafico = mpatches.Patch(
        facecolor=(0,0,0,0),
        label=info_grafico,
        edgecolor='none',
    )                
    ref0 = eje.legend(
        handles=[ref_info_grafico],
        loc='lower right',
        fontsize=tam_tex_ref,
        frameon=False,
        facecolor='white',
        edgecolor='none',
    )
    eje.add_artist(ref0)     

    # REFERENCIAS
    tex_ref_pais_invar = 'Código de país en GINV.'
    tex_ref_vinculo_invar = 'Línea o área que conecta a los miembros de un GINV.'
    tex_ref_pais_gint = 'Código de país en GINT durante el período.'
    tex_ref_pais_sin_gint = 'Código de país sin GINT durante el período.'            
    # GINVS
    ref_paises_invariantes = mpatches.Patch(
        facecolor=to_rgba(color_etq_invariantes, alpha=alfa_etiquetas),
        label=tex_ref_pais_invar,
        edgecolor='none',
    )        
    ref_vinculo_invariantes = mpatches.Patch(
        facecolor=to_rgba(color_vinculo_entre_miembros_invar, alpha=alfa_vinculo_invariantes),
        label=tex_ref_vinculo_invar,
        edgecolor='none',
    )
    # GINTS
    ref_paises_gints = mpatches.Patch(
        facecolor=to_rgba(color_etq_gints, alpha=alfa_etiquetas),
        label=tex_ref_pais_gint,
        edgecolor='none',
    )        
    # Países sin GINT
    ref_paises_sin_gint = mpatches.Patch(
        facecolor=to_rgba(color_etq_sin_gint, alpha=alfa_etiquetas),
        label=tex_ref_pais_sin_gint,
        edgecolor='none',
    )                
    ref1 = eje.legend(
        handles=[
            ref_paises_invariantes,
            ref_vinculo_invariantes,
            ref_paises_gints,
            ref_paises_sin_gint
        ],
        loc='lower left',
        fontsize=tam_tex_ref * 1.4,
        frameon=False,
        facecolor='white',
        edgecolor='none',
        handlelength=ancho_ref,
        handleheight=alto_ref
    )
    eje.add_artist(ref1)
       
    # Configuración final de cada eje        
    for borde_eje in eje.spines.values():
        borde_eje.set_color(color_marco_eje)
        borde_eje.set_linewidth(.5)        
    eje.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

    # TÍTULOS, SUBTÍTULO Y NOTA AL PIE
    fig.suptitle(
        titulo, 
        fontsize=tam_titulo_gral, 
        # fontweight='bold',
        color=color_texto,
        y=0.99,
    )

    if subtitulo:
        fig.text(
            0.5, 
            0.94,
            subtitulo,
            ha='center',
            fontsize=tam_titulo_subtitulo,
            color=color_texto,
            fontstyle='italic'
        )    
    nota_al_pie = textwrap.fill(nota_al_pie, width=caracteres_por_linea)
    fig.text(
        0.01,
        0.001,
        nota_al_pie,
        ha="left",
        va="top",
        fontsize=tam_tex_pie,
        color=color_texto
    )    
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.savefig('resultados/proyeccion_de_coocurrencias_al_plano.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return fig, eje












def graficar_comunidades_ppales(
    grafo_pobla_ext: nx.DiGraph ,
    titulo: str,
    nota_al_pie: str,
    caracteres_por_linea: int = 200,
    **args,
) -> Figure:
    
    # Colores del mapa
    agua = args.get('agua','#fafafa')
    tierra = args.get('tierra', '#bdb1a7')
    fronteras = args.get('fronteras', '#f5e5d8')
    continentes = args.get('continentes', '#90877f')   
    # Colores del grafo
    color_titulo = args.get('color_titulo', '#252a29')
    color_nodos = args.get('color_nodos', '#ba0000')
    color_etq_pais = args.get('color_etq_pais', '#003cff')
    
    # Vsualización del grafo
    fig = plt.figure(figsize=(30,12))
    plt.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.01)
    
    # Configuración de colores
    fig.set_facecolor(agua) 
    eje = fig.add_subplot(1,1,1)
    eje.set_facecolor(agua) 
    eje.axis('off')

    # Hacemos zoom en la región de interés
    posiciones_ingresadas = nx.get_node_attributes(grafo_pobla_ext, 'pos')
    lons = [pos[0] for pos in posiciones_ingresadas.values()]
    lats = [pos[1] for pos in posiciones_ingresadas.values()]
    
    # Márgenes que permiten regular el zoom sobre el mapa
    margen_lat = 5
    margen_lon = 4
    
    # Proyección
    eje = plt.axes(projection=ccrs.PlateCarree())
    
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
        tamaño = 6 if len(etiquetas[nodo]) > 2 else 9
        interlineado = 0.75 if len(etiquetas[nodo]) > 2 else 1
        eje.text(
            x,
            y,
            etiquetas[nodo],
            fontsize=tamaño,
            fontweight='bold',
            fontstyle='italic',
            color=color_etq_pais,
            ha='center',
            va='center',
            linespacing=interlineado,
            alpha=.6
        )
        
    # Colores y texto de las referencias
    tex_ref = 'Código Alfa-2 del origen de comunidad inmigrante.'
    
    ref_pais = mpatches.Patch(color=color_etq_pais, label=tex_ref)
    
    ref1 = eje.legend(
        # title='REFERENCIAS',
        handles=[ref_pais],
        loc='lower left',
        fontsize=11.5,
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

    nota_al_pie = textwrap.fill(nota_al_pie, width=caracteres_por_linea)
    fig.text(
        0.21, 
        0.05,
        nota_al_pie, 
        ha="left",
        va="top",
        fontsize=15,
        color='black'
    )    
    
    eje.set_title(
        titulo, 
        fontsize=19,
        color=color_titulo,
        # fontweight='bold',
        pad=30
    )
    
    plt.savefig("resultados/ppales_comunidades_de_migrantes.png", dpi=300) 
    plt.close()
    
    return fig
