import geopandas as gpd
import numpy as np
import pandas as pd
import requests 

from bs4 import BeautifulSoup
from io import BytesIO, StringIO


# # **Construcción del conjunto de datos**

# 1. **REGIONES, SUBREGIONES, PAÍSES MENOS DESARROLLADOS Y SIN LITORAL**
# Obtenemos los códigos estándar de países o áreas para uso estadístico (M49 ONU)
# y nos quedamos con las columnas de interés
print('1. Procesando regiones, subregiones y códigos de países...')
url_m49 = 'https://unstats.un.org/unsd/methodology/m49/overview/'
headers = {'User-Agent': 'Mozilla/5.0'}
respuesta_m49 = requests.get(url_m49, headers=headers)
contenido_html = BeautifulSoup(respuesta_m49.text, 'html.parser')
tabla_html_EN = contenido_html.find('table', id='downloadTableEN')
tabla_html_ES = contenido_html.find('table', id='downloadTableES')

# Construimos la tabla
df_m49 = pd.read_html(
    StringIO(str(tabla_html_EN)),
    keep_default_na=False, # Previene que el iso2 de Namibia (NA) se lee como NaN 
    na_values=['']
)[0]
df_m49
df_m49 = df_m49.iloc[:, [3, 5, 8, 9, 10, 11, 12, 13]]
df_m49.columns = [
    'region_EN',
    'subregion_EN',
    'pais_EN',
    'cod_m49',
    'iso2_m49',
    'iso3_m49',
    'menos_desarrollado',
    'sin_litoral'
]

# Convertimos a booleanas
df_m49['menos_desarrollado'] = df_m49['menos_desarrollado'] == 'x'
df_m49['sin_litoral'] = df_m49['sin_litoral'] == 'x'

# Agregamos el nombre de país en español
df_m49_pais_ES = pd.read_html(
    StringIO(str(tabla_html_ES)),
    keep_default_na=False,
    na_values=['']
)[0]
df_m49_pais_ES = df_m49_pais_ES.iloc[:, [3, 5, 8, 9]]
df_m49_pais_ES.columns = [
    'region_ES',
    'subregion_ES',
    'pais_ES',
    'cod_m49_ES']

# Juntamos las tablas y ordenamos
df_m49 = (
    df_m49.merge(
        df_m49_pais_ES,
        left_on='cod_m49',
        right_on='cod_m49_ES',
        how='left'
    ).drop(columns=['cod_m49_ES'])
)
columnas_ordenadas = [
    'cod_m49',
    'iso2_m49',
    'iso3_m49',
    'pais_ES',
    'region_ES',
    'subregion_ES',
    'pais_EN',    
    'region_EN',
    'subregion_EN',
    'menos_desarrollado',
    'sin_litoral',
]

df_m49 = df_m49[columnas_ordenadas]

print('Operación finalizada.')


# 2. **LONGITUDES Y LATITUDES**
# Obtenemos los centroides de países desde UNHCR (Alto Comisionado de las Naciones Unidas para los Refugiados)
print('2. Procesando ubicaciones...')
url_coordenadas = 'https://gis.unhcr.org/arcgis/rest/services/core_v2/wrl_polbnd_int_1m_p_unhcr/FeatureServer/0/query'
params = {
    'where': '1=1',
    'outFields':'iso3,secondary_territory,status_un',
    'returnGeometry': 'true',
    'outSR': '4326',
    'f': 'geojson'
}
respuesta_coordenadas = requests.get(url_coordenadas, params=params)
if respuesta_coordenadas.status_code != 200:
    raise RuntimeError('El servidor de UNHCR no respondió correctamente. Ejecutar de nuevo.')

datos_geo = respuesta_coordenadas.json()
gdf = gpd.GeoDataFrame.from_features(datos_geo['features'])
gdf

# Limpiamos los datos
# Descartamos los territorios sin identificación
gdf = gdf[gdf.iso3 != 'AAA']
# Obtenemos las filas con iso3 duplicado
iso3_duplicados = gdf[gdf['iso3'].duplicated(keep=False)].sort_values(['iso3'], ascending=True) 

# De los duplicados, nos quedamos solo con los datos que corresponden a países
iso3_unicos = iso3_duplicados[iso3_duplicados.secondary_territory != 1] 

# Eliminamos los duplicados y juntamos las tablas
gdf = gdf[~gdf.index.isin(iso3_duplicados.index)] 
gdf = pd.concat([gdf, iso3_unicos]).reset_index(drop=True) 

# Obtenemos los centroides
gdf['centroide'] = gdf.geometry.centroid
gdf['lon'] = gdf['centroide'].x
gdf['lat'] = gdf['centroide'].y
df_coordenadas = gdf.drop(columns=['geometry','centroide','secondary_territory'])
df_coordenadas.columns = ['iso3_coord','estatus_geo','lon','lat']

territorios_a_conservar = ['GUF', 'PRI', 'FLK', 'ESH', 'TWN']
df_aux = df_coordenadas[df_coordenadas.iso3_coord.isin(territorios_a_conservar)]

# Territorios de ultramar, autónomos, no soberanos, etc.
territorios = {
    'The City of Vatican',
    'AU Territory',
    'CN Province',
    'DK Self-Governing Territory',
    'DK Territory',    
    'FR Non-Self-Governing Territory',
    'FR Territory',
    'NL Self-Governing Territory',
    'NL Territory',    
    'NZ Non-Self-Governing Territory', 
    'NZ Territory',
    'UK Non-Self-Governing Territory',
    'UK Territory',
    'UK territory',
    'US Non-Self-Governing Territory',
    'US Territory',
    'MU Territory',
    'NO Territory',
    'Non-Self-Governing Territory',
    ' ',   
}
df_coordenadas = df_coordenadas[~df_coordenadas.estatus_geo.isin(territorios)]
df_coordenadas = pd.concat([df_coordenadas, df_aux]).reset_index(drop=True)
# Asignamos un punto en el océano Atlántico Sur solo para poder identificarlo en la exploración
df_coordenadas.loc[201] = ['ZZZ', 'Desconocido', -20.000000, -25.000000]

# Conjunto de códigos ISO 3166 alfa-3
codigos_iso3 = set(df_coordenadas.iso3_coord.unique())

# Conjunto inicial de códigos globales m49
codigos_m49 = set(
    df_m49[df_m49.iso3_m49.isin(codigos_iso3)].cod_m49.unique()
)
# Nos quedamos con los países que estan en el conjunto 
df_m49 = df_m49[df_m49.cod_m49.isin(codigos_m49)]
# Agregamos TWN y Otros
df_aux = pd.DataFrame(
    {
        'cod_m49': [158, 2003],
        'iso2_m49': ['TW', 'ZZ'],
        'iso3_m49': ['TWN', 'ZZZ'],
        'pais_ES': ['Taiwán', 'Otros'], 
        'region_ES': ['Asia', 'Región desconocida'],
        'subregion_ES': ['Asia oriental', 'Subregión desconocida'],
        'pais_EN': ['Taiwan', 'Others'],
        'region_EN': ['Asia', 'Unknown region'],
        'subregion_EN': ['Eastern Asia', 'Unknown subregion'],
        # A Otros le asignamos False por practicidad, aunque no tiene significado
        'menos_desarrollado': [False, False],  
        'sin_litoral': [False, False]
    }
)
df_m49 = (
    pd.concat([df_m49, df_aux])
    .sort_values('iso2_m49', ascending=True)
    .reset_index(drop=True)
)
# Agregamos al conjunto inicial los codigos de TW y ZZ
# Los agregamos después de sumarlos a la tabla para que no haya repetidos
codigos_m49.add(np.int64(158))
codigos_m49.add(np.int64(2003))
# df_m49.to_csv('m49.csv', index=False)


# df auxiliar para agregrar códigos en las tablas
df_alfa3_codigos = df_m49[['cod_m49', 'iso3_m49']]

# Agregamos el código global
df_coordenadas = (
    df_alfa3_codigos.merge(
        df_coordenadas,
        left_on='iso3_m49',
        right_on='iso3_coord',
        how='left',
    ).drop(columns=['iso3_m49'])
    .rename(columns={'cod_m49':'cod_coord'})
    .sort_values('iso3_coord', ascending=True)
    .reset_index(drop=True)
)
# df_coordenadas.to_csv('coordenadas.csv', index=False)
print('Operación finalizada.')


# 3. **DATOS DE POBLACIÓN**
# Descargamos los datos de poblaciones de la página oficial de Naciones Unidas
print('3. Procesando datos de población...')
url_poblaciones = 'https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx'
headers = {'User-Agent': 'Mozilla/5.0'}
respuesta_poblas = requests.get(url_poblaciones, headers=headers)
doc_excel_poblas = BytesIO(respuesta_poblas.content)
poblaciones_90_20 = pd.read_excel(doc_excel_poblas, sheet_name='Estimates', engine='openpyxl')
poblaciones_24 = pd.read_excel(doc_excel_poblas, sheet_name='Medium variant', engine='openpyxl')

# Índices de columnas de interés y lista de años
nombres_columnas = ['iso3_pobla', 'tipo', 'año_pobla', 'poblacion']
indices_columnas = [5, 8, 10, 12]
años_con_datos = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# Para ordenar los datos de población y las estimaciones
def procesar_df(df:pd.DataFrame) -> pd.DataFrame:    
    # Me quedo con las columnas de interés y asigno el encabezado de la tabla
    df = df.iloc[16:, indices_columnas]
    df.columns = nombres_columnas

    # Filtramos los datos de población que coinciden con los años y códigos
    df = (
        df[(df.iso3_pobla.isin(codigos_iso3)) & (df.año_pobla.isin(años_con_datos))]
        .drop(columns=['tipo'])
    )

    # Convertimos las columnas de año y población a enteros
    columnas_a_convertir = ['año_pobla', 'poblacion']    
    df[columnas_a_convertir] = (
        df[columnas_a_convertir].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    )    
    # Convertimos los datos de población al valor real (miles)
    df['poblacion'] = (df['poblacion'] * 1e3).astype(int)

    return df.reset_index(drop=True)

# Obtenemos los datos y las estimaciones de población para el período
df_poblaciones_90_20 = procesar_df(poblaciones_90_20)
df_poblaciones_24 = procesar_df(poblaciones_24)

# Unimos las tablas
df_poblaciones = (
    pd.concat([df_poblaciones_90_20, df_poblaciones_24])
)

# Nos quedamos con los datos de población de países que estan en el conjunto 
df_poblaciones = df_poblaciones[df_poblaciones.iso3_pobla.isin(codigos_iso3)]

# Agregamos el código global a las tablas de poblaciones y coordenadas
df_poblaciones = (
    df_alfa3_codigos.merge(
        df_poblaciones,
        left_on='iso3_m49',
        right_on='iso3_pobla',
        how='left',
    ).drop(columns=['iso3_m49'])
    .dropna(axis=0, how='any')
    .rename(columns={'cod_m49':'cod_pobla'})
    .sort_values(['iso3_pobla','año_pobla'], ascending=[True, True])
    .reset_index(drop=True)
)
# df_poblaciones.to_csv('poblaciones.csv', index=False)

print('Operación finalizada.')


# 4. **DATOS MIGRATORIOS**
# Descargamos los datos de migraciones de la página oficial de Naciones Unidas
print('4. Procesando datos migratorios...')
url_migraciones = 'https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx'
# headers = {'User-Agent': 'Mozilla/5.0'}
headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}
respuesta_migras = requests.get(url_migraciones, headers=headers, timeout=30)

doc_excel_migras = BytesIO(respuesta_migras.content)
df_migras_original = pd.read_excel(doc_excel_migras, sheet_name='Table 1', engine='openpyxl')

# Nos quedamos con las celdas donde están realmente los datos 
df_migras_original = df_migras_original.iloc[9:, :15].reset_index(drop=True)
df_migras_original.columns = df_migras_original.iloc[0].astype(str)
# Datos en bruto
df_migras_original = df_migras_original[1:].reset_index(drop=True)

# Arreglamos los nombres de las columnas
df_migraciones = df_migras_original.copy()
nombres_columnas = df_migras_original.columns.to_list()
nombres_columnas = [f'{col[:-2]}' if col.endswith('.0') else col for col in nombres_columnas]
nombres_columnas[1], nombres_columnas[5] = 'destino', 'origen'
nombres_columnas[4], nombres_columnas[6] = 'cod_des', 'cod_orig'
df_migraciones.columns = nombres_columnas

# Convertimos a enteros los datos de migrantes
for i, col in enumerate(df_migraciones.columns):
    if i in range(7,15):
        df_migraciones[col] = pd.to_numeric(df_migraciones[col], errors='coerce').fillna(0).astype('int')

# Convertimos a enteros los codigos de origen y destino
df_migraciones[['cod_des', 'cod_orig']] = (
    df_migraciones[['cod_des', 'cod_orig']]
    .apply(pd.to_numeric, errors='coerce')
    .fillna(0)
    .astype(int)
)

# Nos quedamos solo con las columnas que me interesan
indices_columnas_de_interes = [1, 4, 5, 6] + list(range(7,15))
df_migraciones = df_migraciones.iloc[:, indices_columnas_de_interes]

# Limpiamos nombres de países y regiones 
df_migraciones['origen'] = df_migraciones['origen'].str.replace('*', '', regex=False)
df_migraciones['destino'] = df_migraciones['destino'].str.replace('*', '', regex=False)

# Filtramos usando los códigos m49
df_migraciones = ( 
    df_migraciones[
        df_migraciones.cod_orig.isin(codigos_m49) &
        df_migraciones.cod_des.isin(codigos_m49)
    ]
)

# Creamos una tabla larga con las migraciones
años = nombres_columnas[7:15] # ['1990', ..., '2024']
df_migraciones = df_migraciones.melt(
    id_vars=['origen', 'cod_orig', 'destino', 'cod_des'],
    value_vars=años, 
    var_name='año',
    value_name='migrantes'
)

# Reemplazamos los nombres de origen y destino 
# para que coincidan con los del df_m49
df_nombres_paises = df_m49[['cod_m49','iso3_m49','pais_ES','pais_EN']]

df_migraciones = (
    df_migraciones.merge(
        df_nombres_paises,
        left_on='cod_orig',
        right_on='cod_m49',
        how='left'
    ).drop(columns=['cod_m49'])
).rename(columns={
    'iso3_m49':'iso3_orig',
    'pais_ES':'origen_ES',
    'pais_EN':'origen_EN',
})

df_migraciones = (
    df_migraciones.merge(
        df_nombres_paises,
        left_on='cod_des',
        right_on='cod_m49',
        how='left'
    ).drop(columns=['cod_m49'])
).rename(columns={
    'iso3_m49':'iso3_des',
    'pais_ES':'destino_ES',
    'pais_EN':'destino_EN',
})

columnas_ordenadas = [
    'cod_orig',
    'iso3_orig',
    'origen_ES',
    'origen_EN',
    'cod_des',
    'iso3_des',
    'destino_ES',
    'destino_EN',
    'año',
    'migrantes',
]
df_migraciones = df_migraciones[columnas_ordenadas]

# Descartamos las filas sin datos
df_migraciones['año'] = pd.to_numeric(df_migraciones['año'], errors='coerce').fillna(0).astype('int')
df_migraciones = (
    df_migraciones[df_migraciones.migrantes != 0]
    .sort_values(['origen_ES','año'], ascending=[True, True])
    .reset_index(drop=True)
)

# Exportamos las tablas
# df_migraciones.to_csv('migraciones.csv', index=False)
# df_migras_original.to_csv('migraciones_original.csv', index=False)
print('Operación finalizada.')


# 5. **UNIFICACIÓN DE LOS DATOS**
# Juntamos los datos obtenidos de las distintas fuentes
print('5. Integrando los datos...')

# Unimos migraciones con los datos de los códigos, regiones, subregiones, etc.
df_migras_90_24 = (
    df_migraciones.merge(
        df_m49,
        left_on=['cod_orig'],
        right_on=['cod_m49'],
        how='left'
    ).drop(columns=['cod_m49','iso3_m49','pais_ES','pais_EN'])
    .rename(
        columns={            
            'region_ES':'region_orig_ES',
            'subregion_ES':'subregion_orig_ES',
            'region_EN':'region_orig_EN',
            'subregion_EN':'subregion_orig_EN',
            'iso2_m49':'iso2_orig',    
            'menos_desarrollado':'menos_desarr_orig',
            'sin_litoral':'sin_litoral_orig',
        }
    )
)
df_migras_90_24 = (
    df_migras_90_24.merge(
        df_m49,
        left_on=['cod_des'],
        right_on=['cod_m49'],
        how='left'
    ).drop(columns=['cod_m49','iso3_m49','pais_ES','pais_EN'])
    .rename(
        columns={
            'region_ES':'region_des_ES',
            'subregion_ES':'subregion_des_ES',
            'region_EN':'region_des_EN',
            'subregion_EN':'subregion_des_EN',
            'iso2_m49':'iso2_des',          
            'menos_desarrollado':'menos_desarr_des',
            'sin_litoral':'sin_litoral_des',
        }
    )
)

# Agregamos las coordenadas de origen y destino 
df_migras_90_24 = (
    df_migras_90_24.merge(
        df_coordenadas,
        left_on=['iso3_orig'],
        right_on=['iso3_coord'],
        how='left'
    ).drop(columns=['iso3_coord','cod_coord','estatus_geo'])
    .rename(
        columns={
            'lon':'lon_orig',
            'lat':'lat_orig',
        }
    )
)

df_migras_90_24 = (
    df_migras_90_24.merge(
        df_coordenadas,
        left_on=['iso3_des'],
        right_on=['iso3_coord'],
        how='left'
    ).drop(columns=['iso3_coord','cod_coord','estatus_geo'])
    .rename(
        columns={
            'lon':'lon_des',
            'lat':'lat_des',
        }
    )
)


# Agregamos los datos de población de los países de origen y destino
df_migras_90_24 = (
    df_migras_90_24.merge(
        df_poblaciones,
        left_on=['iso3_orig', 'año'],
        right_on=['iso3_pobla', 'año_pobla'],
        how='left'
    ).drop(columns=['año_pobla', 'iso3_pobla','cod_pobla'])
    .rename(columns={'poblacion':'poblacion_orig'})
)
df_migras_90_24 = (
    df_migras_90_24.merge(
        df_poblaciones,
        left_on=['iso3_des', 'año'],
        right_on=['iso3_pobla', 'año_pobla'],
        how='left'
    ).drop(columns=['año_pobla', 'iso3_pobla','cod_pobla'])
    .rename(columns={'poblacion':'poblacion_des'})
)

# Ordenamos la tabla
columnas_ordenadas = [
    # Datos de origen
    'cod_orig',
    'iso2_orig',
    'iso3_orig',
    'origen_ES',
    'origen_EN',
    'region_orig_ES',
    'region_orig_EN',
    'subregion_orig_ES',
    'subregion_orig_EN',
    'poblacion_orig',
    'menos_desarr_orig',
    'sin_litoral_orig',
    'lon_orig',
    'lat_orig',
    # Datos de destino
    'cod_des',    
    'iso2_des', 
    'iso3_des',
    'destino_ES',
    'destino_EN', 
    'region_des_ES',
    'region_des_EN',
    'subregion_des_ES',    
    'subregion_des_EN',
    'poblacion_des',
    'menos_desarr_des',
    'sin_litoral_des',
    'lon_des',
    'lat_des',
    #
    'año',
    'migrantes',    
]

df_migras_90_24 = (
    df_migras_90_24[columnas_ordenadas]
    .sort_values(['iso2_orig', 'año'], ascending=[True, True])
    .reset_index(drop=True)
)
df_migras_90_24 = df_migras_90_24.fillna(0)
# df_migras_90_24.to_csv('migras_90_24.csv', index=False)

print('Operación finalizada.')
tex_m49 = '\n• df_m49: códigos y clasificaciones de países, e indicadores estructurales.'
tex_pobla = '\n• df_poblaciones: población por país en los años con datos migratorios.'
tex_coord = '\n• df_coordenadas: centroides geográficos de cada país.'
tex_migras_orig = '\n• df_migras_original: datos migratorios en bruto.'
tex_migras = '\n• df_migraciones: datos migratorios de 202 países.'
tex_migras_90_24 = '\n• df_migras_90_24: reúne los datos anteriores.'
dfs_disponibles = (
    tex_m49 + tex_pobla + tex_coord + tex_migras_orig + tex_migras + tex_migras_90_24
)
print(f'\nDatos disponibles\n{'-'*len('Datos disponibles')}{dfs_disponibles}')

