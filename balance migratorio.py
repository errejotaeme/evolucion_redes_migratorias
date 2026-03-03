import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# ===============================
# 1) Leer CSV
# ===============================

df = pd.read_csv("fuentes-de-datos/migras_90_24.csv")
año = 2024
otro = 2024 if año == 1990 else 1990

# Asegurar tipos correctos
df['año'] = pd.to_numeric(df['año'], errors='coerce')
df['migrantes'] = pd.to_numeric(df['migrantes'], errors='coerce')

# ===============================
# 1) Inicializar diccionario con 0
# ===============================

# Tomamos todos los ISO3 que aparezcan como origen o destino
paises = set(df['iso3_orig'].dropna()) | set(df['iso3_des'].dropna())

# Eliminar código inválido
paises.discard("ZZZ")

balance_dict = {pais: 0 for pais in paises}
balance_dict_otro = {pais: 0 for pais in paises}

# ===============================
# 2) Recorrer dataframe
# ===============================

for _, row in df.iterrows():

    if row['año'] == año:        

        iso_orig = row['iso3_orig']
        iso_dest = row['iso3_des']
        migrantes = row['migrantes']

        if pd.isna(migrantes):
            continue

        if iso_orig != "ZZZ" and iso_orig in balance_dict:
            balance_dict[iso_orig] -= migrantes

        if iso_dest != "ZZZ" and iso_dest in balance_dict:
            balance_dict[iso_dest] += migrantes

    elif row['año'] == otro:
        
        iso_orig = row['iso3_orig']
        iso_dest = row['iso3_des']
        migrantes = row['migrantes']

        if pd.isna(migrantes):
            continue

        if iso_orig != "ZZZ" and iso_orig in balance_dict:
            balance_dict_otro[iso_orig] -= migrantes

        if iso_dest != "ZZZ" and iso_dest in balance_dict:
            balance_dict_otro[iso_dest] += migrantes

# Diccionario ISO3 → valor
#value_dict = balance_dict

# ===============================
# 4) Preparar mapa
# ===============================

fig = plt.figure(figsize=(20,12))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.set_facecolor("#f0f0f0")

ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)

# Colormap verde → rojo
cmap = plt.cm.RdYlGn

# Normalización
vmin = min(min(balance_dict.values()), min(balance_dict_otro.values()))
vmax = max(max(balance_dict.values()), max(balance_dict_otro.values()))

from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdYlGn
# ===============================
# 5) Leer shapefile Natural Earth
# ===============================

shapefile = shpreader.natural_earth(
    resolution='110m',
    category='cultural',
    name='admin_0_countries'
)

reader = shpreader.Reader(shapefile)

# ===============================
# 6) Dibujar países
# ===============================

for country in reader.records():

    iso3 = country.attributes['ISO_A3']
    geom = country.geometry

    # Saltear códigos inválidos
    if iso3 == "-99":
        continue

    value = balance_dict.get(iso3)

    if value is not None:
        facecolor = cmap(norm(value))
    else:
        facecolor = "#d3d3d3"  # gris para países sin datos

    ax.add_geometries(
        [geom],
        ccrs.PlateCarree(),
        facecolor=facecolor,
        edgecolor="black",
        linewidth=0.2
    )

# ===============================
# 7) Barra de color
# ===============================

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
#cbar.set_label("Migrantes (1990)", fontsize=12)

#plt.title("Migrantes por país de origen (1990)", fontsize=18)
plt.savefig(
    f"resultados/balance_migratorio_{año}.png",
    dpi=300,                 # calidad alta para impresión
    bbox_inches="tight",     # elimina márgenes blancos
    pad_inches=0.1,
    facecolor=fig.get_facecolor()
)
print(f"Mapa guardado como: resultados/balance_migratorio_{año}.png")
#plt.show()