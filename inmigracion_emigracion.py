import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def graficar_top20(diccionario):

    # ===============================
    # 1) Convertir diccionario a lista ordenada
    # ===============================

    datos_lista = []

    for iso3, valores in diccionario.items():

        total_2024 = abs(valores["inmigracion_2024"] + valores["emigracion_2024"]-valores["inmigracion_1990"] - valores["emigracion_1990"])

        datos_lista.append({
            "iso3": iso3,
            "nombre": valores["nombre"],
            "inm_1990": valores["inmigracion_1990"],
            "emi_1990": valores["emigracion_1990"],
            "inm_2024": valores["inmigracion_2024"],
            "emi_2024": valores["emigracion_2024"],
            "total_2024": total_2024
        })

    # Ordenar por total 2024 descendente
    datos_lista = sorted(datos_lista, key=lambda x: x["total_2024"], reverse=True)

    # Tomar top 20
    datos_lista = datos_lista[:20]

    # ===============================
    # 2) Preparar datos para gráfico
    # ===============================

    nombres = [d["nombre"] for d in datos_lista]

    inm_1990 = [d["inm_1990"] for d in datos_lista]
    emi_1990 = [d["emi_1990"] for d in datos_lista]

    inm_2024 = [d["inm_2024"] for d in datos_lista]
    emi_2024 = [d["emi_2024"] for d in datos_lista]

    y = np.arange(len(nombres))
    altura = 0.35

    # ===============================
    # 3) Crear figura
    # ===============================

    fig, ax = plt.subplots(figsize=(14, 12))

    # Barras 2024 (arriba)
    ax.barh(y + altura/2, inm_2024, height=altura, color="#4C84C4", label="Inmigración 2024")
    ax.barh(y + altura/2, emi_2024, height=altura, color="#F47C4C", left=inm_2024, label="Emigración 2024")

    # Barras 1990 (abajo)
    ax.barh(y - altura/2, inm_1990, height=altura, color="#4C84C4", alpha=0.5, label="Inmigración 1990")
    ax.barh(y - altura/2, emi_1990, height=altura, color="#F47C4C", alpha=0.5, left=inm_1990, label="Emigración 1990")

    ax.set_yticks(y)
    ax.set_yticklabels(nombres)

    ax.set_xlabel("Número de personas")
    #ax.set_title("Inmigración y Emigración — Top 20 países (1990 y 2024)", fontsize=16)

    ax.legend(loc="lower right")

    ax.invert_yaxis()

    plt.tight_layout()

    plt.savefig("resultados/top20_inmigracion_emigracion_alt.png", dpi=300, bbox_inches="tight")
    print("Gráfico de barras guardado como: resultados/top20_inmigracion_emigracion.png")
    plt.show()


df = pd.read_csv("fuentes-de-datos/migras_90_24.csv")
año = 2024

# ===============================
# 1) Inicializar diccionario con 0
# ===============================

# Tomamos todos los ISO3 que aparezcan como origen o destino
paises = set(df['iso3_orig'].dropna()) | set(df['iso3_des'].dropna())

# Eliminar código inválido
paises.discard("ZZZ")

inmi_emi = {pais: {
    "inmigracion_1990": 0,
    "emigracion_1990": 0,
    "inmigracion_2024": 0,
    "emigracion_2024": 0,
    "nombre": ""
} for pais in paises}

# ===============================
# 2) Recorrer dataframe
# ===============================

for _, row in df.iterrows():
    if row["iso3_orig"] in inmi_emi:
        if inmi_emi[row["iso3_orig"]]["nombre"] == "":
            inmi_emi[row["iso3_orig"]]["nombre"] = row["origen_ES"]
        if row['año'] == 1990:
            inmi_emi[row["iso3_orig"]]["emigracion_1990"] += row["migrantes"]
            inmi_emi[row["iso3_des"]]["inmigracion_1990"] += row["migrantes"]
        elif row['año'] == 2024:
            inmi_emi[row["iso3_orig"]]["emigracion_2024"] += row["migrantes"]
            inmi_emi[row["iso3_des"]]["inmigracion_2024"] += row["migrantes"]

graficar_top20(inmi_emi)