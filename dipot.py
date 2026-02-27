"""
Este módulo está basado en la implementación publicada por
Alexandre Devert (2021) en:

https://gist.github.com/marmakoide/45d5389252683ae09c2df49d0548a627

El código original fue distribuido bajo licencia MIT.
Esta versión incluye modificaciones y extensiones.
"""

import itertools
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
import textwrap

from tqdm.notebook import tqdm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Colormap
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from shapely import minimum_bounding_radius, vectorized
from shapely.geometry import Polygon, Point, MultiPoint, box
from typing import Any, Tuple, TypeAlias


SegmentoVoronoi: TypeAlias = Tuple[
    np.ndarray,  # punto de partida
    np.ndarray,  # dirección normalizada
    int | float | None,  #  distancia desde el punto de partida al inicio del segmento (tmin) 
    int | float | None,  # distancia desde el punto de partida al final del segmento (tmax) 
    ]


class DiagramaDePotencia:
    """
    Implementa una construcción de diagramas de potencia, aprendiendo pesos para aproximar
    el tamaño de las celdas en función de las magnitudes objetivo. 
    Está pensada como herramienta de visualización aproximada para condiciones específicas y
    configuraciones razonables de sitios y dominio. El proceso de optimización se registra en
    un diario y se informa cuando no se alcanza el criterio de convergencia especificado.
    
    Recibe:
    - Dominio convexo como vértices de un polígono simple, sin auto-intersecciones, en [-1, 1] x [-1, 1].
    - Coordenadas únicas y bien distribuidas de los sitios en (-1, 1) x (-1, 1).
    - Magnitudes objetivo (tantas como sitios).
    - Nombres de las celdas (tantos como sitios).      
    """
    _dominio: np.ndarray # Vértices de un polígono     
    _poligono_dominio: Polygon
    _diametro_dominio: float
    _sitios: np.ndarray # Las coordenadas de los sitios proyectadas al dominio. Dimensiones: (|sitios|, 2)
    _nombres_celdas: np.ndarray # Las etiquetas que identifican cada celda    
    _capacidades: np.ndarray # La magnitud o capacidad asociada a cada celda
    _pesos: np.ndarray # Los pesos a aprender que definen el tamaño final de las celdas  
    _sitios_proyectados: np.ndarray # Dimensiones: (|sitios|, 3)
    _ids_caras_inferiores: tuple[list[int], ...] # Los índices de sitios que forman un triángulo en la envolvente de los sitios proyectados
    _vertices_de_celdas: np.ndarray # Dimensiones: (N, 2)
    _celdas_de_diagrama: dict[int, np.ndarray] # Dimensiones de cada arreglo: (|bordes_celda_i|, 2)
    _areas_de_celdas: dict[int, float] # El valor de área de cada celda luego de recortarlas contra el dominio
    _celdas_recortadas: dict[int, np.ndarray] # Vértices de las celdas recortadas
    _cumplio_criterio: bool # Señal para ordenar el gráfico luego de consturir el diagrama
    _diario: str # Guarda el valor de los errores relativos máximos calculados en cada iteración
    _paletas: dict[str, ListedColormap] 
    
    def __init__(
        self,
        dominio: np.ndarray | list[tuple[int | float, int | float]],        
        coordenadas_de_sitios: np.ndarray | list[tuple[int | float, int | float]],
        nombres_de_celdas: np.ndarray | list[str | int | float],
        magnitudes_objetivo: np.ndarray | list[int | float],   
    ) -> None:

        # Dominio          
        if self._validar_domino(dominio):
            self._dominio = self._dominio_en_sentido_ah(np.array(dominio))
            self._poligono_dominio = Polygon(self._dominio)
            self._diametro_dominio = self._calcular_diametro()
        else:            
            raise ValueError(
                'El dominio debe ser un polígono válido, '
                'con vértices en sentido antihorario en [-1,1]x[-1,1].'
            )
        
        # Sitios
        coords = np.array(coordenadas_de_sitios)
        if self._validar_sitios(coords):
            self._sitios = coords
        else:
            raise ValueError(
                'Los sitios deben ser 4 o más, deben estar dentro del rectángulo '
                'que contiene al dominio y no puede haber sitios repetidos.'
            )
        
        # Nombres de celdas        
        nom_celdas = self._validar_nombres_de_celdas(nombres_de_celdas)
        if len(nom_celdas) != self.num_sitios:            
            raise ValueError(
                'type(nombre_celda) -> int, float, str; |nombres_celdas| = |sitios|.'
            )                
        else:
            self._nombres_celdas = np.array(nom_celdas)

        # Magnitudes objetivos
        if self._validar_magnitudes(magnitudes_objetivo):
            self._capacidades = self._calcular_capacidades(np.array(magnitudes_objetivo))
        else:
            raise ValueError(
                'type(magnitud) -> int, float; |magnitudes_objetivo| = |sitios|; magnitud_i > 0.'
            )
        
        # Generados
        self._pesos = np.zeros(self.num_sitios)
        self._diario = ''     
        self._sitios_proyectados = np.array([])
        self._ids_caras_inferiores = ()
        self._vertices_de_celdas = np.array([])
        self._celdas_de_diagrama = dict()
        self._areas_de_celdas = dict()
        self._celdas_recortadas = dict()
        self._cumplio_criterio = False

        # Paletas propias
        self._paletas = {}
        lista_colores1= [ # Marrón
            '#d5d5d5', '#b4b596', '#c2ae6b', '#c29e6b', '#c28973',
            '#998872', '#9aa06d', '#9ab27b', '#b4c28a', '#b1cba6',
        ]
        lista_colores2 = [ # Verde
            '#d5d5d5', '#96b598', '#7bc26b', '#8bc26b', '#a9c273',
            '#829a73', '#6da075', '#79af93', '#8ac299', '#a6cbc1',
        ]
        lista_colores3 = [ # Roja
            '#d5d5d5', '#b5a396', '#c2786b', '#be696c', '#c2738e',
            '#9a7375', '#a0866d', '#b2a87b', '#c2ac8a', '#c8cba6',
        ]       
        self._paletas['Marrón'] = ListedColormap(lista_colores1, name='Marrón')
        self._paletas['Verde'] = ListedColormap(lista_colores2, name='Verde')
        self._paletas['Roja'] = ListedColormap(lista_colores3, name='Roja')

    
    def __repr__(self) -> str:
        return (
            f'DiagramaDePotencia(sitios={self.num_sitios})'
        )

    def __str__(self) -> str:
        return (
            f'DiagramaDePotencia: {self.num_sitios} sitios'
        )


    @property
    def num_sitios(self) -> int:
        return len(self._sitios)

    
    @property
    def area_dominio(self) -> float:
        return self._poligono_dominio.area

    
    def _validar_domino(
        self, 
        dominio: np.ndarray | list[tuple[int | float, int | float]]
    ) -> bool:
        """
        Evalúa las condiciones mínimas que debe cumplir el dominio.
        """
        poligono = Polygon(dominio)
        xmin, ymin, xmax, ymax = poligono.bounds
        en_intervalo = min(xmin, ymin) >= -1 and max(xmax, ymax) <= 1
        sin_agujeros = len(poligono.interiors) == 0
        return poligono.is_valid and en_intervalo and sin_agujeros

    
    def _dominio_en_sentido_ah(self, poligono: np.ndarray) -> np.ndarray:        
        """
        Devuelve los vértices del dominio en sentido antihorario, si no lo están.
        """
        x = poligono[:,0]
        y = poligono[:,1]
        area = 0.5 * np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))
        
        if area < 0:
            return poligono[::-1]
            
        return poligono


    def _calcular_diametro(self) -> float:
        """
        Calcula el diámetro del dominio. Ese valor se utiliza como magnitud 
        para los bordes de celdas infinitas.
        """
        return minimum_bounding_radius(self._poligono_dominio) * 2


    def _validar_sitios(self, coords: np.ndarray) -> bool:
        """
        Evalúa las condiciones mínimas que deben cumplir las coordenadas de los sitios.
        """
        xmin, ymin, xmax, ymax = self._poligono_dominio.bounds
        marco_dominio = box(xmin, ymin, xmax, ymax)
        estan_adentro = vectorized.contains(marco_dominio, coords[:,0], coords[:,1])
        hay_duplicadas = np.unique(coords, axis=0).shape[0] != coords.shape[0]
        return len(coords) > 3 and estan_adentro.all() and not hay_duplicadas

    
    def _validar_nombres_de_celdas(
        self, 
        nombres_de_celdas: np.ndarray | list[str | int | float]
    ) -> list[str]:        
        """
        Evalúa las condiciones mínimas que deben cumplir los nombres de las celdas.
        """
        try:
            return [
                str(elem) 
                for elem in nombres_de_celdas
                if isinstance(elem, (int, float, str))
            ]
        except :
            raise ValueError('nombres_de_celdas debe ser iterable.')


    def _validar_magnitudes(self, magnitudes: np.ndarray | list[int | float]) -> bool:
        """
        Evalúa las condiciones mínimas que deben cumplir las magnitudes objetivo.
        """
        son_numeros = all(
            isinstance(mag, numbers.Real) and not isinstance(mag, bool)
            for mag in magnitudes
        )
        long_correcta = len(magnitudes) == self.num_sitios
        son_positivas_no_nulas = bool(np.all(np.array(magnitudes) > 0))
        return son_numeros and long_correcta and son_positivas_no_nulas
         
        
    def _calcular_capacidades(self, magnitudes: np.ndarray) -> np.ndarray:  
        """
        Retorna el área que le corresponde a cada celda 
        según el valor de la magnitud objetivo asociada.
        """
        return (magnitudes * self.area_dominio) / magnitudes.sum() 
    

    # Triangulación regular
    def _levantamiento_parabolico(self) -> None:
        """
        Eleva los sitios originales en el plano a un parabolide en el espacio,
        desplazando cada punto según el peso correspondiente. 
        """
        z = np.sum(self._sitios**2, axis=1) - self._pesos        
        self._sitios_proyectados = np.column_stack([self._sitios, z])
        

    def _sentido_ah(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """
        Evalúa si los vértices del triángulo en el plano están en sentido antihorario.        
        """
        return np.cross(B - A, C - A) > 0
        
    
    def _caras_inferiores_de_envolvente(self) -> None:
        """
        Extrae los índices en sentido antihorario de los sitios que forman triángulos,
        con caras hacia abajo, en la envolvente convexa de los sitios proyectados.
        """
        S = self._sitios
        envolvente = ConvexHull(self._sitios_proyectados)
        
        self._ids_caras_inferiores = tuple(
            [int(a), int(b), int(c)] # Lista de índices que forman un triángulo en la envolvente
            if self._sentido_ah(S[a], S[b], S[c]) 
            else [int(a), int(c), int(b)]  
            for (a, b, c), implicita_del_plano in zip(envolvente.simplices, envolvente.equations) 
            if implicita_del_plano[2] < 0 # Si componente z de la normal del triángulo es negativa (mira hacia abajo)
        )


    
    # Construcción de celdas
    def _calcular_circuncentro_laguerre(
        self,
        A: np.ndarray, # (3,)
        B: np.ndarray, # (3,)
        C: np.ndarray # (3,)
    ) -> np.ndarray:
        """
        Calcula la proyección al plano XY del punto donde el plano 
        que contiene a los tres vértices de la cara inferior de la 
        envolvente convexa interseca con el parabolide.
        """
        vector_normal_a_cara = np.cross(A, B) + np.cross(B, C) + np.cross(C, A)
        return (-.5 / vector_normal_a_cara[2]) * vector_normal_a_cara[:2]
        
        
    def _vertices_de_celdas_en_el_plano(self) -> None:
        """
        Calcula los vértices de las celdas de Voronoi.
        """
        self._vertices_de_celdas = np.array([
            self._calcular_circuncentro_laguerre(*self._sitios_proyectados[cara])
            for cara in self._ids_caras_inferiores
        ])


    
    def _ordenar_vertices_antihorario(self, vertices: np.ndarray) -> np.ndarray:
        """
        Ordena un conjunto de vértices en sentido antihorario.
        """
        centroide = np.mean(vertices, axis=0)        
        vectores = vertices - centroide
        angulos = np.arctan2(vectores[:,1], vectores[:,0])
        indices_orden = np.argsort(angulos)
        vertices_ordenados = vertices[indices_orden]        
        return vertices_ordenados
        
   
    def _extraer_vertices(
        self,
        segmentos: list[tuple[tuple[int, int], SegmentoVoronoi]]
    ) -> np.ndarray:
        """
        Utiliza los parámetros de los segmentos para extraer los vértices de las celdas.
        """
        vertices: list[tuple[float, float]] = []
        # A = punto; U = dirección; tmin = 0|None; tmax = norma|None
        for lado, (A, U, tmin, tmax) in segmentos: 
            # Si los bordes son infinitos le asigno como magnitud el diámetro del dominio
            # para asegurarme que el vértice cae fuera del dominio (y poder recortar)
            if tmax is None:
                tmax = self._diametro_dominio
            if tmin is None:
                tmin = -self._diametro_dominio
            v1 = A + tmin * U
            v2 = A + tmax * U

            vertices.append(tuple(np.round(v1, 8)))
            vertices.append(tuple(np.round(v2, 8)))
    
        # Elimino duplicados manteniendo el orden
        procesados = set()
        vertices_sin_duplicados = []
        for v in vertices:
            if v not in procesados:
                vertices_sin_duplicados.append(v)
                procesados.add(v)
    
        arr_vertices_sin_duplicados = np.array(vertices_sin_duplicados)
    
        return self._ordenar_vertices_antihorario(arr_vertices_sin_duplicados)


    
    def _ordenar_segmentos(
        self, 
        segmentos: list[tuple[tuple[int, int], SegmentoVoronoi]]
    ) -> list[tuple[tuple[int, int], SegmentoVoronoi]]:
        """
        Ordena los bordes para que formen un ciclo continuo.
        """
        if not segmentos:
            return []
            
        primero = min((seg[0][0], i) for i, seg in enumerate(segmentos))[1]

        segmentos[0], segmentos[primero] = segmentos[primero], segmentos[0]
        for i in range(len(segmentos) -1): #
            for j in range(i + 1, len(segmentos)):
                if segmentos[i][0][1] == segmentos[j][0][0]:
                    segmentos[i+1], segmentos[j] = segmentos[j], segmentos[i+1]
                    break
        return segmentos

    
    def _celdas_voronoi(self) -> None:
        """
        Calcula los vértices que definen cada una de las celdas de Voronoi.
        """
        conj_vertices_triangs = frozenset(itertools.chain(*self._ids_caras_inferiores))

        # Qué triángulos comparten cada lado (par de vértices)
        # lados_de_triangs: dict[tuple[int, int], list[int]] = {} 
        lados_de_triangs: dict[tuple[int, ...], list[int]] = {}
        for i, triang in enumerate(self._ids_caras_inferiores):
            for lado in itertools.combinations(triang, 2): # Combinaciones posibles de vértices
                lado_ordenado = tuple(sorted(lado)) # Ordenadas de menor a mayor
                if lado_ordenado in lados_de_triangs:
                    lados_de_triangs[lado_ordenado].append(i)
                else:
                    lados_de_triangs[lado_ordenado] = [i]


        celdas_de_voronoi: dict[int, list[tuple[tuple[int, int], SegmentoVoronoi]]] = {
            id_vertice: [] for id_vertice in conj_vertices_triangs
        }
        # Para cada grupo de indices de sitios que forman un triángulo en la envolvente
        for i, (a, b, c) in enumerate(self._ids_caras_inferiores):
            # Para cada lado del triángulo
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)): 
                borde = tuple(sorted((u, v))) # Ordenados de menora a mayor
                if len(lados_de_triangs[borde]) == 2: # Lado finito: es interior, lo comparten dos triángulos 
                    j, k = lados_de_triangs[borde] # Los triángulos j, k comparten el lado
                    if k == i:
                        j, k = k, j                        
                    # Parámetros del segmento
                    U = self._vertices_de_celdas[k] - self._vertices_de_celdas[j] # Segmento entre sus circuncentros
                    norma_de_U = float(np.linalg.norm(U)) # Longitud del segmento                    
                    celdas_de_voronoi[u].append( 
                        # ((caras que lo comparten), (origen, dirección, -, tamaño_segmento) 
                        ((j, k), (self._vertices_de_celdas[j], U / norma_de_U, 0, norma_de_U))
                    )
                    
                else: # Lado infinito: el lado pertenece a un triángulo (está en el borde)
                    # Parámetros del segmento
                    # Sitios y circuncentro de Laguerre del triángulo que forman
                    A, B, C, D = self._sitios[u], self._sitios[v], self._sitios[w], self._vertices_de_celdas[i]
                    U = (B - A) / np.linalg.norm(B - A) # Dirección del lado del triángulo
                    I = A + np.dot(D - A, U) * U # Proyección del circuncentro sobre el lado
                    W = (I - D) / np.linalg.norm(I - D) # Dirección perpendicular hacia afuera
                    if np.dot(W, I - C) < 0:
                        W = -W # Orientación: hacia afuera del dominio
                    celdas_de_voronoi[u].append(((lados_de_triangs[borde][0], -1), (D, W, 0, None)))
                    celdas_de_voronoi[v].append(((-1, lados_de_triangs[borde][0]), (D, -W, None, 0)))
        
        celdas_de_voronoi = (
            {
                i: self._ordenar_segmentos(segmentos) 
                for i, segmentos in celdas_de_voronoi.items()
            }
        )
        self._celdas_de_diagrama = (
            {
                int(i): self._extraer_vertices(segmentos) 
                for i, segmentos in celdas_de_voronoi.items()
            }
        )


    
    # Recorte contra el dominio
    def _vertice_adentro(
        self, 
        vert_celda, 
        vert_A_dom, 
        vert_B_dom
    ) -> bool:
        """
        Evalúa si un vértice de la celda está dentro del semiplano
        definido por el lado [vert_A_dom --> vert_B_dom] del dominio.
        Requiere que el dominio esté orientado en sentido antihorario.
        """
        return np.cross(vert_B_dom - vert_A_dom, vert_celda - vert_A_dom) >= 0

    def _intersectar_segmento(
        self, 
        vert_P_celda, 
        vert_Q_celda, 
        vert_A_dom, 
        vert_B_dom
    ) -> np.ndarray:
        """
        Calcula la interseicción entre el borde [P --> Q] de la celda
        y el lado [vert_A_dom --> vert_B_dom] del dominio.
        """
        d = vert_Q_celda - vert_P_celda
        e = vert_B_dom - vert_A_dom
        denom = np.cross(d, e)
        
        # Si el denoinador es cercano a cero devuvlo P
        if abs(denom) < 1e-14: 
            return vert_P_celda

        t = np.cross(vert_A_dom - vert_P_celda, e) / denom

        return vert_P_celda + t*d


    
    def _arreglar_celda(self, vertices) -> np.ndarray:
        """
        Devuelve las coordenadas de la envolvente de la celda recortada,
        orientada en sentido antihorario y sin repetir el último/primer punto.
        """
        vertices = np.array(vertices)
        envolvente = MultiPoint(vertices).convex_hull
        if not envolvente.exterior.is_ccw:
            envolvente = Polygon(list(envolvente.exterior.coords)[::-1])
        return np.array(envolvente.exterior.coords[:-1])


    
    def _recortar_con_semiplano(self, celda, vert_A_dom, vert_B_dom) -> np.ndarray:
        """
        Recorta la celda contra el semiplano definido por el lado 
        [vert_A_dom --> vert_B_dom] del dominio.
        """
        if len(celda) == 0:
            return celda

        recorte = []

        for i in range(len(celda)):
            vert_Q_celda = celda[i]
            vert_P_celda = celda[i-1]

            vert_P_esta_dentro = self._vertice_adentro(
                vert_P_celda, vert_A_dom, vert_B_dom
            )

            vert_Q_esta_dentro = self._vertice_adentro(
                vert_Q_celda, vert_A_dom, vert_B_dom
            )

            if vert_P_esta_dentro:
                if not vert_Q_esta_dentro:
                    recorte.append(
                        self._intersectar_segmento(
                            vert_P_celda, vert_Q_celda, vert_A_dom, vert_B_dom
                        )
                    )
                recorte.append(vert_P_celda)
            
            elif vert_Q_esta_dentro:
                recorte.append(
                    self._intersectar_segmento(
                        vert_P_celda, vert_Q_celda, vert_A_dom, vert_B_dom
                    )
                )
        if len(recorte) == 0:
            return np.empty((0,2))

        celda_arreglada = self._arreglar_celda(recorte)

        return np.array(celda_arreglada)
            

    def _recortar_contra_dominio(self, celda) -> np.ndarray:
        """
        Recorta una celda contra el polígono completo del dominio 
        mediante el algoritmo de Sutherland-Hodgman.
        """
        res = celda.copy()
        dom = self._dominio
        num_vertices_dom = len(dom)
        for i in range(num_vertices_dom):
            vertice_A = dom[i]
            vertice_B = dom[(i + 1) % num_vertices_dom]
            
            res = self._recortar_con_semiplano(res, vertice_A, vertice_B)
            
            if len(res) == 0:
                break
        return res

    
    def _area_de_poligono(self, celda) -> float:
        """
        Devuelve el área real de la celda recortada contra el dominio.
        """
        if len(celda) < 3:
            return 0.0
        x = celda[:,0]
        y = celda[:,1]
        return .5 * abs(
            np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        )
        
    
    def _calcular_area_de_celdas(self) -> None:
        """
        Recorta las celdas contra el dominio, calcula sus áreas
        y almacena los vértices de las celdas recortadas.
        """
        areas = {}
        celdas_recortadas = {} 
        for i, celda in self._celdas_de_diagrama.items():
            celda_recortada = self._recortar_contra_dominio(celda)
            celdas_recortadas[int(i)] = celda_recortada
            areas[int(i)] = self._area_de_poligono(celda_recortada)
        self._areas_de_celdas = areas
        self._celdas_recortadas = celdas_recortadas

        

    
    def _vecinos_desde_triangulacion(self) -> list[tuple[int, ...]]:
        """
        Encuentra los sitios que comparten una arista en el diagrama de Voronoi.
        """
        vecinos = set()    
        for triang in self._ids_caras_inferiores:
            a, b, c = triang
            for i, j in ((a, b), (b, c), (c, a)):
                if i != j: # En gral. es True, pero por si ocurre algo raro al levantar
                    vecinos.add(tuple(sorted((i, j))))

        return list(vecinos) # Pares de indices vecinos
    
    
    def _calcular_jacobiano(self):
        """
        Construye el jacobiano de las áreas respecto de los pesos. 
        Las derivadas parciales se calculan con la fórmula:        
        
        ∂A_i / ∂w_j =  L_ij / (2 * ||x_i - x_j||)     si i y j son vecinos
                      0                               en otro caso
        
        donde:
        - L_ij es la longitud del borde común entre las celdas i y j,
        - x_i, x_j son las posiciones de los sitios.
        """
        J = np.zeros((self.num_sitios, self.num_sitios))

        # Pares (i,j) de índices que comparten un borde
        vecinos = self._vecinos_desde_triangulacion()
        for i, j in vecinos:
            celda_i = self._celdas_recortadas[i]
            celda_j = self._celdas_recortadas[j]
    
            puntos_comunes = []
            # Para cada vértice de la celda i
            for p in celda_i:
                # Para cada vértice de la celda j
                for q in celda_j:
                    if np.linalg.norm(p - q) < 1e-9:
                        puntos_comunes.append(p)  
            
            if len(puntos_comunes) == 2: # Borde común
                L = np.linalg.norm(puntos_comunes[0] - puntos_comunes[1]) # Longitud del borde comùn
                d = np.linalg.norm(self._sitios[i] - self._sitios[j]) # ||x_i - x_j||
                coef = L / (2 * d) # ∂A_i / ∂w_j 
                J[i, j] -= coef
                J[j, i] -= coef
                J[i, i] += coef
                J[j, j] += coef
    
        return J
    
       

    def construir_diagrama(
        self,
        tamaño_del_paso = .1, # Más grande, más rápido converge, pero puede romperse si el paso es grande (probar y mirar el self._diario) 
        max_iteraciones: int = 500,
        umbral_estancamiento: int = 10, # Salir del bucle si el error no mejora o empeora después de {umbral_estancamiento} iteraciones
        error_rel_max_permitido: float = 1e-6, # Criterio: todos las celdas deben tener un error relativo menor 
        imprimir_progreso: bool = False,
        barra: tqdm | None = None, # Barra de progreso 
    ) -> None:
        """
        Aprende los pesos para cumplir con el criterio de convergencia y
        construye el diagrama de potencia.
        """        
        # Esta variable se actualiza en cada iteración
        # El valor incial tiene que ser mayor que el primero que se calcula
        # para que no entre a buscar apenas arranca
        error_rel_max_anterior = 1e12 # Altísimo por las dudas 
        iteracion = 0
        conteo_estancamiento = 0
        self._diario = ''      
        
        # Respaldo para poder explorar cuando se estanca y si no mejroa,
        # devolver la mejor aproximación alcanzada
        pesos_ultima_mejor_aproximacion = self._pesos.copy()
        posible_mejor_error = 1e12
        
        if barra is not None:        
            barra.n = 0
            barra.refresh()

        # Bucle principal: construye el diagrama, calcula áreas y error
        # si mejora avanza, sino busco un paso que mejore el error en un bucle interno
        while iteracion < max_iteraciones:

            # Salgo si se estancó
            if conteo_estancamiento >= umbral_estancamiento:                
                # Restauro los mejores pesos encontrados y recalculo 
                self._pesos = pesos_ultima_mejor_aproximacion
                self._levantamiento_parabolico()
                self._caras_inferiores_de_envolvente()            
                self._vertices_de_celdas_en_el_plano()
                self._celdas_voronoi()
                self._calcular_area_de_celdas()                 
                # Actualizo la barra, imprimo el error alcanzado y retorno
                if barra is not None:        
                    barra.n = 100
                    barra.refresh()                    
                print(f'Error relativo máximo alcanzado: {100 * posible_mejor_error}%.')
                return
            
            # Operaciones para construir el diagrama de potencia
            # Triangulación regular
            self._levantamiento_parabolico()
            self._caras_inferiores_de_envolvente()            
            # Construcción de celdas          
            self._vertices_de_celdas_en_el_plano()
            self._celdas_voronoi()                     
            # Recorte de celdas contra el dominio
            self._calcular_area_de_celdas()    
            
            # Calculo el error máximo reltivo                      
            areas_actuales = np.zeros(self.num_sitios)
            for i, area in self._areas_de_celdas.items():
                areas_actuales[i] = area         
            residuos = areas_actuales - self._capacidades
            errores_rel = np.abs(residuos) / self._capacidades
            error_rel_maximo_actual = np.max(errores_rel)
            
            # Registro el progreso
            datos_progreso = (
                f'Iteración {iteracion}: Err. rel. max.: {100 * error_rel_maximo_actual:.5f}%.'
            )
            self._diario += f'{datos_progreso}\n'
            if imprimir_progreso: 
                print(datos_progreso)
            
            # Corto cuando todas las celdas cumplen con el criterio
            if error_rel_maximo_actual < error_rel_max_permitido:
                self._cumplio_criterio = True
                if barra is not None:        
                    barra.n = 100
                    barra.refresh()
                break                           

            # Actualización de pesos
            J = self._calcular_jacobiano() 
            J_reducido = J[1:, 1:] # Los demás pesos se resuelven en función del primero
            residuos_reducido = residuos[1:]            
            try:
                direccion_de_avance = np.linalg.solve(J_reducido, -residuos_reducido)
            except: # Si la matriz se volvió singular
                direccion_de_avance, *_ = np.linalg.lstsq(J_reducido, -residuos_reducido)   

            # Si mejoro continuo con la magnitud de avance recibida           
            if error_rel_maximo_actual < error_rel_max_anterior:                
                # Respaldo si no mejora
                pesos_ultima_mejor_aproximacion = self._pesos.copy()
                posible_mejor_error = error_rel_maximo_actual 
                # Actualizo los pesos e itero con lso nuevos valores para seguir mejorando
                self._pesos[1:] += tamaño_del_paso * direccion_de_avance
                self._pesos -= np.mean(self._pesos)
                error_rel_max_anterior = error_rel_maximo_actual                
                conteo_estancamiento = 0 

            # Si no avanza busco un paso que mejore
            else:    
                # Suavizo apenas el tamaño del paso original y empiezo a buscar
                paso_mas_corto = tamaño_del_paso * .99
                # Respaldo para poder verificar si mejora
                pesos_originales = self._pesos.copy()

                # Si mejoró acepto los pesos. Si no mejora y el paso es ínfimo corto la búsqeuda 
                # En ese caso, los pesos pueden haber empeorado o estar estancados 
                while (error_rel_maximo_actual >= error_rel_max_anterior and paso_mas_corto >= 5e-4):
                    # Pruebo con un paso un poco menor
                    self._pesos[1:] = pesos_originales[1:] + paso_mas_corto * direccion_de_avance
                    self._pesos -= np.mean(self._pesos)                    
                    # Recalculo areas
                    self._levantamiento_parabolico()
                    self._caras_inferiores_de_envolvente()            
                    self._vertices_de_celdas_en_el_plano()
                    self._celdas_voronoi()
                    self._calcular_area_de_celdas()
                    # Recalculo error
                    areas_nuevas = np.zeros(self.num_sitios)
                    for i, area in self._areas_de_celdas.items():
                        areas_nuevas[i] = area   
                    residuos_nuevo = areas_nuevas - self._capacidades
                    errores_rel_nuevo = np.abs(residuos_nuevo) / self._capacidades                    
                    error_rel_maximo_actual = np.max(errores_rel_nuevo)
                                       
                    # Si no mejoró disminuyo el paso a la mitad y pruebo de nuevo
                    paso_mas_corto *= 0.5

                    # Si los pesos calculados disminuyeron el error, reemplazo a los posibles mejores
                    if error_rel_maximo_actual < posible_mejor_error:                        
                        # Respaldo si no mejora
                        pesos_ultima_mejor_aproximacion = self._pesos.copy()
                        posible_mejor_error = error_rel_maximo_actual 
                    
                # Si la mejora es ínfima
                if error_rel_max_anterior - error_rel_maximo_actual < 1e-5: 
                    # Empiezo el conteo de estancamiento
                    conteo_estancamiento += 1
                # Acutalizo el error antes de volver a empezaar
                error_rel_max_anterior = error_rel_maximo_actual
            
            if barra is not None:        
                barra.n = (1 / (error_rel_maximo_actual / error_rel_max_permitido)) * 100
                barra.refresh()

            iteracion += 1

            


    def _configurar_paleta(self, paleta: Any) -> ListedColormap | Colormap:
        """
        Establece la paleta a utilizar.
        """
        if isinstance(paleta, ListedColormap):
            return paleta
        # Si la cadena coincide con alguna de las propias ('Marrón', 'Verde', 'Roja')
        paleta_defecto = cm.get_cmap('tab20c')
        return self._paletas.get(paleta, paleta_defecto)
        
        
    def _graficar_diagrama(
        self,         
        ancho_tam_fig: int = 10,
        alto_tam_fig: int = 10,
        dpi: int = 300,    
        titulo: str | None = None,
        nota_al_pie: str | None = None,
        num_caracteres: int | None = None,
        ruta_salida: str | None = None,
        eje: Axes | None = None, # Para componer con otros gráficos
        **args,
    ) -> Figure | None:
        """
        Grafica el diagrama sobre una nueva figura o sobre el eje recibido.
        """
        
        # Cuando no se cumplió el criterio
        if not self._cumplio_criterio:
            ids_celdas_vacias = [c for c,v in self._celdas_recortadas.items() if v.shape[0] == 0]
            nombres_de_celdas_vacias = self._nombres_celdas[ids_celdas_vacias]
            for celda in ids_celdas_vacias:
                del self._celdas_recortadas[celda]       
            self._nombres_celdas = np.delete(self._nombres_celdas, ids_celdas_vacias)
            if len(ids_celdas_vacias) > 0:
                print(
                    f'Diagrama incompleto. La(s) celda(s) {nombres_de_celdas_vacias} '
                    'no tiene(n) área asignada. Modifique el dominio o distribuya los sitios.'
                )    

        # Configuración gral.
        color_de_fondo = args.get('color_de_fondo', '#ffffff')
        eje_visible = args.get('eje_visible', False)
        factor_margen_perim = args.get('factor_margen_perim', 1.1)
        
        # Nombres de celdas
        color_nombres_de_celdas = args.get('color_nombres_de_celdas', '#312a2a')
        alfa_nombres_de_celdas = args.get('alfa_nombres_de_celdas', 1)
        tam_min_nombre_celda = args.get('tam_min_nombre_celda', 12) # Tamaño de texto para todas iguales (ccuando factor_aumento = 0)
        factor_aumento = args.get('factor_aumento', 5) # Para nombres de igual tamaño pasar 0
        config_nombres_celdas: dict[str, Any] = {'ha': 'center', 'va': 'center', 'weight': 'bold'}
        
        # Celdas
        paleta_celdas = args.get('paleta_celdas', cm.tab20c) # type: ignore[attr-defined]
        paleta_celdas = self._configurar_paleta(paleta_celdas)        
        colores_celdas = paleta_celdas(np.linspace(0, 1, self.num_sitios))
        alfa_relleno_celdas = args.get('alfa_relleno_celdas', 1.0)
        color_borde_celdas = args.get('color_borde_celdas', '#312a2a')
        grosor_borde_celdas = args.get('grosor_borde_celdas', 1.7)
        alfa_borde_celdas = args.get('alfa_borde_celdas', 1)
        
        # Titulo
        color_titulo = args.get('color_titulo', '#000000')
        tam_titulo = args.get('tam_titulo', 27)
        margen_titulo = args.get('margen_titulo', 5)
        config_titulo = {'fontweight': 'bold', 'alpha': 1.0}

        # Nota al pie
        color_nota_pie = args.get('color_nota_pie', '#000000')
        tam_nota_pie = args.get('tam_nota_pie', 15)
        x_nota_pie = args.get('x_nota_pie', 0.05)
        y_nota_pie = args.get('y_nota_pie', 0.05)
        config_nota_pie: dict[str, Any] = {'ha': 'left', 'va': 'bottom', 'alpha': 1.0}


        if eje is None:
            fig, eje_diagrama = plt.subplots(
                figsize=(ancho_tam_fig, alto_tam_fig), 
                facecolor=color_de_fondo, 
                dpi=dpi
            )
        else:
            eje_diagrama = eje

            
        for i, celda in enumerate(self._celdas_recortadas.values()): # type: ignore

            poligono = Polygon(celda)
            centroide = poligono.centroid
            
            # Celdas
            x, y = poligono.exterior.xy
            eje_diagrama.plot( # Bordes
                x, y, 
                color=color_borde_celdas,
                alpha=alfa_borde_celdas,
                linewidth=grosor_borde_celdas,
            )
            eje_diagrama.fill( # Rellenos
                x, y, 
                alpha=alfa_relleno_celdas, 
                facecolor=colores_celdas[i % len(colores_celdas)]
            )
                
            x_etq, y_etq = centroide.x, centroide.y
            eje_diagrama.text(
                x_etq,
                y_etq,
                self._nombres_celdas[i],
                color=color_nombres_de_celdas,
                alpha=alfa_nombres_de_celdas,
                fontsize=tam_min_nombre_celda * (1 + (factor_aumento * poligono.area)),
                **config_nombres_celdas
            )

        
        # Propiedades del eje
        eje_diagrama.set_facecolor(color_de_fondo)
        
        xmin, xmax = np.min(self._dominio[:,0]), np.max(self._dominio[:,0])
        ymin, ymax = np.min(self._dominio[:,1]), np.max(self._dominio[:,1])
        eje_diagrama.set_xlim((xmin * factor_margen_perim, xmax * factor_margen_perim))
        eje_diagrama.set_ylim((ymin * factor_margen_perim, ymax * factor_margen_perim))
        
        if not eje_visible:
            eje_diagrama.axis('off')

        # Titulo y nota al pie
        if titulo:
            eje_diagrama.set_title(
                titulo, 
                fontsize=tam_titulo, 
                color=color_titulo,
                pad=margen_titulo,
            )
        if nota_al_pie:
            if not num_caracteres:
                raise ValueError('Se debe indicar el número de caracteres por línea.') 
            # import textwrap
            nota_al_pie = textwrap.fill(nota_al_pie, width=num_caracteres)
            eje_diagrama.text(
                xmin + x_nota_pie, 
                ymin - y_nota_pie,
                nota_al_pie, 
                fontsize=tam_nota_pie,
                color=color_nota_pie,
                **config_nota_pie
            )                  
        
        if eje is None:            
            fig.tight_layout()
            if ruta_salida:
                plt.savefig(ruta_salida, bbox_inches='tight', dpi=dpi)
            plt.close()
            return fig
        else:
            return None