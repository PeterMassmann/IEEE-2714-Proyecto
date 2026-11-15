import numpy as np


def estimar_interleave(imagen, R, C):
    alto, ancho = imagen.shape[:2]
    alto_celda = alto // R
    ancho_celda = ancho // C
    filas = np.arange(alto) % alto_celda
    columnas = np.arange(ancho) % ancho_celda

    # Dado que las franjas son cero en todos sus canales, ~any devuelve 1 en las franjas
    filas_cero = ~np.any(imagen, axis=(1, 2))
    columnas_cero = ~np.any(imagen, axis=(0, 2))

    for d in (5, 3, 1):
        q = d // 2

        # Ej d=3: X o o o o o o o X X | X o o o o o o o X X | X o o o o o o X X | ...
        separacion_filas = (filas < q) | (filas >= alto_celda - q - 1)
        separacion_columnas = (columnas < q) | (columnas >= ancho_celda - q - 1)

        if np.all(filas_cero[separacion_filas]) and np.all(columnas_cero[separacion_columnas]):
            return d

    return 0
