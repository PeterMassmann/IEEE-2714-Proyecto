import numpy as np


def extraer_parches(imagen, R, C, d):
    alto, ancho = imagen.shape[:2]
    alto_celda = alto // R
    ancho_celda = ancho // C
    alto_parche = alto_celda - d
    ancho_parche = ancho_celda - d
    q = d // 2

    parches = np.empty((R * C, alto_parche, ancho_parche, 3), dtype=imagen.dtype)

    for r in range(R):
        for c in range(C):
            y = r * alto_celda + q
            x = c * ancho_celda + q
            parches[r * C + c] = imagen[y:y + alto_parche, x:x + ancho_parche]

    return parches
