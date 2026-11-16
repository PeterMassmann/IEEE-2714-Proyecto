import numpy as np

from .interleave import estimar_interleave
from .parches import extraer_parches


def calcular_compatibilidades(parches):
    N = len(parches)
    H = np.empty((N, N), dtype=float)
    V = np.empty((N, N), dtype=float)

    izquierda = parches[:, :, 0, :].astype(float)
    derecha = parches[:, :, -1, :].astype(float)
    arriba = parches[:, 0, :, :].astype(float)
    abajo = parches[:, -1, :, :].astype(float)

    for i in range(N):
        H[i] = np.mean(np.abs(derecha[i] - izquierda), axis=(1, 2)) / 255
        V[i] = np.mean(np.abs(abajo[i] - arriba), axis=(1, 2)) / 255

    np.fill_diagonal(H, np.inf)
    np.fill_diagonal(V, np.inf)
    return H, V


def calcular_matrices(imagen, R, C):
    d = estimar_interleave(imagen, R, C)
    parches = extraer_parches(imagen, R, C, d)
    H, V = calcular_compatibilidades(parches)
    return H, V, d
