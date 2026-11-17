import argparse

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def randomizar_parches(imagen, R, C, d, semilla):
    if imagen.ndim != 3 or imagen.shape[2] != 3 or imagen.dtype != np.uint8:
        raise ValueError("La imagen debe ser RGB de 8 bits")
    if R <= 0 or C <= 0 or d not in (0, 1, 3, 5):
        raise ValueError("R, C o d no son validos")

    alto, ancho = imagen.shape[:2]
    alto_celda = alto // R
    ancho_celda = ancho // C
    alto_parche = alto_celda - d
    ancho_parche = ancho_celda - d

    if not (64 <= alto_parche <= 512 and 64 <= ancho_parche <= 512):
        raise ValueError("Cada lado util del parche debe medir entre 64 y 512 pixeles")

    inicio_y = (alto - R * alto_celda) // 2
    inicio_x = (ancho - C * ancho_celda) // 2
    recorte = imagen[inicio_y:inicio_y + R * alto_celda,
                     inicio_x:inicio_x + C * ancho_celda]

    mosaico = np.zeros_like(recorte)
    P_GT = np.empty(R * C, dtype=int)
    orden = np.random.default_rng(semilla).permutation(R * C)
    q = d // 2

    for k_entrada, k_original in enumerate(orden):
        r_original, c_original = divmod(k_original, C)
        r_entrada, c_entrada = divmod(k_entrada, C)

        y_original = r_original * alto_celda + q
        x_original = c_original * ancho_celda + q
        y_entrada = r_entrada * alto_celda + q
        x_entrada = c_entrada * ancho_celda + q

        mosaico[y_entrada:y_entrada + alto_parche,
                x_entrada:x_entrada + ancho_parche] = recorte[
                    y_original:y_original + alto_parche,
                    x_original:x_original + ancho_parche]
        P_GT[k_original] = k_entrada

    return mosaico, P_GT.reshape(R, C)


def main():
    # Comando: python randomizacion.py imagen.png R C d semilla mosaico.png ground_truth.npy

    parser = argparse.ArgumentParser(description="Genera un mosaico de parches mezclados")
    parser.add_argument("imagen", help="Imagen de entrada RGB")
    parser.add_argument("R", type=int, help="Filas de parches")
    parser.add_argument("C", type=int, help="Columnas de parches")
    parser.add_argument("d", type=int, help="Interleave: 0, 1, 3 o 5")
    parser.add_argument("semilla", type=int, help="Semilla aleatoria")
    parser.add_argument("mosaico", help="Archivo de salida .png")
    parser.add_argument("ground_truth", help="Archivo de salida .npy")
    args = parser.parse_args()

    if not args.mosaico.lower().endswith(".png"):
        raise ValueError("El mosaico debe guardarse como PNG")
    if not args.ground_truth.lower().endswith(".npy"):
        raise ValueError("El ground truth debe guardarse como NPY")

    with Image.open(args.imagen) as archivo:
        if archivo.mode != "RGB":
            raise ValueError("La imagen de entrada debe ser RGB de 8 bits")
        imagen = np.asarray(archivo)

    mosaico, P_GT = randomizar_parches(imagen, args.R, args.C, args.d, args.semilla)
    Image.fromarray(mosaico).save(args.mosaico)
    np.save(args.ground_truth, P_GT)

    # plot both images side by side

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen)
    plt.title("Imagen original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mosaico)
    plt.title("Mosaico de parches mezclados")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
