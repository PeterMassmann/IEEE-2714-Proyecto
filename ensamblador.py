from compatibilidad import calcular_matrices
from parches import extraer_parches
import numpy as np
import cv2
import matplotlib.pyplot as plt

def ensamblador(imagen, R, C, rs, cs, rt, ct):
    H, V, d = calcular_matrices(imagen, R, C)
    parches = extraer_parches(imagen, R, C, d)

    new_image = np.full((R, C), -1) # la imagen parte vacia

    k_anclaje = rs * C + cs

    # se rellena la primera imagen
    new_image[rt,ct] = k_anclaje

    # checkear der
    i = new_image[rt,ct]
    if (ct + 1 < C) and (new_image[rt, ct + 1] == -1):
        costos_der = H[i, :].copy() # revisa todos los posibles valores
        for p in new_image.flatten(): # evitar que se repitan
            if p != -1:
                costos_der[p] = np.inf
        mejor_promedio = np.argmin(costos_der)
        new_image[rt, ct + 1] = mejor_promedio

    # checkear izq
    if (ct - 1 >= 0) and (new_image[rt, ct - 1] == -1):
        costos_izq = H[:, i].copy()
        for p in new_image.flatten():
            if p != -1:
                costos_izq[p] = np.inf
        mejor_promedio = np.argmin(costos_izq)
        new_image[rt, ct - 1] = mejor_promedio

    # checkear arriba
    if (rt - 1 >= 0) and (new_image[rt - 1, ct] == -1):
        costos_arr = V[:, i].copy()
        for p in new_image.flatten():
            if p != -1:
                costos_arr[p] = np.inf
        mejor_promedio = np.argmin(costos_arr)
        new_image[rt - 1, ct] = mejor_promedio
        

    # checkear abajo
    if (rt + 1 < R) and (new_image[rt + 1, ct] == -1):
        costos_ab = V[i, :].copy() 
        for p in new_image.flatten():
            if p != -1:
                costos_ab[p] = np.inf
        mejor_promedio = np.argmin(costos_ab)
        new_image[rt + 1, ct] = mejor_promedio
        

    return new_image


imagen_mosaico = cv2.imread("mosaico.png") # cambiar dependiendo del archivo
imagen_mosaico = cv2.cvtColor(imagen_mosaico, cv2.COLOR_BGR2RGB)
# Esto es entregado, cambiar eventualmente
R, C = 9, 9
rs, cs = 0, 0
rt, ct = 3, 3





def visualizar_avance(new_image, parches, d):
    
    R, C = new_image.shape
    p_h, p_w = parches.shape[1], parches.shape[2]
    
    # reconstuir el tama;o a partir de d 
    s_h = p_h + d
    s_w = p_w + d
    q = d // 2
    
    H = R * s_h
    W = C * s_w
    
    # matriz en blanco 
    canvas = np.zeros((H, W, 3), dtype=parches.dtype)
    
    for r in range(R):
        for c in range(C):
            k = new_image[r, c]
            if k != -1:  # Si la celda tiene una pieza asignada
                y_dest = r * s_h + q
                x_dest = c * s_w + q
                canvas[y_dest : y_dest + p_h, x_dest : x_dest + p_w] = parches[k]

    plt.figure(figsize=(15,8))
    plt.imshow(canvas)
    plt.show()
            
                
    
matriz_resultado = ensamblador(imagen_mosaico, R, C, rs, cs, rt, ct)
    

H, V, d = calcular_matrices(imagen_mosaico, R, C)
parches = extraer_parches(imagen_mosaico, R, C, d)

visualizar_avance(matriz_resultado, parches, d)