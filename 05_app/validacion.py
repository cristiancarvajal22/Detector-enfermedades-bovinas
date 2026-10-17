"""Filtros de calidad orientativos; no verifican que exista una vaca."""
import numpy as np
from PIL import Image


def validar_imagen(imagen, opciones):
    errores = []
    if imagen.width < opciones["ancho_minimo"] or imagen.height < opciones["alto_minimo"]:
        errores.append(opciones["mensajes"]["pequena"])
    # Medir siempre a una escala similar evita comparar texturas de tamaños arbitrarios.
    gris = imagen.convert("L")
    gris.thumbnail((opciones["lado_medicion"], opciones["lado_medicion"]), Image.Resampling.LANCZOS)
    matriz = np.asarray(gris, dtype=np.float32)
    brillo = float(matriz.mean())
    if min(matriz.shape) < 3:
        varianza = 0.0
    else:
        laplaciano = (matriz[:-2, 1:-1] + matriz[2:, 1:-1] + matriz[1:-1, :-2]
                      + matriz[1:-1, 2:] - 4 * matriz[1:-1, 1:-1])
        varianza = float(laplaciano.var())
    if brillo < opciones["brillo_minimo"]:
        errores.append(opciones["mensajes"]["oscura"])
    if varianza < opciones["varianza_laplaciano_minima"]:
        errores.append(opciones["mensajes"]["borrosa"])
    return {"valida": not errores, "errores": errores,
            "medidas": {"ancho": imagen.width, "alto": imagen.height, "brillo": brillo, "varianza_laplaciano": varianza}}
