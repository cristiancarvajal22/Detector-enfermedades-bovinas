"""Lectura de subidas: aplicar los mismos límites antes de previsualizar e inferir."""
import io

from PIL import Image

from comun.imagenes import abrir_rgb


def leer_imagen_subida(archivo, config):
    opciones = config["validacion"]
    contenido = archivo.getvalue()
    if len(contenido) > opciones["max_megabytes"] * 1024 * 1024:
        raise ValueError(config["interfaz"]["textos"]["error_tamano"])
    try:
        return abrir_rgb(io.BytesIO(contenido), opciones["max_pixeles"], opciones["formatos_permitidos"])
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("La imagen supera el límite de píxeles permitido.") from None
