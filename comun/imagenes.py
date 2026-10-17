"""Lectura y huellas; no se modifica ninguna imagen original."""
import hashlib
import warnings
from pathlib import Path

from PIL import Image, ImageOps


def listar_imagenes(carpeta, extensiones):
    carpeta = Path(carpeta)
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in extensiones)


def abrir_rgb(origen, max_pixeles=24000000, formatos=None):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(origen, formats=formatos) as imagen:
            if imagen.width * imagen.height > max_pixeles:
                raise ValueError("La imagen supera el límite de píxeles permitido.")
            rgb = ImageOps.exif_transpose(imagen).convert("RGB")
            # EXIF puede contener coordenadas GPS; la vista previa solo necesita píxeles.
            rgb.info.clear()
            return rgb


def hash_archivo(ruta):
    huella = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


def hash_pixeles(ruta):
    imagen = abrir_rgb(ruta)
    huella = hashlib.sha256(str(imagen.size).encode("ascii"))
    huella.update(imagen.tobytes())
    return huella.hexdigest()
