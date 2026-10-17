"""Identidad del modelo: también incluye pesos para no mezclar evaluaciones."""
import hashlib
import json

from comun.configuracion import ruta_config
from comun.imagenes import hash_archivo


def archivos_modelo(config):
    carpeta = ruta_config(config, "modelo")
    archivos = [carpeta / "saved_model.pb", ruta_config(config, "etiquetas")]
    for nombre in ("variables", "assets"):
        archivos.extend(p for p in (carpeta / nombre).rglob("*") if p.is_file())
    return sorted(set(archivos))


def revision_modelo(config):
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in archivos_modelo(config) if p.is_file())


def identidad_modelo(config):
    carpeta = ruta_config(config, "modelo")
    registros = []
    for archivo in archivos_modelo(config):
        nombre = archivo.relative_to(carpeta).as_posix() if archivo.is_relative_to(carpeta) else "labels.txt"
        registros.append({"archivo": nombre, "sha256": hash_archivo(archivo), "bytes": archivo.stat().st_size})
    datos = json.dumps(registros, sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(datos).hexdigest(),
            "bytes": sum(r["bytes"] for r in registros), "archivos": registros}
