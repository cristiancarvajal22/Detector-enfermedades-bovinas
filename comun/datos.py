"""Inventario y duplicados reutilizados por preparación y evaluación."""
from collections import defaultdict

from comun.configuracion import ruta_config
from comun.imagenes import hash_archivo, hash_pixeles, listar_imagenes


def inventariar(config, raiz=None):
    raiz = raiz or ruta_config(config, "originales")
    registros, errores = [], []
    for clase in config["clases"]:
        for ruta in listar_imagenes(raiz / clase["carpeta"], config["datos"]["extensiones"]):
            try:
                registros.append({"clase": clase["etiqueta"], "carpeta": clase["carpeta"],
                                  "ruta": str(ruta.relative_to(raiz)),
                                  "sha256": hash_archivo(ruta), "pixeles_sha256": hash_pixeles(ruta)})
            except (OSError, ValueError, Warning) as exc:
                errores.append({"ruta": str(ruta), "error": str(exc)})
    return registros, errores


def agrupar_duplicados(registros, campo="pixeles_sha256"):
    grupos = defaultdict(list)
    for registro in registros:
        grupos[registro[campo]].append(registro)
    return [grupo for grupo in grupos.values() if len(grupo) > 1]
