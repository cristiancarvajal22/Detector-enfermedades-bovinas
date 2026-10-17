"""Aprendizaje supervisado: reservar prueba ANTES de entrenar."""
import argparse
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun.configuracion import cargar_config, guardar_json, ruta_config
from comun.datos import agrupar_duplicados, inventariar


def dividir(config):
    origen, destino = ruta_config(config, "originales"), ruta_config(config, "division")
    if destino.exists():
        raise ValueError(f"La salida ya existe: {destino}. Elige una ruta nueva para no sobrescribirla.")
    if destino == origen or origen in destino.parents or destino in origen.parents:
        raise ValueError("Origen y destino no pueden contenerse entre sí.")
    registros, errores = inventariar(config)
    if errores:
        raise ValueError("Hay imágenes ilegibles; revisa detectar_duplicados.py antes de dividir.")
    if any(len({r["clase"] for r in g}) > 1 for g in agrupar_duplicados(registros)):
        raise ValueError("Hay imágenes idénticas con etiquetas distintas; resuelve los conflictos primero.")
    plan, excluidos = [], []
    cantidad = config["datos"]["prueba_por_clase"]
    for clase in config["clases"]:
        unicos = {}
        for registro in registros:
            if registro["clase"] != clase["etiqueta"]:
                continue
            if registro["pixeles_sha256"] in unicos:
                excluidos.append(registro)
            else:
                unicos[registro["pixeles_sha256"]] = registro
        candidatos = sorted(unicos.values(), key=lambda r: r["ruta"])
        if len(candidatos) <= cantidad:
            raise ValueError(f"{clase['etiqueta']}: se requieren más de {cantidad} imágenes únicas para reservar prueba y entrenamiento.")
        random.Random(config["datos"]["semilla"]).shuffle(candidatos)
        for indice, registro in enumerate(candidatos):
            conjunto = "prueba" if indice < cantidad else "entrenamiento"
            relativo = Path(conjunto) / clase["carpeta"] / f"{registro['sha256']}{Path(registro['ruta']).suffix.lower()}"
            plan.append({**registro, "conjunto": conjunto, "destino": relativo.as_posix()})
    # Se valida el plan entero antes de copiar; los originales siempre se conservan.
    destino.mkdir(parents=True, exist_ok=False)
    for registro in plan:
        archivo = destino / registro["destino"]
        archivo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen / registro["ruta"], archivo)
    manifiesto = {"creado_utc": datetime.now(timezone.utc).isoformat(), "semilla": config["datos"]["semilla"],
                  "prueba_por_clase": cantidad, "registros": plan, "duplicados_excluidos": excluidos,
                  "aviso": "Este manifiesto no demuestra que el modelo no haya visto las imágenes de prueba."}
    guardar_json(destino / "manifiesto.json", manifiesto)
    return manifiesto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        manifiesto = dividir(cargar_config(args.config))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"No se pudo dividir: {exc}\n")
    print(f"Copiadas {len(manifiesto['registros'])} imágenes únicas. Consulta manifiesto.json.")
    print(manifiesto["aviso"])


if __name__ == "__main__":
    main()
