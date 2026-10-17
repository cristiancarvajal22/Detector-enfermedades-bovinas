"""Comprensión de datos: cantidad no equivale a representatividad."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun.configuracion import cargar_config, guardar_json, ruta_config
from comun.datos import inventariar


def contar(config):
    registros, errores = inventariar(config)
    conteos = {c["etiqueta"]: sum(r["clase"] == c["etiqueta"] for r in registros) for c in config["clases"]}
    minimo, maximo = min(conteos.values()), max(conteos.values())
    razon = maximo / minimo if minimo else None
    return {"conteos_validos": conteos, "razon_mayor_menor": razon,
            "faltan_clases": minimo == 0,
            "desbalance": minimo == 0 or razon > config["datos"]["razon_desbalance"],
            "ilegibles": errores,
            "limite": "El conteo no mide diversidad de animales, regiones, razas ni condiciones de captura."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = cargar_config(args.config)
    informe = contar(config)
    guardar_json(ruta_config(config, "evaluacion") / "conteos.json", informe)
    for clase, cantidad in informe["conteos_validos"].items():
        print(f"{clase}: {cantidad} imágenes legibles")
    print(f"Razón mayor/menor: {informe['razon_mayor_menor']}")
    if informe["desbalance"]:
        print("AVISO: faltan clases o se supera la razón de desbalance configurada.")
    print(informe["limite"])


if __name__ == "__main__":
    main()
