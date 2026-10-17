"""Auditoría de duplicados por bytes y por píxeles decodificados."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun.configuracion import cargar_config, guardar_json, ruta_config
from comun.datos import agrupar_duplicados, inventariar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = cargar_config(args.config)
    registros, errores = inventariar(config)
    grupos = agrupar_duplicados(registros)
    informe = {"por_bytes": agrupar_duplicados(registros, "sha256"), "por_pixeles": grupos,
               "conflictos_entre_clases": [g for g in grupos if len({r["clase"] for r in g}) > 1],
               "ilegibles": errores,
               "limite": "No detecta todas las fotos similares, recortes, recompresiones ni imágenes del mismo animal."}
    destino = ruta_config(config, "evaluacion") / "duplicados.json"
    guardar_json(destino, informe)
    print(f"Grupos repetidos: {len(grupos)}; conflictos entre clases: {len(informe['conflictos_entre_clases'])}")
    print(f"Informe: {destino}\n{informe['limite']}")


if __name__ == "__main__":
    main()
