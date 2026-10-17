"""Inspecciona la firma real sin suponer compatibilidad por el nombre del modelo."""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from clasificador import crear_clasificador
from comun.configuracion import cargar_config, guardar_json, ruta_config
from comun.modelo import identidad_modelo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = cargar_config(args.config)
    try:
        modelo = crear_clasificador(config)
        informe = modelo.inspeccion()
        informe["modelo_sha256"] = identidad_modelo(config)["sha256"]
        guardar_json(ruta_config(config, "modelo") / "inspeccion.json", informe)
        print(json.dumps(informe, ensure_ascii=False, indent=2))
        print("Revisa además el código de exportación/entrenamiento para confirmar escala, orden RGB y recorte.")
        print("Solo entonces cambia modelo.preprocesamiento.confirmado a true.")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"No se pudo inspeccionar: {exc}\n")


if __name__ == "__main__":
    main()
