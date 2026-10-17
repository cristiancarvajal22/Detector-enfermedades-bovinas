"""Una sola configuración para datos, modelo y despliegue."""
import json
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config(ruta=None):
    archivo = Path(ruta).resolve() if ruta else RAIZ / "config.json"
    config = json.loads(archivo.read_text(encoding="utf-8-sig"))
    clases = config["clases"]
    for campo in ("etiqueta", "carpeta"):
        valores = [c[campo] for c in clases]
        if len(valores) != len(set(valores)) or not all(valores):
            raise ValueError(f"Cada clase necesita {campo} único y no vacío.")
    if len(clases) < 2:
        raise ValueError("Se necesitan al menos dos clases.")
    for clase in clases:
        carpeta = clase["carpeta"]
        if Path(carpeta).name != carpeta or carpeta in (".", ".."):
            raise ValueError("Las carpetas de clase deben ser nombres simples.")
        if clase["riesgo"] not in config["estados"]:
            raise ValueError("Riesgo de clase sin estado configurado.")
    for campo in ("umbral", "margen"):
        valor = config["decision"][campo]
        if not math.isfinite(valor) or not 0 <= valor <= 1:
            raise ValueError(f"{campo} debe estar entre 0 y 1.")
    camara = config["camara"]
    if not 1 <= camara["votos_necesarios"] <= camara["ventana"]:
        raise ValueError("La votación necesita 1 <= N <= M.")
    if not 0 < camara["fps_analisis"] <= 2:
        raise ValueError("La cámara debe analizar más de 0 y hasta 2 cuadros/s.")
    if config["datos"]["prueba_por_clase"] < 1:
        raise ValueError("Se requiere al menos una imagen de prueba por clase.")
    if camara["espera_alertas_segundos"] < 0:
        raise ValueError("El tiempo entre alertas no puede ser negativo.")
    if config["interfaz"]["historial_maximo"] < 1:
        raise ValueError("El historial debe conservar al menos un resultado.")
    preproceso = config["modelo"]["preprocesamiento"]
    if min(preproceso["ancho"], preproceso["alto"]) < 1:
        raise ValueError("Las dimensiones del modelo deben ser positivas.")
    # Se resuelven contra el archivo, nunca contra el directorio de la terminal.
    config["_raiz"] = str(archivo.parent)
    return config


def ruta_config(config, clave):
    return (Path(config["_raiz"]) / config["rutas"][clave]).resolve()


def guardar_json(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
