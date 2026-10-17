"""Adaptador SavedModel; la interfaz consume etiquetas y probabilidades."""
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from comun.configuracion import ruta_config


def leer_etiquetas(ruta):
    lineas = Path(ruta).read_text(encoding="utf-8-sig").splitlines()
    etiquetas = [re.sub(r"^\d+\s+", "", linea.strip()) for linea in lineas if linea.strip()]
    if len(etiquetas) < 2 or len(set(etiquetas)) != len(etiquetas):
        raise ValueError("labels.txt necesita al menos dos etiquetas únicas.")
    # El orden de labels.txt es el orden de las neuronas de salida.
    return etiquetas


def preprocesar(imagen, opciones):
    """Única transformación usada en app, evaluación y cámara continua."""
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    tamano = (opciones["ancho"], opciones["alto"])
    if opciones["ajuste"] == "recorte_central":
        imagen = ImageOps.fit(imagen, tamano, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    elif opciones["ajuste"] == "redimensionar":
        imagen = imagen.resize(tamano, Image.Resampling.LANCZOS)
    else:
        raise ValueError("Ajuste de imagen no soportado.")
    matriz = np.asarray(imagen, dtype=np.float32)
    if opciones["normalizacion"] == "menos_uno_a_uno":
        matriz = matriz / np.float32(127.5) - np.float32(1)
    elif opciones["normalizacion"] == "cero_a_uno":
        matriz = matriz / np.float32(255)
    elif opciones["normalizacion"] != "sin_normalizar":
        raise ValueError("Normalización desconocida.")
    return matriz[np.newaxis, ...]


def validar_probabilidades(valores, cantidad):
    valores = np.asarray(valores, dtype=np.float64)
    if valores.shape != (cantidad,) or not np.all(np.isfinite(valores)):
        raise ValueError("La salida no es un vector finito con una probabilidad por etiqueta.")
    if np.any(valores < 0) or np.any(valores > 1) or not np.isclose(valores.sum(), 1, atol=0.001):
        raise ValueError("La salida no suma 1 o no contiene probabilidades. Revisa tipo_salida.")
    return valores


class ClasificadorSavedModel:
    def __init__(self, config):
        import tensorflow as tf

        self.tf, self.config = tf, config
        self._lock = threading.Lock()
        self.etiquetas = leer_etiquetas(ruta_config(config, "etiquetas"))
        esperadas = {c["etiqueta"] for c in config["clases"]}
        if set(self.etiquetas) != esperadas:
            raise ValueError(f"Las etiquetas de labels.txt y config.json no coinciden. Modelo: {self.etiquetas}; configuración: {sorted(esperadas)}")
        opciones = config["modelo"]
        self.preprocesamiento = opciones["preprocesamiento"]
        carpeta = ruta_config(config, "modelo")
        if not (carpeta / "saved_model.pb").is_file():
            raise FileNotFoundError(f"No se encontró saved_model.pb en {carpeta}.")
        self.modelo = tf.saved_model.load(str(carpeta))
        firma = opciones["firma"]
        if firma not in self.modelo.signatures:
            raise ValueError(f"Firma {firma} ausente. Disponibles: {list(self.modelo.signatures)}")
        self.firma = self.modelo.signatures[firma]
        argumentos, entradas = self.firma.structured_input_signature
        if argumentos or len(entradas) != 1:
            raise ValueError("Este adaptador requiere una entrada de imagen por nombre.")
        self.nombre_entrada = opciones["entrada"] or next(iter(entradas))
        if self.nombre_entrada not in entradas:
            raise ValueError(f"Entrada desconocida: {self.nombre_entrada}.")
        self.spec = entradas[self.nombre_entrada]
        forma = self.spec.shape.as_list()
        esperado = [1, self.preprocesamiento["alto"], self.preprocesamiento["ancho"], 3]
        if len(forma) != 4 or any(a is not None and a != b for a, b in zip(forma, esperado)):
            raise ValueError(f"Entrada incompatible: {forma}; configuración: {esperado}.")
        if self.spec.dtype != tf.float32:
            raise ValueError(f"Se espera float32, se encontró {self.spec.dtype.name}. Revisa el adaptador.")
        salidas = self.firma.structured_outputs
        if not isinstance(salidas, dict) or not salidas:
            raise ValueError("La firma debe devolver un diccionario de tensores.")
        self.nombre_salida = opciones["salida"]
        if not self.nombre_salida:
            if len(salidas) != 1:
                raise ValueError(f"Hay varias salidas: {list(salidas)}. Elige modelo.salida en config.json.")
            self.nombre_salida = next(iter(salidas))
        if self.nombre_salida not in salidas:
            raise ValueError("La salida configurada no existe en la firma.")
        forma_salida = salidas[self.nombre_salida].shape.as_list()
        if len(forma_salida) != 2 or forma_salida[0] not in (None, 1) or forma_salida[1] not in (None, len(self.etiquetas)):
            raise ValueError(f"Salida incompatible: {forma_salida}; etiquetas: {len(self.etiquetas)}.")
        if opciones["tipo_salida"] not in ("probabilidades", "logits"):
            raise ValueError("tipo_salida debe ser probabilidades o logits.")

    def inspeccion(self):
        return {"tensorflow": self.tf.__version__, "firmas": list(self.modelo.signatures),
                "entrada": self.nombre_entrada, "forma": self.spec.shape.as_list(),
                "dtype": self.spec.dtype.name, "salida": self.nombre_salida,
                "etiquetas": self.etiquetas, "preprocesamiento": self.preprocesamiento,
                "aviso": "La firma verifica forma y dtype; no demuestra la normalización ni el recorte del entrenamiento."}

    def predecir(self, imagen):
        lote = preprocesar(imagen, self.preprocesamiento)
        # El recurso cacheado es compartido entre sesiones; se serializa la inferencia.
        with self._lock:
            salida = self.firma(**{self.nombre_entrada: self.tf.convert_to_tensor(lote)})[self.nombre_salida].numpy()
        if salida.shape != (1, len(self.etiquetas)):
            raise ValueError(f"Forma de salida inesperada: {salida.shape}.")
        valores = salida[0].astype(np.float64)
        if self.config["modelo"]["tipo_salida"] == "logits":
            valores = np.exp(valores - valores.max())
            valores = valores / valores.sum()
        return validar_probabilidades(valores, len(self.etiquetas))


def crear_clasificador(config):
    # Un futuro adaptador Keras debe respetar etiquetas/predecir/inspeccion.
    if config["modelo"]["adaptador"] != "savedmodel":
        raise ValueError("Adaptador no implementado. Agrega el adaptador antes de seleccionarlo en config.json.")
    return ClasificadorSavedModel(config)
