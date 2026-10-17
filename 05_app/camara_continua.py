"""Cámara local opcional. No accede a la cámara del navegador remoto."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from clasificador import crear_clasificador
from comun.configuracion import cargar_config, guardar_json, ruta_config
from decision import decidir
from validacion import validar_imagen


class Votacion:
    """Cada cuadro rechazado también ocupa una posición en la ventana."""
    def __init__(self, opciones):
        self.opciones = opciones
        self.ventana = deque(maxlen=opciones["ventana"])
        self.ultima_alerta = float("-inf")

    def actualizar(self, etiqueta, ahora):
        self.ventana.append(etiqueta)
        if len(self.ventana) < self.opciones["ventana"] or etiqueta is None:
            return False
        suficientes = Counter(self.ventana)[etiqueta] >= self.opciones["votos_necesarios"]
        disponible = ahora - self.ultima_alerta >= self.opciones["espera_alertas_segundos"]
        if suficientes and disponible:
            self.ultima_alerta = ahora
            return True
        return False


def registrar_alerta(imagen, resultado, probabilidades, modelo, config):
    ahora = datetime.now(timezone.utc)
    destino = ruta_config(config, "alertas")
    destino.mkdir(parents=True, exist_ok=True)
    nombre = ahora.strftime("%Y%m%dT%H%M%S%fZ")
    imagen.save(destino / f"{nombre}.jpg", quality=92)
    guardar_json(destino / f"{nombre}.json", {"hora_utc": ahora.isoformat(), "captura": f"{nombre}.jpg",
                 "decision": resultado, "probabilidades": dict(zip(modelo.etiquetas, probabilidades.tolist())),
                 "votacion": config["camara"], "aviso": config["proyecto"]["aviso"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError:
        parser.exit(1, "Instala requirements-camara.txt para usar esta función.\n")
    config = cargar_config(args.config)
    opciones = config["camara"]
    modelo = crear_clasificador(config)
    votacion = Votacion(opciones)
    captura = cv2.VideoCapture(opciones["dispositivo"])
    if not captura.isOpened():
        captura.release()
        parser.exit(1, "No se pudo abrir la cámara. Revisa conexión, permisos y dispositivo en config.json.\n")
    titulo = opciones["titulo_ventana"]
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titulo, opciones["ancho_ventana"], opciones["alto_ventana"])
    try:
        fuente = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        fuente = ImageFont.load_default(size=18)
    siguiente, mensaje = 0.0, "Preparando observación…"
    try:
        while True:
            ok, cuadro = captura.read()
            if not ok:
                raise RuntimeError("La cámara dejó de entregar imágenes.")
            imagen = Image.fromarray(cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB))
            ahora = time.monotonic()
            if ahora >= siguiente:
                calidad = validar_imagen(imagen, config["validacion"])
                etiqueta = None
                if calidad["valida"]:
                    probabilidades = modelo.predecir(imagen)
                    resultado = decidir(modelo.etiquetas, probabilidades, config)
                    mensaje = config["estados"][resultado["estado"]]["titulo"]
                    if resultado["estado"] in opciones["estados_alerta"]:
                        etiqueta = resultado["clase"]
                else:
                    mensaje = calidad["errores"][0]
                if votacion.actualizar(etiqueta, ahora):
                    registrar_alerta(imagen, resultado, probabilidades, modelo, config)
                    mensaje += " · Alerta registrada"
                siguiente = time.monotonic() + 1 / opciones["fps_analisis"]
            # El aviso se dibuja debajo de la captura para conservarlo legible.
            vista = Image.new("RGB", (max(imagen.width, 960), imagen.height + 104), "#243B30")
            vista.paste(imagen, (0, 0))
            dibujo = ImageDraw.Draw(vista)
            dibujo.text((12, imagen.height + 8), mensaje, fill="white", font=fuente)
            dibujo.text((12, imagen.height + 40), config["proyecto"]["aviso"], fill="white", font=fuente)
            dibujo.text((12, imagen.height + 72), "Q: cerrar cámara", fill="white", font=fuente)
            cv2.imshow(titulo, cv2.cvtColor(np.asarray(vista), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
