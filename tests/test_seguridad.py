"""Casos de abuso en subidas, errores y publicación de credenciales."""
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from comun.configuracion import cargar_config
from entrada import leer_imagen_subida


class EntradaSeguraTest(unittest.TestCase):
    def setUp(self):
        self.config = cargar_config()

    def test_limite_antes_de_decodificar(self):
        self.config["validacion"]["max_megabytes"] = .0001
        with patch("entrada.abrir_rgb") as abrir:
            with self.assertRaises(ValueError):
                leer_imagen_subida(io.BytesIO(b"A" * 1024), self.config)
            abrir.assert_not_called()

    def test_html_no_se_acepta_como_imagen(self):
        with self.assertRaises(OSError):
            leer_imagen_subida(io.BytesIO(b"<html><script>alert(1)</script></html>"), self.config)

    def test_formato_no_permitido_aunque_se_renombre(self):
        archivo = io.BytesIO()
        Image.new("RGB", (250, 250), "green").save(archivo, "GIF")
        with self.assertRaises(OSError):
            leer_imagen_subida(archivo, self.config)

    def test_metadatos_se_eliminan(self):
        archivo = io.BytesIO()
        exif = Image.Exif()
        exif[270] = "metadatos privados de la fotografia"
        Image.new("RGB", (250, 250), "green").save(archivo, "JPEG", exif=exif)
        imagen = leer_imagen_subida(archivo, self.config)
        self.assertEqual(imagen.info, {})
        self.assertFalse(imagen.getexif())

    def test_limite_de_pixeles_y_bomba(self):
        archivo = io.BytesIO()
        Image.new("RGB", (32, 32)).save(archivo, "PNG")
        self.config["validacion"]["max_pixeles"] = 100
        with self.assertRaises(ValueError):
            leer_imagen_subida(archivo, self.config)
        with patch.object(Image, "MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(ValueError):
                leer_imagen_subida(archivo, self.config)


class PublicacionSeguraTest(unittest.TestCase):
    def test_git_excluye_secretos(self):
        for nombre in (".env", ".env.production", ".streamlit/secrets.toml", "privada.pem", "credentials.json"):
            proceso = subprocess.run(["git", "-C", str(RAIZ), "check-ignore", "-q", "--", nombre], timeout=10)
            self.assertEqual(proceso.returncode, 0, nombre)
        proceso = subprocess.run(["git", "-C", str(RAIZ), "check-ignore", "-q", "--", ".streamlit/config.toml"], timeout=10)
        self.assertEqual(proceso.returncode, 1)

    def test_escaner_no_imprime_valores_detectados(self):
        spec = importlib.util.spec_from_file_location("escaner", RAIZ / "anexos/verificar_secretos.py")
        escaner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(escaner)
        sintetico = "sk-" + "x" * 40
        hallazgos = escaner.detectar(sintetico.encode(), "ejemplo.txt")
        self.assertTrue(hallazgos)
        self.assertNotIn(sintetico, str(hallazgos))

    def test_error_interno_no_se_muestra_al_visitante(self):
        from streamlit.testing.v1 import AppTest
        config = copy.deepcopy(cargar_config())
        config["_prueba_error"] = True
        imagen = Image.fromarray(np.random.default_rng(7).integers(40, 240, (300, 300, 3), dtype=np.uint8))
        archivo = io.BytesIO()
        imagen.save(archivo, "PNG")
        modelo = Mock()
        modelo.predecir.side_effect = RuntimeError("INTERNAL C:/servidor/privado/token.txt")
        with patch("streamlit.file_uploader", return_value=archivo), \
             patch("comun.configuracion.cargar_config", return_value=config), \
             patch("clasificador.crear_clasificador", return_value=modelo):
            app = AppTest.from_file(str(RAIZ / "05_app/app.py"), default_timeout=60).run()
            app.button(key="analizar").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([e.value for e in app.error], [config["interfaz"]["textos"]["error_analisis"]])
        self.assertFalse(any("INTERNAL" in e.value or "token.txt" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
