"""Pruebas de límites de decisión, integridad y flujo, sin inventar métricas clínicas."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from clasificador import crear_clasificador, leer_etiquetas, preprocesar, validar_probabilidades
from comun.configuracion import cargar_config
from comun.imagenes import abrir_rgb, hash_pixeles
from comun.modelo import identidad_modelo
from informacion_modelo import buscar_evaluacion, leer_inspeccion
from decision import decidir
from validacion import validar_imagen
from camara_continua import Votacion


def modulo_archivo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


division = modulo_archivo("division", "02_preparacion/dividir_datos.py")


class ReglasTest(unittest.TestCase):
    def setUp(self):
        self.config = cargar_config()
        self.etiquetas = [c["etiqueta"] for c in self.config["clases"]]

    def test_riesgo_y_abstencion(self):
        for valores, esperado in (([.8, .1, .1], "bajo"), ([.1, .8, .1], "alto"),
                                   ([.6, .3, .1], "incierto"), ([.1, .2, .7], "invalida"),
                                   ([.34, .33, .33], "incierto")):
            with self.subTest(valores=valores):
                valores = valores + [0] * (len(self.etiquetas) - len(valores))
                self.assertEqual(decidir(self.etiquetas, valores, self.config)["estado"], esperado)

    def test_margen_y_frontera_umbral(self):
        self.config["decision"].update(umbral=.5, margen=.15)
        self.assertEqual(decidir(self.etiquetas, [.55, .45] + [0] * (len(self.etiquetas) - 2), self.config)["estado"], "incierto")
        self.config["decision"].update(umbral=.7, margen=.15)
        self.assertEqual(decidir(self.etiquetas, [.7, .2, .1] + [0] * (len(self.etiquetas) - 3), self.config)["estado"], "bajo")

    def test_etiquetas_reordenadas_respetan_salida(self):
        etiquetas = self.etiquetas[::-1]
        valores = [.2 / (len(etiquetas) - 1)] * len(etiquetas)
        valores[etiquetas.index("PielConLSD")] = .8
        self.assertEqual(decidir(etiquetas, valores, self.config)["estado"], "alto")

    def test_clases_oculares_y_abstencion(self):
        for etiqueta, esperado in (("OjoSano", "bajo"), ("OjoRosado", "alto")):
            valores = [.2 / (len(self.etiquetas) - 1)] * len(self.etiquetas)
            valores[self.etiquetas.index(etiqueta)] = .8
            resultado = decidir(self.etiquetas, valores, self.config)
            self.assertEqual(resultado["estado"], esperado)
            self.assertEqual(resultado["clase"], etiqueta)
            self.assertIn("ojo", resultado["recomendacion"].lower())
        valores = [0] * len(self.etiquetas)
        valores[self.etiquetas.index("OjoRosado")] = .6
        valores[self.etiquetas.index("OjoSano")] = .4
        self.assertEqual(decidir(self.etiquetas, valores, self.config)["estado"], "incierto")

    def test_salida_invalida_se_rechaza(self):
        for valores in ([float("nan"), 0, 1], [1, 1, 1], [-.1, .5, .6], [1, 0]):
            with self.assertRaises(ValueError):
                validar_probabilidades(valores, 3)

    def test_preprocesamiento_extremos(self):
        opciones = self.config["modelo"]["preprocesamiento"]
        for tono, esperado in ((0, -1), (255, 1)):
            lote = preprocesar(Image.new("L", (300, 400), tono), opciones)
            self.assertEqual(lote.shape, (1, 224, 224, 3))
            self.assertEqual(lote.dtype, np.float32)
            np.testing.assert_allclose(lote, esperado)

    def test_calidad_oscuridad_tamano_y_textura(self):
        opciones = self.config["validacion"]
        mala = validar_imagen(Image.new("RGB", (100, 100)), opciones)
        self.assertFalse(mala["valida"])
        self.assertEqual(len(mala["errores"]), 3)
        ruido = np.random.default_rng(42).integers(50, 240, (300, 300, 3), dtype=np.uint8)
        self.assertTrue(validar_imagen(Image.fromarray(ruido), opciones)["valida"])

    def test_lectura_indices_y_hash_pixeles(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            (raiz / "labels.txt").write_text("0 Una clase larga\n1 Otra clase\n", encoding="utf-8")
            self.assertEqual(leer_etiquetas(raiz / "labels.txt"), ["Una clase larga", "Otra clase"])
            imagen = Image.new("RGB", (20, 30), "red")
            imagen.save(raiz / "a.png")
            imagen.save(raiz / "b.bmp")
            self.assertEqual(hash_pixeles(raiz / "a.png"), hash_pixeles(raiz / "b.bmp"))
            with self.assertRaises(ValueError):
                abrir_rgb(raiz / "a.png", max_pixeles=50)

    def test_ventana_espera_y_rechazos(self):
        votos = Votacion({"ventana": 4, "votos_necesarios": 3, "espera_alertas_segundos": 10})
        self.assertFalse(votos.actualizar("a", 0))
        self.assertFalse(votos.actualizar(None, 1))
        self.assertFalse(votos.actualizar("a", 2))
        self.assertTrue(votos.actualizar("a", 3))
        self.assertFalse(votos.actualizar("a", 4))
        self.assertTrue(votos.actualizar("a", 13))
        self.assertFalse(votos.actualizar(None, 24))


class PreparacionTest(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temporal.name)
        self.config = cargar_config()
        self.config["_raiz"] = str(self.raiz)
        self.config["datos"]["prueba_por_clase"] = 2
        self.origen = self.raiz / self.config["rutas"]["originales"]
        for indice, clase in enumerate(self.config["clases"]):
            carpeta = self.origen / clase["carpeta"]
            carpeta.mkdir(parents=True)
            for i in range(5):
                tono = indice * 255 // (len(self.config["clases"]) - 1)
                Image.new("RGB", (24, 24), (tono, i * 40, 80)).save(carpeta / f"{i}.png")

    def tearDown(self):
        self.temporal.cleanup()

    def test_division_reproducible_sin_fuga(self):
        carpeta = self.origen / self.config["clases"][0]["carpeta"]
        (carpeta / "copia.png").write_bytes((carpeta / "0.png").read_bytes())
        primero = division.dividir(self.config)
        self.assertEqual(len(primero["duplicados_excluidos"]), 1)
        self.assertEqual(sum(r["conjunto"] == "prueba" for r in primero["registros"]), 2 * len(self.config["clases"]))
        self.config["rutas"]["division"] += "_segunda"
        segundo = division.dividir(self.config)
        self.assertEqual(primero["registros"], segundo["registros"])
        a = {r["pixeles_sha256"] for r in primero["registros"] if r["conjunto"] == "prueba"}
        b = {r["pixeles_sha256"] for r in primero["registros"] if r["conjunto"] == "entrenamiento"}
        self.assertFalse(a & b)
        self.assertTrue((carpeta / "copia.png").exists())
        with self.assertRaises(ValueError):
            division.dividir(self.config)

    def test_conflicto_de_etiquetas_se_detiene(self):
        a, b = [self.origen / c["carpeta"] for c in self.config["clases"][:2]]
        (b / "conflicto.png").write_bytes((a / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "etiquetas distintas"):
            division.dividir(self.config)
        self.assertFalse((self.raiz / self.config["rutas"]["division"]).exists())

    def test_pocos_datos_se_detiene_antes_de_copiar(self):
        self.config["datos"]["prueba_por_clase"] = 5
        with self.assertRaises(ValueError):
            division.dividir(self.config)
        self.assertFalse((self.raiz / self.config["rutas"]["division"]).exists())

    def test_evaluacion_detecta_imagen_modificada_y_adicional(self):
        evaluacion = modulo_archivo("evaluacion", "04_evaluacion/evaluar.py")
        manifiesto = division.dividir(self.config)
        registros = evaluacion.verificar_division(self.config)
        self.assertEqual(len(registros), 5 * len(self.config["clases"]))
        destino = self.raiz / self.config["rutas"]["division"]
        extra = destino / "prueba" / self.config["clases"][0]["carpeta"] / "extra.png"
        Image.new("RGB", (24, 24), "white").save(extra)
        with self.assertRaisesRegex(ValueError, "fuera del manifiesto"):
            evaluacion.verificar_division(self.config)
        extra.unlink()
        imagen = destino / manifiesto["registros"][0]["destino"]
        imagen.write_bytes(b"modificada")
        with self.assertRaisesRegex(ValueError, "modificada"):
            evaluacion.verificar_division(self.config)

    def test_evaluacion_exporta_metricas_y_grafico(self):
        evaluacion = modulo_archivo("evaluacion_exportacion", "04_evaluacion/evaluar.py")
        division.dividir(self.config)
        registros = evaluacion.verificar_division(self.config)
        etiquetas = [c["etiqueta"] for c in self.config["clases"]]

        class ModeloControlado:
            def predecir(self, imagen):
                return np.array([.9] + [.1 / (len(etiquetas) - 1)] * (len(etiquetas) - 1))

        modelo = ModeloControlado()
        modelo.etiquetas = etiquetas
        salida = self.raiz / "evidencia_prueba_unitaria"
        salida.mkdir()
        resumen = evaluacion.evaluar_conjunto(modelo, registros, self.config, salida, "prueba")
        self.assertEqual(resumen["cantidad"], 2 * len(etiquetas))
        self.assertAlmostEqual(resumen["exactitud"], 1 / len(etiquetas))
        self.assertAlmostEqual(resumen["por_clase"][etiquetas[0]]["precision"], 1 / len(etiquetas))
        self.assertEqual(resumen["aceptadas_app"], 0)
        self.assertIsNone(resumen["exactitud_aceptadas_app"])
        self.assertTrue((salida / "matriz_prueba.png").is_file())
        self.assertTrue((salida / "predicciones_prueba.csv").is_file())


class IntegracionModeloTest(unittest.TestCase):
    @unittest.skipUnless((RAIZ / "03_modelo/saved_model.pb").exists(), "Requiere el modelo real")
    def test_savedmodel_real_carga_y_predice(self):
        config = cargar_config()
        modelo = crear_clasificador(config)
        ruido = np.random.default_rng(8).integers(0, 256, (320, 480, 3), dtype=np.uint8)
        valores = modelo.predecir(Image.fromarray(ruido))
        self.assertEqual(len(valores), len(modelo.etiquetas))
        self.assertAlmostEqual(float(valores.sum()), 1, places=3)


class InterfazTest(unittest.TestCase):
    def test_pantalla_inicial_e_historial(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(RAIZ / "05_app/app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Bovino", app.title[0].value)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual([tab.label for tab in app.tabs], ["Analizar imagen", "Modelo y evaluación"])
        self.assertTrue(any("no reemplaza" in elemento.value for elemento in app.caption))
        self.assertFalse(any("aviso-permanente" in elemento.value for elemento in app.markdown))
        visibles = "\n".join(str(elemento.value) for grupo in
                             (app.markdown, app.caption, app.info, app.subheader) for elemento in grupo)
        for texto in ("La confianza es una salida del modelo", "Firma verificada", "Tamaño y tipo de entrada comprobados",
                      "Evaluación del modelo", "Ojo rosado (ojo no sano)", "SavedModel"):
            self.assertNotIn(texto, visibles)
        self.assertNotIn("Formato", [metrica.label for metrica in app.metric])
        for titulo in ("Cómo generar las métricas para la defensa", "Qué se puede explicar en la defensa", "Firma técnica y versión del modelo"):
            self.assertNotIn(titulo, [elemento.label for elemento in app.expander])
        config = cargar_config()
        app.session_state["historial"] = [{"estado": "bajo", "clase": config["clases"][0]["etiqueta"],
                                           "confianza": .9, "hora": "2026-10-16 10:00:00 -04"}]
        app.run()
        self.assertEqual(len(app.session_state["historial"]), 1)
        app.button(key="borrar_historial").click().run()
        self.assertFalse(app.session_state["historial"])
        self.assertEqual(len(app.exception), 0)

    @unittest.skipUnless((RAIZ / "03_modelo/saved_model.pb").exists(), "Requiere el modelo real")
    def test_flujo_imagen_a_resultado_con_modelo_real(self):
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        ruido = np.random.default_rng(15).integers(40, 240, (300, 400, 3), dtype=np.uint8)
        contenido = io.BytesIO()
        Image.fromarray(ruido).save(contenido, format="PNG")
        with patch("streamlit.file_uploader", return_value=contenido):
            app = AppTest.from_file(str(RAIZ / "05_app/app.py"), default_timeout=90).run()
            self.assertFalse(app.button(key="analizar").disabled)
            app.button(key="analizar").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            resultado = app.session_state["resultado"]
            self.assertIsNotNone(resultado)
            self.assertAlmostEqual(sum(resultado["probabilidades"]), 1, places=3)
            self.assertEqual(len(app.session_state["historial"]), 1)
            self.assertEqual(len(app.get("progress")), len(cargar_config()["clases"]))

    def test_resultado_ocular_muestra_clase_y_ficha(self):
        from streamlit.testing.v1 import AppTest
        config = cargar_config()
        etiquetas = [c["etiqueta"] for c in config["clases"]]
        valores = [.025] * len(etiquetas)
        valores[etiquetas.index("OjoRosado")] = 1 - .025 * (len(etiquetas) - 1)
        app = AppTest.from_file(str(RAIZ / "05_app/app.py"), default_timeout=30).run()
        app.session_state["resultado"] = {
            "decision": decidir(etiquetas, valores, config), "etiquetas": etiquetas,
            "probabilidades": valores, "calidad": {"medidas": {}}}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("progress")), len(etiquetas))
        self.assertTrue(any("Queratoconjuntivitis infecciosa bovina" in m.value for m in app.markdown))
        self.assertTrue(any("Ojo sano" in m.value for m in app.markdown))
        self.assertFalse(any("La confianza es una salida del modelo" in m.value for m in app.caption))
        self.assertTrue(any("doloroso" in m.value for m in app.markdown))


class VersionModeloTest(unittest.TestCase):
    def test_cambiar_pesos_invalida_inspeccion_y_metricas(self):
        from comun.configuracion import guardar_json
        with tempfile.TemporaryDirectory() as temporal:
            config = cargar_config()
            config["_raiz"] = temporal
            carpeta = Path(temporal) / config["rutas"]["modelo"]
            carpeta.mkdir()
            (carpeta / "variables").mkdir()
            (carpeta / "saved_model.pb").write_bytes(b"grafo")
            pesos = carpeta / "variables" / "variables.data-00000-of-00001"
            pesos.write_bytes(b"pesos anteriores")
            (carpeta / "labels.txt").write_text("0 A\n1 B\n", encoding="utf-8")
            anterior = identidad_modelo(config)["sha256"]
            guardar_json(carpeta / "inspeccion.json", {"modelo_sha256": anterior})
            ruta = Path(temporal) / config["rutas"]["evaluacion"] / "20260101" / "resumen.json"
            guardar_json(ruta, {"modelo_sha256": anterior, "configuracion": config})
            self.assertIsNotNone(leer_inspeccion(config, anterior))
            self.assertIsNotNone(buscar_evaluacion(config, anterior)[0])
            pesos.write_bytes(b"pesos nuevos")
            nueva = identidad_modelo(config)["sha256"]
            self.assertNotEqual(anterior, nueva)
            self.assertIsNone(leer_inspeccion(config, nueva))
            self.assertIsNone(buscar_evaluacion(config, nueva)[0])

    def test_cambiar_preprocesamiento_invalida_metricas(self):
        from comun.configuracion import guardar_json
        with tempfile.TemporaryDirectory() as temporal:
            config = cargar_config()
            config["_raiz"] = temporal
            ruta = Path(temporal) / config["rutas"]["evaluacion"] / "20260101" / "resumen.json"
            guardar_json(ruta, {"modelo_sha256": "huella", "configuracion": config})
            config["modelo"]["preprocesamiento"]["ajuste"] = "redimensionar"
            self.assertIsNone(buscar_evaluacion(config, "huella")[0])


if __name__ == "__main__":
    unittest.main()
