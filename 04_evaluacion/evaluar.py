"""Evaluación cerrada multiclase y cobertura de las reglas de abstención."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from clasificador import crear_clasificador
from decision import decidir
from validacion import validar_imagen
from comun.configuracion import cargar_config, guardar_json, ruta_config
from comun.imagenes import abrir_rgb, hash_archivo, hash_pixeles, listar_imagenes
from comun.modelo import identidad_modelo


def verificar_division(config):
    raiz = ruta_config(config, "division")
    manifiesto = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    registros = manifiesto["registros"]
    if not registros:
        raise ValueError("El manifiesto está vacío.")
    vistos = {"entrenamiento": set(), "prueba": set()}
    rutas = set()
    for registro in registros:
        relativo = Path(registro["destino"])
        archivo = (raiz / relativo).resolve()
        if raiz not in archivo.parents or registro["conjunto"] not in vistos:
            raise ValueError("Ruta o conjunto inválido en el manifiesto.")
        clase = next((c for c in config["clases"] if c["etiqueta"] == registro["clase"]), None)
        if not clase or relativo.parts[:2] != (registro["conjunto"], clase["carpeta"]):
            raise ValueError("La clase no coincide con la carpeta del manifiesto.")
        if archivo in rutas or hash_archivo(archivo) != registro["sha256"]:
            raise ValueError(f"Imagen modificada, faltante o repetida en manifiesto: {archivo}")
        rutas.add(archivo)
        pixel = hash_pixeles(archivo)
        if pixel != registro["pixeles_sha256"]:
            raise ValueError(f"Píxeles modificados: {archivo}")
        vistos[registro["conjunto"]].add(pixel)
    reales = {p.resolve() for conjunto in vistos for p in listar_imagenes(raiz / conjunto, config["datos"]["extensiones"])}
    if reales != rutas:
        raise ValueError("Hay imágenes fuera del manifiesto o faltan imágenes en la división.")
    if vistos["entrenamiento"] & vistos["prueba"]:
        raise ValueError("Fuga de datos: hay imágenes idénticas entre entrenamiento y prueba.")
    for conjunto in vistos:
        for clase in config["clases"]:
            if not any(r["conjunto"] == conjunto and r["clase"] == clase["etiqueta"] for r in registros):
                raise ValueError(f"Falta {clase['etiqueta']} en {conjunto}.")
    return registros


def evaluar_conjunto(modelo, registros, config, salida, conjunto):
    seleccion = [r for r in registros if r["conjunto"] == conjunto]
    reales, predichas, filas, aceptadas = [], [], [], []
    estados_firmes = {c["riesgo"] for c in config["clases"] if c["rol"] != "invalida"}
    for registro in seleccion:
        imagen = abrir_rgb(ruta_config(config, "division") / registro["destino"], config["validacion"]["max_pixeles"])
        probabilidades = modelo.predecir(imagen)
        predicha = modelo.etiquetas[int(np.argmax(probabilidades))]
        decision = decidir(modelo.etiquetas, probabilidades, config)
        calidad = validar_imagen(imagen, config["validacion"])
        aceptada = calidad["valida"] and decision["estado"] in estados_firmes
        # Se evalúa el modelo sobre TODAS las imágenes; no se ocultan casos difíciles.
        reales.append(registro["clase"])
        predichas.append(predicha)
        aceptadas.append(aceptada)
        fila = {"archivo": registro["destino"], "real": registro["clase"], "predicha": predicha,
                "estado": decision["estado"], "calidad_valida": calidad["valida"],
                "motivos_calidad": " | ".join(calidad["errores"]), "aceptada_app": aceptada}
        fila.update({f"p_{etiqueta}": float(p) for etiqueta, p in zip(modelo.etiquetas, probabilidades)})
        filas.append(fila)
    reporte = classification_report(reales, predichas, labels=modelo.etiquetas, output_dict=True, zero_division=0)
    matriz = confusion_matrix(reales, predichas, labels=modelo.etiquetas)
    indices = [i for i, aceptada in enumerate(aceptadas) if aceptada]
    resumen = {"cantidad": len(reales), "exactitud": float(accuracy_score(reales, predichas)),
               "por_clase": {etiqueta: reporte[etiqueta] for etiqueta in modelo.etiquetas},
               "promedio_macro": reporte["macro avg"], "etiquetas": modelo.etiquetas,
               "matriz_confusion": matriz.tolist(), "aceptadas_app": len(indices),
               "cobertura_app": len(indices) / len(reales),
               "exactitud_aceptadas_app": float(accuracy_score([reales[i] for i in indices], [predichas[i] for i in indices])) if indices else None}
    guardar_json(salida / f"metricas_{conjunto}.json", resumen)
    with (salida / f"predicciones_{conjunto}.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    fig, ax = plt.subplots(figsize=(11, 8))
    # Índices cortos mantienen legibles nombres largos en el gráfico exportado.
    nombres = [f"C{i}" for i in range(len(modelo.etiquetas))]
    ConfusionMatrixDisplay(matriz, display_labels=nombres).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set(xlabel="Clase predicha", ylabel="Clase real", title=f"Matriz de confusión · {conjunto} · conteo de imágenes")
    fig.text(0.02, 0.01, "\n".join(f"C{i}: {e}" for i, e in enumerate(modelo.etiquetas)), fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(salida / f"matriz_{conjunto}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        config = cargar_config(args.config)
        registros = verificar_division(config)
        modelo = crear_clasificador(config)
        salida = ruta_config(config, "evaluacion") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        salida.mkdir(parents=True, exist_ok=False)
        resultados = {c: evaluar_conjunto(modelo, registros, config, salida, c) for c in ("entrenamiento", "prueba")}
        entrenamiento, prueba = resultados["entrenamiento"]["exactitud"], resultados["prueba"]["exactitud"]
        brecha = entrenamiento - prueba
        avisos = []
        if not config["modelo"]["preprocesamiento"]["confirmado"]:
            avisos.append("Normalización y recorte basados en el ejemplo oficial de Teachable Machine; falta comprobar equivalencia con la exportación concreta.")
        if brecha > config["evaluacion"]["brecha_alerta"]:
            avisos.append("Posible sobreajuste o cambio de distribución: la brecha supera el umbral configurado.")
        if max(entrenamiento, prueba) < config["evaluacion"]["exactitud_baja"]:
            avisos.append("Desempeño bajo en ambos conjuntos: compatible con subajuste, ruido de etiquetas o preprocesamiento incorrecto; no lo demuestra.")
        if not config["datos"]["prueba_independiente_confirmada"]:
            avisos.append("PRUEBA NO CONFIRMADA COMO INDEPENDIENTE: resultados exploratorios; no justifican generalización.")
        informe = {"resultados": resultados, "brecha_exactitud": brecha, "avisos": avisos,
                   "modelo_sha256": identidad_modelo(config)["sha256"],
                   "modelo": modelo.inspeccion(), "configuracion": {k: v for k, v in config.items() if not k.startswith("_")},
                   "modelo_pb_sha256": hash_archivo(ruta_config(config, "modelo") / "saved_model.pb"),
                   "manifiesto_sha256": hash_archivo(ruta_config(config, "division") / "manifiesto.json"),
                   "limite": "Una brecha pequeña no prueba ausencia de sobreajuste; revisar independencia por animal, fuente y sesión."}
        guardar_json(salida / "resumen.json", informe)
        print(f"Exactitud entrenamiento: {entrenamiento:.2%}; prueba: {prueba:.2%}; brecha: {brecha:.2%}")
        for aviso in avisos:
            print(f"AVISO: {aviso}")
        print(f"Resultados: {salida}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluación detenida: {exc}\n")


if __name__ == "__main__":
    main()
