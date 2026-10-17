"""Información del modelo para las personas que utilizan el tamizaje."""
import json

import streamlit as st

from clasificador import leer_etiquetas
from comun.configuracion import ruta_config
from comun.modelo import identidad_modelo, revision_modelo
from presentacion import nombre_clase


@st.cache_data(show_spinner=False)
def identidad_cacheada(config_serializada, revision):
    return identidad_modelo(json.loads(config_serializada))


def leer_inspeccion(config, huella):
    archivo = ruta_config(config, "modelo") / "inspeccion.json"
    try:
        informe = json.loads(archivo.read_text(encoding="utf-8"))
        if informe.get("modelo_sha256") == huella:
            return informe
    except (OSError, ValueError, AttributeError):
        pass
    return None


def buscar_evaluacion(config, huella):
    raiz = ruta_config(config, "evaluacion")
    for archivo in sorted(raiz.glob("*/resumen.json"), reverse=True):
        try:
            informe = json.loads(archivo.read_text(encoding="utf-8"))
            misma_config = all(informe["configuracion"].get(c) == config[c]
                               for c in ("modelo", "decision", "validacion"))
            if informe.get("modelo_sha256") == huella and misma_config:
                return informe, archivo.parent
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None, None


def mostrar_informacion_modelo(config):
    textos = config["interfaz"]["textos"]
    st.subheader(textos["titulo_modelo"])
    st.write(textos["descripcion_modelo"])
    if not (ruta_config(config, "modelo") / "saved_model.pb").is_file() or not ruta_config(config, "etiquetas").is_file():
        st.warning(textos["modelo_pendiente"])
        return
    try:
        identidad = identidad_cacheada(json.dumps(config, sort_keys=True), revision_modelo(config))
        etiquetas = leer_etiquetas(ruta_config(config, "etiquetas"))
    except (OSError, ValueError):
        st.error("No se pudo leer la información del modelo. Contacta al responsable de la aplicación.")
        return
    a, b = st.columns(2)
    a.metric("Clases de salida", len(etiquetas))
    b.metric("Tamaño del modelo", f"{identidad['bytes'] / 1024**2:.2f} MiB")
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        st.subheader(textos["clases_modelo"])
        for indice, etiqueta in enumerate(etiquetas):
            st.write(f"**{indice + 1}. {nombre_clase(etiqueta, config)}**")
    with derecha:
        st.subheader(textos["preprocesamiento_modelo"])
        preproceso = config["modelo"]["preprocesamiento"]
        st.write(f"**Imagen:** {preproceso['ancho']} × {preproceso['alto']} píxeles, tres canales RGB.")
        ajustes = {"recorte_central": "Recorte central conservando las proporciones", "redimensionar": "Redimensionado directo"}
        escalas = {"menos_uno_a_uno": "De −1 a 1 (x / 127.5 − 1)", "cero_a_uno": "De 0 a 1 (x / 255)", "sin_normalizar": "De 0 a 255"}
        st.write(f"**Ajuste:** {ajustes.get(preproceso['ajuste'], preproceso['ajuste'])}.")
        st.write(f"**Normalización:** {escalas.get(preproceso['normalizacion'], preproceso['normalizacion'])}.")
    st.divider()
    st.subheader(textos["reglas_modelo"])
    a, b = st.columns(2)
    a.metric("Confianza mínima", f"{config['decision']['umbral']:.0%}")
    b.metric("Separación mínima entre clases", f"{config['decision']['margen'] * 100:.0f} puntos")
    st.write("Si no cumple ambos límites, responde «No estoy seguro». Si gana la clase de imagen no válida, solicita otra fotografía.")
