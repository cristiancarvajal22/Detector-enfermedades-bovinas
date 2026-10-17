"""Despliegue Streamlit: organiza la entrada y los componentes visuales."""
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import UnidentifiedImageError

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "05_app")]
from clasificador import crear_clasificador
from comun.configuracion import cargar_config, ruta_config
from comun.modelo import revision_modelo
from decision import decidir
from entrada import leer_imagen_subida
from informacion_modelo import mostrar_informacion_modelo
from presentacion import aplicar_estilo, aviso, mostrar_ficha, mostrar_historial, mostrar_resultado
from validacion import validar_imagen


@st.cache_resource(show_spinner=False)
def modelo_cacheado(config_serializada, revision):
    return crear_clasificador(json.loads(config_serializada))


def analizar(archivo, config):
    imagen = leer_imagen_subida(archivo, config)
    calidad = validar_imagen(imagen, config["validacion"])
    if not calidad["valida"]:
        return None, calidad
    config_serializada = json.dumps(config, sort_keys=True)
    modelo = modelo_cacheado(config_serializada, revision_modelo(config))
    probabilidades = modelo.predecir(imagen)
    resultado = {"decision": decidir(modelo.etiquetas, probabilidades, config), "etiquetas": modelo.etiquetas,
                 "probabilidades": probabilidades.tolist(), "calidad": calidad}
    return resultado, calidad


def mostrar_analisis(config):
    textos = config["interfaz"]["textos"]
    disponible = (ruta_config(config, "modelo") / "saved_model.pb").is_file() and ruta_config(config, "etiquetas").is_file()
    if not disponible:
        st.warning(textos["modelo_pendiente"])
    st.subheader(textos["entrada"])
    aviso(config)
    st.caption(config["proyecto"]["guia_captura"])
    st.caption(textos["alcance_analisis"])
    fuentes = config["interfaz"]["fuentes_imagen"]
    fuente = st.radio(textos["origen"], fuentes, horizontal=True)
    if fuente == fuentes[0]:
        archivo = st.file_uploader(textos["cargar"], type=config["interfaz"]["tipos_archivo"])
    else:
        st.caption(textos["permiso_camara"])
        archivo = st.camera_input(textos["camara"])
    huella = hashlib.sha256(archivo.getvalue()).hexdigest() if archivo else None
    # No se conserva un resultado que pertenezca a otra imagen o configuración.
    clave = (huella, hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), revision_modelo(config))
    if st.session_state.get("clave_imagen") != clave:
        st.session_state["resultado"] = None
        st.session_state["clave_imagen"] = clave
    izquierda, derecha = st.columns([1, 1.05], gap="large")
    with izquierda:
        if archivo:
            try:
                imagen = leer_imagen_subida(archivo, config)
                st.image(imagen, width="stretch")
            except (OSError, ValueError, Warning, UnidentifiedImageError) as exc:
                logging.getLogger(__name__).warning("Se rechazó una imagen (%s).", type(exc).__name__)
                st.error(textos["error_lectura"])
                archivo = None
        if st.button(textos["analizar"], key="analizar", type="primary", disabled=archivo is None or not disponible, width="stretch"):
            try:
                with st.spinner(textos["cargando"]):
                    resultado, calidad = analizar(archivo, config)
                st.session_state["resultado"] = resultado
                if resultado:
                    registro = {**resultado["decision"], "hora": datetime.now(ZoneInfo(config["proyecto"]["zona_horaria"])).strftime("%Y-%m-%d %H:%M:%S %Z")}
                    st.session_state["historial"].append(registro)
                    st.session_state["historial"] = st.session_state["historial"][-config["interfaz"]["historial_maximo"]:]
                else:
                    for error in calidad["errores"]:
                        st.error(error)
            except Exception as exc:
                st.session_state["resultado"] = None
                logging.getLogger(__name__).error("Falló el análisis (%s).", type(exc).__name__)
                st.error(textos["error_analisis"])
    with derecha:
        if st.session_state.get("resultado"):
            mostrar_resultado(st.session_state["resultado"], config)
        else:
            st.info(textos["sin_imagen"])
    mostrar_historial(config)
    for clase in config["clases"]:
        if clase.get("ficha"):
            with st.expander(f"{textos['ficha']}: {clase.get('nombre', clase['etiqueta'])}"):
                mostrar_ficha(clase["ficha"])


def main():
    config = cargar_config()
    st.set_page_config(page_title=config["proyecto"]["nombre"], layout=config["interfaz"]["ancho"])
    aplicar_estilo(config)
    st.title(config["proyecto"]["nombre"])
    st.write(config["proyecto"]["subtitulo"])
    revision = revision_modelo(config)
    # Una nueva exportación invalida resultados previos aunque conserve las etiquetas.
    if st.session_state.get("version_sesion") != revision:
        st.session_state.update(version_sesion=revision, resultado=None, historial=[])
    textos = config["interfaz"]["textos"]
    analisis, modelo = st.tabs([textos["pestana_analisis"], textos["pestana_modelo"]])
    with analisis:
        mostrar_analisis(config)
    with modelo:
        mostrar_informacion_modelo(config)


if __name__ == "__main__":
    main()
