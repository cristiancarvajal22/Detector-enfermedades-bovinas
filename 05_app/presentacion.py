"""Componentes visuales; no contienen reglas de clasificación."""
import html
import json

import streamlit as st


def aplicar_estilo(config):
    ui = config["interfaz"]
    arrastrar = json.dumps(ui["textos"]["arrastrar"], ensure_ascii=False)
    explorar = json.dumps(ui["textos"]["explorar"], ensure_ascii=False)
    # Los colores provienen de configuración local; los datos de usuario se escapan.
    st.markdown(f"""<style>
    .stApp {{background:{ui['color_fondo']}; color:{ui['color_texto']};}}
    .block-container {{max-width:1180px; padding-top:2.5rem; padding-bottom:3rem;}}
    h1 {{font-size:clamp(2rem,4vw,3rem)!important; letter-spacing:-.045em; line-height:1.12!important;}}
    h2 {{font-size:1.45rem!important; letter-spacing:-.02em;}}
    h3 {{font-size:1.25rem!important;}}
    [data-testid="stImage"] img {{border-radius:{ui['radio_px']}px;}}
    [data-testid="stBaseButton-primary"] {{background:{ui['color_acento']}; border-color:{ui['color_acento']}; border-radius:{ui['radio_px']}px;}}
    /* Se conserva la app local sin los controles de publicación de Streamlit. */
    [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu {{display:none!important;}}
    .resultado {{border-radius:{ui['radio_px']}px; padding:1.7rem; margin:.5rem 0 1rem;}}
    .resultado-icono {{font-size:2.2rem; font-weight:700; line-height:1;}}
    .resultado h2 {{margin:.6rem 0;}}
    .resultado p {{line-height:1.65; margin-bottom:.5rem;}}
    .probabilidad {{display:flex; justify-content:space-between; gap:1rem; margin:.8rem 0 .25rem;}}
    .historial-fila {{padding:.9rem 0; border-bottom:1px solid #D8DED5;}}
    /* El control nativo no expone traducción de estas dos etiquetas. */
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {{display:none;}}
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{content:{arrastrar};}}
    [data-testid="stFileUploaderDropzone"] button {{font-size:0;}}
    [data-testid="stFileUploaderDropzone"] button::after {{content:{explorar}; font-size:.875rem;}}
    @media(max-width:640px) {{.block-container {{padding:1.25rem 1rem 3rem;}} .resultado {{padding:1.2rem;}}}}
    </style>""", unsafe_allow_html=True)


def aviso(config):
    st.caption(config["proyecto"]["aviso"])


def nombre_clase(etiqueta, config):
    clase = next(c for c in config["clases"] if c["etiqueta"] == etiqueta)
    return clase.get("nombre", etiqueta)


def mostrar_resultado(resultado, config):
    decision = resultado["decision"]
    estado = config["estados"][decision["estado"]]
    titulo = html.escape(estado["titulo"])
    recomendacion = html.escape(decision["recomendacion"])
    textos = config["interfaz"]["textos"]
    # Se distingue la clase sugerida del nivel de riesgo, en especial entre piel y ojos.
    st.write(f"**{textos['clase_sugerida']}:** {nombre_clase(decision['clase'], config)}")
    st.markdown(f'''<section class="resultado" role="status" style="background:{estado['fondo']};color:{estado['color']}">
    <div class="resultado-icono" aria-hidden="true">{html.escape(estado['icono'])}</div>
    <h2>{titulo}</h2><p>{recomendacion}</p></section>''', unsafe_allow_html=True)
    if decision["estado"] == "invalida":
        st.info(config["proyecto"]["guia_captura"])
    textos = config["interfaz"]["textos"]
    st.subheader(textos["barras"])
    for etiqueta, valor in zip(resultado["etiquetas"], resultado["probabilidades"]):
        st.markdown(f'<div class="probabilidad"><span>{html.escape(nombre_clase(etiqueta, config))}</span><strong>{valor:.1%}</strong></div>', unsafe_allow_html=True)
        st.progress(float(valor))
    with st.expander(textos["detalle"]):
        st.write(decision["motivo"])
        st.write({"clase_con_mayor_puntuacion": nombre_clase(decision["clase"], config), "confianza": f"{decision['confianza']:.1%}",
                  "margen": f"{decision['margen']:.1%}", "umbral_configurado": f"{config['decision']['umbral']:.0%}",
                  "margen_minimo": f"{config['decision']['margen']:.0%}", "calidad": resultado["calidad"]["medidas"]})
    clase = next(c for c in config["clases"] if c["etiqueta"] == decision["clase"])
    if clase.get("ficha"):
        with st.expander(textos["ficha"]):
            mostrar_ficha(clase["ficha"])


def mostrar_ficha(ficha):
    for titulo, texto in ficha.items():
        if titulo not in ("nota_revision", "fuentes"):
            st.markdown(f"**{titulo}**")
            st.write(texto)
    st.warning(ficha["nota_revision"])
    for fuente in ficha["fuentes"]:
        st.link_button(fuente["titulo"], fuente["url"])


def mostrar_historial(config):
    textos = config["interfaz"]["textos"]
    st.divider()
    st.subheader(textos["historial"])
    if st.button(textos["borrar"], key="borrar_historial"):
        st.session_state["historial"] = []
    historial = st.session_state.get("historial", [])
    if not historial:
        st.caption(textos["sin_historial"])
    for registro in reversed(historial):
        estado = config["estados"][registro["estado"]]
        st.markdown(f'<div class="historial-fila"><strong>{html.escape(estado["icono"])} {html.escape(estado["titulo"])}</strong><br>{html.escape(registro["hora"])} · {html.escape(nombre_clase(registro["clase"], config))} · {registro["confianza"]:.1%}</div>', unsafe_allow_html=True)
