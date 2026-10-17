"""La decisión de tamizaje es independiente de TensorFlow y Streamlit."""
import numpy as np

from clasificador import validar_probabilidades


def decidir(etiquetas, probabilidades, config):
    valores = validar_probabilidades(probabilidades, len(etiquetas))
    orden = np.argsort(-valores, kind="stable")
    primero, segundo = int(orden[0]), int(orden[1])
    confianza, margen = float(valores[primero]), float(valores[primero] - valores[segundo])
    clase = next(c for c in config["clases"] if c["etiqueta"] == etiquetas[primero])
    # La clase de rechazo siempre solicita otra foto, incluso con poca confianza.
    if clase["rol"] == "invalida":
        estado, motivo = "invalida", config["decision"]["motivos"]["invalida"]
        recomendacion = clase["recomendacion"]
    elif confianza + 1e-12 < config["decision"]["umbral"] or margen + 1e-12 < config["decision"]["margen"]:
        estado, motivo = "incierto", config["decision"]["motivos"]["incierto"]
        recomendacion = config["estados"][estado]["recomendacion"]
    else:
        estado, motivo = clase["riesgo"], config["decision"]["motivos"]["aceptada"]
        recomendacion = clase["recomendacion"]
    return {"estado": estado, "clase": etiquetas[primero], "confianza": confianza,
            "margen": margen, "motivo": motivo, "recomendacion": recomendacion}
