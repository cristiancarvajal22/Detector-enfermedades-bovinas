"""Reúne archivos completos, en orden, para leerlos o copiarlos uno por uno."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = [
    "config.json", "requirements.txt", "requirements-camara.txt", "requirements-lock.txt",
    ".gitignore", ".streamlit/config.toml", "comun/__init__.py", "comun/configuracion.py",
    "comun/imagenes.py", "comun/datos.py", "comun/modelo.py", "01_datos/contar_clases.py", "01_datos/detectar_duplicados.py",
    "02_preparacion/dividir_datos.py", "03_modelo/inspeccionar_modelo.py", "03_modelo/README.md",
    "04_evaluacion/evaluar.py", "05_app/clasificador.py", "05_app/decision.py", "05_app/validacion.py",
    "05_app/presentacion.py", "05_app/entrada.py", "05_app/informacion_modelo.py", "05_app/camara_continua.py", "05_app/app.py",
    "anexos/forward_propagation.py", "anexos/verificar_secretos.py", "anexos/exportar_entrega.py",
    "tests/test_proyecto.py", "tests/test_seguridad.py", "PUBLICAR.md", "README.md",
]


def main():
    partes = ["# Entrega completa del proyecto\n",
              "Los archivos ya están creados en el repositorio. Esta copia reúne su contenido íntegro en el orden de entrega; los pesos binarios proceden del ZIP proporcionado y no se representan como código.\n",
              "El README al final incluye la estructura, instalación, recorrido CRISP-DM y guía para la defensa.\n"]
    for nombre in ARCHIVOS:
        archivo = RAIZ / nombre
        lenguaje = {".py": "python", ".json": "json", ".toml": "toml", ".md": "markdown"}.get(archivo.suffix, "text")
        partes.append(f"## {nombre}\n\n````{lenguaje}\n{archivo.read_text(encoding='utf-8').rstrip()}\n````\n")
    destino = RAIZ / "ENTREGA_COMPLETA.md"
    destino.write_text("\n".join(partes), encoding="utf-8")
    print(destino)


if __name__ == "__main__":
    main()
