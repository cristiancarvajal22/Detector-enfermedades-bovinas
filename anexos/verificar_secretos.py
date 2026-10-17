"""Revisa archivos y el historial local sin imprimir posibles credenciales."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

RAIZ = Path(__file__).resolve().parents[1]
OMITIR = {".git", ".venv", "__pycache__", "archive", "node_modules", ".seguridad"}
EXTENSIONES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg",
               ".env", ".pem", ".key", ".ps1", ".sh", ".js", ".ts", ".tsx", ".pb", ".index"}
PATRONES = {
    "clave_privada": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "token_openai": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b",
    "token_anthropic": r"\bsk-ant-[A-Za-z0-9_-]{24,}\b",
    "clave_google": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "clave_aws": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "token_github": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b",
    "token_slack": r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "url_con_credenciales": r"https?://[^\s/:\"']+:[^\s/@\"']+@",
    "credencial_asignada": r"(?im)[\"']?(?:api[_-]?key|client[_-]?secret|access[_-]?token|password)[\"']?\s*[:=]\s*[\"'][^\s\"']{12,}[\"']",
}


def detectar(contenido, ubicacion):
    """Solo devuelve nombres de reglas y líneas; nunca guarda la coincidencia."""
    texto = contenido.decode("utf-8", errors="replace")
    return [{"archivo": ubicacion, "linea": texto.count("\n", 0, m.start()) + 1, "regla": nombre}
            for nombre, patron in PATRONES.items() for m in re.finditer(patron, texto)]


def git(raiz, *argumentos):
    return subprocess.check_output(["git", "-C", str(raiz), *argumentos], timeout=30)


def revisar(raiz=RAIZ):
    raiz = Path(raiz).resolve()
    hallazgos, archivos = [], 0
    for carpeta, directorios, nombres in os.walk(raiz):
        directorios[:] = [d for d in directorios if d not in OMITIR]
        for nombre in nombres:
            ruta = Path(carpeta) / nombre
            especial = nombre.startswith((".env", "variables.data-", "id_rsa", "id_ed25519")) or nombre in (".gitignore", "config")
            if ruta.suffix.lower() not in EXTENSIONES and not especial:
                continue
            if ruta.is_symlink() or ruta.stat().st_size > 32 * 1024 * 1024:
                continue
            relativo = ruta.relative_to(raiz).as_posix()
            archivos += 1
            encontrados = detectar(ruta.read_bytes(), relativo)
            if encontrados:
                ignorado = subprocess.run(["git", "-C", str(raiz), "check-ignore", "-q", "--", relativo], timeout=10).returncode == 0
                hallazgos.extend({**h, "ignorado_por_git": ignorado} for h in encontrados)
    # .git/config puede tener una URL con credenciales aunque nunca se publique.
    configuracion_git = raiz / ".git" / "config"
    if configuracion_git.is_file():
        hallazgos.extend({**h, "ignorado_por_git": True} for h in detectar(configuracion_git.read_bytes(), ".git/config"))
    commits = git(raiz, "rev-list", "--all").decode().splitlines()
    historial, vistos = [], set()
    for commit in commits:
        for entrada in git(raiz, "ls-tree", "-r", commit).decode("utf-8", errors="replace").splitlines():
            datos, nombre = entrada.split("\t", 1)
            _, tipo, objeto = datos.split()
            if tipo != "blob" or objeto in vistos:
                continue
            vistos.add(objeto)
            contenido = git(raiz, "cat-file", "blob", objeto)
            historial.extend(detectar(contenido, f"{commit[:12]}:{nombre}"))
    return {"archivos_revisados": archivos, "commits_revisados": len(commits),
            "objetos_git_revisados": len(vistos), "hallazgos_archivos": hallazgos,
            "hallazgos_historial": historial,
            "limite": "Revisión por patrones del proyecto y del historial local. No demuestra ausencia de todo secreto ni revisa cuentas externas o el historial remoto no descargado."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    informe = revisar()
    contenido = json.dumps(informe, ensure_ascii=False, indent=2)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(contenido, encoding="utf-8")
    print(contenido)
    raise SystemExit(1 if informe["hallazgos_archivos"] or informe["hallazgos_historial"] else 0)


if __name__ == "__main__":
    main()
