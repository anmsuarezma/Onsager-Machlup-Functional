"""Resultados crudos del Bloque B en .npz con sus metadatos (CLAUDE.md §6, aclaración 2, D26)."""

import json
import os
import platform
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]


def _git(*argumentos: str) -> str:
    return subprocess.run(["git", *argumentos], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def metadatos_corrida(parametros: dict, semilla: int, info_hilos: dict) -> dict:
    """Metadatos de una corrida: parámetros, semilla, fecha, commit, hilos y versiones.

    `arbol_modificado` indica si había cambios sin commit, incluidos archivos nuevos no
    versionados (los ignorados por .gitignore, como los resultados, no cuentan): en ese caso
    el hash no basta para reproducir la corrida. Se normaliza con una ida y vuelta por JSON,
    para que lo que se guarda sea exactamente lo que se lee.
    """
    meta = {
        "parametros": parametros,
        "semilla": semilla,
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_modificado": bool(_git("status", "--porcelain")),
        "hilos": info_hilos["hilos"],
        "capa_hilos": info_hilos["capa_hilos"],
        "versiones": {p: version(p) for p in ("numpy", "numba", "llvmlite", "scipy")}
        | {"python": platform.python_version()},
    }
    return json.loads(json.dumps(meta, default=_a_json))


def _a_json(objeto):
    if isinstance(objeto, np.generic):
        return objeto.item()
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    raise TypeError(f"no serializable: {type(objeto)}")


def guardar_resultado(ruta: Path, datos: dict, metadatos: dict) -> None:
    """Guarda arreglos y metadatos (JSON en la clave `metadatos`) de forma atómica (D26).

    Se escribe en un archivo temporal del mismo directorio y se renombra con os.replace:
    un apagado durante la escritura no deja un .npz truncado con el nombre final.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f".{ruta.name}.tmp")
    with open(temporal, "wb") as f:
        np.savez(f, metadatos=np.array(json.dumps(metadatos, ensure_ascii=False)), **datos)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporal, ruta)


def cargar_resultado(ruta: Path) -> tuple[dict, dict]:
    """Lee un resultado guardado con guardar_resultado: (arreglos, metadatos)."""
    with np.load(ruta, allow_pickle=False) as f:
        datos = {k: f[k] for k in f.files if k != "metadatos"}
        metadatos = json.loads(str(f["metadatos"]))
    return datos, metadatos
