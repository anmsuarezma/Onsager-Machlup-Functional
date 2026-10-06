"""Resultados del Bloque A en .npz con metadatos (CLAUDE.md §6).

Duplica a propósito el guardado del bloque estocástico: los dos experimentos no comparten
código fuera de taller.analitico (CLAUDE.md §3).
"""

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


def metadatos_corrida(parametros: dict, semilla: int, hilos: int) -> dict:
    """Parámetros, semilla, fecha, commit, si el árbol tenía cambios sin commit, hilos y versiones."""
    meta = {
        "parametros": parametros,
        "semilla": semilla,
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_modificado": bool(_git("status", "--porcelain")),
        "hilos": hilos,
        "versiones": {p: version(p) for p in ("numpy", "torch", "scipy")} | {"python": platform.python_version()},
    }
    return json.loads(json.dumps(meta))


def guardar_resultado(ruta: Path, datos: dict, metadatos: dict) -> None:
    """Guarda arreglos y metadatos (JSON en la clave `metadatos`) con escritura atómica."""
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
