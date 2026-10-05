"""Resultados .npz con metadatos (CLAUDE.md §6 y aclaración 2): ida y vuelta, campos obligatorios y escritura atómica (D26)."""

import re
import subprocess

import numpy as np
from oraculo_estocastico import RAIZ

from taller.estocastico.guardado import cargar_resultado, guardar_resultado, metadatos_corrida

CAMPOS = {"parametros", "semilla", "fecha", "commit", "arbol_modificado", "hilos", "capa_hilos", "versiones"}


def test_metadatos_tienen_los_campos_obligatorios() -> None:
    meta = metadatos_corrida({"D": [0.25], "dt": 1e-3}, semilla=7, info_hilos={"hilos": 16, "capa_hilos": "tbb"})
    assert CAMPOS <= set(meta)
    assert {"numpy", "numba", "llvmlite"} <= set(meta["versiones"])
    assert meta["semilla"] == 7 and meta["hilos"] == 16


def test_commit_es_el_head_de_git() -> None:
    meta = metadatos_corrida({}, semilla=0, info_hilos={"hilos": 1, "capa_hilos": "x"})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}", meta["commit"]) and meta["commit"] == head


def test_ida_y_vuelta(tmp_path) -> None:
    datos = {"t_cima": np.arange(5.0), "t_pozo": np.arange(5.0) * 2}
    meta = metadatos_corrida({"D": [0.25]}, semilla=3, info_hilos={"hilos": 2, "capa_hilos": "omp"})
    ruta = tmp_path / "prueba.npz"
    guardar_resultado(ruta, datos, meta)
    datos2, meta2 = cargar_resultado(ruta)
    assert all(np.array_equal(datos[k], datos2[k]) for k in datos)
    assert meta2 == meta


def test_escritura_atomica_sin_restos(tmp_path) -> None:
    """D26: se escribe con nombre temporal y se renombra; al terminar solo queda el .npz final,
    y reescribir un resultado existente lo reemplaza."""
    meta = metadatos_corrida({}, semilla=1, info_hilos={"hilos": 1, "capa_hilos": "x"})
    ruta = tmp_path / "e4_D0.250.npz"
    guardar_resultado(ruta, {"t_pozo": np.zeros(3)}, meta)
    guardar_resultado(ruta, {"t_pozo": np.ones(3)}, meta)
    assert [p.name for p in tmp_path.iterdir()] == ["e4_D0.250.npz"]
    assert np.array_equal(cargar_resultado(ruta)[0]["t_pozo"], np.ones(3))
