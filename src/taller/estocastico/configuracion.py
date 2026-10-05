"""Configuración de las corridas del Bloque B: archivos YAML, hilos de Numba y semillas.

Todo lo aleatorio recibe su semilla de un archivo de configs/estocastico/ (CLAUDE.md §6).
"""

from pathlib import Path

import numba
import numpy as np
import yaml


def cargar_config(ruta: Path) -> dict:
    """Lee un archivo YAML de configuración."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_hilos(config: dict, hilos_cli: int | None) -> int:
    """Número de hilos: la bandera --hilos tiene prioridad sobre la configuración (D23)."""
    return int(hilos_cli) if hilos_cli is not None else int(config["hilos"])


def aplicar_hilos(n: int) -> dict:
    """Fija los hilos de Numba y devuelve lo que se guarda en los metadatos.

    La capa de hilos solo se conoce después de ejecutar una función paralela, así que se
    ejecuta una mínima antes de consultarla.
    """
    numba.set_num_threads(n)
    _arranque_paralelo(np.zeros(n))
    return {"hilos": n, "capa_hilos": numba.threading_layer()}


@numba.njit(parallel=True, cache=True)
def _arranque_paralelo(a: np.ndarray) -> float:
    s = 0.0
    for i in numba.prange(a.size):
        s += a[i]
    return s


def semillas_trayectorias(semilla: int, N: int) -> np.ndarray:
    """Una semilla por trayectoria, derivada de la semilla de configuración (D27).

    `SeedSequence` produce semillas estadísticamente independientes de 64 bits, y cada
    trayectoria inicia su propio generador con la suya (D28): el resultado no depende del
    número de hilos. Con 32 bits, 10⁵ semillas tendrían ~1 colisión esperada (N²/2³³).
    """
    semillas = np.random.SeedSequence(semilla).generate_state(N, dtype=np.uint64)
    if len(np.unique(semillas)) < N:  # colisión en 64 bits: ~N²/2⁶⁵, se detecta igual
        raise ValueError(f"semillas repetidas para semilla={semilla}, N={N}")
    return semillas


def n_trayectorias(config: dict, D: float) -> int:
    """N de un D en E4: el valor general, salvo excepción en N_por_D (aclaración 9)."""
    por_D = config.get("N_por_D") or {}
    return int(por_D.get(D, config["N"]))
