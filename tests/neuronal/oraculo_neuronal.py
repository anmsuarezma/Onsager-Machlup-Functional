"""Oráculo independiente de las pruebas del hito 02 (CLAUDE.md §2, decisión D9).

Escrito a mano a partir de la especificación 02 y NO importa nada de `taller`.
"""

from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "configs" / "neuronal" / "entrenamiento.yaml"
RESULTADOS = RAIZ / "results" / "neuronal"

# --- Referencias (CLAUDE.md §2) ---------------------------------------------------------
S0 = 4 * np.sqrt(2) / 3  # 1.8856180831641267
S_D_MIN = 1.0  # S_min·D = ΔV


def x_om(t: np.ndarray) -> np.ndarray:
    """−1/√(1 + e^{8t}), escrito a mano (el overflow en t grande da −0, inofensivo)."""
    with np.errstate(over="ignore"):
        return -1 / np.sqrt(1 + np.exp(8 * np.asarray(t, dtype=float)))


def x_kink(tau: np.ndarray) -> np.ndarray:
    """tanh(√2 τ)."""
    return np.tanh(np.sqrt(2) * np.asarray(tau, dtype=float))


# --- Parámetros de la especificación 02 -------------------------------------------------
T_ESCAPE = 3.0
T_INSTANTON = 4.0
PUNTOS_MALLA = 2001
ARQUITECTURAS = {"defecto": [32, 32], "tres_capas_32": [32, 32, 32], "cuatro_capas_64": [64, 64, 64, 64]}
# Parámetros de cada perceptrón 1 → capas → 1, contados a mano: Σ (entradas + 1)·salidas.
PARAMETROS = {
    "defecto": (1 + 1) * 32 + (32 + 1) * 32 + (32 + 1) * 1,  # 1153
    "tres_capas_32": (1 + 1) * 32 + 2 * (32 + 1) * 32 + (32 + 1) * 1,  # 2209
    "cuatro_capas_64": (1 + 1) * 64 + 3 * (64 + 1) * 64 + (64 + 1) * 1,  # 12673
}

# Alineación (misma convención del hito 00 y de E7)
X_ALINEACION_ESCAPE = -1 / np.sqrt(2)
X_ALINEACION_INSTANTON = 0.0
INTERVALO_RMS_ESCAPE = (-1.0, 0.5)
INTERVALO_RMS_INSTANTON = (-2.0, 2.0)

# --- Tolerancias de la especificación 02 ------------------------------------------------
TOL_ACCION = 0.01  # |S·D − 1| y |S_E − S0|/S0
TOL_RMS = 0.01
TOL_MALLA = 1e-4  # diferencia relativa al duplicar la malla
TOL_COTA = 1e-4  # S·D ≥ 1 − 1e-4, S_E ≥ S0 − 1e-4


def cargar_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
