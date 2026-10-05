"""Oráculo independiente de las pruebas del hito 01 (CLAUDE.md §2, decisión D9).

Escrito a mano a partir de la especificación 01 y NO importa nada de `taller`.
Las tablas de T son las mismas de tests/analitico/test_tiempo_primer_paso.py (mpmath,
40 dígitos); se copian aquí porque pytest no comparte módulos entre carpetas de pruebas.
"""

from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIGS = RAIZ / "configs" / "estocastico"
RESULTADOS = RAIZ / "results" / "estocastico"

# --- Parámetros de la especificación 01 -------------------------------------------------
VALORES_D = [0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5]  # decisión 3
VALORES_DT_E3 = [1e-2, 5e-3, 1e-3, 5e-4]  # decisión 4
D_E3 = 0.25
D_E7 = [0.1, 0.15, 0.25]
X0 = -1.0
X_ALINEACION = -1 / np.sqrt(2)  # decisión 7: x_om(0)
MUESTREO_VENTANA = 0.01
MUESTRAS_VENTANA = 351  # aclaración 11: t_cima − 3, ..., t_cima + 0.5 (cada 0.01)
INDICE_CIMA = 300  # muestra en t_cima (corregido)
MALLA_E7 = (-0.5, 0.25)  # aclaración 11: intervalo del criterio RMS, x_om ≤ −0.345
HILOS_POR_DEFECTO = 10  # aclaración 6 (D23), reemplaza los 16 de D22
N_E3 = 100_000  # aclaración 8
N_E4 = 100_000
N_E4_D01 = 20_000  # aclaración 9

# E2 (aclaración 13)
D_E2 = 0.5
DT_E2 = 1e-3
N_E2 = 100_000
T_EQUILIBRIO_E2 = 50.0
INTERVALO_E2 = 5.0
MUESTRAS_E2 = 20
ANCHO_E2 = 0.05

# --- Validación del puente browniano (aclaración 7) -------------------------------------
# Movimiento browniano sin deriva dx = √(2D) dW desde x = 0, nivel a > 0. Principio de
# reflexión: P(τ_a ≤ t) = 2·P(x_t ≥ a) = erfc(a/√(4 D t)).
D_BROWNIANO = 0.5
NIVEL_BROWNIANO = 1.0
DT_BROWNIANO = 0.05  # 20 pasos hasta t = 1: el sesgo sin corregir es grande
P_ALCANCE_BROWNIANO = {
    0.5: 0.15729920705028516,  # erfc(1)
    1.0: 0.31731050786291415,  # erfc(1/√2)
}

# --- Tolerancias de la especificación 01 ------------------------------------------------
P_KS_E1 = 0.01
L1_E2 = 0.02
ERROR_DT_E3 = 0.02
ERROR_E4 = 0.03
PENDIENTE_E5 = 0.05
CV_E6 = 0.05

# --- T(−1 → b) exacto (mpmath, 40 dígitos) ----------------------------------------------
T_CIMA = {
    0.1: 12762.675627150007,
    0.125: 1749.1908845846764,
    0.15: 467.17537566755423,
    0.2: 90.475939875231091,
    0.25: 33.945744287679815,
    0.35: 11.019610799182002,
    0.5: 4.6065396131071025,
}
T_POZO = {
    0.1: 25527.091364292459,
    0.125: 3500.0348111867892,
    0.15: 935.93210455964312,
    0.2: 182.4176729855807,
    0.25: 69.263644760836284,
    0.35: 23.259751015597875,
    0.5: 10.258570067184178,
}


def x_om(t: np.ndarray) -> np.ndarray:
    """Camino de Onsager-Machlup −1/√(1 + e^{8t}), escrito a mano (overflow inofensivo a −0)."""
    with np.errstate(over="ignore"):
        return -1 / np.sqrt(1 + np.exp(8 * np.asarray(t, dtype=float)))


def semillas_de_pruebas() -> dict:
    """Semillas de las pruebas, leídas de configs/estocastico/pruebas.yaml (CLAUDE.md §6)."""
    with open(CONFIGS / "pruebas.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["semillas"]
