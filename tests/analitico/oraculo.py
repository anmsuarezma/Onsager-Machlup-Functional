"""Oráculo independiente de las pruebas del hito 00 (CLAUDE.md §2, decisión D9).

Todo lo que hay aquí está escrito a mano a partir de la especificación y NO importa nada
de `taller`: es el valor esperado contra el que se valida el módulo. Los símbolos se
definen con los mismos nombres y supuestos que la interfaz del módulo (decisión D11);
sympy considera iguales dos símbolos con el mismo nombre y los mismos supuestos.

Es un módulo normal (no `conftest.py`) porque pytest desaconseja importar desde conftest.
"""

import numpy as np
import sympy as sp

# --- Símbolos (mismos nombres y supuestos que taller.analitico.potencial) -------------
x, v, t, tau = sp.symbols("x v t tau", real=True)
D = sp.Symbol("D", positive=True)

# --- Potencial de doble pozo y derivadas (CLAUDE.md §2) --------------------------------
V_ESP = (x**2 - 1) ** 2
DV_ESP = 4 * x * (x**2 - 1)
D2V_ESP = 12 * x**2 - 4
D3V_ESP = 24 * x

# --- Soluciones cerradas, ramas y constantes (especificación 00) ------------------------
X_OM_ESP = -1 / sp.sqrt(1 + sp.exp(8 * t))
X_KINK_ESP = sp.tanh(sp.sqrt(2) * tau)
RAMA_KINK_ESP = sp.sqrt(2) * (1 - x**2)  # √(2V) en |x| < 1
S0_ESP = 4 * sp.sqrt(2) / 3
S_MIN_ESP = 1 / D
TAU_KRAMERS_ESP = 2 * sp.pi / sp.sqrt(32) * sp.exp(1 / D)

# Condiciones iniciales en el origen de tiempo (fijan el modo cero).
X_OM_0 = -1 / np.sqrt(2)
X_KINK_0 = 0.0

# Lados derechos de las EDO de primer orden, en numpy.
def rhs_om(x_: np.ndarray) -> np.ndarray:
    """ẋ = V′(x) = 4x(x² − 1)."""
    return 4 * x_ * (x_**2 - 1)


def rhs_kink(x_: np.ndarray) -> np.ndarray:
    """ẋ = √(2V) = √2(1 − x²), válido en |x| < 1."""
    return np.sqrt(2) * (1 - x_**2)


VALORES_D = [0.1, 0.15, 0.25, 0.35, 0.5]


# --- Malla determinista irregular (aclaración 9) -----------------------------------------
def malla_irregular(n: int, a: float, b: float, alfa: float) -> np.ndarray:
    """Puntos {k·alfa mod 1} reescalados a [a, b], con alfa irracional: espaciado irregular y sin semilla.

    Se eliminan los puntos a menos de MARGEN de los puntos fijos −1, 0, +1, donde V′ = 0
    y las identidades se cumplirían trivialmente.
    """
    k = np.arange(1, n + 1)
    puntos = a + (b - a) * ((k * alfa) % 1.0)
    lejos = np.min(np.abs(puntos[:, None] - PUNTOS_FIJOS[None, :]), axis=1) >= MARGEN
    return puntos[lejos]


PUNTOS_FIJOS = np.array([-1.0, 0.0, 1.0])
MARGEN = 1e-2
ALFA_X = (np.sqrt(5) - 1) / 2  # razón áurea
ALFA_V = np.sqrt(2) - 1
