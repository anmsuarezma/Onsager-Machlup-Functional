"""Cantidades de referencia en numpy, escritas a mano (decisión D10).

Es lo que importan los bloques estocástico y neuronal: no depende de sympy. La prueba del
criterio 9 compara cada función con la expresión simbólica de sympy evaluada con lambdify.

Notación: tau es el tiempo imaginario del instantón; el tiempo medio de escape es τ_esc.
"""

import numpy as np

DELTA_V = 1.0  # altura de la barrera V(0) − V(−1)
CURVATURA_POZO = 8.0  # V″(±1)
CURVATURA_BARRERA = -4.0  # V″(0)

S0 = 4.0 * np.sqrt(2.0) / 3.0
"""Acción del instantón S0 = ∫_{−1}^{1} √(2V) dx = 4√2/3."""


def V(x: np.ndarray) -> np.ndarray:
    """Potencial de doble pozo V(x) = (x² − 1)²."""
    return (x**2 - 1.0) ** 2


def dV(x: np.ndarray) -> np.ndarray:
    """V′(x) = 4x(x² − 1); la deriva de Langevin es −V′."""
    return 4.0 * x * (x**2 - 1.0)


def d2V(x: np.ndarray) -> np.ndarray:
    """V″(x) = 12x² − 4."""
    return 12.0 * x**2 - 4.0


def x_om(t: np.ndarray) -> np.ndarray:
    """Camino más probable de escape x_om(t) = −1/√(1 + e^{8t}), con x_om(0) = −1/√2.

    Se evalúa sin desbordamiento: con e = e^{−8|t|}, 1/(1 + e^{8t}) es 1/(1 + e) si t < 0
    y e/(1 + e) si t ≥ 0.
    """
    t = np.asarray(t, dtype=float)
    e = np.exp(-8.0 * np.abs(t))
    fraccion = np.where(t >= 0, e / (1.0 + e), 1.0 / (1.0 + e))
    return -np.sqrt(fraccion)


def x_kink(tau: np.ndarray) -> np.ndarray:
    """Kink del instantón x_kink(τ) = tanh(√2 τ), con x_kink(0) = 0."""
    return np.tanh(np.sqrt(2.0) * tau)


def S_min(D: np.ndarray) -> np.ndarray:
    """Acción mínima de escape térmico S_min = ΔV/D."""
    return DELTA_V / np.asarray(D, dtype=float)


def tau_kramers(D: np.ndarray) -> np.ndarray:
    """Tiempo medio de escape de Kramers ⟨τ_esc⟩ = 2π/√(V″(−1)|V″(0)|) · e^{ΔV/D}.

    El nombre usa "tau" por costumbre, pero es el tiempo de escape τ_esc, no el tiempo
    imaginario τ del instantón. El prefactor corresponde a la transición al otro pozo; el
    tiempo para llegar a la cima es asintóticamente la mitad. Asintótica para D → 0.
    """
    D = np.asarray(D, dtype=float)
    prefactor = 2.0 * np.pi / np.sqrt(CURVATURA_POZO * abs(CURVATURA_BARRERA))
    return prefactor * np.exp(DELTA_V / D)
