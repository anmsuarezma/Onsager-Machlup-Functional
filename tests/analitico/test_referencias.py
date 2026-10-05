"""Criterio 9 del hito 00: referencias.py (numpy a mano) frente a sympy evaluado con lambdify.

Tolerancias (aclaración 4): error absoluto < 1e-12 para V, dV, d2V, x_om, x_kink y S0;
error relativo < 1e-12 para S_min y tau_kramers.
Mallas: x ∈ [−1.5, 1.5] (301 puntos); t, τ ∈ [−2, 2] (401 puntos); D ∈ {0.1, 0.15, 0.25, 0.35, 0.5}.

Por diseño, este criterio compara dos implementaciones del módulo entre sí (sympy y numpy
escrito a mano, decisión D10); los valores absolutos se validan contra el oráculo en las
demás pruebas.
"""

import numpy as np
import pytest
import sympy as sp
from oraculo import D, t, tau, x

from taller.analitico import referencias as ref
from taller.analitico.escape_termico import S_min, camino_om
from taller.analitico.instanton import S0, camino_kink
from taller.analitico.kramers import tiempo_kramers
from taller.analitico.potencial import V, d2V, dV

TOL = 1e-12
MALLA_X = np.linspace(-1.5, 1.5, 301)
MALLA_T = np.linspace(-2.0, 2.0, 401)
VALORES_D = np.array([0.1, 0.15, 0.25, 0.35, 0.5])


@pytest.mark.parametrize(
    "nombre, funcion_ref, expresion",
    [
        ("V", ref.V, V),
        ("dV", ref.dV, dV),
        ("d2V", ref.d2V, d2V),
    ],
)
def test_potencial_absoluto(nombre, funcion_ref, expresion) -> None:
    """V, V′ y V″ en la malla de x, con error absoluto < 1e-12."""
    simbolico = sp.lambdify(x, expresion(), "numpy")(MALLA_X)
    error = np.max(np.abs(funcion_ref(MALLA_X) - simbolico))
    assert error < TOL, f"{nombre}: {error:.3e}"


def test_x_om_absoluto() -> None:
    """x_om(t) en la malla de t, con error absoluto < 1e-12."""
    simbolico = sp.lambdify(t, camino_om(), "numpy")(MALLA_T)
    error = np.max(np.abs(ref.x_om(MALLA_T) - simbolico))
    assert error < TOL, f"x_om: {error:.3e}"


def test_x_kink_absoluto() -> None:
    """x_kink(τ) en la malla de τ, con error absoluto < 1e-12."""
    simbolico = sp.lambdify(tau, camino_kink(), "numpy")(MALLA_T)
    error = np.max(np.abs(ref.x_kink(MALLA_T) - simbolico))
    assert error < TOL, f"x_kink: {error:.3e}"


def test_S0_absoluto() -> None:
    """La constante S0 de referencias coincide con S0 de sympy, con error absoluto < 1e-12."""
    assert abs(ref.S0 - float(S0())) < TOL


def test_S_min_relativo() -> None:
    """S_min(D) con error relativo < 1e-12 para cada D."""
    simbolico = sp.lambdify(D, S_min(), "numpy")(VALORES_D)
    error = np.max(np.abs(ref.S_min(VALORES_D) - simbolico) / np.abs(simbolico))
    assert error < TOL, f"S_min: {error:.3e}"


def test_tau_kramers_relativo() -> None:
    """tau_kramers(D) con error relativo < 1e-12 para cada D."""
    simbolico = sp.lambdify(D, tiempo_kramers(), "numpy")(VALORES_D)
    error = np.max(np.abs(ref.tau_kramers(VALORES_D) - simbolico) / np.abs(simbolico))
    assert error < TOL, f"tau_kramers: {error:.3e}"
