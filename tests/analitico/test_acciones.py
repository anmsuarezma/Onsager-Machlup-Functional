"""Criterios 4 y 5 del hito 00: acciones mínimas y cotas por completar cuadrados.

Criterio 4: S_min·D = 1 y S0 = 4√2/3 exactos en sympy; con scipy.quad, error relativo < 1e-10.
Criterio 5: las identidades de completar cuadrados (OM y Bogomolny) se simplifican a cero.
Aclaración (9): verificación numérica complementaria de las identidades en una malla
determinista irregular que evita los puntos fijos.

Los valores esperados y los integrandos vienen de `oraculo.py`. Del módulo solo se
importan las cantidades que se prueban.
"""

import numpy as np
import pytest
import sympy as sp
from oraculo import (
    ALFA_V,
    ALFA_X,
    D,
    DV_ESP,
    MARGEN,
    PUNTOS_FIJOS,
    S0_ESP,
    V_ESP,
    VALORES_D,
    malla_irregular,
    t,
    v,
    x,
)
from scipy.integrate import quad

from taller.analitico.escape_termico import S_min, accion_om, camino_om, identidad_cuadrados_om
from taller.analitico.instanton import S0, identidad_bogomolny

TOL_QUAD = 1e-10

# Lagrangiano OM de la especificación: L = (ẋ + V′)² / (4D).
L_OM_ESP = (v + DV_ESP) ** 2 / (4 * D)


# --- Criterio 4: exacto en sympy -----------------------------------------------


def test_S_min_exacto() -> None:
    """S_min · D = 1, es decir, S_min = ΔV/D."""
    assert sp.simplify(S_min() * D - 1) == 0


def test_accion_om_sobre_camino_exacta() -> None:
    """La acción de Onsager-Machlup integrada sobre x_om en t ∈ (−∞, ∞) vale exactamente 1/D."""
    assert sp.simplify(accion_om(camino_om()) * D - 1) == 0


def test_S0_exacto() -> None:
    """S0 = ∫ √(2V) dx en [−1, 1] = 4√2/3."""
    assert sp.simplify(S0() - S0_ESP) == 0


# --- Criterio 4: scipy.quad ----------------------------------------------------


@pytest.mark.parametrize("D_val", VALORES_D)
def test_S_min_quad(D_val: float) -> None:
    """quad de (ẋ + V′)²/(4D) evaluado sobre el x_om del módulo reproduce 1/D con error relativo < 1e-10.

    Se integra en t ∈ [−10, 10]: fuera de ese intervalo el integrando decae como e^{16t}
    (t → −∞) y e^{−8t} (t → +∞), así que las colas aportan menos de e^{−80} y se evita el
    desbordamiento de e^{8t} en numpy.
    """
    xo = camino_om()
    integrando = L_OM_ESP.subs({v: sp.diff(xo, t), x: xo})
    f = sp.lambdify((t, D), integrando, "numpy")
    valor, _ = quad(lambda s: f(s, D_val), -10, 10, epsabs=0, epsrel=1e-13, limit=200)
    esperado = 1 / D_val
    assert abs(valor - esperado) / esperado < TOL_QUAD


def test_S0_quad() -> None:
    """quad de √(2V) = √(2(x²−1)²) en [−1, 1] reproduce 4√2/3 y el S0 del módulo, con error relativo < 1e-10."""
    valor, _ = quad(lambda s: np.sqrt(2 * (s**2 - 1) ** 2), -1, 1, epsabs=0, epsrel=1e-13)
    esperado = float(S0_ESP)
    assert abs(valor - esperado) / esperado < TOL_QUAD
    assert abs(valor - float(S0())) / esperado < TOL_QUAD


# --- Criterio 5 -----------------------------------------------------------------


def test_identidad_om() -> None:
    """(ẋ + V′)² = (ẋ − V′)² + 4ẋV′: los dos lados son los de la especificación y la diferencia es cero."""
    izquierda, derecha = identidad_cuadrados_om()
    assert sp.simplify(izquierda - (v + DV_ESP) ** 2) == 0
    assert sp.simplify(derecha - ((v - DV_ESP) ** 2 + 4 * v * DV_ESP)) == 0
    assert sp.simplify(izquierda - derecha) == 0


def test_identidad_bogomolny() -> None:
    """½ẋ² + V = ½(ẋ − √(2V))² + ẋ√(2V): los dos lados son los de la especificación y la diferencia es cero."""
    izquierda, derecha = identidad_bogomolny()
    raiz_2V = sp.sqrt(2 * V_ESP)
    assert sp.simplify(izquierda - (v**2 / 2 + V_ESP)) == 0
    assert sp.simplify(derecha - ((v - raiz_2V) ** 2 / 2 + v * raiz_2V)) == 0
    assert sp.simplify(izquierda - derecha) == 0


# --- Aclaración (9): verificación numérica complementaria -------------------------


@pytest.mark.parametrize("identidad", [identidad_cuadrados_om, identidad_bogomolny])
def test_identidades_en_malla_irregular(identidad) -> None:
    """Los dos lados de cada identidad coinciden numéricamente (error relativo < 1e-12).

    Malla determinista sin semilla: x = {k·φ mod 1} en [−1.4, 1.4] con φ la razón áurea y
    v = {k·(√2−1) mod 1} en [−3, 3]. Se excluyen los puntos a menos de 0.01 de −1, 0, +1,
    donde V′ = 0 y las identidades se cumplirían trivialmente.
    """
    xs = malla_irregular(400, -1.4, 1.4, ALFA_X)
    vs = (-3 + 6 * ((np.arange(1, xs.size + 1) * ALFA_V) % 1.0))
    assert np.min(np.abs(xs[:, None] - PUNTOS_FIJOS[None, :])) >= MARGEN
    assert np.unique(np.round(np.diff(np.sort(xs)), 6)).size > 1  # espaciado irregular

    izquierda, derecha = identidad()
    f_izq = sp.lambdify((x, v), izquierda, "numpy")
    f_der = sp.lambdify((x, v), derecha, "numpy")
    a, b = f_izq(xs, vs), f_der(xs, vs)
    assert np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0)) < 1e-12
