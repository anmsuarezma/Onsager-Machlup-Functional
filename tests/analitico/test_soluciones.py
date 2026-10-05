"""Criterios 2 y 8 del hito 00: soluciones cerradas x_om(t) y x_kink(τ).

Criterio 2: residuos simbólicos exactamente cero.
Criterio 8: solve_ivp (DOP853, rtol=1e-12, atol=1e-14) desde t=0 hacia adelante y hacia
atrás en [−2, 2] reproduce la solución cerrada con error absoluto máximo < 1e-8.
Aclaración (2): en |x| < 1 se usa √(2V) = √2(1 − x²); aquí se verifica además esa igualdad.

Los valores esperados, los lados derechos de las EDO y las condiciones iniciales vienen
de `oraculo.py`. Del módulo solo se importan las soluciones y la rama que se prueban.
"""

import numpy as np
import pytest
import sympy as sp
from oraculo import (
    D2V_ESP,
    DV_ESP,
    RAMA_KINK_ESP,
    X_KINK_0,
    X_KINK_ESP,
    X_OM_0,
    X_OM_ESP,
    rhs_kink,
    rhs_om,
    t,
    tau,
    x,
)
from scipy.integrate import solve_ivp

from taller.analitico.escape_termico import camino_om
from taller.analitico.instanton import camino_kink, rama_kink

TOL_IVP = 1e-8


# --- Formas cerradas (fijan el origen de tiempo) ---------------------------------


def test_x_om_forma_cerrada() -> None:
    """x_om(t) = −1/√(1 + e^{8t}).

    Fija el origen de tiempo (modo cero de la traslación temporal): x_om(0) = −1/√2.
    Esta convención la usa después el bloque neuronal.
    """
    assert sp.simplify(camino_om() - X_OM_ESP) == 0


def test_x_om_limites() -> None:
    """x_om va de −1 (t → −∞) a 0 (t → +∞) y cruza −1/√2 en t = 0."""
    xo = camino_om()
    assert sp.limit(xo, t, -sp.oo) == -1
    assert sp.limit(xo, t, sp.oo) == 0
    assert sp.simplify(xo.subs(t, 0) + 1 / sp.sqrt(2)) == 0


def test_x_kink_forma_cerrada() -> None:
    """x_kink(τ) = tanh(√2 τ).

    Fija el origen de tiempo imaginario (modo cero de la traslación): x_kink(0) = 0.
    Esta convención la usa después el bloque neuronal.
    """
    assert sp.simplify(camino_kink() - X_KINK_ESP) == 0


def test_rama_kink_forma() -> None:
    """La rama usada para el kink es √2(1 − x²)."""
    assert sp.simplify(rama_kink() - RAMA_KINK_ESP) == 0


def test_rama_coincide_con_raiz_de_2V() -> None:
    """√(2V) = √2(1 − x²) en (−1, 1): simbólicamente con u = 1 − x² > 0, y numéricamente en una malla."""
    u = sp.Symbol("u", positive=True)
    # En (−1, 1): V = (x² − 1)² = u², con u = 1 − x² > 0.
    assert sp.simplify(sp.sqrt(2 * u**2) - sp.sqrt(2) * u) == 0

    xs = np.linspace(-1, 1, 1001)[1:-1]
    raiz_2V = np.sqrt(2 * (xs**2 - 1) ** 2)
    assert np.max(np.abs(raiz_2V - rhs_kink(xs))) < 1e-14


# --- Criterio 2 -------------------------------------------------------------


def test_x_om_residuo_primer_orden() -> None:
    """ẋ − V′(x) = 0 sobre x_om."""
    xo = camino_om()
    assert sp.simplify(sp.diff(xo, t) - DV_ESP.subs(x, xo)) == 0


def test_x_om_residuo_el() -> None:
    """ẍ − V′V″ = 0 sobre x_om."""
    xo = camino_om()
    assert sp.simplify(sp.diff(xo, t, 2) - (DV_ESP * D2V_ESP).subs(x, xo)) == 0


def test_x_kink_residuo_primer_orden() -> None:
    """ẋ − √2(1 − x²) = 0 sobre x_kink (rama √(2V) en |x| < 1)."""
    xk = camino_kink()
    assert sp.simplify(sp.diff(xk, tau) - RAMA_KINK_ESP.subs(x, xk)) == 0


def test_x_kink_residuo_el() -> None:
    """ẍ − V′ = 0 sobre x_kink."""
    xk = camino_kink()
    assert sp.simplify(sp.diff(xk, tau, 2) - DV_ESP.subs(x, xk)) == 0


# --- Criterio 8 -------------------------------------------------------------


def _error_ivp(rhs, x0: float, solucion, s) -> float:
    """Integra ẋ = rhs(x) (oráculo) desde s = 0, con x(0) = x0 (oráculo), hasta +2 y hasta −2.

    Devuelve el error absoluto máximo frente a la solución cerrada del módulo.
    """
    f_sol = sp.lambdify(s, solucion, "numpy")
    error = 0.0
    for fin in (2.0, -2.0):
        s_eval = np.linspace(0.0, fin, 201)
        res = solve_ivp(
            lambda _s, y: rhs(y),
            (0.0, fin),
            [x0],
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            t_eval=s_eval,
        )
        assert res.success, res.message
        error = max(error, float(np.max(np.abs(res.y[0] - f_sol(s_eval)))))
    return error


@pytest.mark.parametrize("caso", ["om", "kink"])
def test_solve_ivp_reproduce_solucion(caso: str) -> None:
    """ẋ = 4x(x²−1) desde x(0) = −1/√2 reproduce x_om; ẋ = √2(1 − x²) desde x(0) = 0 reproduce x_kink.

    Error absoluto máximo < 1e-8 en [−2, 2].
    """
    if caso == "om":
        error = _error_ivp(rhs_om, X_OM_0, camino_om(), t)
    else:
        error = _error_ivp(rhs_kink, X_KINK_0, camino_kink(), tau)
    print(f"{caso}: error absoluto máximo = {error:.3e}")
    assert error < TOL_IVP
