"""Criterios 6 y 7 del hito 00: Fokker-Planck, corriente estacionaria y puente a Schrödinger.

Convenciones (especificación §0 y §4):
- Fokker-Planck: ∂p/∂t = ∂(V′p)/∂x + D ∂²p/∂x² = −∂J/∂x, con J = −V′p − D p′.
- Con p = e^{−V/2D} ψ se obtiene ∂ψ/∂t = −Hψ, con H = −D∂² + V′²/(4D) − V″/2.

Los valores esperados vienen de `oraculo.py`. Del módulo solo se importan los operadores
y las funciones que se prueban.
"""

import numpy as np
import pytest
import sympy as sp
from oraculo import (
    ALFA_X,
    D,
    D2V_ESP,
    DV_ESP,
    MARGEN,
    PUNTOS_FIJOS,
    V_ESP,
    VALORES_D,
    malla_irregular,
    x,
)

from taller.analitico.potencial import corriente_fp, densidad_estacionaria
from taller.analitico.schrodinger import hamiltoniano_efectivo, operador_fp, psi0

f = sp.Function("f")(x)


# --- Criterio 6 -----------------------------------------------------------------


def test_operador_fp() -> None:
    """El operador de Fokker-Planck es L_FP p = ∂(V′p)/∂x + D ∂²p/∂x²."""
    esperado = sp.diff(DV_ESP * f, x) + D * sp.diff(f, x, 2)
    assert sp.simplify(operador_fp(f) - esperado) == 0


def test_hamiltoniano() -> None:
    """H coincide con −D∂² + V′²/(4D) − V″/2 actuando sobre una función arbitraria."""
    esperado = -D * sp.diff(f, x, 2) + (DV_ESP**2 / (4 * D) - D2V_ESP / 2) * f
    assert sp.simplify(hamiltoniano_efectivo(f) - esperado) == 0


def test_hamiltoniano_es_conjugacion_de_fp() -> None:
    """H se obtiene de L_FP con p = e^{−V/2D} ψ: L_FP(e^{−V/2D} ψ) = −e^{−V/2D} Hψ."""
    peso = sp.exp(-V_ESP / (2 * D))
    diferencia = operador_fp(peso * f) + peso * hamiltoniano_efectivo(f)
    assert sp.simplify(diferencia / peso) == 0


def test_psi0_forma() -> None:
    """ψ0 = e^{−V/2D}, la raíz cuadrada de la densidad de Boltzmann."""
    assert sp.simplify(psi0() - sp.exp(-V_ESP / (2 * D))) == 0


def test_psi0_energia_cero() -> None:
    """Hψ0 = 0: ψ0 es el estado fundamental de energía cero."""
    assert sp.simplify(hamiltoniano_efectivo(psi0())) == 0


@pytest.mark.parametrize("D_val", VALORES_D)
def test_psi0_energia_cero_diferencias_finitas(D_val: float) -> None:
    """Aclaración (9): Hψ0 = 0 numéricamente, con H escrito a mano y ψ0 del módulo.

    ∂² se aproxima con diferencias centradas, de error O(h²). Se exige que el residuo
    relativo |Hψ0| / (|Dψ0″| + |Wψ0|) sea < 1e-4 con h = 1e-4 y que baje al menos un
    factor 50 al pasar de h = 1e-3 a h = 1e-4 (convergencia a cero como h²).
    Malla determinista sin semilla: x = {k·φ mod 1} en [−1.4, 1.4], sin los puntos a menos
    de 0.01 de −1, 0, +1.
    """
    xs = malla_irregular(200, -1.4, 1.4, ALFA_X)
    assert np.min(np.abs(xs[:, None] - PUNTOS_FIJOS[None, :])) >= MARGEN

    psi = sp.lambdify(x, psi0().subs(D, D_val), "numpy")
    W = (4 * xs * (xs**2 - 1)) ** 2 / (4 * D_val) - (12 * xs**2 - 4) / 2

    def residuo(h: float) -> float:
        psi_pp = (psi(xs + h) - 2 * psi(xs) + psi(xs - h)) / h**2
        H_psi = -D_val * psi_pp + W * psi(xs)
        escala = np.abs(D_val * psi_pp) + np.abs(W * psi(xs))
        return float(np.max(np.abs(H_psi) / escala))

    r_grueso, r_fino = residuo(1e-3), residuo(1e-4)
    assert r_fino < 1e-4
    assert r_grueso / r_fino > 50


# --- Criterio 7 -----------------------------------------------------------------


def test_densidad_estacionaria_boltzmann() -> None:
    """p_s ∝ e^{−V/D}: el cociente p_s / e^{−V/D} es una constante no nula."""
    cociente = sp.simplify(densidad_estacionaria() * sp.exp(V_ESP / D))
    assert sp.simplify(sp.diff(cociente, x)) == 0
    assert cociente != 0


def test_corriente_fp_forma() -> None:
    """J[p] = −V′p − D p′."""
    esperado = -DV_ESP * f - D * sp.diff(f, x)
    assert sp.simplify(corriente_fp(f) - esperado) == 0


def test_corriente_nula_en_equilibrio() -> None:
    """J[p_s] = 0: la distribución de Boltzmann no transporta probabilidad (equilibrio detallado)."""
    assert sp.simplify(corriente_fp(densidad_estacionaria())) == 0
