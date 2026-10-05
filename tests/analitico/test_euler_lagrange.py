"""Criterios 1 y 3 del hito 00: ecuaciones de Euler-Lagrange e integrales primeras.

Los valores esperados vienen de `oraculo.py`, escrito a mano desde la especificación.
Del módulo solo se importan los lagrangianos y las funciones variacionales que se prueban.
"""

import pytest
import sympy as sp
from oraculo import D, D2V_ESP, D3V_ESP, DV_ESP, V_ESP, v, x

from taller.analitico.escape_termico import lagrangiano_om
from taller.analitico.instanton import lagrangiano_euclideo
from taller.analitico.variacional import ecuacion_el, integral_primera

# ẍ esperado para cada lagrangiano (especificación §1–§2, §3 y variante completa).
XPP_ESP = {
    "om": DV_ESP * D2V_ESP,  # 4x(x²−1)(12x²−4)
    "euclideo": DV_ESP,  # 4x(x²−1)
    "om_completo": DV_ESP * D2V_ESP - D * D3V_ESP,  # 4x(x²−1)(12x²−4) − 24Dx
}

# Integral primera esperada H = ẋ ∂L/∂ẋ − L.
H_ESP = {
    "om": (v**2 - DV_ESP**2) / (4 * D),
    "euclideo": v**2 / 2 - V_ESP,
    "om_completo": (v**2 - DV_ESP**2) / (4 * D) + D2V_ESP / 2,
}


def _lagrangiano(nombre: str) -> sp.Expr:
    if nombre == "om":
        return lagrangiano_om()
    if nombre == "om_completo":
        return lagrangiano_om(completo=True)
    return lagrangiano_euclideo()


# --- Criterio 1 -------------------------------------------------------------


def test_el_om() -> None:
    """Onsager-Machlup (Freidlin-Wentzell): ẍ = V′V″ = 4x(x²−1)(12x²−4)."""
    assert sp.simplify(ecuacion_el(lagrangiano_om()) - XPP_ESP["om"]) == 0


def test_el_euclideo() -> None:
    """Acción euclídea: ẍ = V′ = 4x(x²−1)."""
    assert sp.simplify(ecuacion_el(lagrangiano_euclideo()) - XPP_ESP["euclideo"]) == 0


def test_el_om_completo() -> None:
    """Onsager-Machlup completo (con −½V″): ẍ = V′V″ − D V‴ = 4x(x²−1)(12x²−4) − 24Dx."""
    xpp = ecuacion_el(lagrangiano_om(completo=True))
    assert sp.simplify(xpp - XPP_ESP["om_completo"]) == 0


# --- Criterio 3 -------------------------------------------------------------


@pytest.mark.parametrize("nombre", ["om", "euclideo", "om_completo"])
def test_integral_primera_forma(nombre: str) -> None:
    """H = ẋ ∂L/∂ẋ − L coincide con la forma esperada (∝ ẋ² − V′² en OM; ½ẋ² − V en el euclídeo)."""
    H = integral_primera(_lagrangiano(nombre))
    assert sp.simplify(H - H_ESP[nombre]) == 0


@pytest.mark.parametrize("nombre", ["om", "euclideo", "om_completo"])
def test_integral_primera_conservada(nombre: str) -> None:
    """dH/dt = ∂H/∂x · ẋ + ∂H/∂v · ẍ se anula al sustituir ẍ por la ecuación de Euler-Lagrange esperada."""
    H = integral_primera(_lagrangiano(nombre))
    dHdt = sp.diff(H, x) * v + sp.diff(H, v) * XPP_ESP[nombre]
    assert sp.simplify(dHdt) == 0
