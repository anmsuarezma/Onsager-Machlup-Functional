"""Aclaración (10) del hito 00: potencial, equilibrio, tasas asintóticas y Kramers.

- Puntos fijos exactamente {−1, 0, 1}; V″(±1) = 8; V″(0) = −4; ΔV = 1.
- Tasas del camino OM, obtenidas como límites con sympy: 8 al alejarse de −1 y 4 al llegar a 0.
- Tasa del kink: 2√2 = √V″(±1).
- tau_kramers(D) = (2π/√32)·e^{1/D}, con error relativo < 1e-12; prefactor 2π/√32 ≈ 1.1107.

Los valores esperados están escritos aquí o vienen de `oraculo.py`; no se construyen con el módulo.
"""

import numpy as np
import sympy as sp
from oraculo import D, TAU_KRAMERS_ESP, VALORES_D, t, tau, x

from taller.analitico import referencias as ref
from taller.analitico.escape_termico import camino_om, tasas_om
from taller.analitico.instanton import camino_kink, tasa_kink
from taller.analitico.kramers import tiempo_kramers
from taller.analitico.potencial import V, altura_barrera, d2V, dV, puntos_fijos

VALORES_D = np.array(VALORES_D)


# --- Potencial y equilibrio -------------------------------------------------------


def test_potencial_y_derivadas() -> None:
    """V = (x²−1)², V′ = 4x(x²−1), V″ = 12x² − 4 (CLAUDE.md §2)."""
    assert sp.simplify(V() - (x**2 - 1) ** 2) == 0
    assert sp.simplify(dV() - 4 * x * (x**2 - 1)) == 0
    assert sp.simplify(d2V() - (12 * x**2 - 4)) == 0


def test_puntos_fijos() -> None:
    """Los puntos fijos de ẋ = −V′ son exactamente {−1, 0, 1}."""
    assert set(puntos_fijos()) == {sp.Integer(-1), sp.Integer(0), sp.Integer(1)}


def test_curvaturas() -> None:
    """V″(±1) = 8 (mínimos estables) y V″(0) = −4 (barrera inestable)."""
    assert d2V(sp.Integer(-1)) == 8
    assert d2V(sp.Integer(1)) == 8
    assert d2V(sp.Integer(0)) == -4


def test_altura_barrera() -> None:
    """ΔV = V(0) − V(−1) = 1."""
    assert altura_barrera() == 1


# --- Tasas asintóticas ------------------------------------------------------------


def test_tasas_om_como_limites() -> None:
    """x_om se aleja de −1 como e^{8t} y llega a 0 como e^{−4t}; 8 = V″(−1), 4 = |V″(0)|."""
    xo = camino_om()
    xdot = sp.diff(xo, t)
    salida = sp.limit(xdot / (xo + 1), t, -sp.oo)  # d/dt ln(x + 1) cuando t → −∞
    llegada = -sp.limit(xdot / xo, t, sp.oo)  # −d/dt ln|x| cuando t → +∞
    assert salida == 8  # = V″(−1)
    assert llegada == 4  # = |V″(0)|
    assert tuple(tasas_om()) == (8, 4)


def test_tasa_kink_como_limite() -> None:
    """x_kink llega a +1 como e^{−2√2 τ}; 2√2 = √V″(±1), la frecuencia del oscilador en cada pozo."""
    xk = camino_kink()
    tasa = sp.limit(sp.diff(xk, tau) / (1 - xk), tau, sp.oo)  # −d/dτ ln(1 − x) cuando τ → +∞
    assert sp.simplify(tasa - 2 * sp.sqrt(2)) == 0
    assert sp.simplify(tasa - sp.sqrt(8)) == 0  # √V″(±1), con V″(±1) = 8
    assert sp.simplify(tasa_kink() - 2 * sp.sqrt(2)) == 0


# --- Kramers ----------------------------------------------------------------------


def test_tiempo_kramers_simbolico() -> None:
    """⟨τ_esc⟩ = (2π/√32)·e^{1/D} exactamente en sympy."""
    assert sp.simplify(tiempo_kramers() - TAU_KRAMERS_ESP) == 0


def test_tau_kramers_referencia() -> None:
    """referencias.tau_kramers(D) = (2π/√32)·e^{1/D} con error relativo < 1e-12."""
    esperado = 2 * np.pi / np.sqrt(32) * np.exp(1 / VALORES_D)
    error = np.max(np.abs(ref.tau_kramers(VALORES_D) - esperado) / esperado)
    assert error < 1e-12, f"tau_kramers: {error:.3e}"


def test_prefactor_kramers() -> None:
    """Prefactor 2π/√32 ≈ 1.1107 (= 2π/√(V″(−1)|V″(0)|))."""
    prefactor = ref.tau_kramers(VALORES_D) * np.exp(-1 / VALORES_D)
    assert np.allclose(prefactor, 1.1107, rtol=0, atol=5e-5)
