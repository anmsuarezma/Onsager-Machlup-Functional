"""Tunelamiento cuántico: instantón de la acción euclídea (plan del taller, §3).

En tiempo imaginario τ, la amplitud de tunelamiento está dominada por el mínimo de
S_E[x] = ∫ [½ẋ² + V(x)] dτ con x(−∞) = −1 y x(+∞) = +1: el kink, que cruza la barrera
de pozo a pozo.
"""

import sympy as sp

from taller.analitico.potencial import V, d2V, tau, v, x
from taller.analitico.variacional import resolver_separable

# Origen de tiempo imaginario (modo cero de la traslación): x_kink(0) = 0.
X_KINK_ORIGEN = sp.Integer(0)


def lagrangiano_euclideo() -> sp.Expr:
    """Lagrangiano euclídeo L_E(x, v) = ½v² + V(x): mecánica en el potencial invertido −V."""
    return v**2 / 2 + V()


def rama_kink() -> sp.Expr:
    """Rama positiva de la integral primera ½ẋ² − V = 0 en |x| < 1: ẋ = √(2V) = √2(1 − x²) (§3).

    En (−1, 1) se cumple 1 − x² > 0. Con u = 1 − x² > 0, V = u² y √(2V) = √2·u sin valor
    absoluto. La rama positiva (ẋ > 0) es la que va de −1 a +1.
    """
    u = sp.Symbol("u", positive=True)
    raiz_2V = sp.sqrt(2 * V().subs(x**2, 1 - u))
    return sp.expand(raiz_2V.subs(u, 1 - x**2))


def camino_kink() -> sp.Expr:
    """Kink del instantón x_kink(τ): solución de ẋ = √2(1 − x²) con x_kink(0) = 0 (§3).

    Resultado esperado: tanh(√2 τ).
    """
    return resolver_separable(rama_kink(), tau, X_KINK_ORIGEN, sp.Integer(0))


def S0() -> sp.Expr:
    """Acción del instantón S0 = ∫_{−1}^{1} √(2V) dx, con √(2V) = √2(1 − x²) en (−1, 1) (§3).

    Por la cota de Bogomolny, S_E ≥ S0 para toda trayectoria de −1 a +1, con igualdad en el kink.
    """
    return sp.integrate(rama_kink(), (x, -1, 1))


def accion_euclidea(camino: sp.Expr) -> sp.Expr:
    """Acción euclídea S_E = ∫ (½ẋ² + V) dτ sobre un camino x(τ), en τ ∈ (−∞, ∞)."""
    integrando = lagrangiano_euclideo().subs({v: sp.diff(camino, tau), x: camino})
    return sp.simplify(sp.integrate(sp.simplify(integrando), (tau, -sp.oo, sp.oo)))


def identidad_bogomolny() -> tuple[sp.Expr, sp.Expr]:
    """Identidad ½ẋ² + V = ½(ẋ − √(2V))² + ẋ√(2V), que da la cota S_E ≥ S0 (§3).

    Devuelve (lado izquierdo, lado derecho). Usa √(2V) ≥ 0 en toda la recta, así que la cota
    vale para cualquier trayectoria: ẋ√(2V) = dW/dτ con W(x) = ∫ √(2V) dx.
    """
    raiz_2V = sp.sqrt(2 * V())
    izquierda = v**2 / 2 + V()
    derecha = (v - raiz_2V) ** 2 / 2 + v * raiz_2V
    return izquierda, derecha


def tasa_kink() -> sp.Expr:
    """Tasa con que el kink llega a +1: 1 − x ~ e^{−ω τ}, con ω = √V″(±1) = 2√2 (§3).

    Se calcula como el límite de −d/dτ ln(1 − x) cuando τ → +∞. ω es la frecuencia del
    oscilador armónico en el fondo de cada pozo.
    """
    xk = camino_kink()
    return sp.simplify(sp.limit(sp.diff(xk, tau) / (1 - xk), tau, sp.oo))


def frecuencia_pozo() -> sp.Expr:
    """Frecuencia del oscilador en el fondo de cada pozo, √V″(±1)."""
    return sp.sqrt(d2V(sp.Integer(1)))


def potencial_efectivo_euclideo() -> sp.Expr:
    """Potencial de la mecánica ficticia del instantón, U_E = −V: cimas solo en ±1."""
    return -V()
