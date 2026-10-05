"""Escape térmico: funcional de Onsager-Machlup en forma de Freidlin-Wentzell (plan del taller, §1–§2).

Para dx = −V′ dt + √(2D) dW, la probabilidad de un camino es ∝ e^{−S[x]} con
S[x] = (1/4D) ∫ (ẋ + V′)² dt (ruido débil). El camino más probable de escape va de
x = −1 (t → −∞) a la cima x = 0 (t → +∞).
"""

import sympy as sp

from taller.analitico.potencial import D, d2V, dV, t, v, x
from taller.analitico.variacional import integral_primera, resolver_separable

# Origen de tiempo (modo cero de la traslación temporal): x_om(0) = −1/√2.
X_OM_ORIGEN = -1 / sp.sqrt(2)


def lagrangiano_om(completo: bool = False) -> sp.Expr:
    """Lagrangiano de Onsager-Machlup L(x, v) = (v + V′)²/(4D).

    Con `completo=True` incluye el término jacobiano −½V″ (CLAUDE.md §10). Por defecto se usa
    la forma de ruido débil (Freidlin-Wentzell), sin ese término.
    """
    L = (v + dV()) ** 2 / (4 * D)
    if completo:
        L = L - d2V() / 2
    return L


def ramas_energia_cero() -> list[sp.Expr]:
    """Velocidades v(x) con integral primera nula: v = ±V′ (§1–§2).

    v = −V′ es la relajación determinista (acción cero). v = +V′ es la dinámica invertida
    en el tiempo: el camino de escape, que sube el potencial.
    """
    return sp.solve(sp.Eq(integral_primera(lagrangiano_om()), 0), v)


def camino_om() -> sp.Expr:
    """Camino más probable de escape x_om(t): solución de ẋ = +V′ que va de −1 a 0 (§1–§2).

    Se resuelve la ecuación separable en (−1, 0), con el origen de tiempo en x_om(0) = −1/√2.
    Resultado esperado: −1/√(1 + e^{8t}).
    """
    return resolver_separable(dV(), t, X_OM_ORIGEN, sp.Rational(-1, 2))


def accion_om(camino: sp.Expr, completo: bool = False) -> sp.Expr:
    """Acción de Onsager-Machlup S = ∫ L dt sobre un camino x(t), integrada en t ∈ (−∞, ∞)."""
    integrando = lagrangiano_om(completo).subs({v: sp.diff(camino, t), x: camino})
    return sp.simplify(sp.integrate(sp.simplify(integrando), (t, -sp.oo, sp.oo)))


def S_min() -> sp.Expr:
    """Acción mínima de escape S_min = ΔV/D, por el argumento de la derivada total (§2).

    Sobre la rama ẋ = V′: L = (2V′)²/(4D) = V′ẋ/D = (1/D) dV/dt, así que
    S = (1/D) ∫_{−1}^{0} V′ dx = [V(0) − V(−1)]/D.
    """
    return sp.integrate(dV(), (x, -1, 0)) / D


def identidad_cuadrados_om() -> tuple[sp.Expr, sp.Expr]:
    """Identidad (ẋ + V′)² = (ẋ − V′)² + 4ẋV′, que da la cota S ≥ ΔV/D (§2).

    Devuelve (lado izquierdo, lado derecho). Integrando: S = (1/4D)∫(ẋ − V′)² dt + ΔV/D
    para cualquier camino de −1 a 0; el primer término es ≥ 0 y se anula solo si ẋ = V′.
    """
    izquierda = (v + dV()) ** 2
    derecha = (v - dV()) ** 2 + 4 * v * dV()
    return izquierda, derecha


def tasas_om() -> tuple[sp.Expr, sp.Expr]:
    """Tasas asintóticas de x_om: (salida de −1, llegada a 0), calculadas como límites.

    x + 1 ~ e^{λ₋ t} cuando t → −∞ y x ~ −e^{−λ₀ t} cuando t → +∞. Se espera λ₋ = V″(−1) = 8
    (la rama ẋ = +V′ linealizada en −1) y λ₀ = |V″(0)| = 4.
    """
    xo = camino_om()
    xdot = sp.diff(xo, t)
    salida = sp.limit(xdot / (xo + 1), t, -sp.oo)
    llegada = -sp.limit(xdot / xo, t, sp.oo)
    return salida, llegada


def potencial_efectivo_om() -> sp.Expr:
    """Potencial de la mecánica ficticia de OM, U_OM = −½V′²: Euler-Lagrange es ẍ = −dU_OM/dx.

    Sus cimas (U_OM = 0) están en −1, 0 y +1, los ceros de V′.
    """
    return -dV() ** 2 / 2
