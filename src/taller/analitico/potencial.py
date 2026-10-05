"""Potencial de doble pozo, equilibrio y Fokker-Planck estacionario (plan del taller, §0).

Define los símbolos de todo el módulo analítico y las expresiones básicas del sistema
físico (CLAUDE.md §2). Todas las expresiones son de sympy.

Símbolos:
    x    posición de la partícula
    v    velocidad ẋ (variable independiente del lagrangiano L(x, v))
    t    tiempo real (escape térmico)
    tau  tiempo imaginario (instantón)
    D    intensidad del ruido k_B T / γ, positiva
"""

import sympy as sp

x, v, t, tau = sp.symbols("x v t tau", real=True)
D = sp.Symbol("D", positive=True)


def V(q: sp.Expr = x) -> sp.Expr:
    """Potencial de doble pozo V(q) = (q² − 1)²: mínimos metaestables en ±1 y barrera en 0."""
    return (q**2 - 1) ** 2


def dV(q: sp.Expr = x) -> sp.Expr:
    """Fuerza cambiada de signo V′(q): la deriva de Langevin es −V′."""
    return sp.diff(V(x), x).subs(x, q)


def d2V(q: sp.Expr = x) -> sp.Expr:
    """Curvatura V″(q): rigidez del pozo (> 0) o inestabilidad de la barrera (< 0)."""
    return sp.diff(V(x), x, 2).subs(x, q)


def d3V(q: sp.Expr = x) -> sp.Expr:
    """Tercera derivada V‴(q); aparece en la ecuación de Euler-Lagrange de Onsager-Machlup completo."""
    return sp.diff(V(x), x, 3).subs(x, q)


def puntos_fijos() -> list[sp.Expr]:
    """Puntos fijos de la dinámica determinista ẋ = −V′(x), ordenados: raíces de V′ (§0)."""
    return sorted(sp.solve(sp.Eq(dV(), 0), x))


def estabilidad(x0: sp.Expr) -> str:
    """Estabilidad lineal de ẋ = −V′ en x0: δẋ = −V″(x0) δx, estable si V″(x0) > 0."""
    curvatura = d2V(x0)
    if curvatura > 0:
        return "estable"
    if curvatura < 0:
        return "inestable"
    return "marginal"


def altura_barrera() -> sp.Expr:
    """ΔV = V(silla) − V(mínimo): energía que el ruido debe aportar para escapar (§0).

    El mínimo y la silla se identifican por la estabilidad de los puntos fijos.
    """
    fijos = puntos_fijos()
    minimos = [p for p in fijos if estabilidad(p) == "estable"]
    sillas = [p for p in fijos if estabilidad(p) == "inestable"]
    return V(sillas[0]) - V(minimos[0])


def densidad_estacionaria() -> sp.Expr:
    """Densidad de Boltzmann sin normalizar p_s ∝ e^{−V/D}: equilibrio de Fokker-Planck (§0)."""
    return sp.exp(-V() / D)


def corriente_fp(p: sp.Expr) -> sp.Expr:
    """Corriente de probabilidad de Fokker-Planck J[p] = −V′p − D ∂p/∂x (§0).

    Con ella, ∂p/∂t = −∂J/∂x. En equilibrio detallado J = 0.
    """
    return -dV() * p - D * sp.diff(p, x)
