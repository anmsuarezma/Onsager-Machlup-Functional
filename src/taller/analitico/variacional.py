"""Herramientas variacionales comunes a Onsager-Machlup y al instantón (plan del taller, §1–§3).

Los lagrangianos se escriben como L(x, v), con v = ẋ. Para usar `euler_equations` de sympy
se convierten internamente a una función q(s) de un tiempo genérico s.
"""

import sympy as sp
from sympy.calculus.euler import euler_equations

from taller.analitico.potencial import v, x

_s = sp.Symbol("s", real=True)
_q = sp.Function("q")(_s)
_a = sp.Symbol("a", real=True)  # aceleración ẍ
_y = sp.Dummy("y", real=True)


def ecuacion_el(L: sp.Expr) -> sp.Expr:
    """Aceleración ẍ(x, v) que impone la ecuación de Euler-Lagrange d/dt(∂L/∂ẋ) − ∂L/∂x = 0.

    La ecuación se obtiene con `euler_equations` y se despeja ẍ.
    """
    L_q = L.xreplace({x: _q, v: _q.diff(_s)})
    (ecuacion,) = euler_equations(L_q, _q, _s)
    expresion = (
        ecuacion.lhs.subs(_q.diff(_s, 2), _a).subs(_q.diff(_s), v).subs(_q, x)
    )
    (aceleracion,) = sp.solve(expresion, _a)
    return sp.factor(aceleracion)


def integral_primera(L: sp.Expr) -> sp.Expr:
    """Integral primera H = ẋ ∂L/∂ẋ − L, conservada porque L no depende explícitamente del tiempo.

    Es la "energía" de la mecánica ficticia asociada al lagrangiano.
    """
    return sp.factor(v * sp.diff(L, v) - L)


def derivada_temporal(H: sp.Expr, aceleracion: sp.Expr) -> sp.Expr:
    """dH/dt = ∂H/∂x · ẋ + ∂H/∂v · ẍ, con ẍ sustituida por la aceleración dada."""
    return sp.diff(H, x) * v + sp.diff(H, v) * aceleracion


def primitiva_real(f: sp.Expr, punto: sp.Expr) -> sp.Expr:
    """Primitiva real de f(x) en el intervalo que contiene a `punto`.

    sympy integra por fracciones simples y produce log(z), que es complejo si z < 0. En el
    intervalo de interés se reemplaza log(z) por log(|z|), según el signo de z en `punto`;
    ambos difieren en una constante. La primitiva se verifica derivándola.
    """
    f_y = f.subs(x, _y)
    F = sp.integrate(sp.apart(f_y, _y), _y)
    G = F.replace(sp.log, lambda z: sp.log(z if z.subs(_y, punto) > 0 else -z))
    if sp.simplify(sp.diff(G, _y) - f_y) != 0:
        raise ArithmeticError("la primitiva real no reproduce el integrando")
    return G.subs(_y, x)


def resolver_separable(
    rhs: sp.Expr, s: sp.Symbol, x0: sp.Expr, punto: sp.Expr
) -> sp.Expr:
    """Resuelve ẋ = rhs(x) con x(0) = x0 por separación de variables: s = ∫_{x0}^{x} dy / rhs(y).

    `punto` es un punto del intervalo donde vive la solución (fija los signos de los
    logaritmos). Entre las raíces de la ecuación implícita se elige la que cumple x(0) = x0.
    """
    G = primitiva_real(1 / rhs, punto)
    raices = sp.solve(sp.Eq(G - G.subs(x, x0), s), x)
    validas = [r for r in raices if sp.simplify(r.subs(s, 0) - x0) == 0]
    if len(validas) != 1:
        raise ArithmeticError(f"se esperaba una única solución con x(0) = x0, hay {validas}")
    return sp.simplify(validas[0])
