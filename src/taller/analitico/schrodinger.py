"""Puente Fokker-Planck → Schrödinger (plan del taller, §4).

La ecuación de Fokker-Planck ∂p/∂t = L_FP p se transforma, con p = e^{−V/2D} ψ, en
∂ψ/∂t = −Hψ, con un operador H de tipo Schrödinger (hermítico y ≥ 0). Su estado de
energía cero es la raíz cuadrada de la distribución de Boltzmann.
"""

import sympy as sp

from taller.analitico.potencial import D, V, dV, x


def operador_fp(p: sp.Expr) -> sp.Expr:
    """Operador de Fokker-Planck L_FP p = ∂(V′p)/∂x + D ∂²p/∂x², tal que ∂p/∂t = L_FP p (§4)."""
    return sp.diff(dV() * p, x) + D * sp.diff(p, x, 2)


def hamiltoniano_efectivo(psi: sp.Expr) -> sp.Expr:
    """Hψ = −e^{V/2D} L_FP(e^{−V/2D} ψ), con la convención ∂ψ/∂t = −Hψ (§4).

    Resultado esperado: H = −D ∂² + V′²/(4D) − V″/2.
    """
    peso = sp.exp(-V() / (2 * D))
    return sp.expand(sp.simplify(-operador_fp(peso * psi) / peso))


def psi0() -> sp.Expr:
    """ψ0 = e^{−V/2D} = √p_s: estado de energía cero de H, la raíz de la densidad de Boltzmann."""
    return sp.exp(-V() / (2 * D))
