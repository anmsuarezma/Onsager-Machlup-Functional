"""Predicción de Kramers para el escape térmico (referencia para el bloque estocástico).

Notación: τ se reserva para el tiempo imaginario del instantón; el tiempo medio de
escape se escribe τ_esc.
"""

import sympy as sp

from taller.analitico.potencial import D, altura_barrera, d2V, puntos_fijos


def tiempo_kramers() -> sp.Expr:
    """Tiempo medio de escape de Kramers sobreamortiguado, ⟨τ_esc⟩ ≈ 2π/√(V″(−1)|V″(0)|) · e^{ΔV/D}.

    Es una asintótica para D → 0 (D ≪ ΔV). El prefactor corresponde al tiempo de
    TRANSICIÓN (cruzar la barrera y caer al otro pozo). El tiempo para llegar por primera
    vez a la cima x = 0 es asintóticamente la mitad: desde la cima, la partícula cae a cada
    lado con probabilidad ½. Qué definición usa el bloque estocástico no se decide aquí.
    """
    minimo, silla = puntos_fijos()[0], puntos_fijos()[1]
    prefactor = 2 * sp.pi / sp.sqrt(d2V(minimo) * sp.Abs(d2V(silla)))
    return prefactor * sp.exp(altura_barrera() / D)


def tabla_kramers(valores_D: list[float]) -> list[tuple[float, float, float]]:
    """Filas (D, ΔV/D, ⟨τ_esc⟩) evaluadas numéricamente a partir de la expresión simbólica."""
    tiempo = tiempo_kramers()
    return [
        (d, float(altura_barrera() / d), float(tiempo.subs(D, d))) for d in valores_D
    ]
