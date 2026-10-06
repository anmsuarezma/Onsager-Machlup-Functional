"""Funcionales del taller discretizados con la regla del trapecio (Bloque A, hito 02).

V y V′ vienen de taller.analitico.referencias, que también operan con tensores de PyTorch.
"""

import torch

from taller.analitico.referencias import V, dV


def accion_escape(x: torch.Tensor, xdot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """S·D = (1/4)∫(ẋ + V′(x))² dt: Onsager-Machlup en forma de ruido débil (Freidlin-Wentzell, §2).

    D solo multiplica la acción, así que se minimiza S·D. Cota: S·D ≥ ΔV = 1 para caminos de
    −1 a 0 (completar cuadrados, hito 00 §4), con igualdad en x_om.
    """
    return torch.trapezoid(0.25 * (xdot + dV(x)) ** 2, t)


def accion_instanton(x: torch.Tensor, xdot: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """S_E = ∫[½ẋ² + V(x)] dτ: acción euclídea del instantón (§3).

    Cota de Bogomolny: S_E ≥ ∫√(2V) dx = S0 = 4√2/3 para caminos de −1 a +1 (hito 00 §6),
    con igualdad en el kink.
    """
    return torch.trapezoid(0.5 * xdot**2 + V(x), tau)
