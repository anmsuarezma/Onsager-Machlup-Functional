"""Ansatz que impone las fronteras exactamente (Bloque A, hito 02).

x(t) = x_a + (x_b − x_a)(t + T_h)/(2T_h) + [(t + T_h)(T_h − t)/T_h²]·N(t/T_h),

con T_h el horizonte (no T, que en el taller designa el tiempo de escape y la temperatura;
D40). El factor (t + T_h)(T_h − t) se anula en ambos extremos, así que x(−T_h) = x_a y
x(T_h) = x_b para cualquier red N: el optimizador no puede bajar la acción incumpliendo la
frontera, como sí podría con una penalización.
"""

import torch
from torch import nn


def camino(red: nn.Module, t: torch.Tensor, horizonte: float, x_a: float, x_b: float) -> torch.Tensor:
    """Camino del ansatz en los instantes t (tensor 1D en el dispositivo y la precisión de la red)."""
    N = red((t / horizonte).unsqueeze(1)).squeeze(1)
    return (x_a + (x_b - x_a) * (t + horizonte) / (2 * horizonte)
            + (t + horizonte) * (horizonte - t) / horizonte**2 * N)


def camino_y_derivada(
    red: nn.Module, t: torch.Tensor, horizonte: float, x_a: float, x_b: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camino y su derivada temporal ẋ por diferenciación automática.

    `create_graph=True` mantiene ẋ diferenciable respecto de los parámetros, para que la
    acción, que depende de ẋ, pueda optimizarse.
    """
    t = t.detach().requires_grad_(True)
    x = camino(red, t, horizonte, x_a, x_b)
    (xdot,) = torch.autograd.grad(x.sum(), t, create_graph=True)
    return x, xdot
