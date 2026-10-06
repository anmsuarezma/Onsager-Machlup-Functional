"""Ansatz que impone las fronteras exactamente (Bloque A, hito 02).

x(t) = x_a + (x_b − x_a)(t + T)/(2T) + [(t + T)(T − t)/T²]·N(t/T).

El factor (t + T)(T − t) se anula en ambos extremos, así que x(−T) = x_a y x(T) = x_b para
cualquier red N: el optimizador no puede bajar la acción incumpliendo la frontera, como sí
podría con una penalización.
"""

import torch
from torch import nn


def camino(red: nn.Module, t: torch.Tensor, T: float, x_a: float, x_b: float) -> torch.Tensor:
    """Camino del ansatz en los instantes t (tensor 1D, float64)."""
    N = red((t / T).unsqueeze(1)).squeeze(1)
    return x_a + (x_b - x_a) * (t + T) / (2 * T) + (t + T) * (T - t) / T**2 * N


def camino_y_derivada(
    red: nn.Module, t: torch.Tensor, T: float, x_a: float, x_b: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camino y su derivada temporal ẋ por diferenciación automática.

    `create_graph=True` mantiene ẋ diferenciable respecto de los parámetros, para que la
    acción, que depende de ẋ, pueda optimizarse.
    """
    t = t.detach().requires_grad_(True)
    x = camino(red, t, T, x_a, x_b)
    (xdot,) = torch.autograd.grad(x.sum(), t, create_graph=True)
    return x, xdot
