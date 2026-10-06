"""Funcionales del hito 02 (regla del trapecio) contra valores escritos a mano.

- S·D = (1/4)∫(ẋ + V′)² dt sobre x_om en [−3, 3]: como ẋ = V′ y la integral es V(x(T)) −
  V(x(−T)), vale 1 − 7.6e-11 (x_om no llega exactamente a 0 en T = 3).
- S·D sobre la recta de −1 a 0 en [−3, 3]: (3/2)∫_{−1}^{0}(1/6 + 4u³ − 4u)² du = 1991/840.
- S·D sobre x ≡ −1 (reposo en el pozo): 0.
- S_E = ∫[½ẋ² + V] dτ sobre el kink en [−4, 4]: S0 salvo las colas (−8.9e-10 relativo).
- S_E sobre la recta de −1 a 1 en [−4, 4]: ½(1/4)²·8 + 4∫_{−1}^{1}(u² − 1)² du = 1/4 + 64/15.
"""

import numpy as np
import pytest
import torch
from oraculo_neuronal import S0

from taller.neuronal.accion import accion_escape, accion_instanton

N = 2001


def _t(T):
    return torch.linspace(-T, T, N, dtype=torch.float64)


def test_escape_sobre_x_om() -> None:
    t = _t(3.0)
    x = -1 / torch.sqrt(1 + torch.exp(8 * t))
    xdot = 4 * x * (x**2 - 1)  # x_om cumple ẋ = V′(x)
    assert abs(accion_escape(x, xdot, t).item() - 1.0) < 1e-9


def test_escape_sobre_la_recta() -> None:
    t = _t(3.0)
    x = -1 + (t + 3) / 6
    xdot = torch.full_like(t, 1 / 6)
    assert abs(accion_escape(x, xdot, t).item() - 1991 / 840) < 1e-6


def test_escape_en_reposo_en_el_pozo() -> None:
    t = _t(3.0)
    assert accion_escape(torch.full_like(t, -1.0), torch.zeros_like(t), t).item() == 0.0


def test_instanton_sobre_el_kink() -> None:
    tau = _t(4.0)
    x = torch.tanh(np.sqrt(2) * tau)
    xdot = np.sqrt(2) * (1 - x**2)
    assert abs(accion_instanton(x, xdot, tau).item() / S0 - 1) < 1e-8


def test_instanton_sobre_la_recta() -> None:
    tau = _t(4.0)
    x = tau / 4
    xdot = torch.full_like(tau, 0.25)
    assert abs(accion_instanton(x, xdot, tau).item() / (1 / 4 + 64 / 15) - 1) < 1e-5


@pytest.mark.parametrize("funcion", [accion_escape, accion_instanton])
def test_la_accion_es_diferenciable(funcion) -> None:
    t = _t(2.0)
    x = torch.linspace(-1, 0, N, dtype=torch.float64, requires_grad=True)
    funcion(x, torch.zeros_like(t), t).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
