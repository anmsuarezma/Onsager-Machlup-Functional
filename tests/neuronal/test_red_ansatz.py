"""Red y ansatz del hito 02: fronteras exactas, camino inicial recto, derivada por autograd,
float64, número de parámetros y ausencia de simetrías impuestas."""

import numpy as np
import pytest
import torch
from oraculo_neuronal import ARQUITECTURAS, PARAMETROS, cargar_config

from taller.neuronal.ansatz import camino, camino_y_derivada
from taller.neuronal.red import contar_parametros, crear_red

PROBLEMAS = {"escape": (3.0, -1.0, 0.0), "instanton": (4.0, -1.0, 1.0)}


def _red(capas=(32, 32), escala=None):
    c = cargar_config()
    escala = c["escala_ultima_capa"] if escala is None else escala
    return crear_red(list(capas), semilla=c["problemas"]["escape"]["semilla"], escala_ultima_capa=escala)


@pytest.mark.parametrize("problema", PROBLEMAS)
@pytest.mark.parametrize("escala", [None, 1.0], ids=["inicial", "salida_grande"])
def test_fronteras_exactas(problema, escala) -> None:
    """El ansatz impone x(−T) = x_a y x(T) = x_b exactamente, sea cual sea N."""
    T, x_a, x_b = PROBLEMAS[problema]
    x = camino(_red(escala=escala), torch.tensor([-T, T], dtype=torch.float64), T, x_a, x_b)
    assert x[0].item() == x_a and x[1].item() == x_b


@pytest.mark.parametrize("problema", PROBLEMAS)
def test_camino_inicial_es_la_recta(problema) -> None:
    """Con la última capa casi nula, el camino inicial es la recta entre los extremos."""
    T, x_a, x_b = PROBLEMAS[problema]
    t = torch.linspace(-T, T, 2001, dtype=torch.float64)
    x = camino(_red(), t, T, x_a, x_b).detach().numpy()
    recta = x_a + (x_b - x_a) * (t.numpy() + T) / (2 * T)
    assert np.max(np.abs(x - recta)) < 1e-2


@pytest.mark.parametrize("problema", PROBLEMAS)
def test_derivada_autograd_contra_diferencias_finitas(problema) -> None:
    T, x_a, x_b = PROBLEMAS[problema]
    red = _red(escala=1.0)
    t = torch.linspace(-T, T, 401, dtype=torch.float64)
    x, xdot = camino_y_derivada(red, t, T, x_a, x_b)
    h = 1e-5
    fd = (camino(red, t + h, T, x_a, x_b) - camino(red, t - h, T, x_a, x_b)) / (2 * h)
    assert x.dtype == torch.float64 and xdot.dtype == torch.float64
    assert torch.max(torch.abs(xdot - fd)).item() < 1e-7


@pytest.mark.parametrize("nombre", ARQUITECTURAS)
def test_numero_de_parametros(nombre) -> None:
    assert contar_parametros(_red(ARQUITECTURAS[nombre])) == PARAMETROS[nombre]


def test_parametros_en_float64() -> None:
    assert all(p.dtype == torch.float64 for p in _red().parameters())


def test_no_impone_simetria() -> None:
    """La red no es impar por construcción: N(s) + N(−s) ≠ 0 para una inicialización genérica."""
    red = _red(escala=1.0)
    s = torch.linspace(0.1, 1.0, 10, dtype=torch.float64).unsqueeze(1)
    assert torch.max(torch.abs(red(s) + red(-s))).item() > 1e-6


def test_semilla_reproducible() -> None:
    a, b = _red(escala=1.0), _red(escala=1.0)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
