"""Funciones de análisis del Bloque B contra valores escritos a mano o calculados con quad aquí mismo."""

import numpy as np
import pytest
from oraculo_estocastico import MALLA_E7, MUESTRAS_VENTANA, T_POZO, x_om
from scipy.integrate import quad

from taller.estocastico.observables import (
    alinear_ventanas,
    coeficiente_variacion,
    distancia_l1,
    media_y_error,
    mediana_y_banda,
    pendiente_arrhenius,
    probabilidades_boltzmann,
    rms_contra_om,
)


def test_media_y_error() -> None:
    media, error = media_y_error(np.array([1.0, 2.0, 3.0]))
    assert media == 2.0 and abs(error - 1 / np.sqrt(3)) < 1e-15


def test_coeficiente_variacion() -> None:
    """Desviación estándar muestral (ddof = 1) entre la media: [1, 2, 3] → 1/2."""
    assert abs(coeficiente_variacion(np.array([1.0, 2.0, 3.0])) - 0.5) < 1e-15


def test_pendiente_arrhenius_sintetica() -> None:
    D = np.array([0.1, 0.125, 0.15, 0.2])
    pendiente, ordenada = pendiente_arrhenius(D, 3.0 * np.exp(1.0 / D))
    assert abs(pendiente - 1.0) < 1e-12 and abs(ordenada - np.log(3.0)) < 1e-12


def test_pendiente_arrhenius_exacta() -> None:
    """Pendiente de ln T(−1 → +1) frente a 1/D en D ≤ 0.2 (valor de np.polyfit sobre la tabla de mpmath)."""
    D = np.array([0.1, 0.125, 0.15, 0.2])
    pendiente, _ = pendiente_arrhenius(D, np.array([T_POZO[d] for d in D]))
    assert abs(pendiente - 0.9884441226588673) < 1e-10


def test_probabilidades_boltzmann_contra_quad() -> None:
    D = 0.5
    bordes = np.linspace(-2.0, 2.0, 81)
    peso = lambda x: np.exp(-((x**2 - 1) ** 2) / D)
    Z = quad(peso, -np.inf, np.inf, epsabs=0, epsrel=1e-13)[0]
    esperado = np.array([quad(peso, a, b, epsabs=0, epsrel=1e-13)[0] for a, b in zip(bordes[:-1], bordes[1:])]) / Z
    assert np.max(np.abs(probabilidades_boltzmann(bordes, D) - esperado)) < 1e-12


def test_probabilidades_boltzmann_mpmath() -> None:
    """P(0.5 < x < 1.5) con D = 0.5, normalizada en toda la recta (mpmath, 30 dígitos)."""
    p = probabilidades_boltzmann(np.array([-0.5, 0.0, 0.5, 1.5]), 0.5)
    assert abs(p[2] - 0.43048243445418221) < 1e-12
    assert abs(p[0] - p[1]) < 1e-14  # simetría x → −x


def test_distancia_l1() -> None:
    assert abs(distancia_l1(np.array([0.2, 0.3, 0.5]), np.array([0.25, 0.25, 0.5])) - 0.1) < 1e-15


@pytest.fixture(scope="module")
def ventanas_sinteticas():
    """Ventanas construidas con x_om desplazado: alineadas deben reproducir x_om."""
    n = 50
    t_cima = 10.0 + 0.37 * np.arange(n)
    t_alineacion = t_cima - (0.2 + 0.013 * np.arange(n))  # desfases que no caen en la malla
    t_muestra = t_cima[:, None] - 3.0 + 0.01 * np.arange(MUESTRAS_VENTANA)
    ventanas = x_om(t_muestra - t_alineacion[:, None])
    return ventanas, t_cima, t_alineacion


def test_alinear_ventanas_reproduce_x_om(ventanas_sinteticas) -> None:
    ventanas, t_cima, t_alineacion = ventanas_sinteticas
    malla = np.linspace(-0.5, 0.5, 101)
    alineadas = alinear_ventanas(ventanas, t_cima, t_alineacion, malla)
    assert alineadas.shape == (50, 101)
    assert np.nanmax(np.abs(alineadas - x_om(malla)[None, :])) < 1e-3  # interpolación lineal, h = 0.01


def test_alinear_ventanas_nan_fuera_de_la_ventana(ventanas_sinteticas) -> None:
    ventanas, t_cima, t_alineacion = ventanas_sinteticas
    malla = np.array([-5.0, 0.0])  # −5 queda antes del inicio de toda ventana
    alineadas = alinear_ventanas(ventanas, t_cima, t_alineacion, malla)
    assert np.all(np.isnan(alineadas[:, 0])) and np.all(np.isfinite(alineadas[:, 1]))


def test_mediana_y_banda() -> None:
    datos = np.tile(np.arange(11.0)[:, None], (1, 3))  # columnas 0, 1, ..., 10
    mediana, p10, p90 = mediana_y_banda(datos)
    assert np.allclose(mediana, 5.0) and np.allclose(p10, 1.0) and np.allclose(p90, 9.0)


def test_rms_contra_om() -> None:
    malla = np.linspace(*MALLA_E7, 76)
    assert rms_contra_om(malla, x_om(malla)) < 1e-15
    assert abs(rms_contra_om(malla, x_om(malla) + 0.1) - 0.1) < 1e-12
