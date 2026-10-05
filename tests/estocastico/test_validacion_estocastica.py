"""E1 y E2: validación estadística del integrador Numba (marcadas `lento`).

E1: Numba contra la referencia NumPy (generadores y semillas distintos), Kolmogorov-Smirnov
sobre los tiempos a +1 corregidos y sin corregir (D24).
E2: equilibrio de Boltzmann con D = 0.5 y los parámetros de la aclaración 13; las
probabilidades exactas por intervalo se calculan aquí con quad, sin usar
taller.estocastico.observables.
"""

import numpy as np
import pytest
from oraculo_estocastico import (
    ANCHO_E2,
    D_E2,
    DT_E2,
    INTERVALO_E2,
    L1_E2,
    MUESTRAS_E2,
    N_E2,
    P_KS_E1,
    T_EQUILIBRIO_E2,
    semillas_de_pruebas,
)
from scipy.integrate import quad
from scipy.stats import ks_2samp

from taller.estocastico.configuracion import aplicar_hilos, semillas_trayectorias
from taller.estocastico.integrador import simular_equilibrio, simular_escape
from taller.estocastico.referencia_numpy import simular_escape_numpy

pytestmark = pytest.mark.lento


@pytest.fixture(scope="module")
def e1():
    """D = 0.35, dt = 1e-3, 10⁴ trayectorias por implementación."""
    semillas = semillas_de_pruebas()
    aplicar_hilos(10)
    numba_ = simular_escape(0.35, 1e-3, semillas_trayectorias(semillas["e1_numba"], 10_000))
    numpy_ = simular_escape_numpy(0.35, 1e-3, 10_000, semillas["e1_numpy"])
    return numba_, numpy_


@pytest.mark.parametrize("clave", ["t_pozo", "t_pozo_sin_corregir"])
def test_e1_numba_contra_numpy(e1, clave) -> None:
    """E1: KS de dos muestras de los tiempos a +1, p > 0.01."""
    numba_, numpy_ = e1
    p = ks_2samp(numba_[clave], numpy_[clave]).pvalue
    assert p > P_KS_E1, f"{clave}: p = {p:.4f}"


def test_e2_equilibrio_de_boltzmann() -> None:
    """E2: distancia L1 entre probabilidades por intervalo < 0.02.

    10⁵ trayectorias, equilibrado hasta t = 50, 20 muestras separadas 5.0 (tiempo de
    correlación ~10 con D = 0.5), intervalos de ancho 0.05 en [−2, 2]. Unos 100 s con 10 hilos.
    """
    aplicar_hilos(10)
    semillas = semillas_trayectorias(semillas_de_pruebas()["e2"], N_E2)
    x = simular_equilibrio(
        D_E2, DT_E2, semillas, t_equilibrio=T_EQUILIBRIO_E2, intervalo=INTERVALO_E2, n_muestras=MUESTRAS_E2
    )
    assert x.shape == (N_E2, MUESTRAS_E2)
    bordes = np.linspace(-2.0, 2.0, round(4.0 / ANCHO_E2) + 1)
    conteos, _ = np.histogram(x.ravel(), bins=bordes)
    empirica = conteos / x.size
    peso = lambda y: np.exp(-((y**2 - 1) ** 2) / D_E2)
    Z = quad(peso, -np.inf, np.inf, epsabs=0, epsrel=1e-12)[0]
    exacta = np.array([quad(peso, a, b)[0] for a, b in zip(bordes[:-1], bordes[1:])]) / Z
    l1 = np.sum(np.abs(empirica - exacta))
    assert l1 < L1_E2, f"L1 = {l1:.4f}"
