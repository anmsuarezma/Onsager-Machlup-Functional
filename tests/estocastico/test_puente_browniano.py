"""Validación de la corrección de puente browniano (aclaración 7, D24) antes de usarla en E3 y E4.

1. La probabilidad de cruce entre pasos, contra valores escritos a mano.
2. Caso de solución exacta: movimiento browniano sin deriva dx = √(2D) dW desde 0 y nivel
   a = 1. Euler-Maruyama es exacto en los instantes de la malla, así que el único sesgo es
   el de observar solo esos instantes; para este proceso el puente browniano es exacto, de
   modo que la corrección debe eliminar el sesgo por completo. La probabilidad exacta de
   alcanzar a antes de t es erfc(a/√(4 D t)) (principio de reflexión).
3. El sistema del taller (doble pozo, D = 0.25) con un dt grueso, contra T exacto de mpmath:
   sin corregir, T(−1 → 0) tiene un sesgo grande; corregido, ambos destinos cumplen el 2 % de E3.
"""

import numpy as np
import pytest
from oraculo_estocastico import (
    D_BROWNIANO,
    DT_BROWNIANO,
    ERROR_DT_E3,
    NIVEL_BROWNIANO,
    P_ALCANCE_BROWNIANO,
    T_CIMA,
    T_POZO,
    semillas_de_pruebas,
)

from taller.estocastico.configuracion import aplicar_hilos, semillas_trayectorias
from taller.estocastico.integrador import prob_cruce_puente, simular_browniano_libre, simular_escape

N_BROWNIANO = 100_000


@pytest.mark.parametrize(
    "x, xn, b, D, dt, esperado",
    [
        # exp(−(b − x)(b − xn)/(D dt)), evaluado a mano
        (-0.1, -0.05, 0.0, 0.25, 1e-2, 0.13533528323661262),  # exp(−2)
        (-0.1, 0.05, 0.0, 0.25, 1e-2, 1.0),  # cruce observado
        (0.9, 1.2, 1.0, 0.25, 1e-2, 1.0),  # cruce observado en el pozo
        (-1.0, -1.0, 0.0, 0.25, 1e-2, 0.0),  # lejos: exp(−400) se trunca a 0
    ],
)
def test_prob_cruce_puente(x, xn, b, D, dt, esperado) -> None:
    assert abs(prob_cruce_puente(x, xn, b, D, dt) - esperado) < 1e-15


@pytest.fixture(scope="module")
def browniano():
    aplicar_hilos(4)
    semillas = semillas_trayectorias(semillas_de_pruebas()["puente_browniano"], N_BROWNIANO)
    return simular_browniano_libre(D_BROWNIANO, DT_BROWNIANO, NIVEL_BROWNIANO, 1.0, semillas)


@pytest.mark.parametrize("t", sorted(P_ALCANCE_BROWNIANO))
def test_browniano_corregido_es_exacto(browniano, t) -> None:
    """P(τ_a ≤ t) corregida coincide con erfc(a/√(4Dt)) dentro de 4 errores estándar (≈ 0.006 en t = 1)."""
    exacta = P_ALCANCE_BROWNIANO[t]
    p = np.mean(browniano["t_paso"] <= t + 1e-9)
    error_estandar = np.sqrt(exacta * (1 - exacta) / N_BROWNIANO)
    assert abs(p - exacta) < 4 * error_estandar, f"t={t}: {p:.5f} frente a {exacta:.5f}"


@pytest.mark.parametrize("t", sorted(P_ALCANCE_BROWNIANO))
def test_browniano_sin_corregir_esta_sesgado(browniano, t) -> None:
    """Sin corregir se pierden cruces: P(τ_a ≤ t) queda por debajo de la exacta en más de 0.02.

    Con dt = 0.05 el nivel efectivo sube ≈ 0.5826·√(2D dt) ≈ 0.13 y P(τ ≤ 1) baja de 0.317 a ≈ 0.26.
    """
    p = np.mean(browniano["t_paso_sin_corregir"] <= t + 1e-9)
    assert p < P_ALCANCE_BROWNIANO[t] - 0.02, f"t={t}: {p:.5f}"


@pytest.mark.lento
def test_doble_pozo_dt_grueso() -> None:
    """D = 0.25, dt = 5e-3, 10⁵ trayectorias (unos 10 s con 10 hilos).

    Corregido: |T/T_exacto − 1| < 0.02 en ambos destinos. Sin corregir: T(−1 → 0) con sesgo
    > 0.05 (la simulación de prueba de D24 dio +8 % con dt = 5e-3).
    """
    aplicar_hilos(10)
    semillas = semillas_trayectorias(semillas_de_pruebas()["puente_doble_pozo"], 100_000)
    r = simular_escape(0.25, 5e-3, semillas)
    for clave, tabla in (("t_cima", T_CIMA), ("t_pozo", T_POZO)):
        error = abs(r[clave].mean() / tabla[0.25] - 1)
        assert error < ERROR_DT_E3, f"{clave}: {error:.4f}"
    sesgo = r["t_cima_sin_corregir"].mean() / T_CIMA[0.25] - 1
    assert sesgo > 0.05, f"sesgo sin corregir en la cima: {sesgo:+.4f}"
