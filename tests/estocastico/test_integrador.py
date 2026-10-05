"""Integrador Euler-Maruyama en Numba (CPU): paso, registro de primeros pasos (corregidos y sin
corregir), ventanas y reproducibilidad.

Pruebas rápidas y deterministas. La validación del puente browniano está en
test_puente_browniano.py y la estadística (E1, E2) en test_validacion_estocastica.py.
"""

import numpy as np
import pytest
from oraculo_estocastico import INDICE_CIMA, MUESTRAS_VENTANA, X_ALINEACION, semillas_de_pruebas

from taller.estocastico.configuracion import aplicar_hilos, semillas_trayectorias
from taller.estocastico.integrador import paso_em, simular_escape

DT = 1e-3
CLAVES_TIEMPO = ("t_cima", "t_pozo", "t_cima_sin_corregir", "t_pozo_sin_corregir", "t_alineacion")


@pytest.mark.parametrize(
    "x, dt, D, xi, esperado",
    [
        # x − 4x(x² − 1)dt + √(2D dt)·ξ, evaluado a mano
        (0.5, 1e-2, 0.25, 1.0, 0.5857106781186547),
        (-1.2, 1e-3, 0.1, -0.7, -1.2077874949366116),
        (-1.0, 1e-3, 0.3, 0.0, -1.0),  # mínimo: sin ruido no se mueve
    ],
)
def test_paso_em(x, dt, D, xi, esperado) -> None:
    assert abs(paso_em(x, dt, D, xi) - esperado) < 1e-15


@pytest.fixture(scope="module")
def corrida_corta():
    aplicar_hilos(4)
    semillas = semillas_trayectorias(semillas_de_pruebas()["integrador"], 256)
    return simular_escape(0.35, DT, semillas, guardar_ventanas=True)


def test_tiempos_ordenados_y_positivos(corrida_corta) -> None:
    """Para llegar a +1 hay que pasar antes por 0: 0 < t_cima ≤ t_pozo, con y sin corregir."""
    c = corrida_corta
    assert np.all(c["t_cima"] > 0) and np.all(c["t_cima"] <= c["t_pozo"])
    assert np.all(c["t_cima_sin_corregir"] <= c["t_pozo_sin_corregir"])


def test_correccion_solo_adelanta_los_cruces(corrida_corta) -> None:
    """El puente agrega cruces entre pasos: el tiempo corregido nunca es posterior al observado."""
    c = corrida_corta
    assert np.all(c["t_cima"] <= c["t_cima_sin_corregir"])
    assert np.all(c["t_pozo"] <= c["t_pozo_sin_corregir"])


def test_tiempos_en_la_malla_de_pasos(corrida_corta) -> None:
    for clave in CLAVES_TIEMPO:
        k = corrida_corta[clave] / DT
        assert np.max(np.abs(k - np.round(k))) < 1e-6, clave


def test_forma_de_las_ventanas(corrida_corta) -> None:
    assert corrida_corta["ventanas"].shape == (256, MUESTRAS_VENTANA)


def test_muestra_de_la_cima_cuando_el_cruce_es_observado(corrida_corta) -> None:
    """Si el cruce de x = 0 se observó en la malla (corregido = sin corregir), la muestra en t_cima es ≥ 0."""
    c = corrida_corta
    observado = c["t_cima"] == c["t_cima_sin_corregir"]
    assert observado.any()
    assert np.all(c["ventanas"][observado, INDICE_CIMA] >= 0)


def test_ventanas_sin_huecos_salvo_antes_de_t0(corrida_corta) -> None:
    """Las muestras anteriores a t = 0 (si t_cima < 3) son NaN; el resto, hasta t_cima + 0.5, finitas."""
    v = corrida_corta["ventanas"]
    t_muestra = corrida_corta["t_cima"][:, None] - 3.0 + 0.01 * np.arange(MUESTRAS_VENTANA)
    antes = t_muestra < -1e-9
    assert np.all(np.isnan(v[antes])) and np.all(np.isfinite(v[~antes]))


def test_alineacion_antes_de_la_cima(corrida_corta) -> None:
    """El último cruce ascendente por −1/√2 ocurre en (0, t_cima)."""
    t_al, t0 = corrida_corta["t_alineacion"], corrida_corta["t_cima"]
    assert np.all(t_al > 0) and np.all(t_al < t0)


def test_entre_alineacion_y_cima_no_se_vuelve_bajo_menos_raiz(corrida_corta) -> None:
    """Por definición de "último cruce", las muestras en [t_alineacion, t_cima] están sobre −1/√2."""
    v, t_al, t0 = corrida_corta["ventanas"], corrida_corta["t_alineacion"], corrida_corta["t_cima"]
    t_muestra = t0[:, None] - 3.0 + 0.01 * np.arange(MUESTRAS_VENTANA)
    entre = (t_muestra >= t_al[:, None] - 1e-9) & (t_muestra <= t0[:, None] + 1e-9)
    assert np.all(v[entre] >= X_ALINEACION)


def test_reproducible_bit_a_bit_e_independiente_de_los_hilos() -> None:
    """Cada trayectoria tiene su propia semilla (D27): el resultado no depende del número de hilos."""
    semillas = semillas_trayectorias(semillas_de_pruebas()["reproducibilidad"], 64)
    aplicar_hilos(1)
    a = simular_escape(0.5, DT, semillas, guardar_ventanas=True)
    aplicar_hilos(4)
    b = simular_escape(0.5, DT, semillas, guardar_ventanas=True)
    for clave in (*CLAVES_TIEMPO, "ventanas"):
        assert np.array_equal(a[clave], b[clave], equal_nan=True), clave
