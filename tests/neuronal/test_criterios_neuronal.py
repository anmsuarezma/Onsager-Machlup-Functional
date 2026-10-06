"""Criterios de aceptación del hito 02 sobre los resultados guardados de la arquitectura por defecto.

Las pruebas leen results/neuronal/{escape,instanton}_defecto.npz y se omiten si no existen.
Todo se recalcula aquí con numpy a partir del camino guardado (x y ẋ de la red evaluados en
la malla doble de 4001 puntos), sin usar el código de taller.neuronal:
- la acción en la malla de entrenamiento (puntos pares de la malla doble) y en la doble;
- la alineación en el cruce por x = −1/√2 (escape) o x = 0 (instantón);
- el RMS frente a la referencia cerrada en el intervalo de la especificación.

Claves esperadas: t_doble, x_doble, xdot_doble (4001,), accion (malla de 2001 puntos),
accion_doble, y la clave metadatos (JSON).
"""

import json

import numpy as np
import pytest
from oraculo_neuronal import (
    INTERVALO_RMS_ESCAPE,
    INTERVALO_RMS_INSTANTON,
    PUNTOS_MALLA,
    RESULTADOS,
    S0,
    S_D_MIN,
    T_ESCAPE,
    T_INSTANTON,
    TOL_ACCION,
    TOL_COTA,
    TOL_MALLA,
    TOL_RMS,
    X_ALINEACION_ESCAPE,
    X_ALINEACION_INSTANTON,
    x_kink,
    x_om,
)


def _cargar(problema: str) -> dict:
    ruta = RESULTADOS / f"{problema}_defecto.npz"
    if not ruta.exists():
        pytest.skip(f"falta results/neuronal/{ruta.name}: entrenamiento pendiente")
    with np.load(ruta, allow_pickle=False) as f:
        datos = {k: f[k] for k in f.files if k != "metadatos"}
        datos["metadatos"] = json.loads(str(f["metadatos"]))
    return datos


def _V(x):
    return (x**2 - 1) ** 2


def _dV(x):
    return 4 * x * (x**2 - 1)


def _integrando(problema: str, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
    if problema == "escape":
        return 0.25 * (xdot + _dV(x)) ** 2  # S·D
    return 0.5 * xdot**2 + _V(x)  # S_E


def _acciones(problema: str) -> tuple[float, float, dict]:
    d = _cargar(problema)
    t, f = d["t_doble"], _integrando(problema, d["x_doble"], d["xdot_doble"])
    return np.trapezoid(f[::2], t[::2]), np.trapezoid(f, t), d


def _rms_alineado(problema: str) -> float:
    d = _cargar(problema)
    t, x = d["t_doble"], d["x_doble"]
    nivel, (a, b), referencia = (
        (X_ALINEACION_ESCAPE, INTERVALO_RMS_ESCAPE, x_om) if problema == "escape"
        else (X_ALINEACION_INSTANTON, INTERVALO_RMS_INSTANTON, x_kink)
    )
    k = int(np.flatnonzero(x >= nivel)[0])  # primer punto de la malla en o sobre el nivel
    t_c = t[k - 1] + (nivel - x[k - 1]) * (t[k] - t[k - 1]) / (x[k] - x[k - 1])
    assert t[0] <= t_c + a and t_c + b <= t[-1], f"el intervalo alineado sale del dominio (t_c = {t_c:.3f})"
    s = np.linspace(a, b, round((b - a) / 0.01) + 1)
    return float(np.sqrt(np.mean((np.interp(t_c + s, t, x) - referencia(s)) ** 2)))


@pytest.mark.parametrize("problema, T", [("escape", T_ESCAPE), ("instanton", T_INSTANTON)])
def test_resultado_completo(problema, T) -> None:
    d = _cargar(problema)
    assert d["t_doble"].shape == (2 * PUNTOS_MALLA - 1,)
    assert d["t_doble"][0] == -T and d["t_doble"][-1] == T
    assert d["metadatos"]["parametros"]["capas"] == [32, 32]


@pytest.mark.parametrize("problema", ["escape", "instanton"])
def test_accion_guardada_coincide_con_la_recalculada(problema) -> None:
    S, S_doble, d = _acciones(problema)
    assert abs(S / float(d["accion"]) - 1) < 1e-10
    assert abs(S_doble / float(d["accion_doble"]) - 1) < 1e-12


def test_escape_accion() -> None:
    S, _, _ = _acciones("escape")
    assert abs(S - S_D_MIN) < TOL_ACCION, f"S·D = {S:.6f}"


def test_instanton_accion() -> None:
    S, _, _ = _acciones("instanton")
    assert abs(S - S0) / S0 < TOL_ACCION, f"S_E = {S:.6f}, S0 = {S0:.6f}"


@pytest.mark.parametrize("problema", ["escape", "instanton"])
def test_malla_doble(problema) -> None:
    S, S_doble, _ = _acciones(problema)
    assert abs(S_doble - S) / S < TOL_MALLA, f"diferencia relativa {abs(S_doble - S) / S:.2e}"


@pytest.mark.parametrize("problema, cota", [("escape", S_D_MIN), ("instanton", S0)])
def test_cota_analitica(problema, cota) -> None:
    """S·D ≥ 1 − 1e-4 y S_E ≥ S0 − 1e-4 (completar cuadrados; Ritz da cotas superiores)."""
    S, S_doble, _ = _acciones(problema)
    assert S >= cota - TOL_COTA and S_doble >= cota - TOL_COTA, (S, S_doble)


def test_escape_rms_alineado() -> None:
    rms = _rms_alineado("escape")
    assert rms < TOL_RMS, f"RMS = {rms:.5f}"


def test_instanton_rms_alineado() -> None:
    rms = _rms_alineado("instanton")
    assert rms < TOL_RMS, f"RMS = {rms:.5f}"
