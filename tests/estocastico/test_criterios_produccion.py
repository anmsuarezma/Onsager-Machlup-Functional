"""E3-E7: criterios de aceptación evaluados sobre los resultados guardados de las corridas de producción.

Las corridas las ejecutan los revisores (aclaración 5). Estas pruebas leen
results/estocastico/ y se omiten si el resultado todavía no existe. Los valores exactos son
la tabla de mpmath del oráculo; el análisis se reimplementa aquí con numpy para no validar
el código con el mismo código. Los criterios se aplican a los tiempos corregidos con el
puente browniano (aclaración 7); los tiempos sin corregir deben estar guardados.

Archivos esperados (aclaración 14, un archivo por dt y por D):
- e3_dt{dt:.0e}.npz: t_cima, t_pozo, t_cima_sin_corregir, t_pozo_sin_corregir (N,).
- e4_D{D:.3f}.npz: los mismos tiempos; para D en D_E7, además ventanas (N, 351) y
  t_alineacion (N,). Metadatos con parametros.dt (el dt de producción) y parametros.N.
"""

import json

import numpy as np
import pytest
from oraculo_estocastico import (
    CV_E6,
    D_E3,
    D_E7,
    ERROR_DT_E3,
    ERROR_E4,
    MALLA_E7,
    MUESTRAS_VENTANA,
    N_E3,
    N_E4,
    N_E4_D01,
    PENDIENTE_E5,
    RESULTADOS,
    T_CIMA,
    T_POZO,
    VALORES_D,
    VALORES_DT_E3,
    x_om,
)

DESTINOS = [("t_cima", T_CIMA), ("t_pozo", T_POZO)]
SIN_CORREGIR = ("t_cima_sin_corregir", "t_pozo_sin_corregir")


def _cargar(nombre: str):
    ruta = RESULTADOS / nombre
    if not ruta.exists():
        pytest.skip(f"falta results/estocastico/{nombre}: corrida de producción pendiente")
    with np.load(ruta, allow_pickle=False) as f:
        datos = {k: f[k] for k in f.files if k != "metadatos"}
        meta = json.loads(str(f["metadatos"]))
    return datos, meta


def _e3(dt: float):
    return _cargar(f"e3_dt{dt:.0e}.npz")


def _e4(D: float):
    return _cargar(f"e4_D{D:.3f}.npz")


def _error_y_ee(t: np.ndarray, exacto: float) -> tuple[float, float]:
    """Error relativo de la media y su error estándar relativo."""
    return abs(t.mean() / exacto - 1), t.std(ddof=1) / np.sqrt(t.size) / exacto


# --- E3 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", VALORES_DT_E3)
def test_e3_archivo_completo(dt) -> None:
    datos, _ = _e3(dt)
    for clave in ("t_cima", "t_pozo", *SIN_CORREGIR):
        assert datos[clave].shape == (N_E3,), clave


@pytest.mark.parametrize("clave, tabla", DESTINOS, ids=["cima", "pozo"])
def test_e3_error_decrece_con_dt(clave, tabla) -> None:
    """Aclaración 8: monotonía exigida solo entre los dt cuyo error supera 2 errores estándar."""
    errores = [_error_y_ee(_e3(dt)[0][clave], tabla[D_E3]) for dt in VALORES_DT_E3]  # dt decreciente
    significativos = [e for e, ee in errores if e > 2 * ee]
    assert np.all(np.diff(significativos) < 0), errores


def test_e3_dt_de_produccion() -> None:
    """dt de producción = mayor dt con error < 0.02 en ambos destinos corregidos; E4 debe usar ese dt."""
    validos = [
        dt for dt in VALORES_DT_E3
        if all(_error_y_ee(_e3(dt)[0][c], t[D_E3])[0] < ERROR_DT_E3 for c, t in DESTINOS)
    ]
    assert validos, "ningún dt cumple el 2 %: se reporta y se detiene (E3)"
    for D in VALORES_D:
        assert _e4(D)[1]["parametros"]["dt"] == max(validos), D


# --- E4 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("D", VALORES_D)
def test_e4_archivo_completo(D) -> None:
    datos, meta = _e4(D)
    N = N_E4_D01 if D == 0.1 else N_E4
    assert meta["parametros"]["N"] == N
    for clave in ("t_cima", "t_pozo", *SIN_CORREGIR):
        assert datos[clave].shape == (N,), clave
    if D in D_E7:
        assert datos["ventanas"].shape == (N, MUESTRAS_VENTANA) and datos["t_alineacion"].shape == (N,)


@pytest.mark.parametrize("D", VALORES_D)
@pytest.mark.parametrize("clave, tabla", DESTINOS, ids=["cima", "pozo"])
def test_e4_tiempos_contra_exacto(D, clave, tabla) -> None:
    error, ee = _error_y_ee(_e4(D)[0][clave], tabla[D])
    assert error < ERROR_E4, f"D={D}: {error:.4f} (error estándar relativo {ee:.4f})"


# --- E5 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("clave, tabla", DESTINOS, ids=["cima", "pozo"])
def test_e5_pendiente_arrhenius(clave, tabla) -> None:
    D = np.array([d for d in VALORES_D if d <= 0.2])
    T_sim = np.array([_e4(d)[0][clave].mean() for d in D])
    T_exacto = np.array([tabla[d] for d in D])
    pendiente_sim = np.polyfit(1 / D, np.log(T_sim), 1)[0]
    pendiente_exacta = np.polyfit(1 / D, np.log(T_exacto), 1)[0]
    assert abs(pendiente_sim - pendiente_exacta) < PENDIENTE_E5, (pendiente_sim, pendiente_exacta)


# --- E6 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("D", [d for d in VALORES_D if d <= 0.15])
def test_e6_coeficiente_de_variacion(D) -> None:
    t = _e4(D)[0]["t_pozo"]
    cv = t.std(ddof=1) / t.mean()
    assert abs(cv - 1) < CV_E6, f"D={D}: CV = {cv:.4f}"


# --- E7 ---------------------------------------------------------------------------------
def _rms_mediana(D: float) -> float:
    datos, _ = _e4(D)
    v, t0, t_al = datos["ventanas"], datos["t_cima"], datos["t_alineacion"]
    malla = np.linspace(*MALLA_E7, 76)  # paso 0.01
    t_muestra = t0[:, None] - 3.0 + 0.01 * np.arange(v.shape[1]) - t_al[:, None]
    alineadas = np.array([np.interp(malla, tm, fila, left=np.nan, right=np.nan) for tm, fila in zip(t_muestra, v)])
    mediana = np.nanmedian(alineadas, axis=0)
    return float(np.sqrt(np.mean((mediana - x_om(malla)) ** 2)))


def test_e7_tubo_reactivo_se_estrecha() -> None:
    """Aclaración 11: la desviación cuadrática media entre la mediana y x_om en t ∈ [−0.5, 0.25] decrece al disminuir D."""
    rms = [_rms_mediana(D) for D in sorted(D_E7, reverse=True)]
    assert np.all(np.diff(rms) < 0), rms
