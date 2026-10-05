"""Observables y análisis del Bloque B: tiempos medios, Arrhenius, Poisson, Boltzmann y tubo reactivo."""

import numpy as np
from scipy.integrate import quad

from taller.analitico.referencias import V, x_om


def media_y_error(muestras: np.ndarray) -> tuple[float, float]:
    """Media y su error estándar (desviación muestral con ddof = 1 entre √N). E3, E4."""
    return float(muestras.mean()), float(muestras.std(ddof=1) / np.sqrt(muestras.size))


def coeficiente_variacion(muestras: np.ndarray) -> float:
    """Desviación estándar muestral (ddof = 1) entre la media.

    Vale 1 para una exponencial: un escape sin memoria (evento de Poisson) lo cumple (E6).
    """
    return float(muestras.std(ddof=1) / muestras.mean())


def pendiente_arrhenius(D: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Ajuste lineal ln T = pendiente·(1/D) + ordenada (E5).

    Ley de Arrhenius: T ∝ e^{ΔV/D}, así que la pendiente tiende a ΔV = 1 cuando D → 0; a D
    finito las correcciones al prefactor la desvían de 1.
    """
    pendiente, ordenada = np.polyfit(1.0 / np.asarray(D), np.log(np.asarray(T)), 1)
    return float(pendiente), float(ordenada)


def probabilidades_boltzmann(bordes: np.ndarray, D: float) -> np.ndarray:
    """Probabilidad de cada intervalo bajo la densidad estacionaria p_s = e^{−V/D}/Z (E2).

    Z se normaliza en toda la recta, no solo en el rango de los intervalos.
    """
    peso = lambda x: np.exp(-V(x) / D)
    Z = sum(quad(peso, a, b, epsabs=0, epsrel=1e-13, limit=200)[0] for a, b in ((-np.inf, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, np.inf)))
    return np.array([quad(peso, a, b, epsabs=0, epsrel=1e-13, limit=200)[0] for a, b in zip(bordes[:-1], bordes[1:])]) / Z


def distancia_l1(p: np.ndarray, q: np.ndarray) -> float:
    """Σ |p_i − q_i| entre dos distribuciones por intervalos (E2)."""
    return float(np.sum(np.abs(p - q)))


def alinear_ventanas(
    ventanas: np.ndarray, t_cima: np.ndarray, t_alineacion: np.ndarray, malla: np.ndarray
) -> np.ndarray:
    """Interpola cada ventana en la malla de tiempos relativa al último cruce por −1/√2 (E7, decisión 7).

    La muestra k de la ventana i está en t_cima − 3 + 0.01·k; con origen en t_alineacion,
    el mismo origen de tiempo de x_om (x_om(0) = −1/√2). Fuera de la ventana, NaN.
    """
    k = np.arange(ventanas.shape[1])
    alineadas = np.empty((ventanas.shape[0], malla.size))
    for i, fila in enumerate(ventanas):
        t = t_cima[i] - 3.0 + 0.01 * k - t_alineacion[i]
        alineadas[i] = np.interp(malla, t, fila, left=np.nan, right=np.nan)
    return alineadas


def mediana_y_banda(alineadas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mediana y percentiles 10 y 90 punto a punto, ignorando NaN (E7, aclaración 12)."""
    p10, mediana, p90 = np.nanpercentile(alineadas, [10, 50, 90], axis=0)
    return mediana, p10, p90


def rms_contra_om(malla: np.ndarray, curva: np.ndarray) -> float:
    """Desviación cuadrática media entre una curva y el camino de Onsager-Machlup x_om (E7)."""
    return float(np.sqrt(np.mean((curva - x_om(malla)) ** 2)))
