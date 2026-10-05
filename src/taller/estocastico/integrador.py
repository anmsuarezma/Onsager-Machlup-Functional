"""Integrador Euler-Maruyama de la dinámica de Langevin en Numba, en CPU (Bloque B, aclaración 1).

dx = −V′(x) dt + √(2D) dW, con V(x) = (x² − 1)² y x0 = −1. Cada iteración del prange
integra una trayectoria completa y registra sus observables sobre la marcha:

- tiempos de primer paso por la cima (x = 0) y por el otro pozo (x = +1), con y sin la
  corrección de puente browniano (aclaración 7, D24);
- el último cruce ascendente por −1/√2 antes de la cima (alineación de E7);
- opcionalmente, la ventana de E7: posiciones en t_cima − 3 + 0.01·k, k = 0, ..., 350.

Generador: xoshiro256** propio, uno por trayectoria, iniciado con splitmix64 a partir de
su semilla de 64 bits (D28). Normales por el método polar de Marsaglia.
Notación: σ² = 2D (el puente browniano se escribe a veces con σ).
"""

import math

import numba
import numpy as np
from numba import njit, prange

from taller.analitico import referencias

# V′ de la única fuente de verdad, compilada para Numba.
_dV = njit(cache=True)(referencias.dV)

X_ALINEACION = -1.0 / math.sqrt(2.0)  # x_om(0), origen de tiempo de E7 (decisión 7)
DURACION_ANTES = 3.0  # ventana: 3 antes de t_cima (decisión 7)
MUESTRAS_ANTES = 300
MUESTRAS_DESPUES = 50  # 0.5 después de t_cima (aclaración 11)
MUESTREO = 0.01

_U64 = numba.uint64


# --- Generador de números aleatorios --------------------------------------------------------
@njit(cache=True)
def _rotl(x, k):
    return (x << _U64(k)) | (x >> _U64(64 - k))


@njit(cache=True)
def _iniciar(semilla):
    """Estado de xoshiro256** a partir de una semilla de 64 bits, con splitmix64."""
    estado = np.empty(4, dtype=np.uint64)
    z = _U64(semilla)
    for i in range(4):
        z = z + _U64(0x9E3779B97F4A7C15)
        w = z
        w = (w ^ (w >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        w = (w ^ (w >> _U64(27))) * _U64(0x94D049BB133111EB)
        estado[i] = w ^ (w >> _U64(31))
    return estado


@njit(cache=True)
def _siguiente(s):
    resultado = _rotl(s[1] * _U64(5), 7) * _U64(9)
    t = s[1] << _U64(17)
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = _rotl(s[3], 45)
    return resultado


@njit(cache=True)
def _uniforme(s):
    """Uniforme en [0, 1) con 53 bits."""
    return float(_siguiente(s) >> _U64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _par_normal(s):
    """Dos normales estándar independientes (método polar de Marsaglia)."""
    while True:
        u = 2.0 * _uniforme(s) - 1.0
        v = 2.0 * _uniforme(s) - 1.0
        r = u * u + v * v
        if 0.0 < r < 1.0:
            f = math.sqrt(-2.0 * math.log(r) / r)
            return u * f, v * f


@njit(cache=True)
def normales(semilla: int, n: int) -> np.ndarray:
    """n normales estándar de un generador (para validar el generador por separado)."""
    s = _iniciar(semilla)
    z = np.empty(n)
    for i in range(0, n - 1, 2):
        z[i], z[i + 1] = _par_normal(s)
    if n % 2:
        z[n - 1] = _par_normal(s)[0]
    return z


# --- Paso de Euler-Maruyama y puente browniano ----------------------------------------------
@njit(cache=True)
def paso_em(x: float, dt: float, D: float, xi: float) -> float:
    """Un paso de Euler-Maruyama: x − V′(x) dt + √(2D dt) ξ, con ξ normal estándar."""
    return x - _dV(x) * dt + math.sqrt(2.0 * D * dt) * xi


@njit(cache=True)
def prob_cruce_puente(x: float, xn: float, b: float, D: float, dt: float) -> float:
    """Probabilidad de que la trayectoria haya cruzado b entre dos instantes de la malla.

    Si algún extremo está en b o por encima, el cruce se observó (1). Si no, el puente
    browniano con σ² = 2D entre x y xn alcanza b con probabilidad
    exp(−(b − x)(b − xn)/(D dt)) (aclaración 7).
    """
    if x >= b or xn >= b:
        return 1.0
    return math.exp(-(b - x) * (b - xn) / (D * dt))


@njit(cache=True)
def _cruza(x, xn, b, D, dt, s):
    """¿Cruzó b en este paso? Observado o, si no, sorteado con la probabilidad del puente."""
    if xn >= b:
        return True
    exponente = (b - x) * (b - xn) / (D * dt)
    if exponente > 50.0:  # probabilidad < 2e-22: no se sortea (ahorra un número aleatorio)
        return False
    return _uniforme(s) < math.exp(-exponente)


def _pasos_por_muestra(dt: float) -> int:
    m = round(MUESTREO / dt)
    if m < 1 or abs(m * dt - MUESTREO) > 1e-9 * MUESTREO:
        raise ValueError(f"dt = {dt} no divide el muestreo de la ventana ({MUESTREO})")
    return m


# --- Escape desde x0 = −1 ------------------------------------------------------------------
def simular_escape(D: float, dt: float, semillas: np.ndarray, guardar_ventanas: bool = False) -> dict:
    """Integra una trayectoria por semilla desde x0 = −1 hasta caer en +1 (Bloque B, E1 y E3-E7).

    Devuelve un diccionario con arreglos (N,):
    - t_cima, t_pozo: tiempos de primer paso por 0 y por +1, con la corrección de puente;
    - t_cima_sin_corregir, t_pozo_sin_corregir: los mismos, observados solo en la malla;
    - t_alineacion: último cruce ascendente por −1/√2 antes de t_cima (corregido);
    y, si guardar_ventanas, ventanas (N, 351) con las posiciones en
    t_cima − 3 + 0.01·k (NaN antes de t = 0).

    Cada trayectoria sigue hasta su llegada observada a +1 y, si hay ventana, hasta
    t_cima + 0.5; los tiempos son múltiplos de dt (el cruce se asigna al final del paso).
    """
    m = _pasos_por_muestra(dt)
    semillas = np.ascontiguousarray(semillas, dtype=np.uint64)
    t = _simular_escape(D, dt, semillas, guardar_ventanas, m)
    claves = ("t_cima", "t_pozo", "t_cima_sin_corregir", "t_pozo_sin_corregir", "t_alineacion")
    salida = {c: t[0][:, i] for i, c in enumerate(claves)}
    if guardar_ventanas:
        salida["ventanas"] = t[1]
    return salida


@njit(parallel=True, cache=True)
def _simular_escape(D, dt, semillas, guardar_ventanas, m):
    N = semillas.size
    tiempos = np.empty((N, 5))
    n_ventana = MUESTRAS_ANTES + MUESTRAS_DESPUES + 1
    ventanas = np.full((N, n_ventana if guardar_ventanas else 0), np.nan)
    largo = MUESTRAS_ANTES * m + 1  # búfer circular: pasos n_cima − 300m, ..., n_cima
    for i in prange(N):
        s = _iniciar(semillas[i])
        bufer = np.empty(largo if guardar_ventanas else 1)
        x = -1.0
        n = 0
        if guardar_ventanas:
            bufer[0] = x
        n_cima = -1
        n_pozo = -1
        n_cima_obs = -1
        n_pozo_obs = -1
        n_cruce = -1  # último cruce ascendente por −1/√2
        n_fin_ventana = -1
        hay_extra = False
        extra = 0.0
        while True:
            if hay_extra:
                xi = extra
                hay_extra = False
            else:
                xi, extra = _par_normal(s)
                hay_extra = True
            xn = paso_em(x, dt, D, xi)
            n += 1
            if n_cima < 0:
                if x < X_ALINEACION <= xn:
                    n_cruce = n
                if _cruza(x, xn, 0.0, D, dt, s):
                    n_cima = n
                    n_fin_ventana = n + MUESTRAS_DESPUES * m
            if n_cima_obs < 0 and xn >= 0.0:
                n_cima_obs = n
            if n_cima >= 0 and n_pozo < 0 and _cruza(x, xn, 1.0, D, dt, s):
                n_pozo = n
            if n_pozo_obs < 0 and xn >= 1.0:
                n_pozo_obs = n
            if guardar_ventanas:
                if n_cima < 0 or n == n_cima:
                    bufer[n % largo] = xn
                if n == n_cima:  # copia las 301 muestras hasta la cima
                    for k in range(MUESTRAS_ANTES + 1):
                        paso = n_cima - (MUESTRAS_ANTES - k) * m
                        if paso >= 0:
                            ventanas[i, k] = bufer[paso % largo]
                elif n_cima >= 0 and n <= n_fin_ventana and (n - n_cima) % m == 0:
                    ventanas[i, MUESTRAS_ANTES + (n - n_cima) // m] = xn
            x = xn
            if n_pozo_obs >= 0 and (not guardar_ventanas or n >= n_fin_ventana):
                break
        tiempos[i, 0] = n_cima * dt
        tiempos[i, 1] = n_pozo * dt
        tiempos[i, 2] = n_cima_obs * dt
        tiempos[i, 3] = n_pozo_obs * dt
        tiempos[i, 4] = n_cruce * dt
    return tiempos, ventanas


# --- Equilibrio (E2) -----------------------------------------------------------------------
def simular_equilibrio(
    D: float, dt: float, semillas: np.ndarray, t_equilibrio: float, intervalo: float, n_muestras: int
) -> np.ndarray:
    """Posiciones (N, n_muestras) en t_equilibrio + j·intervalo, sin fronteras absorbentes (E2).

    Desde x0 = −1, tras equilibrar ambos pozos; su histograma se compara con la densidad de
    Boltzmann e^{−V/D}/Z, el control físico del integrador.
    """
    semillas = np.ascontiguousarray(semillas, dtype=np.uint64)
    pasos = np.array([round((t_equilibrio + j * intervalo) / dt) for j in range(n_muestras)], dtype=np.int64)
    return _simular_equilibrio(D, dt, semillas, pasos)


@njit(parallel=True, cache=True)
def _simular_equilibrio(D, dt, semillas, pasos):
    N = semillas.size
    x_muestras = np.empty((N, pasos.size))
    for i in prange(N):
        s = _iniciar(semillas[i])
        x = -1.0
        n = 0
        for j in range(pasos.size):
            while n < pasos[j]:
                xi1, xi2 = _par_normal(s)
                x = paso_em(x, dt, D, xi1)
                n += 1
                if n < pasos[j]:
                    x = paso_em(x, dt, D, xi2)
                    n += 1
            x_muestras[i, j] = x
    return x_muestras


# --- Validación del puente: browniano sin deriva -------------------------------------------
def simular_browniano_libre(D: float, dt: float, a: float, t_max: float, semillas: np.ndarray) -> dict:
    """Primer paso por a > 0 de dx = √(2D) dW desde 0, hasta t_max (validación de la aclaración 7).

    Euler-Maruyama es exacto en la malla para este proceso, y el puente browniano también
    es exacto: con la corrección, P(τ_a ≤ t) debe coincidir con erfc(a/√(4Dt)). Devuelve
    t_paso y t_paso_sin_corregir (inf si no se alcanza a antes de t_max).
    """
    semillas = np.ascontiguousarray(semillas, dtype=np.uint64)
    t = _simular_browniano_libre(D, dt, a, round(t_max / dt), semillas)
    return {"t_paso": t[:, 0], "t_paso_sin_corregir": t[:, 1]}


@njit(parallel=True, cache=True)
def _simular_browniano_libre(D, dt, a, n_max, semillas):
    N = semillas.size
    t = np.full((N, 2), np.inf)
    sigma = math.sqrt(2.0 * D * dt)
    for i in prange(N):
        s = _iniciar(semillas[i])
        x = 0.0
        corregido = False
        for n in range(1, n_max + 1):
            xn = x + sigma * _par_normal(s)[0]
            if not corregido and _cruza(x, xn, a, D, dt, s):
                t[i, 0] = n * dt
                corregido = True
            if xn >= a:
                t[i, 1] = n * dt
                break
            x = xn
    return t
