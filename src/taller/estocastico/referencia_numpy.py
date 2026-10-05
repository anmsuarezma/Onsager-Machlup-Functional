"""Referencia en NumPy puro del integrador de escape (Bloque B, E1).

Implementación independiente de integrador.py, más simple y más lenta: vectorizada sobre
trayectorias, con el generador PCG64 de NumPy (`default_rng`) en lugar de xoshiro256**.
Sirve para validar el integrador Numba con pocas trayectorias (E1). No guarda ventanas.
"""

import numpy as np

from taller.analitico.referencias import dV


def simular_escape_numpy(D: float, dt: float, N: int, semilla: int) -> dict:
    """Tiempos de primer paso desde x0 = −1 a 0 y a +1, con y sin corrección de puente browniano.

    Misma convención que integrador.simular_escape: el cruce, observado o sorteado con
    probabilidad exp(−(b − x)(b − xn)/(D dt)), se asigna al final del paso; el paso por +1
    corregido solo se cuenta después del paso corregido por 0. Cada trayectoria se integra
    hasta su llegada observada a +1.
    """
    rng = np.random.default_rng(semilla)
    sigma = np.sqrt(2.0 * D * dt)  # σ² = 2D
    tiempos = {c: np.full(N, np.nan) for c in ("t_cima", "t_pozo", "t_cima_sin_corregir", "t_pozo_sin_corregir")}
    indice = np.arange(N)  # trayectorias activas
    x = np.full(N, -1.0)
    n = 0
    while indice.size:
        n += 1
        xn = x - dV(x) * dt + sigma * rng.standard_normal(indice.size)
        t = n * dt
        for b, corregido, observado in ((0.0, "t_cima", "t_cima_sin_corregir"), (1.0, "t_pozo", "t_pozo_sin_corregir")):
            pendiente = np.isnan(tiempos[corregido][indice])
            if b == 1.0:
                pendiente &= ~np.isnan(tiempos["t_cima"][indice])
            with np.errstate(over="ignore"):
                p = np.where(xn >= b, 1.0, np.exp(-np.maximum(b - x, 0) * np.maximum(b - xn, 0) / (D * dt)))
            cruza = pendiente & (rng.random(indice.size) < p)
            tiempos[corregido][indice[cruza]] = t
            nuevo = np.isnan(tiempos[observado][indice]) & (xn >= b)
            tiempos[observado][indice[nuevo]] = t
        sigue = np.isnan(tiempos["t_pozo_sin_corregir"][indice])
        indice, x = indice[sigue], xn[sigue]
    return tiempos
