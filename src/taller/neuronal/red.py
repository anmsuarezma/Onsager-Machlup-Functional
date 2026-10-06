"""Perceptrón multicapa N(s) del ansatz variacional (Bloque A, hito 02)."""

import torch
from torch import nn


def crear_red(capas: list[int], semilla: int, escala_ultima_capa: float) -> nn.Sequential:
    """Perceptrón 1 → capas → 1 en float64, con tanh en las capas ocultas y salida lineal.

    Inicialización: Xavier uniforme en los pesos de las capas ocultas y sesgos uniformes en
    ±1/√(entradas) (la inicialización por defecto de PyTorch). Los sesgos no pueden ser nulos:
    con tanh impar y sin sesgos, la red sería exactamente impar y, en un funcional simétrico,
    el gradiente conservaría esa simetría, que quedaría impuesta (D34). La última capa se
    inicia con Xavier multiplicado por `escala_ultima_capa` (casi nula) y sesgo nulo, de modo
    que N ≈ 0 y el camino inicial del ansatz es la recta entre los extremos. La semilla fija la
    inicialización con un generador propio, sin tocar el estado global de PyTorch.
    """
    generador = torch.Generator().manual_seed(semilla)
    dimensiones = [1, *capas, 1]
    modulos: list[nn.Module] = []
    for i, (entrada, salida) in enumerate(zip(dimensiones[:-1], dimensiones[1:])):
        lineal = nn.Linear(entrada, salida, dtype=torch.float64)
        nn.init.xavier_uniform_(lineal.weight, generator=generador)
        if i == len(dimensiones) - 2:
            nn.init.zeros_(lineal.bias)
            with torch.no_grad():
                lineal.weight.mul_(escala_ultima_capa)
        else:
            cota = entrada**-0.5
            nn.init.uniform_(lineal.bias, -cota, cota, generator=generador)
        modulos.append(lineal)
        if i < len(dimensiones) - 2:
            modulos.append(nn.Tanh())
    return nn.Sequential(*modulos)


def contar_parametros(red: nn.Module) -> int:
    """Número de parámetros entrenables (pesos y sesgos)."""
    return sum(p.numel() for p in red.parameters() if p.requires_grad)
