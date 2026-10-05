"""Prueba de humo del entorno: el paquete y sus dependencias se importan.

No verifica ningún resultado físico; solo que `uv sync` dejó un entorno utilizable.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "modulo",
    ["taller", "taller.analitico", "sympy", "numpy", "scipy", "matplotlib", "jupytext"],
)
def test_importa(modulo: str) -> None:
    importlib.import_module(modulo)
