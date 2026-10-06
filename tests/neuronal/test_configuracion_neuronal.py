"""Configuración del hito 02: parámetros de la especificación y semillas desde configs/ (CLAUDE.md §6)."""

from oraculo_neuronal import ARQUITECTURAS, HORIZONTE_ESCAPE, HORIZONTE_INSTANTON, PUNTOS_MALLA, cargar_config


def test_horizontes_y_fronteras() -> None:
    c = cargar_config()["problemas"]
    assert c["escape"]["horizonte"] == HORIZONTE_ESCAPE and c["instanton"]["horizonte"] == HORIZONTE_INSTANTON
    assert (c["escape"]["x_a"], c["escape"]["x_b"]) == (-1.0, 0.0)
    assert (c["instanton"]["x_a"], c["instanton"]["x_b"]) == (-1.0, 1.0)


def test_malla_y_arquitecturas() -> None:
    c = cargar_config()
    assert c["malla"] == PUNTOS_MALLA
    assert c["arquitecturas"] == ARQUITECTURAS


def test_semillas_enteras_y_distintas() -> None:
    p = cargar_config()["problemas"]
    semillas = [p["escape"]["semilla"], p["instanton"]["semilla"]]
    assert all(isinstance(s, int) for s in semillas) and len(set(semillas)) == 2


def test_optimizadores() -> None:
    c = cargar_config()
    assert c["adam"]["iteraciones"] >= 1000  # "unas miles de iteraciones"
    assert 1e-4 <= c["adam"]["tasa"] <= 1e-2  # "del orden de 1e-3"
    assert c["lbfgs"]["iteraciones_max"] > 0
