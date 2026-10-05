"""Configuración del Bloque B: hilos (aclaración 6, D23), semillas desde configs/ (CLAUDE.md §6),
parámetros de la especificación y reanudación de las corridas (aclaración 14)."""

import numba
import numpy as np
import pytest
import yaml
from oraculo_estocastico import (
    ANCHO_E2,
    CONFIGS,
    D_E2,
    D_E3,
    DT_E2,
    HILOS_POR_DEFECTO,
    INTERVALO_E2,
    MUESTRAS_E2,
    N_E2,
    N_E3,
    N_E4,
    N_E4_D01,
    T_EQUILIBRIO_E2,
    VALORES_D,
    VALORES_DT_E3,
)

from taller.estocastico.configuracion import (
    aplicar_hilos,
    cargar_config,
    n_trayectorias,
    resolver_hilos,
    semillas_trayectorias,
)
from taller.estocastico.correr import analizar_argumentos, debe_correr, ruta_e3, ruta_e4

ARCHIVOS = sorted(CONFIGS.glob("*.yaml"))


def test_existen_las_configuraciones() -> None:
    nombres = {p.name for p in ARCHIVOS}
    assert {"comun.yaml", "pruebas.yaml", "e2_boltzmann.yaml", "e3_convergencia_dt.yaml", "e4_tiempos.yaml"} <= nombres


# --- Hilos --------------------------------------------------------------------------------
def test_hilos_por_defecto_es_10() -> None:
    assert cargar_config(CONFIGS / "comun.yaml")["hilos"] == HILOS_POR_DEFECTO


def test_resolver_hilos_usa_la_configuracion_sin_bandera() -> None:
    assert resolver_hilos({"hilos": HILOS_POR_DEFECTO}, None) == HILOS_POR_DEFECTO


def test_bandera_hilos_tiene_prioridad() -> None:
    assert resolver_hilos({"hilos": HILOS_POR_DEFECTO}, 4) == 4


def test_aplicar_hilos_fija_numba_y_reporta_la_capa() -> None:
    info = aplicar_hilos(4)
    assert numba.get_num_threads() == 4
    assert info["hilos"] == 4
    assert isinstance(info["capa_hilos"], str) and info["capa_hilos"]


# --- Línea de comandos y reanudación -------------------------------------------------------
def test_cli_banderas() -> None:
    sin = analizar_argumentos(["configs/estocastico/e4_tiempos.yaml"])
    con = analizar_argumentos(["configs/estocastico/e4_tiempos.yaml", "--hilos", "8", "--rehacer"])
    assert sin.hilos is None and sin.rehacer is False
    assert con.hilos == 8 and con.rehacer is True


def test_un_archivo_por_D_y_por_dt(tmp_path) -> None:
    assert ruta_e4(tmp_path, 0.1).name == "e4_D0.100.npz"
    assert ruta_e4(tmp_path, 0.125).name == "e4_D0.125.npz"
    assert ruta_e3(tmp_path, 1e-2).name == "e3_dt1e-02.npz"
    assert ruta_e3(tmp_path, 5e-4).name == "e3_dt5e-04.npz"
    nombres = {ruta_e4(tmp_path, d).name for d in VALORES_D} | {ruta_e3(tmp_path, dt).name for dt in VALORES_DT_E3}
    assert len(nombres) == len(VALORES_D) + len(VALORES_DT_E3)


def test_reanudacion_salta_lo_existente_salvo_rehacer(tmp_path) -> None:
    hecho, pendiente = ruta_e4(tmp_path, 0.1), ruta_e4(tmp_path, 0.125)
    hecho.write_bytes(b"x")
    assert debe_correr(hecho, rehacer=False) is False
    assert debe_correr(pendiente, rehacer=False) is True
    assert debe_correr(hecho, rehacer=True) is True


# --- Semillas -----------------------------------------------------------------------------
@pytest.mark.parametrize("ruta", [p for p in ARCHIVOS if p.name.startswith("e")], ids=lambda p: p.name)
def test_toda_corrida_tiene_semilla_entera(ruta) -> None:
    """Nada aleatorio sin semilla desde configuración."""
    config = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    assert isinstance(config["semilla"], int)


def test_pruebas_yaml_tiene_semillas_enteras_distintas() -> None:
    semillas = yaml.safe_load((CONFIGS / "pruebas.yaml").read_text(encoding="utf-8"))["semillas"]
    valores = list(semillas.values())
    assert all(isinstance(s, int) for s in valores) and len(set(valores)) == len(valores)


def test_semillas_trayectorias_deterministas_y_distintas() -> None:
    a = semillas_trayectorias(12345, 1000)
    b = semillas_trayectorias(12345, 1000)
    c = semillas_trayectorias(12346, 1000)
    assert a.shape == (1000,)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 1000
    assert not np.array_equal(a, c)


# --- Parámetros de la especificación y sus aclaraciones -----------------------------------
def test_parametros_e2() -> None:
    e2 = cargar_config(CONFIGS / "e2_boltzmann.yaml")
    assert e2["D"] == D_E2 and e2["dt"] == DT_E2 and e2["N"] == N_E2
    assert e2["t_equilibrio"] == T_EQUILIBRIO_E2 and e2["intervalo"] == INTERVALO_E2
    assert e2["n_muestras"] == MUESTRAS_E2 and e2["ancho_intervalo"] == ANCHO_E2


def test_parametros_e3() -> None:
    e3 = cargar_config(CONFIGS / "e3_convergencia_dt.yaml")
    assert e3["D"] == D_E3 and e3["dt"] == VALORES_DT_E3 and e3["N"] == N_E3


def test_parametros_e4() -> None:
    e4 = cargar_config(CONFIGS / "e4_tiempos.yaml")
    assert e4["D"] == VALORES_D
    assert n_trayectorias(e4, 0.1) == N_E4_D01
    assert all(n_trayectorias(e4, d) == N_E4 for d in VALORES_D if d != 0.1)
