"""Selección de dispositivo y precisión del hito 02 (revisión de dependencias, D39).

- Automática: cuda si está disponible, si no mps, si no cpu; --dispositivo la fuerza.
- float64 en cuda y cpu; float32 con aviso en mps (MPS no soporta float64).
- El dispositivo y la precisión quedan en los metadatos.
- CLAUDE.md §12: prueba explícita de que el cálculo corre en la GPU, no solo de que la
  instalación no da errores. Se omite si no hay GPU CUDA.
"""

import pytest
import torch
from oraculo_neuronal import cargar_config

from taller.neuronal.accion import accion_escape
from taller.neuronal.ansatz import camino_y_derivada
from taller.neuronal.correr import analizar_argumentos
from taller.neuronal.dispositivo import elegir_dispositivo, precision
from taller.neuronal.guardado import metadatos_corrida
from taller.neuronal.red import crear_red

HAY_CUDA = torch.cuda.is_available()


def test_seleccion_automatica() -> None:
    esperado = "cuda" if HAY_CUDA else ("mps" if torch.backends.mps.is_available() else "cpu")
    assert elegir_dispositivo(None) == esperado


def test_forzar_cpu() -> None:
    assert elegir_dispositivo("cpu") == "cpu"


def test_forzar_un_dispositivo_ausente_falla() -> None:
    ausente = "mps" if not torch.backends.mps.is_available() else ("cuda" if not HAY_CUDA else None)
    if ausente is None:
        pytest.skip("cuda y mps disponibles a la vez")
    with pytest.raises(RuntimeError):
        elegir_dispositivo(ausente)


@pytest.mark.parametrize("dispositivo", ["cpu", "cuda"])
def test_float64_en_cpu_y_cuda(dispositivo) -> None:
    assert precision(dispositivo) == torch.float64


def test_float32_con_aviso_en_mps() -> None:
    with pytest.warns(UserWarning, match="float32"):
        assert precision("mps") == torch.float32


def test_bandera_dispositivo() -> None:
    assert analizar_argumentos(["configs/neuronal/entrenamiento.yaml"]).dispositivo is None
    assert analizar_argumentos(["configs/neuronal/entrenamiento.yaml", "--dispositivo", "cpu"]).dispositivo == "cpu"


def test_metadatos_registran_dispositivo_y_precision() -> None:
    meta = metadatos_corrida({}, semilla=1, hilos=2, dispositivo="cpu", precision="float64")
    assert meta["dispositivo"] == "cpu" and meta["precision"] == "float64"
    assert isinstance(meta["nombre_dispositivo"], str) and meta["nombre_dispositivo"]


@pytest.mark.skipif(not HAY_CUDA, reason="sin GPU CUDA")
def test_el_calculo_corre_en_la_gpu() -> None:
    """La red, el camino y la acción viven en la GPU en float64, y la acción coincide con la de CPU."""
    c = cargar_config()
    semilla = c["problemas"]["escape"]["semilla"]
    gpu = crear_red([32, 32], semilla=semilla, escala_ultima_capa=1.0, dispositivo="cuda", dtype=torch.float64)
    cpu = crear_red([32, 32], semilla=semilla, escala_ultima_capa=1.0, dispositivo="cpu", dtype=torch.float64)
    assert all(p.device.type == "cuda" and p.dtype == torch.float64 for p in gpu.parameters())
    t_gpu = torch.linspace(-3.0, 3.0, 2001, dtype=torch.float64, device="cuda")
    x, xdot = camino_y_derivada(gpu, t_gpu, 3.0, -1.0, 0.0)
    S_gpu = accion_escape(x, xdot, t_gpu)
    assert S_gpu.device.type == "cuda"
    x, xdot = camino_y_derivada(cpu, t_gpu.cpu(), 3.0, -1.0, 0.0)
    S_cpu = accion_escape(x, xdot, t_gpu.cpu())
    assert abs(S_gpu.item() / S_cpu.item() - 1) < 1e-12
    assert "NVIDIA" in torch.cuda.get_device_name(0)
