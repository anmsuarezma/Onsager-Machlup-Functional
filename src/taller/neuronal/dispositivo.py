"""Dispositivo de cómputo y precisión del Bloque A (D39).

Selección automática: cuda si está disponible, si no mps, si no cpu. La precisión es float64
en cuda y cpu. MPS (GPU de Apple) no soporta float64: allí se usa float32 con un aviso, y la
precisión queda registrada en los metadatos.
"""

import warnings

import torch

DISPONIBLE = {
    "cuda": torch.cuda.is_available,
    "mps": torch.backends.mps.is_available,
    "cpu": lambda: True,
}


def elegir_dispositivo(forzado: str | None) -> str:
    """Dispositivo forzado (si está disponible) o el primero disponible entre cuda, mps y cpu."""
    if forzado is not None:
        if forzado not in DISPONIBLE:
            raise ValueError(f"dispositivo desconocido: {forzado}")
        if not DISPONIBLE[forzado]():
            raise RuntimeError(f"se pidió el dispositivo {forzado}, pero no está disponible")
        return forzado
    return next(d for d in ("cuda", "mps", "cpu") if DISPONIBLE[d]())


def precision(dispositivo: str) -> torch.dtype:
    """float64 en cuda y cpu; float32 con aviso en mps, que no soporta float64."""
    if dispositivo == "mps":
        warnings.warn("MPS no soporta float64: se entrena en float32 (menos precisión; queda en los metadatos)",
                      UserWarning, stacklevel=2)
        return torch.float32
    return torch.float64


def nombre_dispositivo(dispositivo: str) -> str:
    """Nombre legible del hardware, para los metadatos."""
    if dispositivo == "cuda":
        return torch.cuda.get_device_name(0)
    if dispositivo == "mps":
        return "Apple MPS"
    import platform

    return platform.processor() or platform.machine()
