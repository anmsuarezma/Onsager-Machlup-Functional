# Hito 00 — Analítico

## Entorno

Registrado el 2026-10-04, al preparar el proyecto.

| Elemento | Valor |
|---|---|
| Sistema operativo | Ubuntu 24.04.5 LTS, kernel 6.14.0-37-generic |
| Python | 3.12.3 (intérprete del sistema, usado por uv en `.venv`) |
| uv | 0.10.11 |
| git | 2.43.0 |
| GPU | NVIDIA GeForce RTX 3060, 12 288 MiB |
| Driver NVIDIA | 595.91.07 |
| CUDA máxima soportada por el driver | 13.2 (según `nvidia-smi`; no hay toolkit de CUDA instalado por el proyecto) |

### Dependencias resueltas por uv

Proyecto:

| Paquete | Versión |
|---|---|
| sympy | 1.14.0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| mpmath (transitiva de sympy) | 1.3.0 |

Desarrollo (grupo `dev`):

| Paquete | Versión |
|---|---|
| pytest | 9.1.1 |
| jupytext | 1.19.6 |
| jupyterlab | 4.6.4 |
| ipykernel | 7.4.0 |

El conjunto completo (109 paquetes instalados en `.venv`, incluido `taller`) está fijado en `uv.lock`.

### Pruebas del entorno

`tests/test_entorno.py`: 7 pruebas de importación, todas pasan (detalle en el resumen del hito).

### Pendientes para hitos futuros

- PyTorch y Numba traen su propio runtime de CUDA (*wheels* `nvidia-*` o `cuda-*`). En sus hitos habrá que verificar que ese runtime es compatible con el driver 595.91.07 (CUDA máxima 13.2) y comprobar explícitamente que usan la GPU (CLAUDE.md §12).
