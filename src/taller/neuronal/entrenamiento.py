"""Minimización directa del funcional con la red (método directo del cálculo de variaciones, tipo Ritz).

Sin datos: la pérdida es solo la acción discretizada del camino del ansatz. Las soluciones
cerradas no intervienen en el entrenamiento; se usan solo para validar (pruebas y cuaderno).
"""

import time

import numpy as np
import torch

from taller.neuronal.accion import accion_escape, accion_instanton
from taller.neuronal.ansatz import camino_y_derivada
from taller.neuronal.red import contar_parametros, crear_red

ACCIONES = {"escape": accion_escape, "instanton": accion_instanton}


def malla(horizonte: float, puntos: int, dispositivo: str = "cpu", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Malla fija equiespaciada en [−T_h, T_h]."""
    return torch.linspace(-horizonte, horizonte, puntos, dtype=dtype, device=dispositivo)


def evaluar_accion(problema: str, red, t: torch.Tensor, horizonte: float, x_a: float, x_b: float) -> torch.Tensor:
    """Acción del camino de la red en la malla t (S·D para el escape, S_E para el instantón)."""
    x, xdot = camino_y_derivada(red, t, horizonte, x_a, x_b)
    return ACCIONES[problema](x, xdot, t)


def _sincronizar(dispositivo: str) -> None:
    """Espera a que termine el trabajo en la GPU, para medir tiempos reales."""
    if dispositivo == "cuda":
        torch.cuda.synchronize()
    elif dispositivo == "mps":
        torch.mps.synchronize()


def entrenar(
    problema: str, config: dict, capas: list[int], dispositivo: str = "cpu", dtype: torch.dtype = torch.float64
) -> dict:
    """Adam y después L-BFGS sobre la acción en la malla fija completa, sin lotes (pérdida determinista).

    Devuelve la red entrenada, la acción en cada evaluación de la pérdida (`historia`, con
    `fase` = 0 para Adam y 1 para L-BFGS), la acción final, el número de parámetros y el
    tiempo de entrenamiento (con la GPU sincronizada).
    """
    p = config["problemas"][problema]
    horizonte, x_a, x_b = p["horizonte"], p["x_a"], p["x_b"]
    red = crear_red(capas, semilla=p["semilla"], escala_ultima_capa=config["escala_ultima_capa"],
                    dispositivo=dispositivo, dtype=dtype)
    t = malla(horizonte, config["malla"], dispositivo, dtype)
    _sincronizar(dispositivo)
    historia: list[float] = []
    fase: list[int] = []
    inicio = time.perf_counter()

    adam = torch.optim.Adam(red.parameters(), lr=config["adam"]["tasa"])
    for _ in range(config["adam"]["iteraciones"]):
        adam.zero_grad()
        S = evaluar_accion(problema, red, t, horizonte, x_a, x_b)
        S.backward()
        adam.step()
        historia.append(S.item())
        fase.append(0)

    c = config["lbfgs"]
    lbfgs = torch.optim.LBFGS(
        red.parameters(), lr=c["tasa"], max_iter=c["iteraciones_max"], max_eval=int(1.25 * c["iteraciones_max"]),
        history_size=c["historia"], tolerance_grad=c["tolerancia_grad"], tolerance_change=c["tolerancia_cambio"],
        line_search_fn="strong_wolfe",
    )

    def cierre():
        lbfgs.zero_grad()
        S = evaluar_accion(problema, red, t, horizonte, x_a, x_b)
        S.backward()
        historia.append(S.item())
        fase.append(1)
        return S

    lbfgs.step(cierre)
    _sincronizar(dispositivo)
    tiempo = time.perf_counter() - inicio
    accion_final = evaluar_accion(problema, red, t, horizonte, x_a, x_b).item()
    return {
        "red": red, "horizonte": horizonte, "x_a": x_a, "x_b": x_b, "capas": capas,
        "dispositivo": dispositivo, "dtype": dtype,
        "historia": np.array(historia), "fase": np.array(fase, dtype=np.int8),
        "accion": accion_final, "n_parametros": contar_parametros(red), "tiempo_s": tiempo,
    }


def evaluar_en_malla_doble(problema: str, resultado: dict, puntos: int) -> dict:
    """Camino, derivada y acción de la red entrenada en la malla doble (2·puntos − 1).

    Verificación de la discretización: la malla doble contiene a la de entrenamiento (sus
    puntos pares) y la acción no debe cambiar más de 1e-4 en relativo.
    """
    horizonte, x_a, x_b, red = resultado["horizonte"], resultado["x_a"], resultado["x_b"], resultado["red"]
    t = malla(horizonte, 2 * puntos - 1, resultado["dispositivo"], resultado["dtype"])
    x, xdot = camino_y_derivada(red, t, horizonte, x_a, x_b)
    S = ACCIONES[problema](x, xdot, t)
    a_numpy = lambda v: v.detach().cpu().double().numpy()
    return {"t_doble": a_numpy(t), "x_doble": a_numpy(x), "xdot_doble": a_numpy(xdot), "accion_doble": S.item()}
