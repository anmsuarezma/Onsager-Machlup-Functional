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


def malla(T: float, puntos: int) -> torch.Tensor:
    """Malla fija equiespaciada en [−T, T], en float64."""
    return torch.linspace(-T, T, puntos, dtype=torch.float64)


def evaluar_accion(problema: str, red, t: torch.Tensor, T: float, x_a: float, x_b: float) -> torch.Tensor:
    """Acción del camino de la red en la malla t (S·D para el escape, S_E para el instantón)."""
    x, xdot = camino_y_derivada(red, t, T, x_a, x_b)
    return ACCIONES[problema](x, xdot, t)


def entrenar(problema: str, config: dict, capas: list[int]) -> dict:
    """Adam y después L-BFGS sobre la acción en la malla fija completa, sin lotes (pérdida determinista).

    Devuelve la red entrenada, la acción en cada evaluación de la pérdida (`historia`, con
    `fase` = 0 para Adam y 1 para L-BFGS), la acción final, el número de parámetros y el
    tiempo de entrenamiento.
    """
    p = config["problemas"][problema]
    T, x_a, x_b = p["T"], p["x_a"], p["x_b"]
    red = crear_red(capas, semilla=p["semilla"], escala_ultima_capa=config["escala_ultima_capa"])
    t = malla(T, config["malla"])
    historia: list[float] = []
    fase: list[int] = []
    inicio = time.perf_counter()

    adam = torch.optim.Adam(red.parameters(), lr=config["adam"]["tasa"])
    for _ in range(config["adam"]["iteraciones"]):
        adam.zero_grad()
        S = evaluar_accion(problema, red, t, T, x_a, x_b)
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
        S = evaluar_accion(problema, red, t, T, x_a, x_b)
        S.backward()
        historia.append(S.item())
        fase.append(1)
        return S

    lbfgs.step(cierre)
    tiempo = time.perf_counter() - inicio
    accion_final = evaluar_accion(problema, red, t, T, x_a, x_b).item()
    return {
        "red": red, "T": T, "x_a": x_a, "x_b": x_b, "capas": capas,
        "historia": np.array(historia), "fase": np.array(fase, dtype=np.int8),
        "accion": accion_final, "n_parametros": contar_parametros(red), "tiempo_s": tiempo,
    }


def evaluar_en_malla_doble(problema: str, resultado: dict, puntos: int) -> dict:
    """Camino, derivada y acción de la red entrenada en la malla doble (2·puntos − 1).

    Verificación de la discretización: la malla doble contiene a la de entrenamiento (sus
    puntos pares) y la acción no debe cambiar más de 1e-4 en relativo.
    """
    T, x_a, x_b, red = resultado["T"], resultado["x_a"], resultado["x_b"], resultado["red"]
    t = malla(T, 2 * puntos - 1)
    x, xdot = camino_y_derivada(red, t, T, x_a, x_b)
    S = ACCIONES[problema](x, xdot, t)
    return {"t_doble": t.detach().numpy(), "x_doble": x.detach().numpy(), "xdot_doble": xdot.detach().numpy(),
            "accion_doble": S.item()}
