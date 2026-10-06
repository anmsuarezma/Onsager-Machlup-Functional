"""Entrena la red variacional para los dos problemas y las tres arquitecturas (hito 02).

Uso:
    uv run python -m taller.neuronal.correr configs/neuronal/entrenamiento.yaml
        [--dispositivo {cuda,mps,cpu}] [--hilos N] [--salida DIRECTORIO] [--rehacer]

Dispositivo automático (cuda, si no mps, si no cpu) salvo que --dispositivo lo fuerce (D39).
Guarda {salida}/{problema}_{arquitectura}.npz (por defecto, results/neuronal/) y salta los
que ya existen, salvo con --rehacer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from taller.analitico.referencias import S0
from taller.neuronal.dispositivo import elegir_dispositivo, precision
from taller.neuronal.entrenamiento import entrenar, evaluar_en_malla_doble
from taller.neuronal.guardado import RAIZ, guardar_resultado, metadatos_corrida

RESULTADOS = RAIZ / "results" / "neuronal"
REFERENCIA = {"escape": 1.0, "instanton": S0}  # S_min·D = ΔV y S0, solo para el informe en consola


def analizar_argumentos(argv: list[str]) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("config", type=Path)
    analizador.add_argument("--dispositivo", choices=["cuda", "mps", "cpu"], default=None,
                            help="fuerza el dispositivo (por defecto, automático)")
    analizador.add_argument("--hilos", type=int, default=None, help="hilos de PyTorch en CPU (por defecto, la configuración)")
    analizador.add_argument("--salida", type=Path, default=RESULTADOS, help="directorio de resultados")
    analizador.add_argument("--rehacer", action="store_true")
    return analizador.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = analizar_argumentos(sys.argv[1:] if argv is None else argv)
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    hilos = args.hilos if args.hilos is not None else config["hilos"]
    torch.set_num_threads(hilos)
    dispositivo = elegir_dispositivo(args.dispositivo if args.dispositivo is not None else config.get("dispositivo"))
    dtype = precision(dispositivo)
    nombre_precision = str(dtype).removeprefix("torch.")
    print(f"{args.config.name}: PyTorch {torch.__version__}, dispositivo {dispositivo}, {nombre_precision}, "
          f"{hilos} hilos de CPU → {args.salida}", flush=True)
    for problema in ("escape", "instanton"):
        for nombre, capas in config["arquitecturas"].items():
            ruta = args.salida / f"{problema}_{nombre}.npz"
            if ruta.exists() and not args.rehacer:
                print(f"  {ruta.name}: ya existe, se salta", flush=True)
                continue
            r = entrenar(problema, config, capas, dispositivo, dtype)
            doble = evaluar_en_malla_doble(problema, r, config["malla"])
            datos = {"historia": r["historia"], "fase": r["fase"], "accion": np.float64(r["accion"]),
                     **{k: (np.float64(v) if np.isscalar(v) else v) for k, v in doble.items()}}
            p = config["problemas"][problema]
            parametros = {"problema": problema, "arquitectura": nombre, "capas": capas, "horizonte": p["horizonte"],
                          "x_a": p["x_a"], "x_b": p["x_b"], "malla": config["malla"],
                          "escala_ultima_capa": config["escala_ultima_capa"], "adam": config["adam"],
                          "lbfgs": config["lbfgs"], "n_parametros": r["n_parametros"],
                          "tiempo_s": round(r["tiempo_s"], 2), "evaluaciones_lbfgs": int((r["fase"] == 1).sum()),
                          "cuda_de_torch": torch.version.cuda}
            guardar_resultado(ruta, datos, metadatos_corrida(parametros, p["semilla"], hilos, dispositivo, nombre_precision))
            ref = REFERENCIA[problema]
            print(f"  {ruta.name}: acción {r['accion']:.10f} (error relativo {r['accion'] / ref - 1:+.2e}), "
                  f"malla doble {abs(doble['accion_doble'] / r['accion'] - 1):.1e}, {r['n_parametros']} parámetros, "
                  f"{r['tiempo_s']:.1f} s, {parametros['evaluaciones_lbfgs']} evaluaciones de L-BFGS", flush=True)


if __name__ == "__main__":
    main()
