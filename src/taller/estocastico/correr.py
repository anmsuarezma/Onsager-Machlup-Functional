"""Corridas de producción del Bloque B (E2, E3, E4) desde un archivo de configuración.

Uso:
    uv run python -m taller.estocastico.correr configs/estocastico/e3_convergencia_dt.yaml [--hilos N] [--rehacer]

Guarda un archivo por dt (E3) o por D (E4) en results/estocastico/ y salta los que ya
existen, salvo con --rehacer (aclaración 14): un apagado solo cuesta el elemento en curso.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from taller.analitico.referencias import tiempo_primer_paso
from taller.estocastico.configuracion import (
    aplicar_hilos,
    cargar_config,
    n_trayectorias,
    resolver_hilos,
    semillas_trayectorias,
)
from taller.estocastico.guardado import RAIZ, guardar_resultado, metadatos_corrida
from taller.estocastico.integrador import simular_equilibrio, simular_escape

RESULTADOS = RAIZ / "results" / "estocastico"
COMUN = RAIZ / "configs" / "estocastico" / "comun.yaml"
PASOS_POR_SEGUNDO_10_HILOS = 3.3e8  # medido con 10 hilos: 3.9e8 sin ventanas, 3.25e8 con ventanas (D28)


def analizar_argumentos(argv: list[str]) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("config", type=Path, help="archivo YAML de configs/estocastico/")
    analizador.add_argument("--hilos", type=int, default=None, help="hilos de Numba (por defecto, comun.yaml)")
    analizador.add_argument("--rehacer", action="store_true", help="recalcula aunque el resultado ya exista")
    return analizador.parse_args(argv)


def ruta_e3(carpeta: Path, dt: float) -> Path:
    return Path(carpeta) / f"e3_dt{dt:.0e}.npz"


def ruta_e4(carpeta: Path, D: float) -> Path:
    return Path(carpeta) / f"e4_D{D:.3f}.npz"


def debe_correr(ruta: Path, rehacer: bool) -> bool:
    """Reanudación: un resultado existente se salta, salvo con --rehacer."""
    return rehacer or not Path(ruta).exists()


def _semilla_elemento(semilla: int, indice: int) -> int:
    """Semilla independiente para el elemento `indice` (un dt o un D) de una corrida."""
    return int(np.random.SeedSequence([semilla, indice]).generate_state(1, dtype=np.uint64)[0])


def _estimar(N: int, T: float, dt: float, hilos: int) -> str:
    pasos = N * T / dt
    segundos = pasos / (PASOS_POR_SEGUNDO_10_HILOS * hilos / 10)
    return f"{pasos:.2e} pasos, ~{segundos / 60:.1f} min (estimado)"


def _correr_escape(ruta, D, dt, N, semilla, indice, ventanas, info_hilos, config):
    semilla_e = _semilla_elemento(semilla, indice)
    print(f"  {ruta.name}: D={D}, dt={dt}, N={N}, ventanas={ventanas}; "
          f"{_estimar(N, tiempo_primer_paso(D, 1.0), dt, info_hilos['hilos'])}", flush=True)
    inicio = time.perf_counter()
    r = simular_escape(D, dt, semillas_trayectorias(semilla_e, N), guardar_ventanas=ventanas)
    duracion = time.perf_counter() - inicio
    if ventanas:
        r["ventanas"] = r["ventanas"].astype(np.float32)
    parametros = {"experimento": config["experimento"], "D": D, "dt": dt, "N": N,
                  "indice": indice, "semilla_elemento": semilla_e, "duracion_s": round(duracion, 1)}
    guardar_resultado(ruta, r, metadatos_corrida(parametros, semilla, info_hilos))
    for clave, b in (("t_cima", 0.0), ("t_pozo", 1.0)):
        t, t_sin = r[clave], r[f"{clave}_sin_corregir"]
        exacto = tiempo_primer_paso(D, b)
        ee = t.std(ddof=1) / np.sqrt(N) / exacto
        print(f"    {clave}: T/T_exacto − 1 = {t.mean() / exacto - 1:+.4f} ± {ee:.4f} "
              f"(sin corregir {t_sin.mean() / exacto - 1:+.4f})", flush=True)
    print(f"    {duracion / 60:.1f} min", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = analizar_argumentos(sys.argv[1:] if argv is None else argv)
    config = cargar_config(args.config)
    info_hilos = aplicar_hilos(resolver_hilos(cargar_config(COMUN), args.hilos))
    print(f"{args.config.name}: {info_hilos['hilos']} hilos ({info_hilos['capa_hilos']})", flush=True)
    if metadatos_corrida({}, 0, info_hilos)["arbol_modificado"]:
        print("AVISO: hay cambios sin commit; el hash guardado en los metadatos no basta para reproducir.", flush=True)
    experimento, semilla = config["experimento"], config["semilla"]

    if experimento == "e3":
        for i, dt in enumerate(config["dt"]):
            ruta = ruta_e3(RESULTADOS, dt)
            if not debe_correr(ruta, args.rehacer):
                print(f"  {ruta.name}: ya existe, se salta", flush=True)
                continue
            _correr_escape(ruta, config["D"], dt, config["N"], semilla, i, False, info_hilos, config)

    elif experimento == "e4":
        if config["dt"] is None:
            sys.exit("e4_tiempos.yaml: dt es null. Fíjalo con el dt de producción elegido en E3.")
        for i, D in enumerate(config["D"]):
            ruta = ruta_e4(RESULTADOS, D)
            if not debe_correr(ruta, args.rehacer):
                print(f"  {ruta.name}: ya existe, se salta", flush=True)
                continue
            ventanas = D in config["D_ventanas"]
            _correr_escape(ruta, D, config["dt"], n_trayectorias(config, D), semilla, i, ventanas, info_hilos, config)

    elif experimento == "e2":
        ruta = RESULTADOS / "e2_boltzmann.npz"
        if not debe_correr(ruta, args.rehacer):
            print(f"  {ruta.name}: ya existe, se salta")
            return
        parametros = {k: config[k] for k in ("D", "dt", "N", "t_equilibrio", "intervalo", "n_muestras", "ancho_intervalo")}
        x = simular_equilibrio(config["D"], config["dt"], semillas_trayectorias(semilla, config["N"]),
                               config["t_equilibrio"], config["intervalo"], config["n_muestras"])
        guardar_resultado(ruta, {"x": x.astype(np.float32)}, metadatos_corrida(parametros, semilla, info_hilos))
        print(f"  {ruta.name}: guardado", flush=True)

    else:
        sys.exit(f"experimento desconocido: {experimento}")


if __name__ == "__main__":
    main()
