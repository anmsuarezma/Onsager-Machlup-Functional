# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hito 02 — Bloque A reducido: la red variacional
#
# El Bloque A resuelve el problema variacional del taller en un tercer espacio de funciones:
# primero fue el analítico (hito 00), luego la simulación del proceso real (hito 01) y
# ahora una red neuronal. Aquí se usa solo la **forma débil**: se minimiza directamente la
# acción sobre la familia de caminos que genera la red (método directo del cálculo de
# variaciones, tipo Ritz), sin pasar por la ecuación de Euler-Lagrange (especificación
# `specs/02_neuronal.md`). Las redes verifican las matemáticas: que el minimizador y su
# acción son los de las soluciones cerradas.
#
# Cada sección sigue la estructura de siempre: **objetivo**, **resultado esperado**,
# **cálculo**, **verificación** e **interpretación física**.
#
# **Este cuaderno no entrena.** Carga los resultados guardados en `results/neuronal/` por
# `python -m taller.neuronal.correr` y genera las figuras en `figures/neuronal/`. Los criterios
# de aceptación se verifican en `tests/neuronal/test_criterios_neuronal.py`.
#
# **Sin datos.** Las soluciones cerradas y la simulación no intervienen en el
# entrenamiento: la pérdida es solo la acción. Aquí se usan para validar.
#
# **Correspondencia con el plan del taller.**
#
# | Cuaderno | Contenido | Plan del taller |
# |---|---|---|
# | A.0 | Método directo: ansatz, red y funcionales | §5′ Fundamento de las redes (Ritz) |
# | A.1 | Escape térmico: la red frente a $x_\mathrm{om}$ | §2 Camino más probable; Bloque A, aplicación |
# | A.2 | Instantón: la red frente a $x_\mathrm{kink}$ | §3 Instantón; Bloque A, aplicación |
# | A.3 | La acción durante el entrenamiento y la cota analítica | §2 y §3 (cotas por completar cuadrados); §5′ (Ritz da cotas superiores) |
# | A.4 | Robustez ante la arquitectura | Bloque A, validación |
# | A.5 | La red sobre el tubo reactivo del ruido (Bloque A + Bloque B) | "Cómo se relacionan las partes": el camino de la red se superpone a las trayectorias reactivas |
# | A.6 | Resumen | §6 Discusión |
#
# **Fuera del alcance** (especificación 02): la pérdida PINN (forma fuerte) y los estudios
# 1 a 3 del plan (horizonte finito, modo cero y selección de rama).
#
# **Nota sobre A.5.** CLAUDE.md §3 reserva para `notebooks/03_integracion/` los cuadernos que
# combinan ambos experimentos; la especificación 02 pide expresamente la superposición aquí
# (D36). El cuaderno carga los resultados de E4 directamente con numpy, sin importar
# `taller.estocastico`, y `src/` sigue sin mezclar los bloques.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from taller.analitico import referencias as ref
from taller.neuronal.guardado import cargar_resultado

# Cada cuaderno se empareja con un .py en formato percent; solo el .py se versiona.
formats = "ipynb,py:percent"

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
RESULTADOS = RAIZ / "results" / "neuronal"
FIGURAS = RAIZ / "figures" / "neuronal"
ARQUITECTURAS = ["defecto", "tres_capas_32", "cuatro_capas_64"]
PROBLEMAS = {
    # problema: (cota analítica, nivel de alineación, referencia cerrada, intervalo del RMS)
    "escape": (1.0, -1 / np.sqrt(2), ref.x_om, (-1.0, 0.5)),
    "instanton": (ref.S0, 0.0, ref.x_kink, (-2.0, 2.0)),
}
R = {(p, a): cargar_resultado(RESULTADOS / f"{p}_{a}.npz") for p in PROBLEMAS for a in ARQUITECTURAS}

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # paleta categórica fija (D16)
TINTA, TINTA_SUAVE, REJILLA = "#1f1f1e", "#5f5e58", "#e4e3dc"
AZULES = {100: "#cde2fb", 250: "#86b6ef", 600: "#184f95"}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": TINTA_SUAVE, "axes.labelcolor": TINTA, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE,
    "lines.linewidth": 1.8, "legend.frameon": False,
})
FIGURAS.mkdir(parents=True, exist_ok=True)


def guardar(fig, nombre: str) -> None:
    for extension in ("pdf", "png"):
        # Sin fecha de creación: el mismo dato produce siempre el mismo archivo.
        fig.savefig(FIGURAS / f"{nombre}.{extension}", bbox_inches="tight", metadata={"CreationDate": None})
    print("guardada:", FIGURAS.relative_to(RAIZ) / nombre, "(.pdf, .png)")


def cruce(t: np.ndarray, x: np.ndarray, nivel: float) -> float:
    """Instante en que el camino cruza `nivel`, por interpolación lineal (convención del hito 00 y de E7)."""
    k = int(np.flatnonzero(x >= nivel)[0])
    return t[k - 1] + (nivel - x[k - 1]) * (t[k] - t[k - 1]) / (x[k] - x[k - 1])


def rms_alineado(problema: str, arquitectura: str) -> tuple[float, float]:
    """RMS entre el camino alineado de la red y la referencia cerrada, y el instante de cruce."""
    datos, _ = R[problema, arquitectura]
    _, nivel, referencia, (a, b) = PROBLEMAS[problema]
    t_c = cruce(datos["t_doble"], datos["x_doble"], nivel)
    s = np.linspace(a, b, round((b - a) / 0.01) + 1)
    return float(np.sqrt(np.mean((np.interp(t_c + s, datos["t_doble"], datos["x_doble"]) - referencia(s)) ** 2))), t_c


meta = R["escape", "defecto"][1]
print(f"commit {meta['commit'][:7]} ({'árbol modificado' if meta['arbol_modificado'] else 'árbol limpio'}), "
      f"{meta['dispositivo']} ({meta['nombre_dispositivo']}), {meta['precision']}, {meta['hilos']} hilos, "
      f"PyTorch {meta['versiones']['torch']}, {meta['fecha']}")

# %% [markdown]
# ---
# ## A.0 El método directo: ansatz, red y funcionales
#
# **Objetivo.** Minimizar cada funcional sobre una familia de caminos dada por una red, sin
# resolver la ecuación de Euler-Lagrange.
#
# **Resultado esperado.** El método de Ritz restringe el mínimo a una familia de prueba. Si
# la familia es lo bastante rica, el mínimo restringido se acerca al verdadero **desde
# arriba**: la acción de cualquier camino admisible es una cota superior del mínimo. Las
# cotas por completar cuadrados del hito 00 dan además cotas inferiores exactas:
# $S\cdot D\ge\Delta V = 1$ para caminos de $-1$ a $0$, y $S_E\ge S_0 = 4\sqrt2/3$ para caminos de
# $-1$ a $+1$ (Bogomolny).
#
# **Cálculo.**
# - Funcionales: $S\cdot D = \frac14\int_{-T_h}^{T_h}(\dot x + V')^2\,dt$, con $x(-T_h) = -1$ y
#   $x(T_h) = 0$ (Freidlin-Wentzell; $D$ solo multiplica la acción); y
#   $S_E = \int_{-T_h}^{T_h}[\tfrac12\dot x^2 + V]\,d\tau$, con $x(-T_h) = -1$ y $x(T_h) = +1$.
#   Horizonte $T_h = 3$ y $T_h = 4$, respectivamente (se llama $T_h$, no $T$, porque en el
#   taller $T$ designa el tiempo medio de escape y la temperatura; D40).
# - Ansatz con fronteras exactas:
#   $x(t) = x_a + (x_b - x_a)\frac{t+T_h}{2T_h} + \frac{(t+T_h)(T_h-t)}{T_h^2}\,N(t/T_h)$. Sin
#   penalizaciones: con la acción como pérdida, una penalización dejaría bajar la acción
#   incumpliendo la frontera.
# - $N$: perceptrón con tanh (por defecto, 2 capas de 32), en float64. Pesos Xavier; sesgos
#   ocultos no nulos, porque una red tanh sin sesgos sería exactamente impar y en el
#   instantón impondría la simetría que la red debe encontrar (D35). Última capa casi nula:
#   el camino inicial es la recta.
# - Malla fija de 2001 puntos, completa en cada iteración (pérdida determinista); $\dot x$ por
#   diferenciación automática; trapecio. Adam (3000 iteraciones, tasa $10^{-3}$) y después
#   L-BFGS con búsqueda de línea de Wolfe fuerte.

# %%
for p in PROBLEMAS:
    _, m = R[p, "defecto"]
    q = m["parametros"]
    print(f"{p}: T_h = {q['horizonte']}, x(−T_h) = {q['x_a']}, x(T_h) = {q['x_b']}, capas {q['capas']}, {q['n_parametros']} parámetros, "
          f"malla {q['malla']}, Adam {q['adam']['iteraciones']} × tasa {q['adam']['tasa']}, "
          f"L-BFGS {q['evaluaciones_lbfgs']} evaluaciones, {q['tiempo_s']} s, semilla {m['semilla']}")

# %% [markdown]
# ---
# ## A.1 Escape térmico: la red frente a $x_\mathrm{om}$ (§2)
#
# **Objetivo.** Comprobar que la red encuentra, solo a partir del funcional, el camino más
# probable de escape y su acción.
#
# **Resultado esperado.** $S\cdot D\to\Delta V = 1$ y, alineado en su cruce por $-1/\sqrt2$, el
# camino coincide con $x_\mathrm{om}(t) = -1/\sqrt{1+e^{8t}}$. Criterios:
# $|S\cdot D - 1| < 0.01$; RMS frente a $x_\mathrm{om}$ en $t\in[-1, 0.5]$ $< 0.01$;
# $S\cdot D\ge 1 - 10^{-4}$.
#
# ## A.2 Instantón: la red frente a $x_\mathrm{kink}$ (§3)
#
# **Resultado esperado.** $S_E\to S_0 = 4\sqrt2/3\approx1.8856181$ y, alineado en su cruce por
# $0$, el camino coincide con $x_\mathrm{kink}(\tau) = \tanh(\sqrt2\,\tau)$. Criterios:
# $|S_E - S_0|/S_0 < 0.01$; RMS en $\tau\in[-2, 2]$ $< 0.01$; $S_E\ge S_0 - 10^{-4}$.
#
# **Verificación de la malla.** La acción del camino final se reevalúa en una malla con el
# doble de puntos (4001, que contiene a la de entrenamiento); la diferencia relativa debe ser
# $< 10^{-4}$.

# %%
for p, (cota, nivel, _, (a, b)) in PROBLEMAS.items():
    datos, _ = R[p, "defecto"]
    S, S2 = float(datos["accion"]), float(datos["accion_doble"])
    rms, t_c = rms_alineado(p, "defecto")
    nombre = "S·D" if p == "escape" else "S_E"
    print(f"{p}: {nombre} = {S:.10f} (referencia {cota:.10f}; error relativo {S / cota - 1:+.2e}, criterio < 1e-2)")
    print(f"    cota: {nombre} − referencia = {S - cota:+.2e} ≥ −1e-4 | malla doble: {S2:.10f}, diferencia relativa {abs(S2 / S - 1):.1e} (< 1e-4)")
    print(f"    cruce por x = {nivel:+.4f} en t = {t_c:+.4f}; RMS alineado en [{a}, {b}] = {rms:.2e} (criterio < 1e-2)")

# %%
fig, ejes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
for col, (p, titulo, eje_t) in enumerate((("escape", "escape térmico (S·D)", "t"), ("instanton", "instantón (S_E)", r"\tau"))):
    datos, _ = R[p, "defecto"]
    _, nivel, referencia, (a, b) = PROBLEMAS[p]
    t_c = cruce(datos["t_doble"], datos["x_doble"], nivel)
    s = datos["t_doble"] - t_c
    nombre_ref = r"$x_\mathrm{om}(t)$" if p == "escape" else r"$x_\mathrm{kink}(\tau)$"
    ax = ejes[0, col]
    ax.plot(s, referencia(s), color=TINTA, lw=3.2, alpha=0.25, label=nombre_ref + " (cerrada)")
    ax.plot(s, datos["x_doble"], color=AZUL, lw=1.4, label="red (alineada)")
    ax.axvspan(a, b, color=REJILLA, alpha=0.5, lw=0, zorder=0)
    ax.set_title(f"A.{col + 1} {titulo}", loc="left")
    ax.set_ylabel("$x$")
    ax.legend(loc="lower right", fontsize=9)
    ax = ejes[1, col]
    ax.plot(s, datos["x_doble"] - referencia(s), color=AZUL, lw=1.2)
    ax.axhline(0, color=TINTA_SUAVE, lw=0.8)
    ax.axvspan(a, b, color=REJILLA, alpha=0.5, lw=0, zorder=0)
    ax.set_ylabel("red − cerrada")
    ax.set_xlabel(rf"${eje_t}$ (origen: cruce por ${'-1/\\sqrt{2}' if p == 'escape' else '0'}$)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
fig.tight_layout()
guardar(fig, "caminos_red")
plt.show()

# %% [markdown]
# **Verificación.** Los seis criterios de la arquitectura por defecto se cumplen con varios
# órdenes de margen: el error de la acción es $5.2\times10^{-7}$ (escape) y $2.2\times10^{-8}$
# (instantón), frente a $10^{-2}$; el RMS alineado es $4.8\times10^{-5}$ y $1.4\times10^{-5}$,
# frente a $10^{-2}$; las cotas se cumplen (la acción queda por encima de la referencia); y al
# duplicar la malla la acción cambia $2.6\times10^{-11}$ y $2.4\times10^{-13}$ en relativo,
# frente a $10^{-4}$.
#
# La franja sombreada es el intervalo del RMS. La diferencia entre la red y la solución
# cerrada (paneles inferiores) es una oscilación repartida por todo el dominio, de amplitud
# $\sim10^{-4}$ (escape) y $\sim5\times10^{-5}$ (instantón). Es el patrón típico del error de
# aproximación y de optimización de la red, no un efecto localizado en los bordes.
#
# **Interpretación física.** Sin ecuación de movimiento y sin datos, la minimización directa
# de la acción reproduce el camino de escape de §2 y el kink de §3: la forma débil contiene
# toda la información del problema. Con fronteras exactas en $\pm T_h$, la red resuelve en
# rigor el problema de "llegar a tiempo", cuya solución difiere de la de horizonte infinito
# en términos exponencialmente pequeños en $T_h$. En el escape se ven en el borde izquierdo:
# como el cruce quedó en $t=-2.00$, tras alinear el dominio empieza en $t=-1$, donde la red
# vale exactamente $-1$ y $x_\mathrm{om}(-1) = -1 + 1.7\times10^{-4}$. Esa es la diferencia
# de $-1.7\times10^{-4}$ del extremo izquierdo del panel. Esos mismos términos de borde son lo
# único que fija la posición del cruce: es el modo cero de traslación temporal (§5′), y por
# eso se alinea antes de comparar.
# En el escape, el cruce quedó en $t=-2.00$, cerca del extremo izquierdo; la ventana alineada
# $[-1, 0.5]$ empieza apenas $7\times10^{-4}$ dentro del dominio.

# %% [markdown]
# ---
# ## A.3 La acción durante el entrenamiento y la cota analítica
#
# **Objetivo.** Ver cómo el método directo baja la acción hacia la cota analítica sin
# cruzarla.
#
# **Resultado esperado.** La acción parte de la recta (exactamente $1991/840\approx2.370$ en el
# escape y $1/4 + 64/15\approx4.517$ en el instantón, en el límite continuo), decrece y se
# detiene sobre la cota: cada evaluación es la acción de un camino admisible, de modo que en
# el continuo nunca puede quedar por debajo de $1$ ni de $S_0$.

# %%
fig, ejes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col")
for col, p in enumerate(PROBLEMAS):
    cota = PROBLEMAS[p][0]
    datos, _ = R[p, "defecto"]
    h, fase = datos["historia"], datos["fase"]
    n = np.arange(1, h.size + 1)
    fin_adam = int((fase == 0).sum())
    nombre = r"$S\cdot D$" if p == "escape" else r"$S_E$"
    cota_txt = r"cota $\Delta V = 1$" if p == "escape" else r"cota $S_0 = 4\sqrt{2}/3$"
    ax = ejes[0, col]
    ax.plot(n, h, color=AZUL, lw=1.4)
    ax.axhline(cota, color=NARANJA, ls="--", lw=1.4, label=cota_txt)
    ax.axvline(fin_adam, color=TINTA_SUAVE, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylabel(nombre)
    ax.set_title(f"{'escape térmico' if p == 'escape' else 'instantón'}", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax = ejes[1, col]
    ax.plot(n, h - cota, color=AZUL, lw=1.4)
    ax.axvline(fin_adam, color=TINTA_SUAVE, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(nombre + " − cota")
    ax.set_xlabel("evaluación de la acción (Adam, luego L-BFGS)")
    ax.text(fin_adam * 1.08, ax.get_ylim()[1] * 0.3, "L-BFGS →", color=TINTA_SUAVE, fontsize=8)
    ax.text(fin_adam * 0.92, ax.get_ylim()[1] * 0.3, "← Adam", color=TINTA_SUAVE, fontsize=8, ha="right")
fig.suptitle("A.3 La acción durante el entrenamiento (arquitectura por defecto)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
guardar(fig, "accion_entrenamiento")
plt.show()
for p in PROBLEMAS:
    cota = PROBLEMAS[p][0]
    h, fase = R[p, "defecto"][0]["historia"], R[p, "defecto"][0]["fase"]
    print(f"{p}: inicial {h[0]:.6f}; fin de Adam {h[fase == 0][-1]:.8f} (exceso {h[fase == 0][-1] - cota:.1e}); "
          f"final {h[-1]:.10f} (exceso {h[-1] - cota:.1e}); mínimo − cota sobre toda la historia: {h.min() - cota:.1e}")

# %% [markdown]
# **Verificación.** Ninguna de las evaluaciones queda por debajo de la cota: el mínimo de la
# historia está $5.2\times10^{-7}$ (escape) y $4.2\times10^{-8}$ (instantón) por encima.
# Adam baja el exceso de $\sim1$ a $\sim5\times10^{-5}$ (escape) y $\sim6\times10^{-5}$
# (instantón); L-BFGS lo baja dos o tres órdenes más.
#
# **Interpretación física.** La curva es el método de Ritz en acción: una sucesión de caminos
# admisibles cuya acción decrece monótonamente hacia el mínimo, siempre por encima de la
# cota que da el completar cuadrados (§2 y §3). Que la red se detenga sobre la cota, y no
# debajo, es una verificación conjunta de tres cosas: de la discretización (una cuadratura
# sesgada podría bajar de la cota), del ansatz (las fronteras exactas impiden "hacer
# trampa" sin llegar a la frontera) y de la propia cota. L-BFGS terminó por el límite de
# evaluaciones en 5 de las 6 redes, no por tolerancia: el exceso restante ($10^{-7}$–$10^{-8}$)
# está muy por debajo de la tolerancia.

# %% [markdown]
# ---
# ## A.4 Robustez ante la arquitectura
#
# **Objetivo.** Comprobar que el resultado no depende de la red elegida.
#
# **Resultado esperado.** Con 3 capas de 32 neuronas y con 4 capas de 64, la acción y el
# camino coinciden con los de la arquitectura por defecto dentro de la tolerancia.

# %%
print(f"{'problema':10} {'arquitectura':16} {'parám.':>6} {'acción':>14} {'error rel.':>10} {'RMS':>9} {'cruce':>7} {'malla doble':>11} {'tiempo s':>8}")
for p, (cota, *_rest) in PROBLEMAS.items():
    for a in ARQUITECTURAS:
        datos, m = R[p, a]
        S = float(datos["accion"])
        rms, t_c = rms_alineado(p, a)
        print(f"{p:10} {a:16} {m['parametros']['n_parametros']:>6} {S:>14.10f} {S / cota - 1:>+10.2e} {rms:>9.2e} {t_c:>+7.3f} "
              f"{abs(float(datos['accion_doble']) / S - 1):>11.1e} {m['parametros']['tiempo_s']:>8.1f}")

# %% [markdown]
# | Problema | Arquitectura | Parámetros | Acción | Error relativo | RMS | Tiempo (s) |
# |---|---|---|---|---|---|---|
# | escape | 2 × 32 (defecto) | 1153 | 1.0000005167 | $+5.2\times10^{-7}$ | $4.8\times10^{-5}$ | 23.6 |
# | escape | 3 × 32 | 2209 | 1.0000001996 | $+2.0\times10^{-7}$ | $3.3\times10^{-5}$ | 35.7 |
# | escape | 4 × 64 | 12673 | 1.0000006030 | $+6.0\times10^{-7}$ | $1.4\times10^{-5}$ | 71.2 |
# | instantón | 2 × 32 (defecto) | 1153 | 1.8856181255 | $+2.2\times10^{-8}$ | $1.4\times10^{-5}$ | 20.2 |
# | instantón | 3 × 32 | 2209 | 1.8856181108 | $+1.5\times10^{-8}$ | $9.3\times10^{-6}$ | 24.1 |
# | instantón | 4 × 64 | 12673 | 1.8856181529 | $+3.7\times10^{-8}$ | $2.1\times10^{-5}$ | 79.3 |
#
# **Verificación.** Las tres arquitecturas coinciden dentro de la tolerancia, con margen: la
# acción difiere entre ellas en menos de $5\times10^{-7}$ (escape) y $3\times10^{-8}$
# (instantón), y el RMS es siempre $\le5\times10^{-5}$.
#
# **Interpretación: qué limita la precisión.** Al nivel de la tolerancia, ni el horizonte ni
# la red: los errores son de $10^{-7}$–$10^{-8}$ frente a $10^{-2}$. Por debajo de eso, el
# límite observado es la **optimización**, no la capacidad de la red: multiplicar por 11 el
# número de parámetros no reduce el error, que no es monótono con el tamaño (la red más
# grande da la acción más alta en ambos problemas), y L-BFGS se detuvo por el límite de
# evaluaciones. La dirección más difícil de optimizar es el modo casi plano de traslación:
# el cruce cae en un lugar distinto con cada arquitectura ($-2.00$, $-1.83$ y $-1.85$ en el
# escape; $+0.17$, $-0.23$ y $-0.03$ en el instantón), porque el funcional solo lo fija a través
# de términos exponencialmente pequeños en $T_h$. Cuánto aporta el horizonte finito a ese
# exceso residual no se puede separar sin variar $T_h$ (estudio 1 del plan, fuera del alcance).

# %% [markdown]
# ---
# ## A.5 La red sobre el tubo reactivo del ruido (Bloque A + Bloque B)
#
# **Objetivo.** Superponer el camino que la red encuentra minimizando el funcional (las
# matemáticas) a las trayectorias reactivas del proceso real (la física, E7 del hito 01).
#
# **Resultado esperado.** Con la misma alineación (cruce por $-1/\sqrt2$), el camino de la red
# cae dentro del tubo reactivo y sobre su mediana en la subida, y el tubo se estrecha
# alrededor de él al bajar $D$.
#
# **Cálculo.** Ventanas de E4 (`results/estocastico/e4_D*.npz`, hito 01) cargadas con numpy
# y alineadas como en E7. Camino de la red: escape térmico, arquitectura por defecto.

# %%
RESULTADOS_E4 = RAIZ / "results" / "estocastico"
malla = np.round(np.arange(-1.5, 0.75 + 1e-9, 0.01), 2)
red_esc, _ = R["escape", "defecto"]
t_c_red = cruce(red_esc["t_doble"], red_esc["x_doble"], -1 / np.sqrt(2))
x_red = np.interp(malla + t_c_red, red_esc["t_doble"], red_esc["x_doble"])
tubo = {}
for D in (0.25, 0.15, 0.1):
    with np.load(RESULTADOS_E4 / f"e4_D{D:.3f}.npz", allow_pickle=False) as f:
        ventanas, t_cima, t_al = f["ventanas"].astype(float), f["t_cima"], f["t_alineacion"]
        meta_e4 = json.loads(str(f["metadatos"]))
    k = np.arange(ventanas.shape[1])
    alineadas = np.array([np.interp(malla, t_cima[i] - 3.0 + 0.01 * k - t_al[i], ventanas[i], left=np.nan, right=np.nan)
                          for i in range(ventanas.shape[0])])
    p10, mediana, p90 = np.nanpercentile(alineadas, [10, 50, 90], axis=0)
    tubo[D] = (mediana, p10, p90)
    subida = (malla >= -0.5) & (malla <= 0.25)
    dentro = np.mean((x_red[subida] >= p10[subida]) & (x_red[subida] <= p90[subida]))
    print(f"D = {D}: {ventanas.shape[0]} ventanas (commit {meta_e4['commit'][:7]}); RMS(mediana − red) en [−0.5, 0.25] = "
          f"{np.sqrt(np.mean((mediana[subida] - x_red[subida]) ** 2)):.4f}; fracción del camino de la red dentro de la banda: {dentro:.2f}")
print(f"RMS(red − x_om) en la misma malla [−0.5, 0.25]: {np.sqrt(np.mean((x_red[subida] - ref.x_om(malla[subida])) ** 2)):.1e}")

# %%
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, D in zip(ejes, (0.25, 0.15, 0.1)):
    mediana, p10, p90 = tubo[D]
    ax.fill_between(malla, p10, p90, color=AZULES[100], lw=0, label="ruido: banda 10–90 %")
    ax.plot(malla, mediana, color=AZULES[600], lw=1.6, label="ruido: mediana")
    ax.plot(malla, x_red, color=NARANJA, lw=1.6, ls="--", label="red variacional")
    ax.axvspan(-0.5, 0.25, color=REJILLA, alpha=0.45, lw=0, zorder=0)
    ax.set_title(rf"$D = {D}$", loc="left")
    ax.set_xlabel(r"$t$ (origen: cruce por $-1/\sqrt{2}$)")
ejes[0].set_ylabel("$x$")
ejes[0].set_ylim(-1.38, 0.98)
ejes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("A.5 El camino de la red sobre el tubo reactivo de la simulación (E7)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
guardar(fig, "red_sobre_tubo_reactivo")
plt.show()

# %% [markdown]
# **Verificación.** La distancia cuadrática media entre la mediana del ruido y la red en
# $t\in[-0.5, 0.25]$ decrece al bajar $D$: 0.120, 0.083 y 0.058. Son los mismos valores del
# criterio E7 frente a $x_\mathrm{om}$, porque la red y $x_\mathrm{om}$ difieren en
# $4\times10^{-5}$ en esa malla. El camino de la red está dentro de la banda 10–90 % en el 80 %,
# el 89 % y el 93 % de ese intervalo. Antes del origen está siempre dentro. Sale de la banda,
# por debajo del percentil 10, solo justo después del origen: en $t\in(0, 0.15]$, $(0, 0.08]$ y
# $(0, 0.05]$ para $D = 0.25$, 0.15 y 0.1. Es un efecto de la alineación, no una discrepancia
# con la teoría: después del *último* cruce por $-1/\sqrt2$, las trayectorias, por definición,
# ya no vuelven a bajar de ese nivel. El conjunto queda condicionado a subir, y la banda está
# pellizcada cerca de $t=0$ (hito 01, D33). El tramo afectado se acorta al bajar $D$.
#
# **Interpretación física.** Dos verificaciones independientes, que no comparten código,
# llegan al mismo camino: la red lo encuentra minimizando la acción, sin simular nada, y el
# proceso real lo recorre cuando escapa. Es la imagen central del taller (§2 del plan): el
# escape térmico ocurre a lo largo del minimizador del funcional de Onsager-Machlup, y el
# ruido lo dispersa en un tubo de ancho $\propto\sqrt D$. Cerca de la cima, donde la deriva se
# anula, el ruido se adelanta al minimizador (hito 01, B.7): la teoría de ruido débil describe
# la subida, no la llegada.

# %% [markdown]
# ---
# ## A.6 Resumen
#
# | Criterio (arquitectura por defecto) | Escape térmico | Instantón | Tolerancia |
# |---|---|---|---|
# | Error relativo de la acción | $5.2\times10^{-7}$ | $2.2\times10^{-8}$ | $10^{-2}$ |
# | RMS alineado frente a la solución cerrada | $4.8\times10^{-5}$ | $1.4\times10^{-5}$ | $10^{-2}$ |
# | Acción − cota analítica | $+5.2\times10^{-7}$ | $+4.2\times10^{-8}$ | $\ge-10^{-4}$ |
# | Malla doble (diferencia relativa) | $2.6\times10^{-11}$ | $2.4\times10^{-13}$ | $<10^{-4}$ |
#
# La red variacional, entrenada solo con la acción, reproduce las soluciones cerradas de §2
# y §3 y sus acciones $\Delta V/D$ y $S_0$, sin cruzar las cotas analíticas, con cualquiera de
# las tres arquitecturas. Superpuesta al tubo reactivo de la simulación, cierra el círculo
# del taller: el camino que minimiza el funcional es el que recorre el proceso real.
