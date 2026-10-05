# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hito 01 — Bloque B: simulación estocástica del escape térmico
#
# El Bloque B del plan del taller es el **experimento físico**: pone a prueba contra el
# proceso real la predicción de la teoría variacional de ruido débil. Esa predicción dice
# que las transiciones siguen el camino de Onsager-Machlup y que su tiempo escala como
# $e^{\Delta V/D}$. Las redes neuronales (Bloque A) verifican las matemáticas; este bloque
# verifica la física (especificación `specs/01_estocastico.md` y sus aclaraciones).
#
# Cada sección sigue la estructura de siempre: **objetivo**, **resultado esperado**,
# **cálculo**, **verificación** e **interpretación física**.
#
# **Este cuaderno no simula.** Carga los resultados guardados en `results/estocastico/`
# por `python -m taller.estocastico.correr` y genera las figuras en `figures/estocastico/`.
# La lógica vive en `src/taller/estocastico/`; los criterios de aceptación se verifican en
# `tests/estocastico/test_criterios_produccion.py`. Aquí se muestran los números y se
# interpretan.
#
# **Sistema y notación** (CLAUDE.md §2). Dinámica de Langevin sobreamortiguada
# $dx = -V'(x)\,dt + \sqrt{2D}\,dW$ con $V(x) = (x^2-1)^2$, $\Delta V = 1$ y condición inicial
# $x_0 = -1$. Esquema de Euler-Maruyama (el mismo esquema discreto del que sale el
# funcional en §1 del plan):
# $$x_{n+1} = x_n - V'(x_n)\,\Delta t + \sqrt{2D\,\Delta t}\;\eta_n,\qquad \eta_n\sim\mathcal N(0,1).$$
# El tiempo medio de escape se escribe $T$ (o $\tau_\mathrm{esc}$); $\tau$ queda reservado al
# tiempo imaginario del instantón.
#
# **Correspondencia con el plan del taller y con la especificación.**
#
# | Cuaderno | Contenido | Plan del taller | Especificación |
# |---|---|---|---|
# | B.0 | Oráculo exacto: tiempo medio de primer paso | §0 Problema físico (respuesta esperada) | E0 |
# | B.1 | Validación del integrador: Numba frente a NumPy | Bloque B (esquema discreto, §1) | E1 |
# | B.2 | Control interno: equilibrio de Boltzmann | Bloque B, control interno; §0 | E2 |
# | B.3 | Frontera discreta: convergencia en $\Delta t$ y puente browniano | Bloque B ("definir con cuidado escape") | E3 |
# | B.4 | Medición 2: tiempos de escape frente al oráculo | Bloque B, medición 2 | E4 |
# | B.5 | Medición 2: ley de Arrhenius | Bloque B, medición 2; §0 y §6 (prefactores) | E5 |
# | B.6 | El escape como evento de Poisson | Bloque B, medición 2; §6 | E6 |
# | B.7 | Medición 1: el tubo reactivo frente a $x_\mathrm{om}$ | Bloque B, medición 1; §2 Camino más probable | E7 |
# | B.8 | Resumen | §6 Discusión | — |
#
# **Diferencias con el plan.** El plan proponía unas mil trayectorias y 3 o 4 valores de
# $D$ en $[0.15, 0.35]$, y advertía que $D = 0.1$ era inviable (unas 24 000 unidades de tiempo
# por evento). Con Numba en CPU (D19, D28) el costo baja lo suficiente para usar $10^5$
# trayectorias por $D$ y siete valores de $D$ entre $0.1$ y $0.5$ ($2\cdot10^4$ en
# $D = 0.1$). La medición 1 se compara con el camino cerrado $x_\mathrm{om}$; la superposición
# con el camino de la red (Bloque A) queda para `notebooks/03_integracion/`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from scipy.integrate import quad
from scipy.stats import ks_2samp

from taller.analitico import referencias as ref
from taller.estocastico.guardado import cargar_resultado
from taller.estocastico.observables import (
    alinear_ventanas,
    coeficiente_variacion,
    distancia_l1,
    media_y_error,
    mediana_y_banda,
    pendiente_arrhenius,
    probabilidades_boltzmann,
    rms_contra_om,
)

# Cada cuaderno se empareja con un .py en formato percent; solo el .py se versiona.
formats = "ipynb,py:percent"

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
RESULTADOS = RAIZ / "results" / "estocastico"
FIGURAS = RAIZ / "figures" / "estocastico"
VALORES_D = [0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5]
VALORES_DT = [1e-2, 5e-3, 1e-3, 5e-4]
DESTINOS = {"t_cima": 0.0, "t_pozo": 1.0}  # b = 0: llegar a la cima; b = +1: caer al otro pozo

# Constante de la corrección de continuidad de Broadie, Glasserman y Kou: −ζ(1/2)/√(2π).
BETA = 0.5826


def cargar(nombre: str) -> tuple[dict, dict]:
    return cargar_resultado(RESULTADOS / nombre)


def error_relativo(t: np.ndarray, exacto: float) -> tuple[float, float]:
    """Error relativo de la media frente al valor exacto, y su error estándar (en %)."""
    media, ee = media_y_error(t)
    return 100 * (media / exacto - 1), 100 * ee / exacto


def describir(meta: dict) -> str:
    estado = "árbol modificado" if meta["arbol_modificado"] else "árbol limpio"
    return f"commit {meta['commit'][:7]} ({estado}), {meta['hilos']} hilos, {meta['fecha']}"


AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # paleta categórica fija (D16)
TINTA, TINTA_SUAVE, REJILLA = "#1f1f1e", "#5f5e58", "#e4e3dc"
AZULES = {100: "#cde2fb", 250: "#86b6ef", 300: "#6da7ec", 450: "#2a78d6", 600: "#184f95", 700: "#0d366b"}

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
        fig.savefig(FIGURAS / f"{nombre}.{extension}", bbox_inches="tight",
                    metadata={"CreationDate": None})
    print("guardada:", FIGURAS.relative_to(RAIZ) / nombre, "(.pdf, .png)")


E4 = {D: cargar(f"e4_D{D:.3f}.npz") for D in VALORES_D}
E3 = {dt: cargar(f"e3_dt{dt:.0e}.npz") for dt in VALORES_DT}
T_EXACTO = {(D, b): ref.tiempo_primer_paso(D, b) for D in VALORES_D for b in (0.0, 1.0)}
for nombre, (_, meta) in [("E3 dt=1e-2", E3[1e-2]), ("E4 D=0.1", E4[0.1])]:
    print(f"{nombre}: {describir(meta)}")

# %% [markdown]
# Los archivos de E4 se generaron con el árbol marcado como modificado. El único cambio era
# el registro de consola `log_E4.log`, que no está versionado; tres de los siete archivos se
# regeneraron con el código del commit y son idénticos bit a bit (D32 y bitácora).

# %% [markdown]
# ---
# ## B.0 Oráculo exacto: el tiempo medio de primer paso
#
# **Objetivo.** Tener una vara exacta, sin aproximaciones asintóticas, contra la que medir
# la simulación (decisión 2 de la especificación).
#
# **Resultado esperado.** En una dimensión, el tiempo medio de primer paso de $x_0$ a una
# frontera absorbente $b$ (con $-\infty$ inalcanzable) es
# $$T(x_0\to b) = \frac1D\int_{x_0}^{b} dy\; e^{V(y)/D}\int_{-\infty}^{y} dz\; e^{-V(z)/D}.$$
# Kramers es su asintótica para $D\to0$ con $b = +1$:
# $T_K = \frac{2\pi}{\sqrt{V''(-1)\,|V''(0)|}}\,e^{\Delta V/D} = \frac{2\pi}{\sqrt{32}}\,e^{1/D}$.
# Llegar a la cima ($b=0$) cuesta asintóticamente la mitad, porque desde la cima la
# partícula cae a cada lado con probabilidad ½.
#
# **Cálculo y verificación.** `referencias.tiempo_primer_paso` (quad anidado) coincide con
# una evaluación independiente de mpmath a 40 dígitos con error relativo máximo de
# $5.6\times10^{-16}$ (criterio E0: $<10^{-8}$; prueba `test_tiempo_primer_paso.py`).

# %%
print(f"{'D':>6} {'T(−1→0)':>12} {'T(−1→+1)':>12} {'T(0)/T(+1)':>11} {'T(+1)/T_K':>10}")
for D in VALORES_D:
    T0, T1 = T_EXACTO[D, 0.0], T_EXACTO[D, 1.0]
    print(f"{D:>6} {T0:>12.4f} {T1:>12.4f} {T0 / T1:>11.4f} {T1 / ref.tau_kramers(D):>10.4f}")

# %% [markdown]
# **Interpretación física.** Kramers queda por debajo del tiempo exacto:
# $T/T_K = 1.043$ en $D = 0.1$ y $1.250$ en $D = 0.5$. Las correcciones son de orden $D/\Delta V$. El exponente es el mismo;
# lo que falla a $D$ finito es el prefactor. El cociente $T(0)/T(+1)$ tiende a ½ cuando
# $D\to0$ (0.49997 en $D = 0.1$) y se aleja de ½ cuando la barrera deja de ser alta frente a
# $D$. Por eso este bloque mide contra $T$ exacto y no contra Kramers: así la desviación de
# Kramers es un resultado, no un error.

# %% [markdown]
# ---
# ## B.1 Validación del integrador: Numba frente a NumPy (E1)
#
# **Objetivo.** Comprobar que el integrador de producción (Numba en CPU, un prange sobre
# trayectorias y un generador xoshiro256** propio por trayectoria, D28) simula el mismo
# proceso que una implementación de referencia simple e independiente (NumPy vectorizado,
# generador PCG64).
#
# **Resultado esperado.** Con $D = 0.35$, $\Delta t = 10^{-3}$ y $10^4$ trayectorias en cada
# implementación, los tiempos de primer paso a $x = +1$ son estadísticamente
# indistinguibles: prueba de Kolmogorov-Smirnov de dos muestras con $p > 0.01$.
#
# **Cálculo.** `e1_validacion.npz`: réplica independiente de la prueba
# `test_validacion_estocastica.py`, con otras semillas (D32).

# %%
e1, meta_e1 = cargar("e1_validacion.npz")
print(describir(meta_e1))
for clave in ("t_pozo", "t_pozo_sin_corregir", "t_cima", "t_cima_sin_corregir"):
    exacto = T_EXACTO[0.35, 0.0 if "cima" in clave else 1.0]
    p = ks_2samp(e1[f"numba_{clave}"], e1[f"numpy_{clave}"]).pvalue
    en, een = error_relativo(e1[f"numba_{clave}"], exacto)
    ep, eep = error_relativo(e1[f"numpy_{clave}"], exacto)
    criterio = "criterio p > 0.01" if clave.startswith("t_pozo") else "informativo"
    print(f"{clave:20s}: KS p = {p:.3f} ({criterio}) | T/T_exacto − 1: Numba {en:+.2f} ± {een:.2f} %, "
          f"NumPy {ep:+.2f} ± {eep:.2f} %")

# %% [markdown]
# **Verificación.** El criterio se cumple en la prueba (p = 0.165 corregido, 0.145 sin
# corregir) y en esta réplica (p = 0.111 en ambos). En la réplica, los tiempos a la cima,
# que no son criterio, dan p más bajos (0.052 y 0.023). Para descartar una diferencia
# sistemática se compararon muestras mucho mayores en una investigación aparte (bitácora):
# con $10^6$ trayectorias de Numba y $10^5$ de NumPy, los cuatro tiempos dan p entre 0.72
# y 0.95, y ambas implementaciones coinciden con $T$ exacto dentro de un error estándar
# (0.10 % en Numba). Los p bajos de las muestras de $10^4$ son fluctuaciones.

# %%
# Figura E1: funciones de supervivencia de los tiempos al otro pozo.
def supervivencia(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orden = np.sort(t)
    return orden, 1.0 - np.arange(1, orden.size + 1) / orden.size


fig, ax = plt.subplots(figsize=(6.2, 3.6))
for impl, color, estilo in (("numba", AZUL, "-"), ("numpy", NARANJA, "--")):
    s, S = supervivencia(e1[f"{impl}_t_pozo"])
    ax.plot(s, S, color=color, ls=estilo, label={"numba": "Numba (producción)", "numpy": "NumPy (referencia)"}[impl])
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.2)
ax.set_xlim(0, 1.02 * max(e1["numba_t_pozo"].max(), e1["numpy_t_pozo"].max()))
ax.set_xlabel(r"tiempo de primer paso a $x=+1$")
ax.set_ylabel(r"$P(T_\mathrm{esc} > t)$")
ax.set_title(r"E1: Numba frente a NumPy ($D=0.35$, $10^4$ trayectorias c/u)", loc="left")
ax.legend()
guardar(fig, "e1_numba_numpy")
plt.show()

# %% [markdown]
# **Interpretación física.** Las dos curvas se superponen hasta $P\sim10^{-2}$. Más abajo
# quedan unas 100 trayectorias en cada muestra y las colas fluctúan, como corresponde a ese
# número de eventos (la comparación con $10^6$ y $10^5$ trayectorias no muestra diferencias).
# El integrador de producción reproduce el mismo proceso estocástico que
# el esquema de Euler-Maruyama escrito de la forma más directa. A partir de aquí, cualquier
# discrepancia con la teoría es física o de discretización, no un error de programación.

# %% [markdown]
# ---
# ## B.2 Control interno: equilibrio de Boltzmann (E2)
#
# **Objetivo.** Verificar que el integrador produce la estadística de equilibrio correcta.
# Es el control físico de que la deriva $-V'$ y la intensidad del ruido $\sqrt{2D}$ están
# bien (§0 del plan: $p_s\propto e^{-V/D}$ anula la corriente de Fokker-Planck).
#
# **Resultado esperado.** A tiempos largos, la densidad de posiciones es
# $p_s(x) = e^{-V(x)/D}/Z$. Con $D = 0.5$ hay muchas transiciones y se equilibran ambos
# pozos; con $D$ pequeño solo se equilibraría dentro de un pozo. Criterio: distancia $L_1$
# entre probabilidades por intervalo $< 0.02$, con $\Delta t = 10^{-3}$.
#
# **Cálculo.** $10^5$ trayectorias desde $x_0 = -1$, equilibradas hasta $t = 50$ (el modo
# lento decae como $e^{-\lambda_1 t}$ con $\lambda_1\approx 2/T(-1\to+1)\approx0.2$), con 20
# muestras por trayectoria separadas 5.0, del orden del tiempo de correlación (aclaración 13).
# Intervalos de ancho 0.05 en $[-2, 2]$.

# %%
e2, meta_e2 = cargar("e2_boltzmann.npz")
print(describir(meta_e2))
x = e2["x"].astype(float).ravel()
bordes = np.linspace(-2.0, 2.0, 81)
empirica = np.histogram(x, bins=bordes)[0] / x.size
exacta = probabilidades_boltzmann(bordes, 0.5)
l1 = distancia_l1(empirica, exacta)
print(f"muestras: {x.size:,}; fuera de [−2, 2]: {np.mean(np.abs(x) > 2):.1e}")
print(f"L1 = {l1:.4f}   (criterio: < 0.02)")
print(f"fracción en x > 0: {np.mean(x > 0):.4f} ± {np.sqrt(0.25 / x.size):.4f}   (exacta: 0.5)")
rng_ruido = np.random.default_rng(meta_e2["semilla"])  # solo para estimar el ruido de L1
l1_ruido = [distancia_l1(rng_ruido.multinomial(x.size, exacta / exacta.sum()) / x.size, exacta) for _ in range(200)]
print(f"L1 esperado solo por ruido estadístico con muestras independientes: {np.mean(l1_ruido):.4f} ± {np.std(l1_ruido):.4f}")

# %%
fig, ax = plt.subplots(figsize=(6.2, 3.6))
centros, ancho = 0.5 * (bordes[1:] + bordes[:-1]), np.diff(bordes)
ax.bar(centros, empirica / ancho, width=ancho * 0.9, color=AZULES[250], label="simulación (histograma)")
xx = np.linspace(-2, 2, 801)
Z = quad(lambda y: np.exp(-ref.V(y) / 0.5), -np.inf, np.inf)[0]  # normalización en toda la recta
ax.plot(xx, np.exp(-ref.V(xx) / 0.5) / Z, color=TINTA, label=r"$e^{-V/D}/Z$ (Boltzmann)")
ax.set_xlabel("$x$")
ax.set_ylabel("densidad")
ax.set_title(rf"E2: equilibrio con $D = 0.5$ ($L_1 = {l1:.4f}$)", loc="left")
ax.set_ylim(0, 0.92)
ax.legend(loc="upper center", ncols=2, fontsize=9)
guardar(fig, "e2_boltzmann")
plt.show()

# %% [markdown]
# **Interpretación física.** El histograma reproduce la densidad de Boltzmann con
# $L_1 = 0.0044$, del orden del ruido estadístico esperado ($\approx0.004$): no se detecta
# ningún sesgo de discretización en el equilibrio. Ambos pozos están igualmente poblados,
# como exige la simetría de $V$. Que el proceso termine en $e^{-V/D}$ confirma que deriva y
# ruido cumplen la relación de fluctuación-disipación: la misma $D$ que fija el ruido fija la
# temperatura de equilibrio. Este es el control interno que pide el plan.

# %% [markdown]
# ---
# ## B.3 Frontera discreta: convergencia en $\Delta t$ y puente browniano (E3)
#
# **Objetivo.** Elegir el paso de tiempo de producción y entender el error de detectar el
# primer paso solo en instantes discretos.
#
# **Resultado esperado.** Euler-Maruyama observa la trayectoria cada $\Delta t$ y pierde los
# cruces entre pasos. En la cima la deriva se anula y la trayectoria se mueve como un
# browniano libre; perder cruces equivale a desplazar la frontera hacia arriba en
# $\delta = \beta\sqrt{2D\,\Delta t}$, con $\beta = -\zeta(1/2)/\sqrt{2\pi}\approx0.5826$
# (corrección de continuidad para barreras discretas). El sesgo predicho del tiempo a la cima
# es $T(-1\to\delta)/T(-1\to0) - 1$, del orden de $\sqrt{\Delta t}$. En el pozo ($x = +1$) la
# deriva empuja hacia la frontera y el efecto es despreciable.
#
# **Cálculo.** Se corrige con un **puente browniano** (aclaración 7, D24): si
# $x_n < b$ y $x_{n+1} < b$, se considera que la trayectoria cruzó $b$ entre ambos instantes
# con probabilidad $\exp[-(b-x_n)(b-x_{n+1})/(D\,\Delta t)]$. La corrección se validó antes
# de usarla: para un browniano sin deriva, donde es exacta, reproduce
# $P(\tau_a\le t) = \operatorname{erfc}(a/\sqrt{4Dt})$ dentro de 1.04 errores estándar,
# mientras que sin corregir el error es de 36 a 38 errores estándar. Se guardan también los
# tiempos sin corregir. $D = 0.25$, $10^5$ trayectorias por $\Delta t$.

# %%
print(describir(E3[1e-3][1]))
print(f"{'dt':>7} | {'cima corr. %':>14} | {'pozo corr. %':>14} | {'cima sin corr. %':>16} | {'predicho %':>10} | {'pozo sin corr. %':>16}")
tabla_e3 = {}
for dt in VALORES_DT:
    datos = E3[dt][0]
    fila = {c: error_relativo(datos[c], T_EXACTO[0.25, b]) for c, b in
            [("t_cima", 0.0), ("t_pozo", 1.0), ("t_cima_sin_corregir", 0.0), ("t_pozo_sin_corregir", 1.0)]}
    delta = BETA * np.sqrt(2 * 0.25 * dt)
    fila["predicho"] = 100 * (ref.tiempo_primer_paso(0.25, delta) / T_EXACTO[0.25, 0.0] - 1)
    tabla_e3[dt] = fila
    f = lambda c: f"{fila[c][0]:+.2f} ± {fila[c][1]:.2f}"
    print(f"{dt:>7.0e} | {f('t_cima'):>14} | {f('t_pozo'):>14} | {f('t_cima_sin_corregir'):>16} | "
          f"{fila['predicho']:>+10.2f} | {f('t_pozo_sin_corregir'):>16}")

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.8))
dts = np.array(VALORES_DT)
dt_fino = np.geomspace(4e-4, 1.2e-2, 60)
predicho = [100 * (ref.tiempo_primer_paso(0.25, BETA * np.sqrt(0.5 * d)) / T_EXACTO[0.25, 0.0] - 1) for d in dt_fino]
ax.plot(dt_fino, predicho, color=TINTA_SUAVE, ls="--", lw=1.4, label=r"predicho: frontera en $\beta\sqrt{2D\Delta t}$")
for clave, color, marca, etiqueta in (
    ("t_cima_sin_corregir", NARANJA, "o", "cima sin corregir"),
    ("t_cima", AZUL, "o", "cima corregida (puente)"),
    ("t_pozo", AQUA, "s", "pozo corregido (puente)"),
):
    y = np.array([tabla_e3[d][clave][0] for d in VALORES_DT])
    e = np.array([tabla_e3[d][clave][1] for d in VALORES_DT])
    ax.errorbar(dts, y, yerr=e, color=color, marker=marca, ms=6, lw=1.4, capsize=2, label=etiqueta)
ax.axhline(0, color=TINTA_SUAVE, lw=0.8)
ax.axhspan(-2, 2, color=REJILLA, alpha=0.5, lw=0, zorder=0)
ax.text(4.3e-4, -1.85, "tolerancia de E3 (±2 %)", color=TINTA_SUAVE, fontsize=8, va="bottom")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$T_\mathrm{sim}/T_\mathrm{exacto} - 1$  (%)")
ax.set_title(r"E3: error del tiempo de primer paso frente a $\Delta t$ ($D = 0.25$)", loc="left")
ax.legend(loc="upper left", fontsize=9)
guardar(fig, "e3_convergencia_dt")
plt.show()

# %% [markdown]
# **Verificación.** Con la corrección, los cuatro $\Delta t$ cumplen el 2 % en ambos
# destinos, y el error decrece al disminuir $\Delta t$ entre los valores significativos
# (aclaración 8; pruebas de E3 aprobadas). Sin corregir, el sesgo de la cima sigue la
# predicción de continuidad: +11.4 %, +8.7 %, +3.7 % y +2.6 % medidos frente a +12.5 %,
# +8.9 %, +4.0 % y +2.8 % predichos.
#
# **Elección del paso de producción: $\Delta t = 10^{-3}$** (D30, aclaración 15). La regla
# literal (mayor $\Delta t$ con error < 2 %) daba $10^{-2}$. Con la corrección queda un sesgo
# de orden $\Delta t$: −1.29 % ± 0.31 % en la cima con $10^{-2}$, y −0.63 % ± 0.11 % con
# $5\times10^{-3}$ (12 lotes independientes, bitácora). Con $10^{-3}$ el sesgo es de
# alrededor de −0.1 %, muy por debajo de la tolerancia de E4 sea cual sea su dependencia
# con $D$.
#
# **Interpretación física.** Dos errores distintos conviven aquí. El de la frontera escala
# como $\sqrt{\Delta t}$ y es grande: es puramente geométrico, viene de mirar un camino
# continuo de forma estroboscópica, y el puente browniano lo elimina. El residual es de
# orden $\Delta t$ y pequeño: es el error débil del esquema en la dinámica misma. Que el
# sesgo sin corregir siga la predicción $\beta\sqrt{2D\Delta t}$ confirma que, en la cima, la
# partícula se comporta como un browniano libre: es la "meseta" de $V$ donde la deriva se
# anula, y es ahí donde el ruido decide.

# %% [markdown]
# ---
# ## B.4 Medición 2: tiempos de escape frente al oráculo exacto (E4)
#
# **Objetivo.** Medir el tiempo medio de escape en todos los $D$ y para las dos
# definiciones de escape, y compararlo con el valor exacto.
#
# **Resultado esperado.** $|T_\mathrm{sim}/T_\mathrm{exacto} - 1| < 3\%$ para los siete $D$ y
# los dos destinos (criterio E4).
#
# **Cálculo.** $\Delta t = 10^{-3}$; $10^5$ trayectorias por $D$, salvo $D = 0.1$ con
# $2\cdot10^4$ (aclaración 9). Un archivo por $D$.

# %%
print(f"{'D':>6} {'N':>7} | {'cima corr. %':>14} | {'pozo corr. %':>14} | {'cima sin corr. %':>16} | {'predicho %':>10} | {'T(0)/T(+1) sim':>14} | {'exacto':>7}")
tabla_e4 = {}
for D in VALORES_D:
    datos = E4[D][0]
    fila = {c: error_relativo(datos[c], T_EXACTO[D, b]) for c, b in
            [("t_cima", 0.0), ("t_pozo", 1.0), ("t_cima_sin_corregir", 0.0), ("t_pozo_sin_corregir", 1.0)]}
    fila["predicho"] = 100 * (ref.tiempo_primer_paso(D, BETA * np.sqrt(2 * D * 1e-3)) / T_EXACTO[D, 0.0] - 1)
    tabla_e4[D] = fila
    f = lambda c: f"{fila[c][0]:+.2f} ± {fila[c][1]:.2f}"
    cociente = datos["t_cima"].mean() / datos["t_pozo"].mean()
    print(f"{D:>6} {datos['t_pozo'].size:>7} | {f('t_cima'):>14} | {f('t_pozo'):>14} | {f('t_cima_sin_corregir'):>16} | "
          f"{fila['predicho']:>+10.2f} | {cociente:>14.4f} | {T_EXACTO[D, 0.0] / T_EXACTO[D, 1.0]:>7.4f}")
peor = max(abs(tabla_e4[D][c][0]) for D in VALORES_D for c in DESTINOS)
print(f"mayor |error| corregido: {peor:.2f} %   (criterio: < 3 %)")

# %% [markdown]
# **Verificación.** Los 14 casos cumplen el 3 % con margen: el mayor error es de 0.62 %. Todos
# los errores corregidos están dentro de unos 2 errores estándar de cero (el mayor, −2.0, en la
# cima con $D = 0.35$). El cociente
# $T(0)/T(+1)$ simulado reproduce el exacto, incluida su desviación de ½ con $D$ grande.

# %%
# Figura E4: sesgo de la cima sin corregir frente a D.
fig, ax = plt.subplots(figsize=(6.4, 3.6))
Ds = np.array(VALORES_D)
D_fino = np.linspace(0.09, 0.52, 60)
pred_fino = [100 * (ref.tiempo_primer_paso(d, BETA * np.sqrt(2 * d * 1e-3)) / ref.tiempo_primer_paso(d, 0.0) - 1) for d in D_fino]
ax.plot(D_fino, pred_fino, color=TINTA_SUAVE, ls="--", lw=1.4, label=r"predicho: frontera en $\beta\sqrt{2D\Delta t}$")
for clave, color, etiqueta in (("t_cima_sin_corregir", NARANJA, "cima sin corregir"), ("t_cima", AZUL, "cima corregida (puente)")):
    ax.errorbar(Ds, [tabla_e4[D][clave][0] for D in VALORES_D], yerr=[tabla_e4[D][clave][1] for D in VALORES_D],
                color=color, marker="o", ms=6, lw=1.4, capsize=2, label=etiqueta)
ax.axhline(0, color=TINTA_SUAVE, lw=0.8)
ax.set_xlabel("$D$")
ax.set_ylabel(r"$T_\mathrm{sim}/T_\mathrm{exacto} - 1$  (%)")
ax.set_title(r"E4: sesgo de la frontera discreta en la cima ($\Delta t = 10^{-3}$)", loc="left")
ax.legend(loc="center right", fontsize=9)
guardar(fig, "e4_sesgo_cima")
plt.show()
sesgos = np.array([tabla_e4[D]["t_cima_sin_corregir"][0] for D in VALORES_D])
print(f"sesgo sin corregir: media {sesgos.mean():+.2f} %, rango [{sesgos.min():+.2f}, {sesgos.max():+.2f}] %; "
      f"predicho entre {min(tabla_e4[D]['predicho'] for D in VALORES_D):+.2f} y {max(tabla_e4[D]['predicho'] for D in VALORES_D):+.2f} %")

# %% [markdown]
# **Interpretación física.** Sin corregir, el tiempo a la cima tiene un sesgo de
# aproximadamente +4 % para todos los $D$, de 0.1 a 0.5, como predice la corrección de
# continuidad. La razón de que no dependa de $D$: el desplazamiento de la frontera crece como
# $\sqrt D$, pero la región donde $T(b)$ es sensible a $b$ (la meseta de la cima, de ancho
# $\sim\sqrt{D/|V''(0)|}$) también, y el cociente no depende de $D$. Con el puente browniano,
# el sesgo desaparece en todos los $D$. El tiempo al otro pozo no necesita la corrección
# (columna "pozo sin corregir" del archivo: dentro de 2 errores estándar de cero).

# %% [markdown]
# ---
# ## B.5 Medición 2: ley de Arrhenius (E5)
#
# **Objetivo.** Verificar la predicción central de la teoría de ruido débil: el tiempo de
# escape escala como $e^{S_\mathrm{min}} = e^{\Delta V/D}$.
#
# **Resultado esperado.** $\ln T$ es lineal en $1/D$ con pendiente $\to\Delta V = 1$ cuando
# $D\to0$. Criterio E5: en $D\le0.2$, la pendiente del ajuste lineal simulado difiere de la
# del exacto en menos de 0.05, para ambos destinos (aclaración 10). Sin criterio, se reporta
# cuánto se aleja la pendiente exacta de 1 y por qué.

# %%
D_arr = np.array([d for d in VALORES_D if d <= 0.2])
pendientes = {}
for clave, b in DESTINOS.items():
    sim = pendiente_arrhenius(D_arr, np.array([E4[d][0][clave].mean() for d in D_arr]))
    exa = pendiente_arrhenius(D_arr, np.array([T_EXACTO[d, b] for d in D_arr]))
    pendientes[clave] = (sim, exa)
    print(f"{clave}: pendiente simulada {sim[0]:.4f}, exacta {exa[0]:.4f}, |Δ| = {abs(sim[0] - exa[0]):.4f} (criterio < 0.05); "
          f"exacta − ΔV = {exa[0] - 1:+.4f}")
kr = pendiente_arrhenius(D_arr, ref.tau_kramers(D_arr))
print(f"Kramers: pendiente {kr[0]:.4f}, ordenada {kr[1]:+.4f} = ln(2π/√32) = {np.log(2 * np.pi / np.sqrt(32)):+.4f}")
# Pendiente local exacta: d ln T / d(1/D) por diferencias finitas centradas
for D in (0.1, 0.15, 0.2, 0.35):
    h = 1e-3
    loc = (np.log(ref.tiempo_primer_paso(1 / (1 / D + h), 1.0)) - np.log(ref.tiempo_primer_paso(1 / (1 / D - h), 1.0))) / (2 * h)
    print(f"  pendiente local exacta (pozo) en D = {D}: {loc:.4f}")

# %%
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
inv = 1 / Ds
inv_fino = np.linspace(1.9, 10.3, 100)
for clave, b, color, marca, nombre in (("t_pozo", 1.0, AQUA, "s", "otro pozo"), ("t_cima", 0.0, AZUL, "o", "cima")):
    ax.plot(inv_fino, [np.log(ref.tiempo_primer_paso(1 / u, b)) for u in inv_fino], color=TINTA, lw=1.0)
    ax.plot(inv, [np.log(E4[D][0][clave].mean()) for D in VALORES_D], marca, color=color, ms=7,
            label=f"simulado, {nombre} (pend. {pendientes[clave][0][0]:.3f})")
ax.plot(inv_fino, np.log(ref.tau_kramers(1 / inv_fino)), color=TINTA_SUAVE, ls="--", lw=1.4, label="Kramers (pend. 1)")
ax.plot([], [], color=TINTA, lw=1.0, label="exacto")
ax.set_xlabel("$1/D$")
ax.set_ylabel(r"$\ln T$")
ax.set_title("E4–E5: ley de Arrhenius", loc="left")
ax.legend(fontsize=8.5)
for clave, b, color, marca in (("t_pozo", 1.0, AQUA, "s"), ("t_cima", 0.0, AZUL, "o")):
    ax2.errorbar(Ds, [tabla_e4[D][clave][0] / 100 + 1 for D in VALORES_D],
                 yerr=[tabla_e4[D][clave][1] / 100 for D in VALORES_D], color=color, marker=marca, ms=6, lw=0, elinewidth=1.2, capsize=2)
ax2.plot(D_fino, [ref.tau_kramers(d) / ref.tiempo_primer_paso(d, 1.0) for d in D_fino], color=TINTA_SUAVE, ls="--", lw=1.4)
ax2.axhline(1, color=TINTA, lw=1.0)
ax2.text(0.47, 0.835, "Kramers", color=TINTA_SUAVE, fontsize=9, ha="right")
ax2.text(0.47, 1.012, "simulado (cima, pozo)", color=TINTA, fontsize=9, ha="right")
ax2.set_xlabel("$D$")
ax2.set_ylabel(r"$T/T_\mathrm{exacto}$")
ax2.set_title("cociente con el valor exacto", loc="left")
fig.tight_layout()
guardar(fig, "e5_arrhenius")
plt.show()

# %% [markdown]
# **Verificación.** Las pendientes simuladas coinciden con las exactas: 0.9887 frente a
# 0.9900 en la cima y 0.9876 frente a 0.9884 en el pozo ($|\Delta|\le0.0013$, criterio
# < 0.05).
#
# **Interpretación física.** La pendiente es $\Delta V = 1$ dentro de un 1.2 %: el tiempo
# de escape escala como $e^{S_\mathrm{min}}$, con $S_\mathrm{min} = \Delta V/D$, la acción
# del camino de Onsager-Machlup (§2 del plan). La pendiente exacta en $D\le0.2$ (0.9884 y
# 0.9900) no es exactamente 1, y eso es un resultado físico, no un error: el prefactor
# también depende de $D$: en el tiempo exacto, $T = A(D)\,e^{1/D}$, con $A(D)$ que tiende al
# prefactor de Kramers $2\pi/\sqrt{32}$ solo cuando $D\to0$. El ajuste de $\ln T = \ln A(D) + 1/D$ en un intervalo finito absorbe
# la variación de $\ln A$ en la pendiente; la pendiente local exacta se acerca a 1 al bajar
# $D$. La teoría variacional fija el exponente; el prefactor es lo que no captura (§6).
# Kramers acierta el exponente y queda por debajo del tiempo exacto: $T/T_K$ va de 1.04
# ($D=0.1$) a 1.25 ($D=0.5$).

# %% [markdown]
# ---
# ## B.6 El escape como evento de Poisson (E6)
#
# **Objetivo.** Comprobar que el escape es un evento raro sin memoria.
#
# **Resultado esperado.** Si $T\gg$ el tiempo de relajación dentro del pozo
# ($1/V''(-1) = 1/8$), la partícula olvida su condición inicial antes de escapar y el
# tiempo de escape es exponencial: coeficiente de variación $\mathrm{CV} = 1$. Criterio E6:
# $|\mathrm{CV} - 1| < 0.05$ para $D\le0.15$.

# %%
for D in VALORES_D:
    t = E4[D][0]["t_pozo"]
    cv = coeficiente_variacion(t)
    marca = "criterio < 0.05" if D <= 0.15 else "informativo"
    print(f"D = {D:<5}: CV = {cv:.4f}, |CV − 1| = {abs(cv - 1):.4f} ({marca}); T/(1/8) = {8 * t.mean():.0f}")

# %%
fig, ax = plt.subplots(figsize=(6.2, 3.8))
u = np.linspace(0, 8, 100)
ax.plot(u, np.exp(-u), color=TINTA, lw=1.2, ls="--", label=r"exponencial $e^{-t/\langle T\rangle}$")
for D, tono in ((0.5, 250), (0.25, 450), (0.1, 700)):
    t = E4[D][0]["t_pozo"]
    s, S = supervivencia(t / t.mean())
    ax.plot(s, S, color=AZULES[tono], label=rf"$D = {D}$ (CV = {coeficiente_variacion(t):.3f})")
ax.set_yscale("log")
ax.set_xlim(0, 8)
ax.set_ylim(3e-4, 1.2)
ax.set_xlabel(r"$t/\langle T\rangle$")
ax.set_ylabel(r"$P(T_\mathrm{esc} > t)$")
ax.set_title("E6: distribución del tiempo de escape al otro pozo", loc="left")
ax.legend(fontsize=9)
guardar(fig, "e6_distribucion_tiempos")
plt.show()

# %% [markdown]
# **Verificación.** En $D\le0.15$, $|\mathrm{CV} - 1|\le0.005$ (criterio < 0.05).
#
# **Interpretación física.** Con barrera alta, la supervivencia es una recta en escala
# logarítmica: el escape es un proceso de Poisson con tasa $1/T$. La partícula visita el pozo
# muchas veces y en cada intento la probabilidad de cruzar es la misma, sin memoria de
# cuánto lleva esperando. Al aumentar $D$, el CV baja de 1 (0.93 en $D = 0.5$) y la curva se
# curva al principio. Aparece un tiempo muerto: el tiempo mínimo de relajación y del viaje
# de ida, que ya no es despreciable frente a $T$ ($T\approx10$ con $D=0.5$). La separación
# de escalas que hace al evento "raro" es la misma que lo hace markoviano, y ambas fallan
# juntas.

# %% [markdown]
# ---
# ## B.7 Medición 1: el tubo reactivo frente al camino de Onsager-Machlup (E7)
#
# **Objetivo.** Verificar que las trayectorias reales de escape siguen el camino que
# minimiza el funcional de Onsager-Machlup (§2 del plan).
#
# **Resultado esperado.** Alineadas en el último cruce por $-1/\sqrt2$ antes de llegar a la
# cima (el mismo origen de tiempo de $x_\mathrm{om}$, con $x_\mathrm{om}(0) = -1/\sqrt2$), las
# trayectorias reactivas se concentran alrededor de
# $x_\mathrm{om}(t) = -1/\sqrt{1+e^{8t}}$, en un tubo que se estrecha al disminuir $D$.
# Criterio E7: la desviación cuadrática media entre la mediana punto a punto y
# $x_\mathrm{om}$ en $t\in[-0.5, 0.25]$ (donde $x_\mathrm{om}\le-0.345$) decrece al
# disminuir $D$ (aclaración 11).
#
# **Cálculo.** Ventanas de $t_\mathrm{cima} - 3$ a $t_\mathrm{cima} + 0.5$, muestreadas
# cada 0.01 y guardadas para $D\in\{0.1, 0.15, 0.25\}$. Las muestras anteriores a $t = 0$
# se guardan como NaN y se usa la mediana sin NaN (aclaración 12).

# %%
malla = np.round(np.arange(-1.5, 0.75 + 1e-9, 0.01), 2)
criterio = (malla >= -0.5 - 1e-9) & (malla <= 0.25 + 1e-9)
tubo = {}
for D in (0.25, 0.15, 0.1):
    datos = E4[D][0]
    alineadas = alinear_ventanas(datos["ventanas"].astype(float), datos["t_cima"], datos["t_alineacion"], malla)
    mediana, p10, p90 = mediana_y_banda(alineadas)
    tubo[D] = (alineadas, mediana, p10, p90)
    rms = rms_contra_om(malla[criterio], mediana[criterio])
    print(f"D = {D:<5}: RMS(mediana − x_om) en [−0.5, 0.25] = {rms:.4f}; "
          f"ventanas con NaN al inicio: {100 * np.mean(np.isnan(datos['ventanas'][:, 0])):.2f} %")

# %% [markdown]
# **Ancho del tubo, lejos de la alineación.** En $t = 0$ todas las trayectorias pasan por
# $x = -1/\sqrt2$ por construcción: allí el ancho solo mide cuánto avanza la trayectoria en
# un intervalo de muestreo, del orden de $\sqrt{2D\cdot0.01}$, y no dice nada del tubo. El ancho
# 10–90 % se mide en $t = -0.25$ (subida) y en $t = -1$ (dentro del pozo), y se divide entre
# $\sqrt D$: si el tubo es gaussiano alrededor del camino, con varianza proporcional a $D$,
# ese cociente es aproximadamente constante. En el pozo se compara con la fluctuación de
# equilibrio armónica, de desviación $\sqrt{D/V''(-1)} = \sqrt{D/8}$ (ancho 10–90 %
# $\approx 2\cdot1.2816\,\sqrt{D/8}$), y con los cuantiles 10–90 % de la densidad de Boltzmann
# exacta restringida al pozo izquierdo ($x<0$), que incluye la anarmonía de $V$.

# %%
def indice(t: float) -> int:
    return int(np.argmin(np.abs(malla - t)))


def ancho_boltzmann_pozo(D: float) -> float:
    """Ancho 10–90 % de e^{−V/D} restringida a x < 0 (cuantiles por integración numérica)."""
    xs = np.linspace(-2.5, 0.0, 200_001)
    cdf = np.cumsum(np.exp(-ref.V(xs) / D))
    q10, q90 = np.interp([0.1, 0.9], cdf / cdf[-1], xs)
    return q90 - q10


print(f"{'D':>5} | {'t = 0':>7} {'√(2D·0.01)':>10} | {'t = −0.25':>9} {'/√D':>6} | {'t = −1':>7} {'/√D':>6} "
      f"{'armónico':>9} {'Boltzmann':>9}")
for D in (0.25, 0.15, 0.1):
    _, _, p10, p90 = tubo[D]
    ancho = {t: p90[indice(t)] - p10[indice(t)] for t in (0.0, -0.25, -1.0)}
    print(f"{D:>5} | {ancho[0.0]:>7.4f} {np.sqrt(2 * D * 0.01):>10.4f} | {ancho[-0.25]:>9.4f} {ancho[-0.25] / np.sqrt(D):>6.3f} | "
          f"{ancho[-1.0]:>7.4f} {ancho[-1.0] / np.sqrt(D):>6.3f} {2 * 1.2816 * np.sqrt(D / 8):>9.4f} {ancho_boltzmann_pozo(D):>9.4f}")

# %% [markdown]
# **Tiempo de llegada a la cima.** Cerca de la cima, $x_\mathrm{om}(t)\approx-e^{-4t}$, con
# $4 = |V''(0)|$. El ruido domina cuando la distancia a la cima es del orden de la
# fluctuación térmica en la meseta, $\sqrt{2D/|V''(0)|} = \sqrt{D/2}$. Igualando,
# $e^{-4t^*} = \sqrt{D/2}$, es decir $t^* = \tfrac18\ln(2/D)$, con
# $1/8 = 1/(2|V''(0)|)$. Es una estimación de orden de magnitud del tiempo que tarda la
# trayectoria reactiva en llegar a la cima desde la alineación.

# %%
for D in (0.25, 0.15, 0.1):
    datos = E4[D][0]
    medido = np.median(datos["t_cima"] - datos["t_alineacion"])
    print(f"D = {D:<5}: t* = ln(2/D)/8 = {np.log(2 / D) / 8:.3f}; mediana medida de t_cima − t_alineación = {medido:.3f}")

# %%
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, D in zip(ejes, (0.25, 0.15, 0.1)):
    _, mediana, p10, p90 = tubo[D]
    ax.fill_between(malla, p10, p90, color=AZULES[100], lw=0, label="banda 10–90 %")
    ax.plot(malla, mediana, color=AZULES[600], label="mediana")
    ax.plot(malla, ref.x_om(malla), color=TINTA, ls="--", lw=1.4, label=r"$x_\mathrm{om}(t)$")
    ax.axvspan(-0.5, 0.25, color=REJILLA, alpha=0.45, lw=0, zorder=0)
    rms = rms_contra_om(malla[criterio], mediana[criterio])
    ax.set_title(rf"$D = {D}$   (RMS = {rms:.3f})", loc="left")
    ax.set_xlabel("$t$ (origen: último cruce por $-1/\\sqrt{2}$)")
ejes[0].set_ylabel("$x$")
ejes[0].legend(loc="upper left", fontsize=8.5)
ejes[0].set_ylim(-1.38, 0.98)
ejes[0].text(-0.13, -1.33, "intervalo\ndel criterio", color=TINTA_SUAVE, fontsize=7.5, ha="center")
fig.suptitle("E7: tubo reactivo frente al camino de Onsager-Machlup", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
guardar(fig, "e7_tubo_reactivo")
plt.show()

# %%
fig, ejes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
bordes_x = np.linspace(-1.6, 1.2, 141)
bordes_t = np.append(malla - 0.005, malla[-1] + 0.005)
for ax, D in zip(ejes, (0.25, 0.15, 0.1)):
    alineadas = tubo[D][0]
    validos = np.isfinite(alineadas)
    tt = np.broadcast_to(malla, alineadas.shape)[validos]
    H, _, _ = np.histogram2d(tt, alineadas[validos], bins=[bordes_t, bordes_x])
    H = H / H.sum(axis=1, keepdims=True).clip(min=1) / np.diff(bordes_x)  # densidad de x en cada t
    ax.pcolormesh(bordes_t, bordes_x, H.T, cmap="Blues", norm=PowerNorm(0.5, vmin=0), rasterized=True)
    ax.plot(malla, ref.x_om(malla), color=NARANJA, ls="--", lw=1.4, label=r"$x_\mathrm{om}(t)$")
    ax.set_title(rf"$D = {D}$", loc="left")
    ax.set_xlabel("$t$ (origen: último cruce por $-1/\\sqrt{2}$)")
    ax.grid(False)
ejes[0].set_ylabel("$x$")
ejes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("E7: densidad de las trayectorias reactivas alineadas (escala de raíz cuadrada)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
guardar(fig, "e7_densidad_ventanas")
plt.show()

# %% [markdown]
# **Verificación.** La desviación cuadrática media entre la mediana y $x_\mathrm{om}$ en
# $t\in[-0.5, 0.25]$ decrece al disminuir $D$: 0.120 ($D = 0.25$), 0.083 ($D = 0.15$) y
# 0.058 ($D = 0.1$). Esa es la evidencia de que el tubo converge al camino.
#
# **Ancho del tubo.** En $t=0$ el ancho (0.066, 0.051 y 0.041) coincide con
# $\sqrt{2D\cdot0.01}$ (0.071, 0.055 y 0.045): es el avance en un intervalo de muestreo y no es
# evidencia del estrechamiento. El "pellizco" de la figura de densidad en $t = 0$ es
# consecuencia de la alineación, no física. Lejos de la alineación, el tubo se estrecha
# como $\sqrt D$: en $t=-0.25$ el ancho es 0.521, 0.415 y 0.333, con ancho$/\sqrt D$ = 1.04,
# 1.07 y 1.05, aproximadamente constante. En $t=-1$, dentro del pozo, el ancho es 0.488,
# 0.373 y 0.298 (ancho$/\sqrt D$ = 0.98, 0.96 y 0.94). Supera al armónico, 0.453, 0.351 y
# 0.287, en un 8 %, 6 % y 4 %, y se acerca a él al bajar $D$: es la anarmonía del pozo. Con
# la densidad de Boltzmann exacta del pozo, el acuerdo es mejor (0.3765 y 0.2987 en
# $D = 0.15$ y 0.1). En $D = 0.25$ esa referencia, 0.523, queda un 7 % por encima de lo
# medido, porque su cola hacia la cima ya pesa. Antes de escapar, la partícula está en
# equilibrio térmico en el pozo.
#
# **Interpretación física.** Antes del origen, las trayectorias reactivas suben desde el pozo
# por el camino $x_\mathrm{om}$: el escape no es una difusión al azar hacia la cima, sino
# una excursión coordinada por la ruta de mínima acción, la imagen invertida en el tiempo de
# la relajación determinista ($\dot x = +V'$). La diferencia está cerca de la cima.
# $x_\mathrm{om}$ tarda un tiempo infinito en llegar a $x=0$, porque la cima es un punto de
# equilibrio, mientras que el ruido la alcanza en un tiempo finito. La estimación
# $t^* = \frac18\ln(2/D)$ (llegada al punto donde la distancia a la cima es la fluctuación
# térmica de la meseta) da 0.26, 0.32 y 0.37, frente a las medianas medidas de
# $t_\mathrm{cima} - t_\mathrm{alineación}$: 0.23, 0.31 y 0.38. El coeficiente
# $1/8 = 1/(2|V''(0)|)$ lo fija la curvatura de la barrera, y $t^*$ diverge
# logarítmicamente cuando $D\to0$. Por eso la mediana va por delante de $x_\mathrm{om}$
# en el tramo final, y el criterio se evalúa donde $x_\mathrm{om}$ describe la subida. A
# medida que $D$ baja, la mediana converge a $x_\mathrm{om}$ y el tubo se estrecha como
# $\sqrt D$: es la
# concentración de la medida de caminos alrededor del minimizador que predice el método de
# Laplace (§1 del plan).

# %% [markdown]
# ---
# ## B.8 Resumen
#
# | Experimento | Criterio | Resultado | |
# |---|---|---|---|
# | E0 oráculo exacto | quad frente a mpmath < $10^{-8}$; factor ½; convergencia a Kramers | $5.6\times10^{-16}$ | pasa |
# | E1 Numba frente a NumPy | KS $p > 0.01$ | $p = 0.165$ y 0.145 (prueba); 0.111 (réplica) | pasa |
# | E2 Boltzmann | $L_1 < 0.02$ | 0.0046 (prueba); 0.0044 (réplica) | pasa |
# | E3 convergencia en $\Delta t$ | error < 2 %; monotonía significativa | los 4 $\Delta t$; $\Delta t = 10^{-3}$ elegido (D30) | pasa |
# | E4 tiempos | error < 3 %, 7 $D$ × 2 destinos | máximo 0.62 % | pasa |
# | E5 Arrhenius | $|\Delta\,\text{pendiente}| < 0.05$ | 0.0013 (cima), 0.0008 (pozo) | pasa |
# | E6 Poisson | $|\mathrm{CV}-1| < 0.05$, $D\le0.15$ | $\le 0.005$ | pasa |
# | E7 tubo reactivo | RMS decrece con $D$ | 0.120 → 0.083 → 0.058 | pasa |
#
# **Qué verifica el Bloque B sobre la física.** El proceso real escapa con el exponente
# $\Delta V/D$ que fija la acción de Onsager-Machlup (B.5), y lo hace siguiendo su camino
# (B.7), como un evento raro y sin memoria (B.6). Lo que la aproximación de ruido débil no
# captura es el prefactor: en este rango de $D$, el tiempo exacto supera a Kramers entre un
# 4 % y un 25 % (B.0, B.5), y la pendiente efectiva de Arrhenius es 0.988–0.990 y no 1. Un resultado numérico
# del bloque es que la frontera discreta sesga el tiempo a la cima en ≈ +4 % (con
# $\Delta t = 10^{-3}$), independiente de $D$ y bien descrito por la corrección de
# continuidad; el puente browniano lo elimina.
