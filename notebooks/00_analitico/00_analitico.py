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
# # Hito 00 — Referencia analítica: escape térmico e instantón en el doble pozo
#
# Este cuaderno construye, con sympy, la referencia simbólica contra la que se contrastan
# los experimentos numéricos del taller (especificación `specs/00_analitico.md`). Cada
# resultado se **deriva** y luego se **verifica dos veces**: simbólicamente (residuo
# exactamente cero) y numéricamente con scipy (una implementación independiente).
#
# Cada sección sigue la misma estructura: **objetivo**, **resultado esperado**,
# **cálculo**, **verificación** e **interpretación física**.
#
# La lógica vive en `src/taller/analitico/`; aquí solo se llama, se verifica y se explica.
#
# **Notación** (CLAUDE.md §2): $V(x) = (x^2-1)^2$, $D = k_BT/\gamma$ es la intensidad del
# ruido, $t$ es el tiempo real y $\tau$ el tiempo imaginario del instantón. El tiempo medio
# de escape se escribe $\tau_\mathrm{esc}$ para no confundirlo con $\tau$.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import Math, display
from scipy.differentiate import derivative
from scipy.integrate import quad, solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, minimize_scalar

from taller.analitico import escape_termico as om
from taller.analitico import instanton as ins
from taller.analitico import kramers, referencias as ref, schrodinger
from taller.analitico.potencial import (
    D,
    V,
    altura_barrera,
    corriente_fp,
    d2V,
    d3V,
    densidad_estacionaria,
    dV,
    estabilidad,
    puntos_fijos,
    t,
    tau,
    v,
    x,
)
from taller.analitico.variacional import derivada_temporal, ecuacion_el, integral_primera

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
FIGURAS = RAIZ / "figures" / "analitico"
VALORES_D = [0.1, 0.15, 0.25, 0.35, 0.5]


def mostrar(nombre: str, expresion) -> None:
    """Muestra `nombre = expresion` en LaTeX."""
    display(Math(f"{nombre} = {sp.latex(expresion)}"))


def es_cero(expresion) -> bool:
    """Verificación simbólica: el residuo se simplifica exactamente a cero."""
    return sp.simplify(expresion) == 0


# Malla determinista irregular que evita los puntos fijos −1, 0, +1 (aclaración 9):
# x_k = a + (b − a)·{k·φ mod 1}, con φ la razón áurea; sin aleatoriedad ni semilla.
def malla_irregular(n: int, a: float, b: float, alfa: float = (np.sqrt(5) - 1) / 2) -> np.ndarray:
    k = np.arange(1, n + 1)
    puntos = a + (b - a) * ((k * alfa) % 1.0)
    lejos = np.min(np.abs(puntos[:, None] - np.array([-1.0, 0.0, 1.0])[None, :]), axis=1) >= 1e-2
    return puntos[lejos]


# %% [markdown]
# ---
# ## §0 El potencial y el equilibrio
#
# **Objetivo.** Caracterizar el paisaje de energía: dónde están los estados metaestables y
# la barrera, cuán rígidos son, y cuál es el equilibrio de Fokker-Planck.
#
# **Resultado esperado.** Puntos fijos de $\dot x = -V'(x)$ en $x \in \{-1, 0, 1\}$;
# curvaturas $V''(\pm1) = 8$ (mínimos estables) y $V''(0) = -4$ (barrera inestable);
# $\Delta V = V(0) - V(-1) = 1$. La densidad $p_s \propto e^{-V/D}$ anula la corriente
# $J = -V'p - D\,p'$.
#
# **Cálculo.** Derivadas del potencial:

# %%
mostrar("V(x)", V())
mostrar("V'(x)", sp.factor(dV()))
mostrar("V''(x)", sp.expand(d2V()))

# %% [markdown]
# Los puntos fijos de la dinámica determinista son las raíces de $V'$. Su estabilidad sale
# de linealizar: si $x = x_0 + \delta$, entonces $\dot\delta = -V''(x_0)\,\delta$. El punto
# es estable si $V''(x_0) > 0$ (la perturbación decae como $e^{-V''(x_0)t}$).

# %%
for p in puntos_fijos():
    print(f"x0 = {str(p):>3}:  V = {V(p)},  V'' = {str(d2V(p)):>3}  ->  {estabilidad(p)}")
mostrar(r"\Delta V", altura_barrera())

# %% [markdown]
# Para el equilibrio: con $p_s = e^{-V/D}$ se tiene $p_s' = -(V'/D)\,p_s$, así que
# $J = -V'p_s - D\,(-V'/D)\,p_s = 0$ término a término.

# %%
p_s = densidad_estacionaria()
mostrar("p_s", p_s)
mostrar("J[p_s]", sp.simplify(corriente_fp(p_s)))

# %% [markdown]
# **Verificación numérica (scipy).** Raíces de $V'$ con `brentq` en intervalos que las
# encierran, curvaturas con `scipy.differentiate.derivative`, y $J[p_s]$ con la derivada
# numérica de $p_s$ en una malla irregular que evita los puntos fijos (donde $V' = 0$ y la
# anulación sería trivial).

# %%
raices = [brentq(ref.dV, a, b) for a, b in [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5)]]
curvaturas = [derivative(ref.dV, r).df for r in raices]
print("raíces de V' (brentq):", np.round(raices, 14))
print("V'' numérica en ellas:", np.round(curvaturas, 10))

xs = malla_irregular(300, -1.4, 1.4)
for D_val in [0.1, 0.5]:
    p_num = lambda y: np.exp(-ref.V(y) / D_val)
    dp = derivative(p_num, xs).df
    J = -ref.dV(xs) * p_num(xs) - D_val * dp
    escala = np.abs(ref.dV(xs) * p_num(xs)) + np.abs(D_val * dp)
    print(f"D = {D_val}: max |J| / escala = {np.max(np.abs(J) / escala):.2e}")

# %% [markdown]
# **Interpretación física.** El sistema tiene dos estados metaestables simétricos
# ($x = \pm1$) separados por una barrera de altura $\Delta V = 1$ en $x = 0$. La curvatura
# del pozo, $V''(-1) = 8$, es la tasa de relajación hacia el mínimo; la de la barrera,
# $|V''(0)| = 4$, es la tasa con que la dinámica determinista se aleja de la cima. Ambas
# controlan el prefactor de Kramers (sección §9). En equilibrio la corriente es nula en
# cada punto (equilibrio detallado): la distribución de Boltzmann no transporta
# probabilidad, aunque cada partícula individual salte entre pozos.

# %% [markdown]
# ---
# ## §1 Lagrangiano de Onsager-Machlup y ecuación de Euler-Lagrange
#
# **Objetivo.** Obtener la ecuación que satisface el camino más probable de escape
# térmico.
#
# **Resultado esperado.** Con $L = (\dot x + V')^2/(4D)$ (forma de ruido débil de
# Freidlin-Wentzell), la ecuación de Euler-Lagrange es $\ddot x = V'V''$.
#
# **Cálculo.** Para la dinámica $dx = -V'dt + \sqrt{2D}\,dW$, el ruido en un intervalo es
# $\sqrt{2D}\,dW = (\dot x + V')\,dt$, y la probabilidad de un camino es
# $\propto \exp\!\left[-\frac{1}{4D}\int (\dot x + V')^2 dt\right]$. El lagrangiano es:

# %%
L_om = om.lagrangiano_om()
mostrar("L_{OM}", L_om)

# %% [markdown]
# A mano: $\partial L/\partial\dot x = (\dot x + V')/(2D)$, cuya derivada total es
# $(\ddot x + V''\dot x)/(2D)$; y $\partial L/\partial x = (\dot x + V')\,V''/(2D)$. Al
# restar, los términos $V''\dot x$ se cancelan y queda $(\ddot x - V'V'')/(2D) = 0$.
# sympy (`euler_equations`) da lo mismo:

# %%
xpp_om = ecuacion_el(L_om)
mostrar(r"\ddot x", xpp_om)
print("ẍ − V'V'' = 0:", es_cero(xpp_om - dV() * d2V()))

# %% [markdown]
# **Verificación numérica (scipy).** Se integra la ecuación de segundo orden
# $\ddot x = V'V''$ con `solve_ivp` desde las condiciones del camino cerrado en $t=0$
# ($x = -1/\sqrt2$, $\dot x = V'(-1/\sqrt2)$) y se compara con $x_\mathrm{om}(t)$ (sección §3).
#
# Es una verificación exigente: el camino es una órbita heteroclínica (conecta dos puntos
# de equilibrio de la mecánica ficticia), que es inestable. Cualquier error en las
# condiciones iniciales crece exponencialmente, por eso el intervalo es corto.

# %%
def integrar_segundo_orden(aceleracion, x0, v0, s_fin):
    sol = solve_ivp(
        lambda _s, y: [y[1], aceleracion(y[0])],
        (0.0, s_fin), [x0, v0], method="DOP853", rtol=1e-12, atol=1e-14,
        t_eval=np.linspace(0.0, s_fin, 201),
    )
    return sol.t, sol.y


x0_om = ref.x_om(0.0)
error_el = 0.0
for fin in (1.5, -1.5):
    s, (xs_num, _) = integrar_segundo_orden(
        lambda y: ref.dV(y) * ref.d2V(y), x0_om, ref.dV(x0_om), fin
    )
    error_el = max(error_el, np.max(np.abs(xs_num - ref.x_om(s))))
print(f"ẍ = V'V'' integrada en [−1.5, 1.5]: error máximo frente a x_om = {error_el:.2e}")

# %% [markdown]
# **Interpretación física.** El camino más probable obedece una mecánica newtoniana
# ficticia, con aceleración $V'V'' = \frac{d}{dx}\left(\tfrac12 V'^2\right)$: es una
# partícula en el potencial efectivo $U_\mathrm{OM} = -\tfrac12 V'^2$ (sección §7). El
# ruido no aparece en esta ecuación: en el límite de ruido débil, $D$ solo multiplica a la
# acción, y por eso el camino óptimo no depende de la temperatura.

# %% [markdown]
# ---
# ## §2 Integral primera y ramas de energía cero
#
# **Objetivo.** Reducir la ecuación de segundo orden a una de primer orden y elegir la
# rama que describe el escape.
#
# **Resultado esperado.** $H = \dot x\,\partial L/\partial\dot x - L \propto \dot x^2 - V'^2$,
# conservada. Con $H = 0$ (la energía de un camino que parte del reposo en $x=-1$), las
# ramas son $\dot x = \pm V'$. La que conecta $x=-1$ ($t\to-\infty$) con $x=0$ ($t\to+\infty$)
# es $\dot x = +V'$.
#
# **Cálculo.** Como $L$ no depende explícitamente de $t$, la función $H$ se conserva
# (teorema de Noether para traslaciones temporales). A mano:
# $H = \dot x\,\frac{\dot x + V'}{2D} - \frac{(\dot x + V')^2}{4D}
#    = \frac{(\dot x + V')(\dot x - V')}{4D} = \frac{\dot x^2 - V'^2}{4D}$.

# %%
H_om = integral_primera(L_om)
mostrar("H_{OM}", H_om)
print("dH/dt = 0 sobre Euler-Lagrange:", es_cero(derivada_temporal(H_om, xpp_om)))
mostrar(r"\dot x \;(H = 0)", om.ramas_energia_cero())

# %% [markdown]
# **¿Qué rama escapa?** En $(-1, 0)$: $x<0$ y $x^2-1<0$, así que $V' = 4x(x^2-1) > 0$.
# - $\dot x = -V' < 0$: la partícula baja hacia $-1$. Es la **relajación determinista**, y
#   su acción es cero porque $\dot x + V' = 0$.
# - $\dot x = +V' > 0$: la partícula sube de $-1$ hacia $0$. Es el **camino de escape**: la
#   relajación invertida en el tiempo.

# %%
for x_prueba in [-0.9, -0.5, -0.1]:
    print(f"x = {x_prueba}:  V'(x) = {ref.dV(x_prueba):+.3f}  ->  la rama ẋ = +V' sube (ẋ > 0)")

# %% [markdown]
# **Verificación numérica (scipy).** A lo largo de la integración numérica de la sección
# §1, $H$ debe mantenerse constante e igual a cero.

# %%
s, (xs_num, vs_num) = integrar_segundo_orden(
    lambda y: ref.dV(y) * ref.d2V(y), x0_om, ref.dV(x0_om), 1.5
)
H_num = (vs_num**2 - ref.dV(xs_num) ** 2) / 4  # H·D, con D = 1 sin pérdida de generalidad
print(f"max |H·D| a lo largo de la integración: {np.max(np.abs(H_num)):.2e}")

# %% [markdown]
# **Interpretación física.** El escape más probable es la imagen especular en el tiempo de
# la relajación: la partícula sube la barrera siguiendo exactamente la trayectoria por la
# que bajaría, pero al revés. Es una propiedad de los sistemas en equilibrio detallado
# (reversibilidad microscópica), y es la razón de que el camino de escape sea tan simple.

# %% [markdown]
# ---
# ## §3 El camino de escape $x_\mathrm{om}(t)$
#
# **Objetivo.** Resolver la ecuación de primer orden $\dot x = V'(x)$ y estudiar su forma
# asintótica.
#
# **Resultado esperado.** $x_\mathrm{om}(t) = -1/\sqrt{1 + e^{8t}}$, con el origen de
# tiempo en $x_\mathrm{om}(0) = -1/\sqrt2$. Se aleja de $-1$ como $e^{8t}$ y llega a $0$
# como $e^{-4t}$.
#
# **Cálculo.** La ecuación es separable: $dt = dx / [4x(x^2-1)]$. Por fracciones simples,
# $\frac{1}{4x(x^2-1)} = -\frac{1}{4x} + \frac{1}{8(x-1)} + \frac{1}{8(x+1)}$, y en
# $(-1, 0)$ una primitiva real es
# $$ t = \tfrac18 \ln(1-x^2) - \tfrac14 \ln|x| + C = \tfrac18 \ln\frac{1-x^2}{x^2} + C. $$
# Con $x(0) = -1/\sqrt2$, $\frac{1-x^2}{x^2} = 1$, así que $C = 0$ y
# $\frac{1-x^2}{x^2} = e^{8t}$, es decir $x^2 = \frac{1}{1+e^{8t}}$. La rama negativa da
# $x_\mathrm{om}$.
#
# **El origen de tiempo es una convención.** La ecuación es invariante bajo
# $t \to t - t_0$: hay una familia de soluciones (el "modo cero"). Fijar $x(0) = -1/\sqrt2$
# elige una. El bloque neuronal usará la misma convención.

# %%
x_om = om.camino_om()
mostrar(r"x_\mathrm{om}(t)", x_om)
print("residuo ẋ − V'(x):     ", sp.simplify(sp.diff(x_om, t) - dV(x_om)))
print("residuo ẍ − V'V''(x):  ", sp.simplify(sp.diff(x_om, t, 2) - (dV() * d2V()).subs(x, x_om)))
print("límites t → −∞, +∞:    ", sp.limit(x_om, t, -sp.oo), sp.limit(x_om, t, sp.oo))
print("x_om(0):               ", x_om.subs(t, 0))

# %% [markdown]
# **Tasas asintóticas.** Cerca de $-1$, linealizando $\dot x = V'(x)$ con $x = -1 + \delta$:
# $\dot\delta \approx V''(-1)\,\delta = 8\,\delta$, de modo que $\delta \sim e^{8t}$. Cerca de
# $0$: $\dot x \approx V''(0)\,x = -4x$, de modo que $x \sim e^{-4t}$. sympy las calcula como
# límites de las derivadas logarítmicas:

# %%
salida, llegada = om.tasas_om()
print(f"tasa de salida de −1:  {salida}   (V''(−1) = {d2V(-1)})")
print(f"tasa de llegada a 0:   {llegada}   (|V''(0)| = {abs(d2V(0))})")

# %% [markdown]
# **Verificación numérica (scipy).** `solve_ivp` (DOP853, rtol $=10^{-12}$, atol $=10^{-14}$)
# integra $\dot x = 4x(x^2-1)$ desde $x(0) = -1/\sqrt2$, hacia adelante y hacia atrás, en
# $t\in[-2,2]$ (criterio 8). Se parte de $t=0$ y no de un punto cercano a $-1$: allí el
# valor de $x + 1$ es tan pequeño que se pierde precisión por cancelación, y como $-1$ es
# inestable para $\dot x = V'$, ese error crecería como $e^{8t}$.

# %%
def error_primer_orden(rhs, x0, solucion):
    error = 0.0
    for fin in (2.0, -2.0):
        s_eval = np.linspace(0.0, fin, 201)
        sol = solve_ivp(lambda _s, y: rhs(y), (0.0, fin), [x0], method="DOP853",
                        rtol=1e-12, atol=1e-14, t_eval=s_eval)
        error = max(error, np.max(np.abs(sol.y[0] - solucion(s_eval))))
    return error


f_x_om = sp.lambdify(t, x_om, "numpy")
print(f"error máximo |x_num − x_om| en [−2, 2]: {error_primer_orden(ref.dV, -1/np.sqrt(2), f_x_om):.2e}"
      "   (criterio: < 1e-8)")

# %% [markdown]
# **Interpretación física.** El camino tarda un tiempo infinito en salir de $-1$ y en
# llegar a $0$: ambos son puntos de equilibrio. En la práctica, la partícula fluctúa en el
# pozo durante un tiempo de orden $\tau_\mathrm{esc}$ y, cuando escapa, lo hace en una
# excursión rápida de duración $\sim 1/8 + 1/4$ que sigue $x_\mathrm{om}$. Las tasas 8 y 4 son
# las curvaturas del potencial: la excursión sale del pozo a la velocidad con que se
# relajaría hacia él, y llega a la cima a la velocidad con que se alejaría de ella.

# %% [markdown]
# ---
# ## §4 La acción mínima $S_\mathrm{min}$ y la cota global
#
# **Objetivo.** Calcular la acción del camino de escape: su exponencial,
# $e^{-S_\mathrm{min}}$, es la escala de la probabilidad de escape.
#
# **Resultado esperado.** $S_\mathrm{min} = \Delta V / D = 1/D$; y
# $S[x] \ge \Delta V/D$ para toda trayectoria de $-1$ a $0$.
#
# **Cálculo (derivada total).** Sobre la rama $\dot x = V'$:
# $L = \frac{(2V')^2}{4D} = \frac{V'^2}{D} = \frac{V'\dot x}{D} = \frac1D \frac{dV}{dt}$.
# Entonces $S = \frac1D \int_{-\infty}^{\infty} \frac{dV}{dt}\,dt = \frac{V(0) - V(-1)}{D}$.

# %%
mostrar(r"S_\mathrm{min}", om.S_min())
mostrar(r"S[x_\mathrm{om}] \;(\text{integral directa en } t)", om.accion_om(x_om))

# %% [markdown]
# **Cota global.** Desarrollando los cuadrados se comprueba la identidad
# $(\dot x + V')^2 = (\dot x - V')^2 + 4\dot x V'$. Integrando sobre cualquier camino de $-1$
# a $0$:
# $$ S[x] = \frac{1}{4D}\int(\dot x - V')^2 dt + \frac1D\int \dot x V' dt
#        = \underbrace{\frac{1}{4D}\int(\dot x - V')^2 dt}_{\ge 0} + \frac{\Delta V}{D}
#        \;\ge\; \frac{\Delta V}{D}. $$
# El término cruzado es una derivada total ($\dot x V' = dV/dt$), así que solo depende de
# los extremos. La igualdad se alcanza si y solo si $\dot x = V'$: el camino de escape es el
# **minimizador global**, no solo un punto estacionario.

# %%
izq, der = om.identidad_cuadrados_om()
mostrar(r"(\dot x + V')^2 - [(\dot x - V')^2 + 4\dot x V']", sp.simplify(izq - der))

# %% [markdown]
# **Verificación numérica (scipy).** (a) `quad` integra el lagrangiano sobre $x_\mathrm{om}$
# para cada $D$ (criterio 4). Se integra en $t \in [-10, 10]$: las colas decaen como
# $e^{16t}$ y $e^{-8t}$, y aportan menos de $e^{-80}$.
#
# (b) La cota en acción: la familia de caminos reescalados $x_\lambda(t) = x_\mathrm{om}(\lambda t)$
# va de $-1$ a $0$ con $\dot x_\lambda = \lambda V'$. Para ella,
# $S = \frac{(\lambda+1)^2}{4\lambda D}\int_{-1}^{0} V'dx = \frac{(\lambda+1)^2}{4\lambda}\,\frac1D$,
# que es mínima (y vale $1/D$) solo en $\lambda = 1$.

# %%
xo_s = om.camino_om()
integrando = sp.lambdify((t, D), om.lagrangiano_om().subs({v: sp.diff(xo_s, t), x: xo_s}), "numpy")
for D_val in VALORES_D:
    valor, _ = quad(lambda s: integrando(s, D_val), -10, 10, epsabs=0, epsrel=1e-13, limit=200)
    print(f"D = {D_val:<4}:  quad = {valor:.12f},  1/D = {1/D_val:.12f},  error relativo = {abs(valor*D_val - 1):.1e}")

print()
for lam in [0.5, 0.8, 1.0, 1.25, 2.0]:
    # x_λ(t) = x_om(λt): ẋ_λ = λ V'(x_λ); D = 1
    f = lambda s: (lam * ref.dV(ref.x_om(lam * s)) + ref.dV(ref.x_om(lam * s))) ** 2 / 4
    valor, _ = quad(f, -20, 20, epsabs=0, epsrel=1e-12, limit=400)
    print(f"λ = {lam:<4}:  S·D = {valor:.10f}   (λ+1)²/(4λ) = {(lam+1)**2/(4*lam):.10f}")

# %% [markdown]
# **Interpretación física.** La probabilidad de escapar por unidad de tiempo escala como
# $e^{-S_\mathrm{min}} = e^{-\Delta V/D} = e^{-\Delta V/k_BT}$ (con $\gamma = 1$): es la **ley de
# Arrhenius**, obtenida aquí como el valor mínimo de un funcional. El exponente solo depende
# de la altura de la barrera, no de la forma del potencial; la forma entra en el prefactor
# (sección §9). Cualquier otro camino, por ejemplo uno más rápido o más lento
# ($\lambda \neq 1$), tiene acción mayor y es exponencialmente menos probable.

# %% [markdown]
# ---
# ## §5 El instantón: ecuación de movimiento y kink
#
# **Objetivo.** Encontrar el camino dominante del tunelamiento: el mínimo de la acción
# euclídea entre los dos pozos.
#
# **Resultado esperado.** Con $L_E = \tfrac12\dot x^2 + V$, Euler-Lagrange da $\ddot x = V'$.
# La integral primera con energía cero es $\tfrac12\dot x^2 - V = 0$, y la rama
# $\dot x = \sqrt{2V} = \sqrt2(1-x^2)$ en $|x|<1$ da el kink $x_\mathrm{kink}(\tau) = \tanh(\sqrt2\,\tau)$,
# con tasa asintótica $2\sqrt2 = \sqrt{V''(\pm1)}$.
#
# **Cálculo.** En tiempo imaginario $t = -i\tau$, la energía cinética cambia de signo y la
# acción euclídea es la de una partícula en el potencial **invertido** $-V$:

# %%
L_E = ins.lagrangiano_euclideo()
mostrar("L_E", L_E)
xpp_E = ecuacion_el(L_E)
mostrar(r"\ddot x", xpp_E)
print("ẍ − V' = 0:", es_cero(xpp_E - dV()))

H_E = integral_primera(L_E)
mostrar("H_E", H_E)
print("dH_E/dτ = 0 sobre Euler-Lagrange:", es_cero(derivada_temporal(H_E, xpp_E)))

# %% [markdown]
# **Elección de la rama.** $H_E = \tfrac12\dot x^2 - V = 0$ (la partícula parte del reposo
# en la cima $x=-1$ de $-V$, donde $V = 0$) da $\dot x = \pm\sqrt{2V}$. En general
# $\sqrt{2V} = \sqrt2\,|x^2-1|$. El kink vive en $(-1, 1)$, donde $1 - x^2 > 0$, así que
# allí $\sqrt{2V} = \sqrt2(1-x^2)$ sin valor absoluto. La rama positiva ($\dot x > 0$) es la
# que va de $-1$ a $+1$.

# %%
mostrar(r"\dot x", ins.rama_kink())

# %% [markdown]
# Separando variables, $d\tau = \frac{dx}{\sqrt2(1-x^2)}$, y en $(-1,1)$:
# $\sqrt2\,\tau = \operatorname{artanh} x + C$. Con $x(0) = 0$ (convención del origen, el
# modo cero), $C = 0$ y $x = \tanh(\sqrt2\,\tau)$.

# %%
x_kink = ins.camino_kink()
mostrar(r"x_\mathrm{kink}(\tau)", x_kink)
print("residuo ẋ − √2(1 − x²): ", sp.simplify(sp.diff(x_kink, tau) - ins.rama_kink().subs(x, x_kink)))
print("residuo ẍ − V'(x):      ", sp.simplify(sp.diff(x_kink, tau, 2) - dV(x_kink)))
print("límites τ → −∞, +∞:     ", sp.limit(x_kink, tau, -sp.oo), sp.limit(x_kink, tau, sp.oo))
print(f"tasa asintótica:         {ins.tasa_kink()}  =  √V''(±1) = {ins.frecuencia_pozo()}")

# %% [markdown]
# **Verificación numérica (scipy).** (a) Primer orden: $\dot x = \sqrt2(1-x^2)$ desde
# $x(0)=0$ en $\tau\in[-2,2]$ (criterio 8). (b) Segundo orden: $\ddot x = V'$ desde
# $x(0) = 0$, $\dot x(0) = \sqrt2$; como en §1, la órbita es heteroclínica e inestable, así
# que se usa un intervalo corto.

# %%
f_x_kink = sp.lambdify(tau, x_kink, "numpy")
print(f"(a) error máximo |x_num − x_kink| en [−2, 2]: "
      f"{error_primer_orden(lambda y: np.sqrt(2)*(1 - y**2), 0.0, f_x_kink):.2e}   (criterio: < 1e-8)")
error_el_E = 0.0
for fin in (1.5, -1.5):
    s, (xs_num, _) = integrar_segundo_orden(ref.dV, 0.0, np.sqrt(2), fin)
    error_el_E = max(error_el_E, np.max(np.abs(xs_num - ref.x_kink(s))))
print(f"(b) ẍ = V' integrada en [−1.5, 1.5]: error máximo frente a x_kink = {error_el_E:.2e}")

# %% [markdown]
# **Interpretación física.** En el potencial invertido $-V$, los pozos $\pm1$ se convierten
# en **cimas** y la barrera en un **valle**. El instantón es una partícula que parte (en
# $\tau\to-\infty$) de la cima $-1$, cruza el valle y llega (en $\tau\to+\infty$) a la cima
# $+1$, sin energía sobrante. La tasa $2\sqrt2 = \sqrt{V''(\pm1)} = \omega$ es la frecuencia del
# oscilador armónico en el fondo de cada pozo: el ancho del kink, $\sim 1/\omega$, es el
# "tiempo" de tunelamiento.

# %% [markdown]
# ---
# ## §6 La acción del instantón $S_0$ y la cota de Bogomolny
#
# **Objetivo.** Calcular la acción del instantón, que fija la escala exponencial
# $e^{-S_0/\hbar}$ de la amplitud de tunelamiento.
#
# **Resultado esperado.** $S_0 = \int_{-1}^{1}\sqrt{2V}\,dx = 4\sqrt2/3 \approx 1.8856$;
# y $S_E \ge S_0$ para toda trayectoria de $-1$ a $+1$.
#
# **Cálculo.** En $(-1,1)$, $\sqrt{2V} = \sqrt2(1-x^2)$:
# $S_0 = \sqrt2\left[x - \tfrac{x^3}{3}\right]_{-1}^{1} = \sqrt2\left(2 - \tfrac23\right) = \tfrac{4\sqrt2}{3}$.

# %%
mostrar("S_0", ins.S0())
print(f"S0 ≈ {float(ins.S0()):.10f}")
mostrar(r"S_E[x_\mathrm{kink}] \;(\text{integral directa en } \tau)", ins.accion_euclidea(x_kink))

# %% [markdown]
# **Cota de Bogomolny.** Completando el cuadrado:
# $\tfrac12\dot x^2 + V = \tfrac12\left(\dot x - \sqrt{2V}\right)^2 + \dot x\sqrt{2V}$. El último
# término es una derivada total, $\dot x\sqrt{2V} = \frac{dW}{d\tau}$ con
# $W(x) = \int\sqrt{2V}\,dx$. Para cualquier trayectoria de $-1$ a $+1$:
# $$ S_E = \tfrac12\int\left(\dot x - \sqrt{2V}\right)^2 d\tau + W(1) - W(-1) \;\ge\; S_0, $$
# con igualdad si y solo si $\dot x = \sqrt{2V}$: el kink. Aquí $\sqrt{2V} \ge 0$ en toda la
# recta, así que la cota no requiere suponer $|x| < 1$.

# %%
izq, der = ins.identidad_bogomolny()
mostrar(r"\tfrac12\dot x^2 + V - [\tfrac12(\dot x - \sqrt{2V})^2 + \dot x\sqrt{2V}]", sp.simplify(izq - der))

# %% [markdown]
# **Verificación numérica (scipy).** (a) `quad` de $\sqrt{2V}$ en $[-1,1]$ (criterio 4).
# (b) `quad` de $S_E$ sobre el kink reescalado $x_\lambda(\tau) = x_\mathrm{kink}(\lambda\tau)$.
# Con $\dot x_\lambda = \lambda\sqrt{2V}$, se obtiene $S_E = \tfrac12(\lambda + 1/\lambda)\,S_0$,
# mínima solo en $\lambda = 1$.

# %%
valor, _ = quad(lambda y: np.sqrt(2 * ref.V(y)), -1, 1, epsabs=0, epsrel=1e-13)
print(f"(a) quad = {valor:.14f},  4√2/3 = {4*np.sqrt(2)/3:.14f},  error relativo = {abs(valor/ref.S0 - 1):.1e}")
print()
for lam in [0.5, 0.8, 1.0, 1.25, 2.0]:
    f = lambda s: 0.5 * (lam * np.sqrt(2) * (1 - ref.x_kink(lam * s) ** 2)) ** 2 + ref.V(ref.x_kink(lam * s))
    valor, _ = quad(f, -30, 30, epsabs=0, epsrel=1e-12, limit=400)
    print(f"(b) λ = {lam:<4}:  S_E/S0 = {valor/ref.S0:.10f}   (λ + 1/λ)/2 = {(lam + 1/lam)/2:.10f}")

# %% [markdown]
# **Interpretación física.** La amplitud de tunelamiento entre los pozos escala como
# $e^{-S_0/\hbar}$, con $S_0 = 4\sqrt2/3$ en estas unidades. Igual que en el caso térmico, el
# exponente es el valor mínimo de una acción y se calcula sin resolver la ecuación de
# movimiento: basta la integral $\int\sqrt{2V}\,dx$, la misma que aparece en la fórmula WKB.
# La diferencia está en qué controla el exponente. En el caso térmico es la **altura** de la
# barrera ($\Delta V$); en el cuántico es su **área efectiva** ($\int\sqrt{2V}\,dx$, que
# depende del ancho).

# %% [markdown]
# ---
# ## §7 Potenciales efectivos: por qué un camino termina en la cima y el otro cruza
#
# **Objetivo.** Entender, con la analogía mecánica, la diferencia cualitativa entre los dos
# caminos.
#
# **Resultado esperado.** $U_\mathrm{OM} = -\tfrac12 V'^2$ tiene cimas (máximos, con
# $U = 0$) en $-1$, $0$ y $+1$. $U_E = -V$ tiene cimas solo en $\pm1$, y en $0$ tiene un valle.
#
# **Cálculo.** Las dos ecuaciones de Euler-Lagrange tienen forma newtoniana,
# $\ddot x = -dU/dx$: $\ddot x = V'V'' = -\frac{d}{dx}\left(-\tfrac12V'^2\right)$ y
# $\ddot x = V' = -\frac{d}{dx}(-V)$.

# %%
U_om, U_E = om.potencial_efectivo_om(), ins.potencial_efectivo_euclideo()
mostrar(r"U_\mathrm{OM}", U_om)
mostrar("U_E", U_E)
print("−dU_OM/dx = V'V'':", es_cero(-sp.diff(U_om, x) - dV() * d2V()))
print("−dU_E/dx  = V'   :", es_cero(-sp.diff(U_E, x) - dV()))
for nombre, U in [("U_OM", U_om), ("U_E", U_E)]:
    criticos = sp.solve(sp.diff(U, x), x)
    clasificacion = {c: ("máximo (cima)" if sp.diff(U, x, 2).subs(x, c) < 0 else
                         "mínimo (valle)" if sp.diff(U, x, 2).subs(x, c) > 0 else "degenerado")
                     for c in criticos}
    print(f"{nombre}: {clasificacion}")

# %% [markdown]
# Los puntos críticos de $U_\mathrm{OM}$ son los ceros de $V'V''$. En los ceros de $V'$
# ($-1$, $0$, $+1$) se tiene $U_\mathrm{OM}'' = -(V''^2 + V'V''') = -V''^2 < 0$: son cimas, todas
# a la misma altura $U = 0$. En los ceros de $V''$ ($x = \pm1/\sqrt3$, donde $|V'|$ es máximo)
# hay valles. Para $U_E = -V$, las cimas son los mínimos de $V$ y el valle es la barrera.
# Verificación numérica con `minimize_scalar` sobre $-U$:

# %%
f_U_om = sp.lambdify(x, U_om, "numpy")
f_U_E = sp.lambdify(x, U_E, "numpy")
for nombre, f in [("U_OM", f_U_om), ("U_E", f_U_E)]:
    maximos = []
    for a, b in [(-1.3, -0.7), (-0.3, 0.3), (0.7, 1.3)]:
        r = minimize_scalar(lambda y: -f(y), bounds=(a, b), method="bounded",
                            options={"xatol": 1e-10})
        es_interior = a + 1e-6 < r.x < b - 1e-6
        maximos.append(f"{r.x:+.6f}" if es_interior else "—")
    print(f"máximos de {nombre} en (−1.3,−0.7), (−0.3,0.3), (0.7,1.3): {maximos}")

# %% [markdown]
# **Interpretación física.** Con energía cero, la partícula ficticia parte en reposo de una
# cima y solo puede detenerse (en tiempo infinito) en otra cima de la misma altura.
# - **Escape térmico** ($U_\mathrm{OM}$): partiendo de $-1$, la **primera** cima que
#   encuentra es $x=0$, y ahí se detiene. El camino óptimo termina en la barrera. Desde allí,
#   bajar hasta $+1$ no cuesta nada: es la relajación determinista, de acción cero. El costo
#   del escape está todo en la subida.
# - **Tunelamiento** ($U_E = -V$): la barrera es un **valle** de $-V$; la partícula lo
#   atraviesa sin detenerse y solo se detiene en la cima $+1$. El instantón cruza de pozo a
#   pozo.
#
# Es la diferencia física central: la activación térmica **sube** la barrera (y el exponente
# mide su altura), mientras que el tunelamiento la **atraviesa** (y el exponente mide su
# ancho y su altura a la vez).

# %% [markdown]
# ---
# ## §8 Puente Fokker-Planck → Schrödinger
#
# **Objetivo.** Mostrar que la dinámica de la probabilidad térmica es una ecuación de
# Schrödinger en tiempo imaginario, el puente formal entre las dos partes del taller.
#
# **Resultado esperado.** Con $p = e^{-V/2D}\,\psi$, la ecuación de Fokker-Planck se
# convierte en $\partial_t\psi = -H\psi$, con
# $H = -D\,\partial_x^2 + \frac{V'^2}{4D} - \frac{V''}{2}$; y $\psi_0 = e^{-V/2D}$ cumple
# $H\psi_0 = 0$.
#
# **Cálculo.** Fokker-Planck: $\partial_t p = \partial_x(V'p) + D\,\partial_x^2 p$. Con
# $p = e^{-V/2D}\psi$: $p' = e^{-V/2D}\left(\psi' - \frac{V'}{2D}\psi\right)$ y
# $p'' = e^{-V/2D}\left(\psi'' - \frac{V'}{D}\psi' + \left[\frac{V'^2}{4D^2} - \frac{V''}{2D}\right]\psi\right)$.
# Sustituyendo y dividiendo por $e^{-V/2D}$, los términos en $\psi'$ se cancelan:
# $\partial_t\psi = D\psi'' - \left(\frac{V'^2}{4D} - \frac{V''}{2}\right)\psi \equiv -H\psi$.

# %%
f = sp.Function("f")(x)
mostrar(r"H f", sp.collect(schrodinger.hamiltoniano_efectivo(f), f))
esperado = -D * sp.diff(f, x, 2) + (dV() ** 2 / (4 * D) - d2V() / 2) * f
print("H coincide con −D∂² + V'²/(4D) − V''/2:", es_cero(schrodinger.hamiltoniano_efectivo(f) - esperado))
mostrar(r"\psi_0", schrodinger.psi0())
mostrar(r"H\psi_0", sp.simplify(schrodinger.hamiltoniano_efectivo(schrodinger.psi0())))

# %% [markdown]
# **Verificación numérica (scipy).** Se discretiza $H$ en una malla con diferencias
# centradas (matriz tridiagonal simétrica) y se diagonaliza con `eigh_tridiagonal`. El
# autovalor más bajo debe ser $\approx 0$, y su autovector debe coincidir con $\psi_0$
# normalizado. Es independiente de sympy: solo usa $V'$ y $V''$ en numpy.

# %%
for D_val in [0.1, 0.25, 0.5]:
    malla = np.linspace(-2.5, 2.5, 4001)
    h = malla[1] - malla[0]
    W = ref.dV(malla) ** 2 / (4 * D_val) - ref.d2V(malla) / 2
    diagonal = 2 * D_val / h**2 + W
    fuera = -D_val / h**2 * np.ones(malla.size - 1)
    autovalores, autovectores = eigh_tridiagonal(diagonal, fuera, select="i", select_range=(0, 0))
    psi_num = np.abs(autovectores[:, 0])
    psi_exacta = np.exp(-ref.V(malla) / (2 * D_val))
    psi_exacta /= np.linalg.norm(psi_exacta)
    print(f"D = {D_val}:  E0 = {autovalores[0]:+.2e},  max |ψ_num − ψ0| = {np.max(np.abs(psi_num - psi_exacta)):.2e}")

# %% [markdown]
# **Interpretación física.** $H$ es un hamiltoniano de mecánica cuántica supersimétrica, con
# "superpotencial" $V'/(2\sqrt D)$. Su estado fundamental, de energía exactamente cero, es la
# raíz de la distribución de Boltzmann: el equilibrio térmico es el vacío de un problema
# cuántico. La evolución de Fokker-Planck es la de Schrödinger en tiempo imaginario, y la
# temperatura $D$ hace el papel de $\hbar$. Por eso el escape térmico y el tunelamiento se
# tratan con la misma maquinaria variacional: en ambos, un exponente $e^{-S/(\text{parámetro pequeño})}$
# está dominado por un camino de acción mínima.

# %% [markdown]
# ---
# ## §9 Predicción de Kramers (referencia para el bloque estocástico)
#
# **Objetivo.** Dejar la predicción cuantitativa del tiempo medio de escape que medirá el
# bloque estocástico.
#
# **Resultado esperado.**
# $\langle\tau_\mathrm{esc}\rangle \approx \frac{2\pi}{\sqrt{V''(-1)\,|V''(0)|}}\,e^{\Delta V/D}
#  = \frac{2\pi}{\sqrt{32}}\,e^{1/D}$, con prefactor $2\pi/\sqrt{32} \approx 1.1107$.
#
# **Cálculo.** El exponente es $S_\mathrm{min} = \Delta V/D$ (sección §4). El prefactor sale
# de aproximar como gaussianas las integrales de la fórmula exacta del tiempo medio de
# primer paso (abajo), centradas en el pozo (curvatura $V''(-1)$) y en la barrera
# (curvatura $|V''(0)|$).

# %%
mostrar(r"\langle\tau_\mathrm{esc}\rangle", kramers.tiempo_kramers())
print(f"prefactor 2π/√32 = {2*np.pi/np.sqrt(32):.6f}")

# %% [markdown]
# **Advertencias para el bloque estocástico.**
# 1. **Es una asintótica para $D \to 0$** ($D \ll \Delta V$). Las correcciones son de orden
#    $D/\Delta V$: en $D = 0.35$ y $D = 0.5$ **se esperan desviaciones** visibles respecto de
#    esta fórmula.
# 2. **El prefactor depende de qué se llame "escape".** La fórmula da el tiempo de
#    **transición** (cruzar la barrera y caer al otro pozo). El tiempo para **llegar por
#    primera vez a la cima** $x=0$ es asintóticamente **la mitad**: desde la cima, la
#    partícula cae a cada lado con probabilidad ½, y en promedio necesita dos llegadas a la
#    cima para completar una transición. **La definición que usará el bloque estocástico no
#    se decide en este hito.**
#
# **Verificación numérica (scipy).** En una dimensión, el tiempo medio de primer paso desde
# $x_0 = -1$ hasta un punto absorbente $b$ (con $-\infty$ reflejante) tiene una fórmula
# exacta:
# $$ T(b) = \frac1D\int_{-1}^{b} dy\; e^{V(y)/D}\int_{-\infty}^{y} dz\; e^{-V(z)/D}. $$
# Se evalúa con `quad` anidado (el límite inferior $-\infty$ se toma en $-3$, donde
# $e^{-V/D} < e^{-640}$) para $b = 0$ (llegar a la cima) y $b = +1$ (llegar al otro pozo), y
# se compara con Kramers.

# %%
def tiempo_primer_paso(D_val: float, b: float) -> float:
    interior = lambda y: quad(lambda z: np.exp(-ref.V(z) / D_val), -3.0, y,
                              epsabs=0, epsrel=1e-11, limit=200)[0]
    exterior, _ = quad(lambda y: np.exp(ref.V(y) / D_val) * interior(y), -1.0, b,
                       epsabs=0, epsrel=1e-10, limit=200)
    return exterior / D_val


print(f"{'D':>5} {'ΔV/D':>6} {'Kramers':>12} {'T(b=0) exacto':>14} {'T(b=+1) exacto':>15}"
      f" {'T(+1)/Kramers':>14} {'T(0)/T(+1)':>11}")
for d, exponente, tk in kramers.tabla_kramers(VALORES_D):
    T0, T1 = tiempo_primer_paso(d, 0.0), tiempo_primer_paso(d, 1.0)
    print(f"{d:>5} {exponente:>6.2f} {tk:>12.4f} {T0:>14.4f} {T1:>15.4f} {T1/tk:>14.4f} {T0/T1:>11.4f}")

# %% [markdown]
# **Interpretación física.** La tabla muestra tres cosas:
# - La predicción de Kramers se acerca a $T(+1)$, el tiempo exacto de transición, a medida
#   que $D$ disminuye. La diferencia relativa decrece con $D$, como corresponde a una
#   asintótica.
# - En $D = 0.35$ y $0.5$ la diferencia es grande: allí la barrera es comparable a la
#   energía térmica y la aproximación gaussiana del prefactor ya no vale.
# - El tiempo para llegar a la cima, $T(0)$, tiende a la mitad del tiempo de transición
#   cuando $D \to 0$.
#
# El dato que domina todo es la exponencial $e^{1/D}$: entre $D = 0.5$ y $D = 0.1$ el tiempo
# crece más de tres órdenes de magnitud, lo que define el costo computacional del bloque
# estocástico.

# %% [markdown]
# ---
# ## §10 Variante: Onsager-Machlup completo (solo derivación)
#
# **Objetivo.** Ver qué cambia si se conserva el término jacobiano del funcional de
# Onsager-Machlup (CLAUDE.md §10: por defecto **no** se usa).
#
# **Resultado esperado.** Con $L = \frac{(\dot x + V')^2}{4D} - \tfrac12V''$, la ecuación de
# Euler-Lagrange es $\ddot x = V'V'' - D\,V'''$.
#
# **Cálculo.** El término $-\tfrac12V''(x)$ no depende de $\dot x$; solo contribuye a
# $\partial L/\partial x$ con $-\tfrac12V'''$. La ecuación pasa a ser
# $\frac{\ddot x - V'V''}{2D} + \tfrac12V''' = 0$, es decir $\ddot x = V'V'' - D\,V'''$.

# %%
L_completo = om.lagrangiano_om(completo=True)
mostrar(r"L_\mathrm{OM}^\mathrm{completo}", L_completo)
xpp_completo = ecuacion_el(L_completo)
mostrar(r"\ddot x", xpp_completo)
print("ẍ − (V'V'' − D V''') = 0:", es_cero(xpp_completo - (dV() * d2V() - D * d3V())))
H_completo = integral_primera(L_completo)
mostrar(r"H^\mathrm{completo}", H_completo)
print("dH/dt = 0 sobre Euler-Lagrange:", es_cero(derivada_temporal(H_completo, xpp_completo)))

# %% [markdown]
# **Interpretación física.** La corrección $-D\,V''' = -24Dx$ es **proporcional al ruido**.
# Con el término jacobiano, el camino "más probable" depende de $D$; sin él (ruido débil) es
# universal. Para $D \to 0$ la corrección desaparece y se recupera el resultado de
# Freidlin-Wentzell, que es el que fija el exponente de Arrhenius. El término jacobiano
# afecta al prefactor, no al exponente. Esta variante queda solo derivada; no se usa en
# otros hitos salvo que se decida.

# %% [markdown]
# ---
# ## §11 Figuras
#
# Las tres figuras se generan con las funciones de `taller.analitico.referencias` (que la
# prueba del criterio 9 valida contra sympy) y se guardan en PDF y PNG en
# `figures/analitico/`. No hay simulación: son curvas analíticas.

# %%
AZUL, NARANJA = "#2a78d6", "#eb6834"  # escape térmico, instantón (paleta categórica fija)
TINTA, TINTA_SUAVE, REJILLA = "#1f1f1e", "#5f5e58", "#e4e3dc"

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
        fig.savefig(FIGURAS / f"{nombre}.{extension}", bbox_inches="tight")
    print("guardada:", FIGURAS.relative_to(RAIZ) / nombre, "(.pdf, .png)")


# %%
# Figura 1: el potencial de doble pozo.
xs_fig = np.linspace(-1.6, 1.6, 600)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(xs_fig, ref.V(xs_fig), color=TINTA)
ax.plot([-1, 1], [0, 0], "o", color=TINTA, ms=6)
ax.plot([0], [1], "o", color=TINTA, ms=6, mfc="white")
ax.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color=TINTA_SUAVE))
ax.text(0.06, 0.45, r"$\Delta V = 1$", color=TINTA)
ax.text(-1, -0.17, "metaestable\n" r"$V''=8$", ha="center", va="top", color=TINTA_SUAVE, fontsize=9)
ax.text(1, -0.17, "metaestable\n" r"$V''=8$", ha="center", va="top", color=TINTA_SUAVE, fontsize=9)
ax.text(0, 1.08, r"barrera, $V''=-4$", ha="center", color=TINTA_SUAVE, fontsize=9)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-0.55, 2.0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$V(x)$")
ax.set_title(r"Potencial de doble pozo $V(x) = (x^2-1)^2$")
guardar(fig, "potencial")
plt.show()

# %%
# Figura 2: camino de escape térmico y kink del instantón.
ss = np.linspace(-2.0, 2.0, 600)
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.axhline(0, color=TINTA_SUAVE, lw=0.8)
ax.plot(ss, ref.x_om(ss), color=AZUL, label=r"$x_\mathrm{om}(t) = -1/\sqrt{1+e^{8t}}$ (escape térmico)")
ax.plot(ss, ref.x_kink(ss), color=NARANJA, ls="--", label=r"$x_\mathrm{kink}(\tau) = \tanh(\sqrt{2}\,\tau)$ (instantón)")
ax.plot([0], [ref.x_om(0.0)], "o", color=AZUL, ms=6)
ax.plot([0], [0.0], "o", color=NARANJA, ms=6)
ax.text(1.0, -0.13, r"$x_\mathrm{om}\to 0$ (cima)", color=TINTA, fontsize=9)
ax.text(0.95, 0.62, r"$x_\mathrm{kink}\to +1$ (otro pozo)", color=TINTA, fontsize=9)
ax.text(0.1, ref.x_om(0.0) - 0.12, r"$x_\mathrm{om}(0)=-1/\sqrt{2}$", color=TINTA_SUAVE, fontsize=9)
ax.set_xlabel(r"$t$ (tiempo real)  /  $\tau$ (tiempo imaginario)")
ax.set_ylabel(r"$x$")
ax.set_ylim(-1.1, 1.1)
ax.set_title("Caminos de acción mínima desde el pozo $x=-1$")
ax.legend(loc="upper left", fontsize=8.5)
guardar(fig, "caminos")
plt.show()

# %%
# Figura 3: potenciales efectivos (dos paneles, sin doble eje).
xs_fig = np.linspace(-1.35, 1.35, 600)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)
ax1.plot(xs_fig, f_U_om(xs_fig), color=AZUL)
ax1.plot([-1, 0, 1], [0, 0, 0], "o", color=AZUL, ms=6)
ax1.set_title(r"Escape térmico: $U_\mathrm{OM} = -\frac{1}{2}V'^2$")
ax1.text(0, 0.12, "cimas en $-1$, $0$, $+1$", ha="center", color=TINTA, fontsize=9)
ax1.set_ylim(-2.4, 0.45)
ax2.plot(xs_fig, f_U_E(xs_fig), color=NARANJA, ls="--")
ax2.plot([-1, 1], [0, 0], "o", color=NARANJA, ms=6)
ax2.plot([0], [-1], "o", color=NARANJA, ms=6, mfc="white")
ax2.set_title(r"Instantón: $U_E = -V$")
ax2.text(0, -1.2, "valle en $0$", ha="center", color=TINTA, fontsize=9)
ax2.text(0, 0.1, "cimas en $\\pm1$", ha="center", color=TINTA, fontsize=9)
ax2.set_ylim(-1.35, 0.35)
for ax in (ax1, ax2):
    ax.set_xlabel(r"$x$")
ax1.set_ylabel(r"$U(x)$")
fig.suptitle("Mecánica ficticia con energía cero: de cima a cima", y=1.02)
fig.tight_layout()
guardar(fig, "potenciales_efectivos")
plt.show()

# %% [markdown]
# ---
# ## Resumen
#
# | Cantidad | Resultado (sympy) | Verificación numérica (scipy) |
# |---|---|---|
# | Puntos fijos, curvaturas | $\{-1,0,1\}$; $V''(\pm1)=8$, $V''(0)=-4$; $\Delta V=1$ | `brentq`, `derivative` |
# | Euler-Lagrange OM | $\ddot x = V'V''$ | `solve_ivp` de 2.º orden |
# | Camino de escape | $x_\mathrm{om} = -1/\sqrt{1+e^{8t}}$ | `solve_ivp`, error $< 10^{-8}$ |
# | Acción mínima térmica | $S_\mathrm{min} = \Delta V/D = 1/D$ | `quad`; familia $\lambda$ |
# | Euler-Lagrange euclídea | $\ddot x = V'$ | `solve_ivp` de 2.º orden |
# | Kink | $x_\mathrm{kink} = \tanh(\sqrt2\tau)$ | `solve_ivp`, error $< 10^{-8}$ |
# | Acción del instantón | $S_0 = 4\sqrt2/3$ | `quad`; familia $\lambda$ |
# | Schrödinger | $H = -D\partial^2 + V'^2/4D - V''/2$, $H\psi_0 = 0$ | `eigh_tridiagonal`: $E_0\approx0$ |
# | Kramers | $\langle\tau_\mathrm{esc}\rangle \approx (2\pi/\sqrt{32})\,e^{1/D}$ | MFPT exacto con `quad` |
# | OM completo | $\ddot x = V'V'' - D\,V'''$ | — (solo derivación) |
