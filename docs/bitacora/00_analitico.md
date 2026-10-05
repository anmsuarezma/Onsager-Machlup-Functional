# Hito 00 — Analítico

Especificación: `specs/00_analitico.md` (con sus "Aclaraciones aprobadas"). Fechas: 2026-10-04 y 2026-10-05.

## 0. Objetivo y resultado físico que verifica

Construir con sympy la referencia simbólica contra la que se contrastan los experimentos numéricos:
- el camino más probable de escape térmico (Onsager-Machlup en forma de Freidlin-Wentzell) y su acción mínima `S_min = ΔV/D`, que es el exponente de Arrhenius;
- el instantón del tunelamiento y su acción `S0 = 4√2/3`;
- el puente Fokker-Planck → Schrödinger;
- la predicción de Kramers para el bloque estocástico.

Cada resultado se deriva con sympy y se verifica dos veces: simbólicamente (residuo exactamente cero) y numéricamente con scipy.

## 1. Entorno

Registrado el 2026-10-04, al preparar el proyecto. Actualizado el mismo día al migrar al entorno compartido FMA (el `.venv` local inicial usaba el Python 3.12.3 del sistema y se eliminó).

| Elemento | Valor |
|---|---|
| Sistema operativo | Ubuntu 24.04.5 LTS, kernel 6.14.0-37-generic |
| Python | 3.12.13, gestionado por uv (`~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu`) |
| Entorno | FMA: `~/Documents/Mauricio/Doctorado Fisica/Fisica Matematica Avanzada/FMA`, vía `UV_PROJECT_ENVIRONMENT` |
| Kernel de Jupyter | `FMA` (`~/.local/share/jupyter/kernels/fma`) |
| uv | 0.10.11 |
| git | 2.43.0 |
| GPU | NVIDIA GeForce RTX 3060, 12 288 MiB |
| Driver NVIDIA | 595.91.07 |
| CUDA máxima soportada por el driver | 13.2 (según `nvidia-smi`; no hay toolkit de CUDA instalado por el proyecto) |

### 1.1 Dependencias resueltas por uv

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

El conjunto completo (109 paquetes instalados, incluido `taller`) está fijado en `uv.lock`.

Las versiones no cambiaron al sincronizar contra FMA: `uv sync` instaló exactamente lo fijado en `uv.lock`.

### 1.2 Pruebas del entorno

`tests/test_entorno.py`: 7 pruebas de importación, todas pasan, tanto en el `.venv` inicial como en FMA.

## 2. Qué se implementó

### 2.1 Módulo `src/taller/analitico/`

| Archivo | Funciones principales |
|---|---|
| `potencial.py` | Símbolos `x, v, t, tau` (reales) y `D` (positivo); `V, dV, d2V, d3V` (derivadas con `diff`); `puntos_fijos`, `estabilidad` (linealización de ẋ = −V′), `altura_barrera` (identifica mínimo y silla por estabilidad), `densidad_estacionaria`, `corriente_fp` |
| `variacional.py` | `ecuacion_el` (`euler_equations` + despeje de ẍ), `integral_primera` (H = v ∂L/∂v − L), `derivada_temporal`, `primitiva_real`, `resolver_separable` |
| `escape_termico.py` | `lagrangiano_om(completo)`, `ramas_energia_cero`, `camino_om`, `accion_om` (integral directa en t), `S_min` (vía derivada total), `identidad_cuadrados_om`, `tasas_om` (límites), `potencial_efectivo_om` |
| `instanton.py` | `lagrangiano_euclideo`, `rama_kink` (√(2V) con u = 1 − x² > 0), `camino_kink`, `S0`, `accion_euclidea`, `identidad_bogomolny`, `tasa_kink`, `frecuencia_pozo`, `potencial_efectivo_euclideo` |
| `schrodinger.py` | `operador_fp`, `hamiltoniano_efectivo` (por conjugación de L_FP con e^{−V/2D}; convención ∂ψ/∂t = −Hψ), `psi0` |
| `kramers.py` | `tiempo_kramers` (desde curvaturas y ΔV del módulo), `tabla_kramers` |
| `referencias.py` | Numpy escrito a mano, sin sympy (D10): `V, dV, d2V, x_om` (forma sin desbordamiento), `x_kink, S_min, S0, tau_kramers` |

Ningún resultado de la especificación está escrito en el módulo simbólico: las soluciones `x_om` y `x_kink` salen de resolver la ecuación separable, `S_min` y `S0` de integrar, y `H` de conjugar el operador de Fokker-Planck.

### 2.2 Pruebas `tests/analitico/` (congeladas en el commit `0dd7804`)

`oraculo.py` (expresiones esperadas escritas a mano, sin importar `taller`) y seis archivos de prueba: `test_euler_lagrange.py`, `test_soluciones.py`, `test_acciones.py`, `test_fokker_planck.py`, `test_referencias.py` y `test_potencial.py`.

### 2.3 Cuaderno `notebooks/00_analitico/00_analitico.py`

Tiene 11 secciones (§0 a §10) más la de figuras (§11). Cada una sigue la estructura objetivo → resultado esperado → cálculo (con los pasos a mano en LaTeX) → verificación con sympy y con scipy → interpretación física.

## 3. Pruebas

**Resultado:** `uv run pytest` da **70 passed in 7.47 s** (63 del hito más 7 del entorno). Ninguna falla. Los valores medidos, frente a los criterios:

| Criterio | Qué se mide | Obtenido | Criterio |
|---|---|---|---|
| 1 | ẍ de Euler-Lagrange (OM, euclídeo, OM completo) | residuo simbólico 0 | = 0 |
| 2 | Residuos de `x_om` y `x_kink` (primer orden y Euler-Lagrange) | 0 exacto | = 0 |
| 3 | dH/dt sobre Euler-Lagrange (los 3 lagrangianos) | 0 exacto | = 0 |
| 4 | `S_min·D`, `S0` en sympy | `1`, `4√2/3` exactos | exacto |
| 4 | `quad` de `S_min` (5 valores de D) | error relativo ≤ 2.2e-16 | < 1e-10 |
| 4 | `quad` de `S0` | error relativo 1.1e-16 | < 1e-10 |
| 5 | Identidades OM y Bogomolny | diferencia 0 | = 0 |
| 6 | `H` frente a la expresión esperada; `Hψ0` | 0; 0 | = 0 |
| 7 | `J[p_s]` | 0 | = 0 |
| 8 | `solve_ivp` (DOP853) frente a `x_om` en [−2, 2] | 1.795e-12 | < 1e-8 |
| 8 | `solve_ivp` (DOP853) frente a `x_kink` en [−2, 2] | 2.831e-12 | < 1e-8 |
| 9 | `V, dV, d2V` (absoluto) | 0.0 | < 1e-12 |
| 9 | `x_om` / `x_kink` (absoluto) | 1.1e-16 / 0.0 | < 1e-12 |
| 9 | `S0` (absoluto) | 2.2e-16 | < 1e-12 |
| 9 | `S_min` / `tau_kramers` (relativo) | 0.0 / 0.0 | < 1e-12 |
| 10 | Puntos fijos, `V″(±1)`, `V″(0)`, `ΔV` | {−1, 0, 1}, 8, −4, 1 | exacto |
| 10 | Tasas OM como límites | 8 y 4 | 8 y 4 |
| 10 | Tasa del kink | 2√2 | 2√2 |
| 10 | `tau_kramers` frente al oráculo; prefactor | relativo 0.0; 1.110721 | < 1e-12; ≈ 1.1107 |
| Acl. 9 | Identidades en la malla irregular (391 puntos, distancia mínima a los puntos fijos 0.0109) | 8.1e-15 (OM), 8.9e-16 (Bogomolny) | < 1e-12 |
| Acl. 9 | `Hψ0` por diferencias finitas, h = 1e-4 | máximo 5.4e-6 (D = 0.5) | < 1e-4 |
| Acl. 9 | Convergencia `r(1e-3)/r(1e-4)` | 94.3 a 104.6 | > 50 (orden h²) |

### 3.1 Verificaciones adicionales del cuaderno (no son criterios)

Son números reales de la ejecución:
- **§0:** raíces de V′ con `brentq` en {−1, 0, 1}; V″ numérica 8, −4, 8; `|J|/escala` ≤ 4.7e-11.
- **§1 y §5:** integración de las ecuaciones de **segundo orden** en [−1.5, 1.5]: error 1.85e-11 frente a `x_om` y 4.36e-12 frente a `x_kink`. H se mantiene en ≤ 1.9e-12 a lo largo de la órbita.
- **§4:** la familia reescalada `x_om(λt)` da `S·D = (λ+1)²/(4λ)` con 10 cifras (1.125, 1.0125, 1, 1.0125, 1.125). La cota se satura solo en λ = 1.
- **§6:** la familia `x_kink(λτ)` da `S_E/S0 = (λ + 1/λ)/2` con 10 cifras.
- **§8:** diagonalización de H discretizada (`eigh_tridiagonal`, 4001 puntos): `E0` = −1.6e-5, −6.6e-6 y −3.5e-6 para D = 0.1, 0.25 y 0.5 (cero dentro del error de discretización O(h²)). Autovector frente a ψ0: ≤ 1.5e-7.
- **§9:** tiempo medio de primer paso exacto (fórmula integral en una dimensión, `quad` anidado):

| D | Kramers | T(b=0) exacto | T(b=+1) exacto | T(+1)/Kramers | T(0)/T(+1) |
|---|---|---|---|---|---|
| 0.1 | 24465.25 | 12762.68 | 25527.09 | 1.0434 | 0.5000 |
| 0.15 | 872.77 | 467.18 | 935.93 | 1.0724 | 0.4992 |
| 0.25 | 60.64 | 33.95 | 69.26 | 1.1421 | 0.4901 |
| 0.35 | 19.34 | 11.02 | 23.26 | 1.2027 | 0.4738 |
| 0.5 | 8.21 | 4.61 | 10.26 | 1.2500 | 0.4490 |

Kramers se acerca al tiempo exacto de transición cuando D → 0 (desviación de 4 % en D = 0.1 y de 25 % en D = 0.5). El tiempo para llegar a la cima tiende a la mitad del de transición. Las dos cosas confirman las aclaraciones (7) y (8).

## 4. Problemas encontrados y cómo se resolvieron

1. **`dsolve` no resuelve las ecuaciones de primer orden con condición inicial.** Para ẋ = √2(1 − x²) con x(0) = 0 lanza `NotImplementedError: Initial conditions produced too many solutions for constants`. Se resolvió con separación de variables explícita (`resolver_separable`, decisión D14).
2. **La primitiva de sympy usa logaritmos complejos.** `integrate(1/V′)` devuelve `log(x)` y `log(x² − 1)`, que son complejos en (−1, 0). Se reemplaza `log(z)` por `log(|z|)` según el signo de z en el intervalo físico (difieren en una constante), y la primitiva se verifica derivándola.
3. **El oráculo dependía del módulo en cinco puntos** (revisión de los revisores):
   - la condición inicial del criterio 8;
   - el integrando de `quad` para `S_min` y para `S0`;
   - las tasas comparadas con `d2V` del módulo.
   Se corrigieron antes de congelar las pruebas, con `oraculo.py`.
4. **Viabilidad de las pruebas antes de congelarlas.** En el scratchpad, y con las formas de la especificación, se comprobó que:
   - sympy reduce los residuos a cero y calcula los límites;
   - los criterios numéricos se alcanzan con holgura (errores de ~1e-12 frente a 1e-8);
   - el residuo de diferencias finitas es de ~5e-6 con h = 1e-4. Con eso se fijó la tolerancia de 1e-4 y la exigencia de convergencia h².
5. **La sucesión de razón áurea pasa a 1.8e-4 del punto fijo −1.** Por eso la malla irregular excluye explícitamente un margen de 0.01 alrededor de −1, 0 y +1.
6. **Ejecución del cuaderno:**
   - un `f-string` con formato `>3` sobre un entero de sympy lanzó `TypeError`; se corrigió con `str()`;
   - en dos figuras había etiquetas solapadas con las curvas; se movieron;
   - un párrafo de §7 decía que la segunda derivada de U_OM se anula en −1, 0 y +1; era falso (vale −V″² < 0) y se corrigió antes de la ejecución final.
7. **Entorno:** `uv python install` dejó un enlace `~/.local/bin/python3.12` en el PATH. Se eliminó con autorización.
8. **Corrección posterior a la aprobación (2026-10-05, etiqueta `hito-00.1`, D18).** La revisión encontró un error conceptual y tres imprecisiones en celdas markdown del cuaderno. No se cambió código, pruebas ni figuras.
   - **§8, error conceptual.** El texto decía que "la temperatura D hace el papel de ℏ" y que "por eso" ambas partes usan la misma maquinaria variacional. Era incorrecto. D hace un papel análogo a ℏ, pero el potencial cuántico de H es V′²/(4D) − V″/2 y no V. El escape térmico en V no es el tunelamiento en V, y §7 lo muestra con U_OM frente a U_E. La razón real de la maquinaria común es otra: el peso e^{−S/parámetro pequeño} (D o ℏ) está dominado por el mínimo de la acción. Se conservó lo correcto: el estado fundamental de energía cero es la raíz de Boltzmann, con estructura supersimétrica y superpotencial V′/(2√D).
   - **Correspondencia con el plan.** Se agregó al inicio una tabla que relaciona las secciones §0 a §11 del cuaderno con las del plan del taller (§0 a §4).
   - **§5.** "Tiempo de tunelamiento" pasó a ser "anchura del kink en tiempo imaginario", con la aclaración de que el tiempo de tunelamiento en tiempo real es un concepto distinto y discutido.
   - **§3.** La escala 1/8 + 1/4 de la excursión se presenta como indicativa: la duración real depende logarítmicamente de D, según dónde se fijen el inicio y el fin de la excursión.

   Tras la corrección, el cuaderno se ejecutó desde una sesión limpia sin errores (12.2 s) y `uv run pytest` dio 70 passed en 7.50 s. Al ejecutarlo, los PDF de las figuras solo cambiaron en el metadato `CreationDate`, así que se restauraron las versiones del repositorio.

   **Continuación de D18 (2026-10-05, sin etiqueta nueva).** Dos ajustes de texto más, también en celdas markdown:
   - El objetivo de §8 ya no dice "el puente formal entre las dos partes del taller". Ahora dice que el mapeo conecta la dinámica térmica con un problema cuántico en el potencial V′²/(4D) − V″/2, no con el tunelamiento en V.
   - En la tabla de correspondencia, la fila de §9 (Kramers) pasa a "§0 Problema físico (respuesta esperada) y §6 Discusión (prefactores); referencia del Bloque B".

Todas las pruebas pasaron al primer intento después de implementar el módulo. No se modificó ninguna prueba ni tolerancia después del commit de congelación.

## 5. Figuras generadas

Todas están en `figures/analitico/`, en PDF y PNG. Se generan en §11 del cuaderno a partir de `referencias.py`, sin simulación.

| Archivo | Qué muestra |
|---|---|
| `potencial.{pdf,png}` | V(x) con los mínimos metaestables (V″ = 8), la barrera (V″ = −4) y ΔV = 1 |
| `caminos.{pdf,png}` | `x_om(t)` (azul, continua) termina en la cima 0; `x_kink(τ)` (naranja, discontinua) cruza hasta +1. Están marcados los orígenes x_om(0) = −1/√2 y x_kink(0) = 0 |
| `potenciales_efectivos.{pdf,png}` | Dos paneles: U_OM = −½V′², con cimas en −1, 0, +1; U_E = −V, con cimas en ±1 y valle en 0. Explica por qué un camino se detiene en la barrera y el otro la cruza |

Colores: los dos primeros de la paleta categórica de referencia, más el estilo de línea como codificación secundaria (D16).

**Ejecución del cuaderno:** `uv run jupytext --to ipynb --execute notebooks/00_analitico/00_analitico.py` desde una sesión limpia termina sin errores en 12.3 s. El `.ipynb` resultante no se versiona.

## 6. Pendientes y dudas para la revisión

- **Verificaciones más allá de la especificación** (D15): la tabla de tiempo medio de primer paso exacto (§9), la diagonalización de H (§8) y las familias reescaladas λ (§4 y §6) no estaban pedidas. **Resuelto en la revisión (2026-10-05):** se mantienen las tres, marcadas en el cuaderno como "verificaciones complementarias, fuera de la especificación" (D17).
- **Pendiente para el hito 01:** el tiempo medio de primer paso exacto queda como candidato a referencia del bloque estocástico (sin aproximación asintótica, para ambas definiciones de "escape"; mejor punto de comparación que Kramers en D = 0.35 y 0.5). Si se incorpora a `referencias.py` se decidirá en el hito 01, no ahora.
- **Posible verificación futura (no implementada):** el primer autovalor excitado de H, λ₁, debería aproximarse a 2/⟨τ_esc⟩ (relajación entre los dos pozos). Conecta §8 con §9, pero no se pidió.
- **Paleta:** el validador de paleta de la guía de visualización requiere `node`, que no está instalado. Se usaron colores de una paleta ya validada más codificación secundaria; no se validó en esta máquina.
- **Pendiente de hitos futuros:** PyTorch y Numba traen su propio runtime de CUDA (*wheels* `nvidia-*` o `cuda-*`). En sus hitos habrá que verificar que ese runtime es compatible con el driver 595.91.07 (CUDA máxima 13.2) y comprobar explícitamente que usan la GPU (CLAUDE.md §12).
