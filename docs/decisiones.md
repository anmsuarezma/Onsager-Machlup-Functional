# Registro de decisiones

Cada entrada: fecha, decisión, alternativas consideradas y justificación.

---

## 2026-10-04 — Hito 00: preparación del entorno

### D1. Python 3.12

- **Alternativas:** 3.11 (mínimo del CLAUDE.md), 3.13 (instalada por uv en la máquina).
- **Justificación:** es la versión con soporte más maduro para las dependencias de hitos futuros (PyTorch, Numba con CUDA, CuPy). Fijarla desde ahora evita migrar a mitad del taller. Se fija con `.python-version` y `requires-python = ">=3.12"`. uv usa el intérprete del sistema (3.12.3).

### D2. Backend de construcción `uv_build` con layout `src/`

- **Alternativas:** `hatchling`, `setuptools`.
- **Justificación:** viene con uv, sin configuración extra. Con `name = "taller"` encuentra `src/taller/` por convención. uv instala el paquete en modo editable, así `import taller` funciona en pruebas y cuadernos sin tocar `sys.path`.

### D3. Dependencias del proyecto frente a las de desarrollo

- **Proyecto:** `sympy` (derivación simbólica), `numpy` (evaluación numérica de expresiones simbólicas), `scipy` (verificación numérica independiente de sympy: cuadraturas, EDO; se reutilizará en el bloque estocástico), `matplotlib` (figuras).
- **Desarrollo (grupo `dev`):** `pytest` (pruebas), `jupytext` (emparejar cuadernos con `.py`), `jupyterlab` (interfaz de cuadernos para quien clone el repositorio sin VS Code), `ipykernel` (núcleo de Python para los cuadernos).
- **Alternativa descartada:** prescindir de `scipy` y verificar solo con `mpmath`. Se mantiene `scipy` por ser una implementación independiente de sympy.
- **Excluidos por ahora:** PyTorch, Numba y librerías de GPU; llegan en sus hitos (CLAUDE.md §12).

### D4. Cuadernos: se versiona solo el `.py` de jupytext

- **Alternativas:** versionar `.ipynb` sin salidas (con `nbstripout`) junto al `.py`.
- **Justificación:** un `.ipynb` sin salidas no aporta nada en GitHub y duplica el `.py`. Lo visible del proyecto serán las figuras versionadas en `figures/`. Los `.ipynb` van al `.gitignore` y `nbstripout` se elimina del plan. Se ajustó en consecuencia la línea de cuadernos de la §6 del CLAUDE.md (cambio autorizado).

### D5. Carpetas creadas de forma incremental

- **Creadas en el hito 00:** `src/taller/analitico/`, `tests/analitico/`, `notebooks/00_analitico/`, `figures/analitico/`, `specs/`, `docs/bitacora/`, `docs/teoria/`, `docs/plan/`.
- **Pospuestas:** `configs/` y `results/` (el hito 00 no tiene aleatoriedad ni resultados crudos), y todo lo de `estocastico` y `neuronal`.
- **Justificación:** CLAUDE.md §3, crear solo lo que pide el hito en curso.

### D6. Prueba de humo del entorno

- `tests/test_entorno.py` solo verifica que `taller`, `taller.analitico` y las dependencias se importan. No es una prueba de física. Vive en `tests/` (no en `tests/analitico/`) porque concierne al entorno, no a un experimento.

### D7. Entorno compartido del semestre: FMA

- **Decisión:** el entorno de Python vive fuera del repositorio, en `~/Documents/Mauricio/Doctorado Fisica/Fisica Matematica Avanzada/FMA`, creado con `uv venv --managed-python --python 3.12` (CPython 3.12.13 gestionado por uv). El proyecto lo usa mediante la variable `UV_PROJECT_ENVIRONMENT`. Se registra como kernel de Jupyter `FMA`. Se eliminó el `.venv` local.
- **Alternativas:** `.venv` local por repositorio (lo que había antes); entorno con el Python del sistema.
- **Justificación:** desde FMA se harán otros experimentos del curso durante el semestre, así que conviene un único entorno con un kernel conocido para VS Code y JupyterLab. El Python gestionado por uv no cambia con las actualizaciones de Ubuntu. `pyproject.toml` y `uv.lock` siguen siendo la fuente de verdad de las versiones.
- **Consecuencias a vigilar:**
  - Sin `UV_PROJECT_ENVIRONMENT` definido, `uv run` o `uv sync` recrean silenciosamente un `.venv` local.
  - `uv sync` es exacto por defecto: desinstala de FMA todo paquete que no esté en `uv.lock`. Lo que otros experimentos del semestre instalen a mano en FMA se borrará en el siguiente `uv sync` de este repositorio, salvo que se use `uv sync --inexact`.

### D8. Cómo se define `UV_PROJECT_ENVIRONMENT`

- **Decisión:**
  - Un `.envrc` de direnv en la carpeta del curso (`~/Documents/Mauricio/Doctorado Fisica/Fisica Matematica Avanzada/.envrc`), fuera del repositorio, autorizado con `direnv allow`. El hook de direnv ya estaba en `~/.bashrc`, así que ese archivo no se modificó.
  - Además, la misma variable en la sección `env` de `.claude/settings.local.json`, porque los comandos de shell de Claude Code no pasan por el hook de direnv.
  - `.claude/settings.local.json` se agregó al `.gitignore` del repositorio. Ya lo ignoraba el gitignore global del usuario, pero no conviene depender de eso.
- **Alternativas:**
  - `.envrc` en la raíz del repositorio: la ruta personal terminaría en GitHub y no la heredarían otros experimentos del semestre.
  - `export` en `~/.bashrc`: afectaría a todos los proyectos con uv de la máquina.
- **Justificación:** la variable se hereda en toda la carpeta del curso (este taller y los experimentos futuros) sin versionar rutas personales. Quien clone el repositorio sin la variable obtiene un `.venv` local con las mismas versiones de `uv.lock`, y eso es lo esperado.

### D9. Las pruebas son el oráculo independiente

- **Decisión:** las pruebas de `tests/` escriben explícitamente los valores y expresiones esperados de las especificaciones (`4*sqrt(2)/3`, `V′V″`, `−1/√(1+e^{8t})`…), sin importarlos de `src/`. La regla de "no copiar valores de referencia" del CLAUDE.md §2 se refiere al código fuente; se aclaró allí con una línea (cambio autorizado).
- **Alternativas:** que las pruebas importen el valor esperado del propio módulo `analitico/`.
- **Justificación:** una prueba que compara un valor del módulo con otro valor del mismo módulo no verifica nada. El valor escrito en la prueba es el criterio de aceptación de la especificación.

### D10. `referencias.py` escrito a mano en numpy

- **Decisión:** las versiones numéricas de referencia se escriben a mano en numpy, sin `lambdify` y sin importar sympy.
- **Alternativa:** generarlas con `lambdify` desde las expresiones simbólicas.
- **Justificación:** el criterio 9 compara entonces dos implementaciones independientes (sympy y numpy a mano). Con `lambdify` se compararía consigo mismo. Además, los bloques estocástico y neuronal importan `referencias.py` sin cargar sympy.

### D11. Interfaz del módulo analítico fijada por las pruebas

- **Decisión:** las pruebas del hito 00 se escriben antes del módulo y fijan su interfaz pública: expresiones en los símbolos `x` (posición), `v` (velocidad `ẋ`), `t`, `tau` (reales) y `D` (positivo), exportados por `potencial.py`. Los lagrangianos e integrales primeras se expresan en `(x, v)`. `ecuacion_el(L)` devuelve `ẍ` como función de `(x, v)`. El operador `hamiltoniano_efectivo(psi)` actúa sobre una expresión en `x`, con la convención `∂ψ/∂t = −Hψ`.
- **Alternativa:** expresar todo con `Function('x')(t)`, como lo maneja internamente `euler_equations`.
- **Justificación:** con símbolos planos las pruebas escriben el oráculo de forma directa (por ejemplo `4*x*(x**2 - 1)*(12*x**2 - 4)`) y no dependen de la representación interna de sympy. La conversión a `Function` queda como detalle interno de `variacional.py`.

## 2026-10-05 — Hito 00: congelación de las pruebas

### D12. Verificación numérica complementaria en malla determinista irregular

- **Decisión:** la verificación numérica de las identidades y de `Hψ0 = 0` (aclaración 9) usa una malla determinista `x_k = a + (b−a)·{k·φ mod 1}`, con `φ` la razón áurea (y `√2 − 1` para `v`). Se eliminan los puntos a menos de 0.01 de −1, 0 y +1. `Hψ0` se evalúa por diferencias centradas y se exige residuo relativo < 1e-4 con h = 1e-4 y convergencia como h² (factor > 50 entre h = 1e-3 y h = 1e-4).
- **Alternativas:**
  - (b) puntos aleatorios con semilla en `configs/`;
  - una malla uniforme (`linspace`), que pasa por 0 y ±1 o queda simétrica alrededor de ellos.
- **Justificación:** sin aleatoriedad no hace falta semilla ni crear `configs/` en este hito (CLAUDE.md §6). En los puntos fijos `V′ = 0`, y las identidades se cumplirían trivialmente. Una sucesión de baja discrepancia cubre el intervalo con espaciado irregular. Las tolerancias de diferencias finitas se fijaron tras medir el residuo en el scratchpad (~5e-6 con h = 1e-4).

### D13. Oráculo de las pruebas en `tests/analitico/oraculo.py`

- **Decisión:** las expresiones esperadas (`V`, `V′`, `V″`, `V‴`, soluciones cerradas, condiciones iniciales, lados derechos de las EDO, `S0`, `S_min`, `⟨τ_esc⟩`) y los símbolos se escriben a mano en `oraculo.py`, que no importa nada de `taller`. Las pruebas solo importan del módulo la cantidad que verifican.
- **Alternativa:** `conftest.py`, sugerido por los revisores.
- **Justificación:** pytest desaconseja importar desde `conftest.py`, que está pensado para *fixtures*; un módulo normal se importa sin ambigüedad. El criterio 9 es la única excepción deliberada: compara sympy contra numpy, ambos del módulo (D10).

## 2026-10-05 — Hito 00: implementación

### D14. Soluciones cerradas por separación de variables con primitiva real verificada

- **Decisión:** `x_om` y `x_kink` se obtienen con `resolver_separable`: s = ∫ dy / rhs(y) desde el origen fijado. La primitiva de sympy (fracciones simples, con `log` complejos) se convierte en real reemplazando `log(z)` por `log(|z|)` según el signo de z en el intervalo físico, y se verifica derivándola. Entre las raíces de la ecuación implícita se elige la que cumple la condición inicial.
- **Alternativas:** `dsolve` con `ics` (falla: `NotImplementedError: Initial conditions produced too many solutions for constants`); escribir a mano la primitiva (sería copiar el resultado).
- **Justificación:** el resultado se deriva, no se copia, y cada paso se verifica.

### D15. Verificaciones numéricas adicionales en el cuaderno

- **Decisión:** además de lo pedido, el cuaderno incluye:
  - las familias reescaladas `x(λt)`, que ilustran que la cota de acción se satura solo en el minimizador;
  - la diagonalización de H discretizado (`eigh_tridiagonal`) como verificación de `Hψ0 = 0`;
  - el tiempo medio de primer paso exacto en una dimensión (`quad` anidado) como verificación de Kramers y de la razón ½ entre llegar a la cima y transitar.
- **Alternativa:** limitarse a `quad` y `solve_ivp`.
- **Justificación:** la especificación pide contrastar cada resultado simbólico con una verificación numérica independiente con scipy. Para `Hψ0` y Kramers, estas son las verificaciones naturales. No se agregan pruebas ni código a `src/`: viven solo en el cuaderno. Se señalan en la bitácora por si los revisores prefieren quitarlas.

### D16. Estilo de las figuras

- **Decisión:** azul `#2a78d6` para el escape térmico (línea continua) y naranja `#eb6834` para el instantón (línea discontinua), los dos primeros colores de una paleta categórica de referencia ya validada para daltonismo. El estilo de línea y las etiquetas directas son codificación secundaria. Los potenciales efectivos van en dos paneles, nunca con doble eje Y. Texto en tinta neutra, rejilla tenue.
- **Alternativa:** los colores por defecto de matplotlib.
- **Justificación:** se mantiene la identidad de cada experimento en todas las figuras del proyecto. El validador de paleta no se pudo ejecutar (no hay `node`).
