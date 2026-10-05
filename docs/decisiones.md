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

## 2026-10-05 — Hito 00: aprobación

### D17. Hito 00 aprobado

- **Decisión:** los revisores aprueban el hito 00 y se etiqueta `hito-00`. Decisiones de la revisión:
  - **D15 se confirma:** las tres verificaciones extra del cuaderno (familias reescaladas λ en §4 y §6, diagonalización de H en §8, tiempo medio de primer paso exacto en §9) se mantienen, marcadas en el cuaderno como "verificaciones complementarias, fuera de la especificación".
  - **Tiempo medio de primer paso exacto:** queda como candidato a referencia del bloque estocástico. Si se incorpora a `referencias.py` se decide en el hito 01, no ahora. Anotado como pendiente en la bitácora.
- **Alternativas:** quitar las verificaciones extra del cuaderno; incorporar ya el tiempo de primer paso a `referencias.py`.
- **Justificación:** las verificaciones son independientes de sympy y no agregan código a `src/` ni pruebas. Incorporar una referencia nueva al módulo corresponde al hito que la usará.

## 2026-10-05 — Hito 00: corrección del cuaderno (`hito-00.1`)

### D18. Corrección de texto del cuaderno analítico después de la aprobación

- **Decisión:** se corrigen cuatro celdas markdown de `notebooks/00_analitico/00_analitico.py`. No cambian código, pruebas ni figuras:
  1. **§8:** se corrige el error conceptual "D hace el papel de ℏ, y por eso…". El potencial del problema cuántico es V′²/(4D) − V″/2, no V, así que el escape térmico en V no equivale al tunelamiento en V. La maquinaria común se debe a que en ambos casos el peso es e^{−S/parámetro pequeño}.
  2. **Inicio del cuaderno:** se agrega una tabla de correspondencia entre las secciones del cuaderno (§0 a §11) y las del plan del taller (§0 Problema físico, §1 Del ruido al funcional, §2 Camino más probable, §3 Instantón, §4 Comparación estructural).
  3. **§5:** "tiempo de tunelamiento" pasa a "anchura del kink en tiempo imaginario".
  4. **§3:** la duración 1/8 + 1/4 de la excursión se matiza como indicativa, con dependencia logarítmica en D.
- **Alternativas:** dejar el cuaderno como se aprobó y corregirlo en el hito 01; mover la etiqueta `hito-00`.
- **Justificación:** el error de §8 contradice un punto central del taller y conviene corregirlo antes de que el bloque estocástico y la parte teórica lo usen como referencia. Se crea la etiqueta nueva `hito-00.1` y `hito-00` se mantiene en el commit aprobado, para conservar la historia de la revisión.

## 2026-10-05 — Hito 01: dependencias y cómputo

### D19. El Bloque B se implementa en CPU con Numba, no en GPU

- **Decisión (de los revisores):** la simulación estocástica usa Numba en CPU (`@njit(parallel=True)` con `prange` sobre trayectorias), más una referencia en NumPy puro. No se usa GPU en este bloque.
- **Problema encontrado con la GPU.** Se instaló `numba-cuda[cu12]` 0.30.4, con CUDA 12.9: el extra `cu13` resolvía a `cuda-toolkit` 13.4, más nuevo que el CUDA 13.2 del driver 595.91.07. Con numpy 2.5.3, compilar cualquier kernel fallaba con `AttributeError: module 'numpy' has no attribute 'row_stack'`: numpy 2.5 eliminó `np.row_stack` y numba-cuda la sigue registrando.
  - NVIDIA puso numba-cuda en modo de mantenimiento el 2026-06-25 (issue NVIDIA/numba-cuda#902): solo corregirá fallos críticos y de seguridad durante la vida de CUDA 13.
  - Cerró como *not planned* el soporte de numpy 2.5 (issue #907, 2026-09-10) y recomienda migrar a `numba-cuda-mlir`.
- **Alternativas probadas** (en entornos temporales, fuera de FMA):
  - (a) numba-cuda 0.30.4 con numpy 2.4.6: funcionaba (kernel mínimo con error 0.0; Ornstein-Uhlenbeck con varianza 0.50007 frente a 0.5; 2.3e9 pasos/s en float64).
  - (b) numba-cuda-mlir 0.5.4 con numpy 2.5.3: el kernel mínimo funcionaba, pero `xoroshiro128p_normal_float64` no compilaba dentro de un kernel (`Untyped global name`) y la inicialización de estados corría en Python interpretado (8.4 s para 4096 estados).
- **Justificación:** un kernel CUDA, con una dependencia en modo de mantenimiento, cuesta más complejidad de la que aporta a este experimento. En CPU, cualquiera puede reproducirlo sin GPU NVIDIA. Numba en CPU con 16 hilos da 5.3e8 pasos/s en float64 (prueba de Ornstein-Uhlenbeck: media y varianza compatibles con los valores exactos de Euler-Maruyama a 0.63 errores estándar). Es suficiente para el hito.
- **Nota para el hito neuronal (Bloque A):** PyTorch trae su propio runtime de CUDA en sus *wheels* (no depende del de Numba ni del toolkit del sistema); solo exige un driver compatible. Allí se ofrecerá elegir el dispositivo `cuda`, `mps` o `cpu`, y la prueba de GPU de CLAUDE.md §12 se hará con `torch.cuda.is_available()` y el nombre del dispositivo.

### D20. numpy fijado en `>=2.4,<2.5` y regresión del hito 00

- **Decisión (de los revisores):** `numpy>=2.4,<2.5` en `pyproject.toml`, de forma preventiva: Numba suele ir atrasado respecto a numpy. uv resolvió numpy 2.4.6 (antes 2.5.3). El grupo `estocastico` queda con `numba` 0.68.0 (trae `llvmlite` 0.50.0) y `pyyaml` 6.0.3; PyYAML ya estaba instalado como dependencia transitiva de Jupyter. No cambió ninguna otra versión.
- **Regresión del hito 00 con numpy 2.4.6:**
  - `uv run pytest`: 70 passed.
  - El cuaderno del hito 00 se ejecuta sin errores desde una sesión limpia.
  - Los 262 números impresos por el cuaderno son idénticos como texto a los de la ejecución con numpy 2.5.3.
  - 38 valores clave recalculados a precisión completa son idénticos bit a bit en los dos entornos (diferencia relativa máxima 0): errores de `solve_ivp` de primer y segundo orden, `quad` de `S_min` y `S0`, familias λ, `E0` de `eigh_tridiagonal`, Kramers y tiempos de primer paso `T(0)` y `T(+1)`. La versión con numpy 2.5.3 se ejecutó en un entorno temporal con las versiones del lockfile anterior.
  - Las seis figuras regeneradas son idénticas byte a byte.

### D21. `default-groups = ["dev", "estocastico"]`

- **Decisión:** el grupo `estocastico` se instala por defecto.
- **Alternativa:** pasar `--group estocastico` en cada `uv sync` o `uv run`.
- **Justificación:** `uv sync` es exacto y, sin esto, desinstalaba Numba del entorno FMA (comprobado con `--dry-run`).

### D22. Hilos de Numba (reemplazada por D23)

- **Decisión (de los revisores):** 16 hilos por defecto (8 núcleos con hyperthreading), no la máquina completa (20 hilos). El valor va en la configuración; la bandera `--hilos` de los scripts tiene prioridad. Se aplica con `numba.set_num_threads`. Los metadatos de cada resultado guardan el número de hilos y la capa de hilos de Numba (en esta máquina, `tbb`, con la `libtbb.so.12` del sistema).

## 2026-10-05 — Hito 01: revisión del plan

### D23. 10 hilos por defecto (reemplaza a D22)

- **Decisión (de los revisores):** el valor por defecto de `configs/estocastico/comun.yaml` pasa de 16 a 10 hilos. La bandera `--hilos` sigue teniendo prioridad; se sigue aplicando con `numba.set_num_threads` y los metadatos siguen guardando hilos y capa de hilos.
- **Alternativas:** mantener 16; usar los 20 hilos de la máquina.
- **Justificación:** la máquina se ha apagado varias veces bajo carga y no se quiere exigirla de más. Rendimiento medido con 10 hilos (integrador de prueba, D = 0.25, dt = 1e-3, 4·10⁴ trayectorias): 1.9·10⁸ pasos/s sin la corrección de puente y 1.57·10⁸ con ella (con 16 hilos: 2.4·10⁸ sin corrección).

### D24. Corrección de puente browniano en los tiempos de primer paso

- **Problema:** en x = 0 la deriva se anula y Euler-Maruyama pierde cruces entre pasos. Con una simulación de prueba (D = 0.25, 2·10⁴ trayectorias), T(−1 → 0) sale sesgado en +3.9 % (dt = 1e-3) y +3.4 % (dt = 5e-4), con error estándar de 0.7 %; la estimación teórica es ≈ 0.93·√(2 dt), independiente de D. T(−1 → +1) no tenía sesgo medible. Ningún dt de la lista cumplía el 2 % de E3 en la cima.
- **Decisión (de los revisores, opción A del plan):** si x_n < b y x_{n+1} < b, se considera un cruce entre pasos con probabilidad exp(−(b − x_n)(b − x_{n+1})/(D dt)), asignado al final del paso. Se aplica a ambos destinos; se valida antes con el movimiento browniano sin deriva (solución exacta por el principio de reflexión); se guardan también los tiempos sin corregir. Los criterios de E3 y E4 se aplican a los tiempos corregidos.
- **Detalles de implementación (de Claude):** la trayectoria se integra hasta el último de estos instantes: llegada sin corregir a +1 y t_cima + 0.5 (fin de la ventana de E7), para registrar ambos tiempos y la ventana completa. La ventana y la alineación de E7 se refieren al t_cima corregido. La prueba E1 compara Numba y NumPy tanto en tiempos corregidos como sin corregir, así que la referencia NumPy implementa también la corrección.
- **Alternativas:** (B) aplicar E3 y E4 solo al pozo; (C) bajar dt a ≈ 1e-4, fuera de la lista de la especificación.
- **Justificación:** con la corrección, el sesgo desaparece incluso con dt = 1e-2 en la simulación de prueba (−1.4 % ± 0.7 %), lo que abarata E4 en un factor de 10 a 20 frente a dt = 1e-3 o 5e-4.

### D25. Criterios de E3, E5 y E7, y parámetros de E2 y E4

- **Decisión (de los revisores):** E3 con N = 10⁵ y monotonía exigida solo entre los dt con error > 2 errores estándar; E4 con N = 2·10⁴ en D = 0.1; E5 con ambos destinos, reportando las pendientes exactas 0.9884 (pozo) y 0.9900 (cima); ventana de E7 hasta t_cima + 0.5 (351 muestras) con el criterio RMS en t ∈ [−0.5, 0.25]; NaN y nanmedian para muestras anteriores a t = 0; E2 con 10⁵ trayectorias, equilibrado hasta t = 50, 20 muestras separadas 5.0 e intervalos de 0.05.
- **Justificación:** ver las aclaraciones (8)-(13) de `specs/01_estocastico.md`. En particular, en E7 x_om nunca alcanza la cima y cerca de ella el ruido domina; en E2 el tiempo de correlación con D = 0.5 es ~10.

### D26. División del trabajo y reanudación de las corridas

- **Decisión (de los revisores):** Claude escribe código, pruebas y scripts y ejecuta solo pruebas y validaciones pequeñas; los revisores ejecutan las corridas de producción, con una estimación de tiempo con 10 hilos para cada una. El script de producción guarda un archivo por D (E4) y por dt (E3) y salta los que ya existen, salvo con `--rehacer`.
- **Detalle de implementación (de Claude):** cada archivo se escribe primero con un nombre temporal y luego se renombra con `os.replace` (operación atómica en el mismo sistema de archivos). Así un apagado durante la escritura no deja un `.npz` truncado que la reanudación tomaría por terminado.
- **Justificación:** con un archivo por D, un apagado solo cuesta el D en curso.

### D27. Una semilla por trayectoria

- **Decisión (de Claude, propuesta en el plan del hito sin objeciones):** cada trayectoria recibe su propia semilla, derivada con `numpy.random.SeedSequence` de la semilla de configuración, y reinicia el generador de Numba al empezar. La referencia NumPy usa `default_rng` (PCG64), un generador distinto.
- **Alternativas:** un generador por hilo (no reproducible al cambiar el número de hilos).
- **Justificación:** el resultado es idéntico bit a bit con cualquier número de hilos, lo que importa ahora que el valor por defecto cambió y `--hilos` puede variar entre corridas. Usar generadores distintos en Numba y NumPy hace más independiente la comparación E1.
