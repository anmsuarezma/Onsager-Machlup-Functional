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
- **Justificación:** ver las aclaraciones (8)-(13) de `specs/01_estocastico.md`. En particular, en E7 x_om nunca alcanza la cima y cerca de ella el ruido domina; en [−0.5, 0.25] se cumple x_om ≤ x_om(0.25) ≈ −0.345 (la primera versión de la aclaración decía −0.35, valor que x_om alcanza en t ≈ 0.246; corregido en la revisión de las pruebas, el intervalo no cambia). En E2 el tiempo de correlación con D = 0.5 es ~10.

### D26. División del trabajo y reanudación de las corridas

- **Decisión (de los revisores):** Claude escribe código, pruebas y scripts y ejecuta solo pruebas y validaciones pequeñas; los revisores ejecutan las corridas de producción, con una estimación de tiempo con 10 hilos para cada una. El script de producción guarda un archivo por D (E4) y por dt (E3) y salta los que ya existen, salvo con `--rehacer`.
- **Detalle de implementación (de Claude):** cada archivo se escribe primero con un nombre temporal y luego se renombra con `os.replace` (operación atómica en el mismo sistema de archivos). Así un apagado durante la escritura no deja un `.npz` truncado que la reanudación tomaría por terminado.
- **Justificación:** con un archivo por D, un apagado solo cuesta el D en curso.

### D27. Una semilla por trayectoria

- **Decisión (de Claude, propuesta en el plan del hito sin objeciones):** cada trayectoria recibe su propia semilla, derivada con `numpy.random.SeedSequence` de la semilla de configuración, y reinicia el generador de Numba al empezar. La referencia NumPy usa `default_rng` (PCG64), un generador distinto.
- **Alternativas:** un generador por hilo (no reproducible al cambiar el número de hilos).
- **Justificación:** el resultado es idéntico bit a bit con cualquier número de hilos, lo que importa ahora que el valor por defecto cambió y `--hilos` puede variar entre corridas. Usar generadores distintos en Numba y NumPy hace más independiente la comparación E1.

## 2026-10-05 — Hito 01: implementación

### D28. Generador xoshiro256** propio en Numba, con semillas de 64 bits

- **Problema:** el plan (D27) reiniciaba el generador de Numba (`np.random.seed`, MT19937) con la semilla de cada trayectoria, pero ese generador solo acepta semillas de 32 bits. Con 10⁵ trayectorias se esperan ~1.2 colisiones (N²/2³³): trayectorias idénticas. Se detectó al implementar, antes de usarlo.
- **Decisión (de Claude):** `semillas_trayectorias` devuelve semillas de 64 bits (`SeedSequence.generate_state(N, uint64)`, con detección de colisiones), y el integrador usa un generador propio por trayectoria: xoshiro256** iniciado con splitmix64 (el método que recomiendan sus autores), con normales por el método polar de Marsaglia. La interfaz y las pruebas congeladas no cambian. La referencia NumPy sigue con PCG64.
- **Validación antes de usarlo (scratchpad):** los bits de splitmix64 y xoshiro256** coinciden con una implementación independiente en Python puro (3 semillas, 1000 salidas), y splitmix64(0) da el valor publicado 0xe220a8397b1dcdaf. Con 2·10⁶ normales: media +0.00018 (EE 0.00071), varianza 0.99938, KS contra N(0, 1) con p = 0.70, correlación consecutiva −0.0003. Entre 200 corrientes, la correlación máxima es |r| = 0.064, la esperada por azar.
- **Alternativas:** MT19937 de Numba con semillas de 32 bits (colisiones); un generador por hilo (no reproducible al cambiar los hilos); `np.random.Generator` dentro de Numba (un objeto por trayectoria no cabe en un prange).
- **Efecto en el rendimiento (10 hilos, capa tbb):** 3.9·10⁸ pasos/s sin ventanas y 3.25·10⁸ con ventanas, frente a 1.6·10⁸ del integrador de prueba con MT19937 (D23). Las estimaciones de tiempo se rehacen con estos valores.

### D29. Metadatos: `arbol_modificado` cuenta los archivos no versionados

- **Decisión (de Claude):** `arbol_modificado` usa `git status --porcelain` completo. La primera versión ignoraba los archivos no versionados y daba `False` con todo el código nuevo sin commit (detectado en la prueba de humo de `correr.py`). `correr.py` avisa al empezar si el árbol está modificado. Los temporales de la escritura atómica (`results/**/.*.tmp`) se ignoran en git, para que un apagado no marque el árbol como modificado.

## 2026-10-05 — Hito 01: resultados de E3 y dt de producción

### D30. dt de producción = 1e-3, no el dt de la regla literal de E3

- **Resultados de E3** (D = 0.25, N = 10⁵ por dt, commit 2f273cf, 10 hilos; error relativo de la media frente a T exacto ± error estándar, en %):

  | dt | cima corregida | pozo corregido | cima sin corregir | pozo sin corregir |
  |---|---|---|---|---|
  | 1e-2 | −1.29 ± 0.31 | −0.54 ± 0.31 | +11.36 ± 0.35 | −0.39 ± 0.31 |
  | 5e-3 | −0.33 ± 0.31 | −0.11 ± 0.31 | +8.74 ± 0.34 | +0.02 ± 0.31 |
  | 1e-3 | −0.10 ± 0.31 | +0.08 ± 0.31 | +3.71 ± 0.32 | +0.12 ± 0.31 |
  | 5e-4 | −0.13 ± 0.31 | +0.18 ± 0.31 | +2.56 ± 0.32 | +0.23 ± 0.31 |

  Las pruebas de E3 que no dependen de E4 pasan: los cuatro archivos están completos, y el error decrece con dt en ambos destinos (aclaración 8).
- **Sesgo sin corregir frente a la corrección de continuidad para barreras discretas:** la barrera efectiva se desplaza δ = 0.5826·√(2D dt) (0.5826 = −ζ(1/2)/√(2π)), y el sesgo predicho es T(−1 → δ)/T(−1 → 0) − 1, evaluado con `tiempo_primer_paso`. Predicho: +12.5 %, +8.9 %, +4.0 % y +2.8 %. Medido: +11.4 %, +8.7 %, +3.7 % y +2.6 %. Escala como √dt. En el mensaje de los revisores el primer valor predicho aparece como +12.6 %; con la evaluación exacta da +12.5 %.
- **Decisión (de los revisores):** el dt de producción de E4 es 1e-3. La regla original de la especificación (mayor dt con error < 2 % en ambos destinos) elegía 1e-2, porque los cuatro dt la cumplen. La regla se precisó después (aclaración 15, D31): basta con que el dt de E4 esté validado por E3.
- **Justificación (de los revisores, corregida tras la investigación de los 12 lotes):** E3 establece un sesgo de orden dt en los tiempos corregidos: −1.29 % ± 0.31 % en la cima con dt = 1e-2 (4.2 errores estándar) y −0.63 % ± 0.11 % con dt = 5e-3, según los 12 lotes independientes de abajo. E3 no establece ni su mecanismo ni su dependencia con D. La elección de dt = 1e-3 es conservadora: deja el sesgo en alrededor de −0.1 %, muy por debajo de la tolerancia de E4 (3 %), sea cual sea su dependencia con D. E4 cuesta ~47 min.
- **Justificación original (reemplazada):** atribuía el sesgo al calentamiento numérico de Euler-Maruyama y suponía que crecería al disminuir D a través de e^{ΔV/D}. Se reemplazó porque E3 no permite afirmar ni el mecanismo ni la dependencia con D (ver la observación de abajo).
- **Alternativas:** dt = 1e-2 (regla literal, ~5 min); dt = 5e-3 (~9 min).
- **Investigación de la discrepancia con la validación del puente (de Claude, sin cambiar nada):** la validación (`test_doble_pozo_dt_grueso`, D = 0.25, dt = 5e-3, N = 10⁵) dio −1.02 % ± 0.31 % en el pozo corregido, y E3 con los mismos D, dt y N dio −0.11 % ± 0.31 %.
  - **Causa: semillas distintas, nada más.** Ambas corridas usan el mismo N y el mismo generador: la validación se ejecutó con el integrador de D28, no con el generador anterior. La validación usa la semilla 707 de `pruebas.yaml`; E3 usa la semilla derivada de 20261006. Reproducir E3 con el código actual da un resultado idéntico bit a bit al archivo.
  - Con 12 lotes adicionales independientes (semillas 1000-1011, N = 10⁵ cada uno), el pozo corregido promedia **−0.48 % ± 0.07 %** (desviación entre lotes 0.23 %) y la cima corregida −0.63 % ± 0.11 %. Hay un sesgo real de ≈ −0.5 % con dt = 5e-3, que una sola corrida de 10⁵ (error estándar 0.31 %) no resuelve. La validación cayó 1.7 errores estándar por debajo y E3 1.2 por encima: dos fluctuaciones de signo opuesto alrededor del mismo sesgo. La diferencia entre ambas, 0.91 %, son 2.1 errores estándar combinados (√2·0.31 %), no 3.
  - Este sesgo de ≈ −0.5 % con dt = 5e-3 es coherente con un sesgo de orden dt (≈ −1 % con 1e-2, ≈ −0.1 % con 1e-3), y apoya la elección de dt = 1e-3.
  - **Corrección del cálculo de la discrepancia:** en la revisión se estimó una diferencia de unos 3 errores estándar, comparándola con el error estándar de una sola corrida (0.91 %/0.31 % ≈ 2.9). Como ambas corridas fluctúan, el error estándar de la diferencia es √2·0.31 % = 0.44 %, y la discrepancia es de unos 2 errores estándar (2.1).
- **Observación (de Claude):** la estimación ingenua del calentamiento dentro del pozo (en un pozo armónico con V″ = 8, Euler-Maruyama equilibra a D_eff = D/(1 − 4 dt)) predice ≈ −8 % con dt = 5e-3 y D = 0.25, mucho más que el ≈ −0.5 % medido. El mecanismo y su dependencia con D no quedan establecidos por E3; E4 con dt = 1e-3 lo mostrará en todos los D.
- **Conflicto con una prueba congelada:** `test_e3_dt_de_produccion` exigía el mayor dt válido según la regla literal (1e-2) y habría fallado con dt = 1e-3. Resuelto en D31.

### D31. Modificación autorizada de una prueba congelada: `test_e3_dt_de_produccion`

- **Qué se cambió:** `tests/estocastico/test_criterios_produccion.py::test_e3_dt_de_produccion`, congelada en el commit 2b3de27. Antes exigía que el dt de los metadatos de E4 fuera **igual al mayor** dt de E3 con error corregido < 2 % en ambos destinos. Ahora exige que ese dt (a) sea uno de los dt evaluados en E3 y (b) cumpla en E3 |error corregido| < 2 % en ambos destinos. Sigue fallando si ningún dt cumple el 2 %.
- **Autorización:** los revisores la autorizaron explícitamente el 2026-10-05, después de E3 y antes de lanzar E4. Es la única prueba congelada que se ha modificado en este hito. Ninguna tolerancia cambió.
- **Por qué la regla original era incompleta:** el propósito del criterio es garantizar que E4 use un dt validado por E3, no forzar el mayor. La regla "el mayor dt con error < 2 %" no permitía una elección conservadora: un dt menor, también validado, con menos sesgo de discretización (D30). Elegir un dt menor que el máximo válido está permitido.
- **Alternativas:** mantener la prueba y usar dt = 1e-2; mantenerla y aceptar el fallo documentado.
- **Especificación:** la regla se precisa en la aclaración (15) de `specs/01_estocastico.md`. El texto original de E3 queda sin modificar.

## 2026-10-05 — Hito 01: trazabilidad de E4 y datos para el cuaderno

### D32. Los registros `*.log` se ignoran en git; E1 y E2 guardan resultados para las figuras

- **Problema:** E4 se generó con `arbol_modificado = True`. Investigado: el único cambio era `log_E4.log`, el registro de consola de la propia corrida, creado sin versionar en la raíz antes de que arrancara Python. Ningún archivo versionado se modificó después de 4a8ce76. Tres archivos de E4 (D = 0.5, 0.35 y 0.25, este con ventanas) se regeneraron con el código de HEAD y son idénticos bit a bit, aunque la corrida original usó 14 hilos (D27). Los resultados son válidos (detalle en la bitácora).
- **Decisión (de Claude):** `*.log` se agrega a `.gitignore`, para que un registro de consola no marque el árbol como modificado. Alternativa: versionar los registros en `results/`; no se elige porque los metadatos de cada `.npz` ya guardan lo necesario.
- **E1 y E2 con datos guardados:** las pruebas lentas no escriben en `results/`. Para que las figuras del cuaderno salgan de resultados guardados, `correr.py` corre ahora también E1 (`configs/estocastico/e1_validacion.yaml`) y E2 (`e2_boltzmann.yaml`), con semillas propias distintas de las de `pruebas.yaml`: son réplicas independientes de las pruebas, no los mismos números.

## 2026-10-05 — Hito 01: revisión final y cierre

### D33. Correcciones de interpretación en E7 y B.5, y aprobación del hito 01

- **Decisión (de los revisores):** antes de cerrar el hito se corrige el cuaderno del Bloque B y la bitácora:
  1. **Error de interpretación en E7:** el ancho 10-90 % de la banda en t = 0 no es evidencia del estrechamiento del tubo. En t = 0 todas las trayectorias pasan por x = −1/√2 por construcción de la alineación; ese ancho (0.066, 0.051, 0.041) solo mide el avance en un intervalo de muestreo, √(2D·0.01) = 0.071, 0.055 y 0.045. Se elimina como evidencia, se explica que el pellizco de la figura de densidad en t = 0 es un efecto de la alineación, y el ancho se mide lejos de ella: en t = −0.25 (ancho/√D = 1.04, 1.07, 1.05) y en t = −1.0, dentro del pozo (0.98, 0.96, 0.94 frente a 0.906 del equilibrio armónico 2·1.2816·√(D/8)). La evidencia principal sigue siendo el criterio E7 (RMS de la mediana frente a x_om).
  2. **Escala logarítmica de E7:** el ajuste empírico "~0.17 por unidad de ln(1/D)" se reemplaza por la estimación de orden de magnitud t* = (1/8)·ln(2/D): distancia a la cima e^{−4t}, con 4 = |V″(0)|, igual a la fluctuación térmica de la meseta √(2D/|V″(0)|) = √(D/2). Da 0.26, 0.32 y 0.37, frente a 0.23, 0.31 y 0.38 medidos.
  3. **B.5:** el prefactor A(D) de T = A(D) e^{1/D} es el del tiempo exacto, que tiende al prefactor de Kramers 2π/√32 solo cuando D → 0 (antes se atribuía a Kramers).
  4. El mecanismo del sesgo residual de orden dt con la corrección de puente queda como pregunta abierta en la bitácora: con dt = 1e-3 no afecta a ningún resultado.
- **Alcance:** cambian el texto y las tablas del cuaderno y la bitácora. No cambian el código de `src/`, las pruebas ni los resultados. Las figuras regeneradas son idénticas byte a byte.
- **Aprobación:** los revisores aprueban el hito 01 con estas correcciones; se crea la etiqueta anotada `hito-01`.
- **Observación (de Claude):** en el pozo, el ancho medido coincide mejor con los cuantiles de la densidad de Boltzmann exacta restringida a x < 0 que con el valor armónico (0.3765 y 0.2987 frente a 0.373 y 0.298 en D = 0.15 y 0.1). Con D = 0.25 esa referencia queda un 7 % por encima de lo medido (0.523 frente a 0.488), porque su cola hacia la cima ya pesa.

## 2026-10-05 — Hito 02: Bloque A reducido (red variacional)

### D34. PyTorch solo para CPU, con el índice de PyTorch explícito

- **Decisión (de los revisores, configuración de Claude):** grupo `neuronal` con `torch` 2.14.1+cpu. El índice `https://download.pytorch.org/whl/cpu` se declara con `explicit = true` y se asigna solo a `torch` en `[tool.uv.sources]`; todo lo demás sigue viniendo de PyPI. El grupo se agrega a `default-groups` (como en D21). PyTorch usa 10 hilos (`torch.set_num_threads`, como D23).
- **Problema encontrado:** `uv add --index pytorch-cpu=…` sin `explicit` hacía que uv buscara también en ese índice las demás dependencias (por ejemplo, `requests`) y la resolución fallaba.
- **Efecto en el entorno:** llegan torch, filelock, fsspec, networkx y setuptools; ninguna versión existente cambia (numpy sigue en 2.4.6). `torch.version.cuda` es `None`. Suite completa tras instalar: 185 passed.

### D35. Sesgos ocultos no nulos: una red tanh sin sesgos impone la imparidad

- **Problema:** la primera implementación iniciaba los sesgos en cero. Un perceptrón con tanh (función impar) y sin sesgos es exactamente impar, N(−s) = −N(s). En el instantón, el funcional es simétrico bajo x(τ) → −x(−τ) y, partiendo de una red impar, el gradiente conserva esa simetría: quedaría impuesta, contra la especificación. Lo detectó la prueba congelada `test_no_impone_simetria`, antes de cualquier entrenamiento.
- **Decisión (de Claude):** los sesgos ocultos se inician uniformes en ±1/√(entradas) (la inicialización por defecto de PyTorch); los pesos ocultos siguen siendo Xavier, como pide la especificación. La última capa conserva sesgo nulo y pesos Xavier × 1e-3, de modo que el camino inicial sigue siendo la recta.
- **Alternativa:** sesgos nulos (impone la simetría).

### D36. La superposición red–ruido va en el cuaderno del hito 02

- **Decisión (de los revisores):** la figura del camino de la red sobre el tubo reactivo de E7 va en `notebooks/02_neuronal/`, aunque CLAUDE.md §3 reserva para `notebooks/03_integracion/` los cuadernos que combinan ambos experimentos. Para respetar el propósito de §3, el cuaderno carga los resultados de E4 directamente con numpy, sin importar `taller.estocastico`, y `src/` no mezcla los bloques.
- **Alternativa:** mover esa sección a `notebooks/03_integracion/`.

### D37. L-BFGS se detuvo por el límite de evaluaciones

- **Observación (de Claude):** con `iteraciones_max = 5000` (`max_eval` = 6250), L-BFGS terminó por el límite de evaluaciones en 5 de las 6 redes y por tolerancia en una (escape, 3 × 32). No es "hasta convergencia" en sentido estricto, pero el exceso de la acción sobre la cota ya es de 1e-7 a 1e-8, frente a una tolerancia de 1e-2. No se cambió la configuración.

### D38. PyTorch estándar (con CUDA en Linux) en lugar del índice CPU

- **Decisión (de los revisores):** `uv add torch` sin índices especiales; reemplaza a D34. Resolvió torch 2.14.1+cu130 (runtime CUDA 13.0, compatible con el driver 595.91, CUDA 13.2). Solo cambió torch entre los paquetes existentes (+cpu → +cu130); numpy sigue en 2.4.6.
- **Verificación:** GPU comprobada con un cálculo real en float64 (`test_el_calculo_corre_en_la_gpu`, CLAUDE.md §12). Suite completa tras instalar: 231 passed (sesión del 2026-10-06, antes del apagado).
- **Registro tardío:** el commit 94f9db3 cita D38, D39 y D40, pero la máquina se apagó antes de escribirlas aquí; se registran el 2026-10-06 a partir de los mensajes de commit y de las instrucciones de los revisores.

### D39. Selección de dispositivo y precisión

- **Decisión (de los revisores):** automática (cuda, si no mps, si no cpu), con `--dispositivo` para forzarla y la clave `dispositivo` en la configuración. float64 en cuda y cpu; float32 con aviso en mps, que no soporta float64. Dispositivo, nombre del hardware y precisión van en los metadatos. Los tiempos en GPU se miden con `torch.cuda.synchronize()` antes de detener el reloj. `--salida` permite guardar una corrida de comparación sin pisar los resultados oficiales.

### D40. El horizonte de las redes se llama `horizonte` (T_h), no T

- **Decisión (de los revisores):** en el taller T designa el tiempo medio de escape y la temperatura. Se renombra en el código, la configuración y el cuaderno.
- **Modificación autorizada de pruebas congeladas:** solo la clave `T` → `horizonte` que leen las pruebas y los nombres de constantes del oráculo (`T_ESCAPE`, `T_INSTANTON` → `HORIZONTE_*`). Ningún valor esperado ni tolerancia cambia.

## 2026-10-06 — Apagado inesperado durante el hito 02

### D41. Solo CPU, con 8 hilos de PyTorch, hasta nuevo aviso

- **Contexto:** la máquina se apagó (≈ 00:13) mientras entrenaba en CPU con 10 hilos la corrida de comparación de tiempos (`--dispositivo cpu --salida results/neuronal/cpu`), justo después de entrenar las seis redes en la GPU. Hay apagados repetidos bajo carga (ver D23).
- **Decisión (de los revisores):** todo el Bloque A corre en CPU (`dispositivo: cpu` en la configuración; `--dispositivo` sigue disponible) con `torch.set_num_threads(8)` (`hilos: 8`; `--hilos` tiene prioridad). No se usa la GPU. La medición de tiempos en CUDA queda pendiente.
- **Resultados:** los seis entrenados en la GPU y el único completado de la comparación en CPU con 10 hilos se apartan (sin borrar) en `results/neuronal/previos_apagado/`; los resultados oficiales se reentrenan en CPU.
- **Pruebas:** la suite corre con `CUDA_VISIBLE_DEVICES=""` para no tocar la GPU; `test_el_calculo_corre_en_la_gpu` se omite por su propio `skipif`. Las pruebas del bloque estocástico siguen usando los 10 hilos de Numba de `configs/estocastico/comun.yaml` (D23); limitar Numba a 8 con `NUMBA_NUM_THREADS` choca con esa configuración y hace fallar 4 pruebas, así que no se limita.

### D42. Modificación autorizada de una prueba congelada: RMS del escape en la intersección con el dominio

- **Causa:** la posición de la transición del escape la fija solo débilmente el horizonte finito (modo cero de traslación). Con T_h = 3, la ventana alineada [−1, 0.5] solo cabe si el cruce cae en t ≥ −2, y las redes quedan justo en ese límite: −1.9993 (CPU, 10 hilos), −1.9999 (CPU, 8 hilos) y −2.0007 (GPU). Con la GPU, `test_escape_rms_alineado` falló porque la ventana salía 7e-4 del dominio, aunque el camino era igual de bueno (RMS 4.6e-5 en la parte disponible). Es un defecto de diseño del criterio, no del resultado.
- **Decisión (de los revisores):** el RMS del escape se evalúa en la intersección de [−1, 0.5] (tiempo alineado) con el dominio disponible, y la prueba exige que el núcleo [−0.5, 0.5] quede entero dentro; si no, falla. La tolerancia (< 0.01) no cambia. El instantón no cambia.
- **Cambios:** `tests/neuronal/oraculo_neuronal.py` (constante `NUCLEO_RMS_ESCAPE = (-0.5, 0.5)`) y `_rms_alineado` en `tests/neuronal/test_criterios_neuronal.py`. Los puntos del RMS son los mismos de antes (paso 0.01), solo que se descartan los que caen fuera del dominio: cuando la ventana cabe entera, el valor no cambia. El cuaderno usa la misma regla.
- **Validación de la prueba modificada:** resultados vigentes en CPU, RMS 4.771e-5 (igual que antes); resultados de la GPU, antes en falla, RMS 4.637e-5; camino sintético x_om con cruce en −2.6 (núcleo fuera del dominio), falla como debe; con cruce en −2.4 pasa.

### D43. Alcance tras el apagado: hilos de Numba y tiempos en CUDA

- **Decisión (de los revisores):** el bloque estocástico conserva sus 10 hilos de Numba (D23); no se cambian ni su configuración ni sus pruebas. La medición de tiempos en CUDA queda fuera del alcance del hito 02, porque no responde ninguna pregunta física del taller. La opción `--dispositivo` se mantiene funcional, y D41 (solo CPU, 8 hilos de PyTorch) sigue vigente.
