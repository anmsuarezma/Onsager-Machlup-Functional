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
