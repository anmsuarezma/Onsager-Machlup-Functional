# Experimentos numéricos — Taller FMA

Parte numérica del taller de Física Matemática Avanzada: método variacional aplicado al escape de un estado metaestable en el doble pozo `V(x) = (x² - 1)²`, por activación térmica (funcional de Onsager-Machlup) y por tunelamiento (instantón). El marco completo del proyecto está en [`CLAUDE.md`](CLAUDE.md).

## Uso

Requiere [uv](https://docs.astral.sh/uv/).

### Instalación estándar (quien clona el repositorio)

```bash
uv sync                 # crea .venv local con Python 3.12 y las versiones exactas de uv.lock
uv run pytest           # ejecuta las pruebas
uv run jupyter lab      # abre los cuadernos
```

Sin más configuración, uv crea un `.venv` dentro del repositorio con **las mismas versiones fijadas en `uv.lock`**. Es lo esperado: los resultados no dependen de dónde viva el entorno, sino del lockfile.

### Entorno compartido del semestre (FMA)

En la máquina del autor, el proyecto no usa un `.venv` local: se instala en el entorno compartido **FMA**, fuera del repositorio. uv lo encuentra mediante la variable `UV_PROJECT_ENVIRONMENT`, que se define en un `.envrc` (direnv) de la carpeta del curso, no versionado:

```bash
# .envrc en la carpeta padre del curso
export UV_PROJECT_ENVIRONMENT="<ruta>/FMA"
```

Primera vez, para crear FMA con el Python 3.12 gestionado por uv y registrarlo como kernel de Jupyter:

```bash
uv python install 3.12
uv venv --managed-python --python 3.12 "$UV_PROJECT_ENVIRONMENT"
uv sync
uv run python -m ipykernel install --user --name FMA --display-name "FMA"
```

Para activar FMA en una terminal sin uv: `source "$UV_PROJECT_ENVIRONMENT/bin/activate"`. En VS Code o JupyterLab, selecciona el kernel **FMA**.

**Cuidado:** `uv sync` elimina de FMA los paquetes que no estén en `uv.lock`; usa `uv sync --inexact` si FMA tiene paquetes de otros experimentos.

### Cuadernos

Los cuadernos se versionan como `.py` (formato percent de jupytext). Para abrirlos como cuaderno en JupyterLab: clic derecho → *Open With → Notebook*; jupytext crea el `.ipynb` emparejado localmente (no se versiona).

**Nota:** la ruta del proyecto puede contener espacios; usa comillas en los comandos de shell.
