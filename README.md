# Experimentos numéricos — Taller FMA

Parte numérica del taller de Física Matemática Avanzada: método variacional aplicado al escape de un estado metaestable en el doble pozo `V(x) = (x² - 1)²`, por activación térmica (funcional de Onsager-Machlup) y por tunelamiento (instantón). El marco completo del proyecto está en [`CLAUDE.md`](CLAUDE.md).

## Uso

Requiere [uv](https://docs.astral.sh/uv/). El proyecto no usa un `.venv` local: se instala en el entorno compartido del semestre **FMA**.

```bash
# Ruta del entorno FMA (fuera del repositorio)
export UV_PROJECT_ENVIRONMENT="$HOME/Documents/Mauricio/Doctorado Fisica/Fisica Matematica Avanzada/FMA"

# Primera vez: crear FMA con el Python 3.12 gestionado por uv
uv python install 3.12
uv venv --managed-python --python 3.12 "$UV_PROJECT_ENVIRONMENT"

uv sync                 # instala en FMA exactamente lo fijado en uv.lock
uv run pytest           # ejecuta las pruebas
uv run jupyter lab      # abre los cuadernos

# Registrar FMA como kernel de Jupyter (una vez)
uv run python -m ipykernel install --user --name FMA --display-name "FMA"
```

Para activar FMA en una terminal sin uv: `source "$UV_PROJECT_ENVIRONMENT/bin/activate"`. En VS Code o JupyterLab, selecciona el kernel **FMA**.

**Cuidado:** si `UV_PROJECT_ENVIRONMENT` no está definida, uv crea un `.venv` local. `uv sync` además elimina de FMA los paquetes que no estén en `uv.lock`; usa `uv sync --inexact` si FMA tiene paquetes de otros experimentos.

Los cuadernos se versionan como `.py` (formato percent de jupytext). Para abrirlos como cuaderno en JupyterLab: clic derecho → *Open With → Notebook*; jupytext crea el `.ipynb` emparejado localmente (no se versiona).

**Nota:** la ruta del proyecto puede contener espacios; usa comillas en los comandos de shell.
