# Experimentos numéricos — Taller FMA

Parte numérica del taller de Física Matemática Avanzada: método variacional aplicado al escape de un estado metaestable en el doble pozo `V(x) = (x² - 1)²`, por activación térmica (funcional de Onsager-Machlup) y por tunelamiento (instantón). El marco completo del proyecto está en [`CLAUDE.md`](CLAUDE.md).

## Uso

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # crea .venv con Python 3.12 y todas las dependencias
uv run pytest           # ejecuta las pruebas
uv run jupyter lab      # abre los cuadernos
```

Los cuadernos se versionan como `.py` (formato percent de jupytext). Para abrirlos como cuaderno en JupyterLab: clic derecho → *Open With → Notebook*; jupytext crea el `.ipynb` emparejado localmente (no se versiona).

**Nota:** la ruta del proyecto puede contener espacios; usa comillas en los comandos de shell.
