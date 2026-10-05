# CLAUDE.md — Experimentos numéricos del Taller FMA

Este archivo es el contexto permanente del proyecto. Léelo completo al inicio de cada sesión. Las tareas concretas están en `specs/`; este archivo define el marco, las reglas y las convenciones que aplican a todas ellas.

## 1. Qué es este proyecto

Taller del curso **Física Matemática Avanzada** (posgrado en física). Aplicación del método variacional al problema del escape de un estado metaestable, en dos versiones:

- **Escape térmico (Parte 1):** partícula browniana sobreamortiguada en un doble pozo. El camino más probable de escape minimiza el funcional de Onsager-Machlup.
- **Tunelamiento cuántico (Parte 2):** la misma partícula, cuántica, en tiempo imaginario. El camino dominante es el instantón, que minimiza la acción euclídea.

**Objetivo primario:** determinar, con el método variacional, el mecanismo y la escala exponencial con que un sistema abandona un estado metaestable, contrastar el escape térmico con el tunelamiento, y verificarlo numéricamente.

Este repositorio contiene **solo la parte numérica**. La parte teórica (derivaciones) la desarrolla otra persona y se incorporará en `docs/teoria/`.

**Principio rector:** la física manda y las matemáticas sirven. Cada pieza de código existe para verificar o ilustrar un resultado físico concreto. Si no puedes decir qué resultado físico verifica una función, probablemente no debería existir.

## 2. El sistema físico (referencia autoritativa)

Todo el proyecto usa unidades adimensionales y esta notación. No la cambies.

| Símbolo | Definición | Valor o expresión |
|---|---|---|
| `V(x)` | Potencial de doble pozo | `(x**2 - 1)**2` |
| `V'(x)` | Derivada | `4*x*(x**2 - 1)` |
| `V''(x)` | Segunda derivada | `12*x**2 - 4` |
| Mínimos | Estados metaestables | `x = ±1`, `V = 0`, `V'' = 8` |
| Barrera | Punto silla | `x = 0`, `V = 1`, `V'' = -4` |
| `ΔV` | Altura de la barrera | `1` |
| `D` | Intensidad del ruido (`k_B T / γ`) | parámetro, régimen `D ≪ ΔV` |
| `T` | Horizonte temporal finito (redes) | parámetro |

**Dinámica de Langevin (Parte 1):** `dx = -V'(x) dt + sqrt(2D) dW`

**Funcional de Onsager-Machlup, forma de ruido débil (Freidlin-Wentzell):**
`S[x] = (1/(4D)) ∫ (ẋ + V'(x))² dt`, fronteras `x(-∞) = -1`, `x(+∞) = 0`.

**Acción euclídea (Parte 2):** `S_E[x] = ∫ [½ ẋ² + V(x)] dτ`, fronteras `x(-∞) = -1`, `x(+∞) = +1`.

**Resultados de referencia (verificados con sympy):**

| Cantidad | Expresión | Valor |
|---|---|---|
| Camino de Onsager-Machlup | `x_om(t) = -1/sqrt(1 + exp(8t))` | cumple `ẋ = V'(x)` |
| Kink del instantón | `x_kink(τ) = tanh(sqrt(2) τ)` | cumple `ẋ = sqrt(2V(x))` |
| Acción mínima térmica | `S_min = ΔV / D` | `1/D` |
| Acción del instantón | `S0 = ∫ sqrt(2V) dx` en `[-1, 1]` | `4*sqrt(2)/3 ≈ 1.8856180832` |

Estos valores son **la verdad contra la que se valida todo**. Viven en un único lugar del código (`src/taller/analitico/`) y el resto del proyecto los importa. Nunca los copies a mano en otro archivo.

## 3. Estructura del repositorio

```
experimentos_numericos/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── specs/                 # qué se planea hacer y con qué criterios se acepta
├── src/taller/
│   ├── analitico/         # potencial, Euler-Lagrange, soluciones cerradas, S_min, S0, Kramers
│   ├── estocastico/       # Euler-Maruyama (NumPy de referencia y CUDA), observables
│   └── neuronal/          # ansatz de frontera, pérdidas PINN y Ritz, entrenamiento
├── tests/
│   ├── analitico/
│   ├── estocastico/
│   └── neuronal/
├── notebooks/
│   ├── 00_analitico/
│   ├── 01_estocastico/
│   ├── 02_neuronal/
│   └── 03_integracion/
├── configs/{estocastico,neuronal}/
├── results/{estocastico,neuronal}/
├── figures/{analitico,estocastico,neuronal,integracion}/
└── docs/
    ├── plan/              # plan del taller y grafos
    ├── teoria/            # derivaciones teóricas (otra persona)
    ├── bitacora/          # una entrada por hito
    └── decisiones.md
```

**Crea solo las carpetas que pide el hito en curso.** Esta estructura es el destino, no una orden de crear todo ahora.

### Dirección de las dependencias (regla científica, no estética)

- `estocastico/` y `neuronal/` importan de `analitico/`.
- `estocastico/` y `neuronal/` **nunca se importan entre sí**, ni comparten utilidades fuera de `analitico/`.
- Solo los cuadernos de `notebooks/03_integracion/` combinan ambos experimentos.

**Razón:** los dos experimentos son verificaciones independientes. Las redes verifican las matemáticas (que el minimizador y su acción son los correctos). La simulación verifica la física (que el proceso real sigue ese camino y escala como Arrhenius). Si compartieran código, un error común contaminaría ambas verificaciones a la vez.

## 4. Flujo de trabajo por hitos

El proyecto avanza **un hito a la vez**. Cada hito tiene una especificación en `specs/`.

1. Lee la especificación completa del hito. Si algo es ambiguo, **pregunta antes de implementar**; no supongas.
2. Implementa solo lo que pide la especificación.
3. Ejecuta las pruebas del hito y reporta los resultados reales.
4. Escribe la entrada de bitácora del hito (sección 7).
5. **Detente y espera revisión.** No empieces el siguiente hito sin aprobación explícita.

Al terminar cada hito, entrega un resumen breve: qué se implementó, qué pruebas pasaron o fallaron (con los números), qué figuras se generaron y qué quedó pendiente o dudoso.

## 5. Reglas que no se rompen

1. **No modifiques sin autorización explícita:** los archivos de `tests/`, los valores de referencia de la sección 2, las tolerancias de las especificaciones ni los archivos de `specs/`. Si crees que una prueba o una tolerancia está mal, dilo y explica por qué; no la cambies.
2. **Nunca ajustes una prueba, una tolerancia o un parámetro para que algo pase.** Si una prueba falla, repórtalo con los números. Un fallo bien documentado es un resultado; un fallo escondido es un error científico.
3. **No inventes resultados.** Todo número o figura que reportes debe salir de código ejecutado en esta sesión.
4. **Valida antes de usar.** Cada pieza nueva se compara contra algo conocido (una solución cerrada, una distribución exacta, una implementación de referencia más simple) antes de construir encima de ella.
5. **Respeta la dirección de dependencias** de la sección 3.
6. **No sobrediseñes.** Nada de integración continua, empaquetado elaborado, documentación web ni abstracciones "por si acaso". El plazo es corto; la prioridad es corrección y claridad.
7. **No instales dependencias que el hito no necesite.** Si hace falta una nueva, justifícala en el resumen del hito.

## 6. Reproducibilidad

- **Entorno:** gestionado con `uv` y `pyproject.toml`. Versiones de dependencias fijadas en el lockfile.
- **Semillas:** todo lo aleatorio recibe su semilla desde un archivo de `configs/`, nunca codificada dentro de una función.
- **Resultados crudos:** se guardan en `results/` (formato `.npz` o similar) junto con sus metadatos: parámetros usados, semilla, fecha y hash del commit de git.
- **Figuras:** se generan **desde los resultados guardados**, nunca directamente de una simulación en vivo. Cualquier figura debe poder regenerarse sin volver a simular.
- **GPU:** los generadores aleatorios en GPU pueden no ser deterministas bit a bit. La reproducibilidad exigida es **estadística**, salvo que una especificación diga otra cosa. Documenta cuál se cumple.
- **Cuadernos:** emparejados con jupytext (formato `.py` percent). En git se versiona **solo el `.py`**; los `.ipynb` están en `.gitignore` y se regeneran localmente. La lógica vive en `src/`; los cuadernos solo orquestan, grafican y explican.

## 7. Documentación del proceso

Tres tipos de documentos, con funciones distintas:

- **`specs/`**: lo que se *planea* hacer y sus criterios de aceptación. Se escribe antes de implementar y no lo editas tú.
- **`docs/bitacora/`**: lo que *realmente pasó* en cada hito. Un archivo por hito, `NN_nombre.md`, con esta estructura:
  - Objetivo del hito y resultado físico que verifica.
  - Qué se implementó (archivos y funciones principales).
  - Pruebas: cuáles pasaron, cuáles fallaron, con los valores numéricos obtenidos frente a los esperados.
  - Problemas encontrados y cómo se resolvieron (incluidos los intentos fallidos).
  - Figuras generadas y qué muestran.
  - Pendientes y dudas para la revisión.
- **`docs/decisiones.md`**: cada decisión de diseño con su fecha, alternativas consideradas y justificación. Si durante un hito tomas una decisión no prevista en la especificación, regístrala aquí y señálala en el resumen.

## 8. Convenciones de código

- Python 3.11 o superior, con anotaciones de tipo.
- **Idioma:** documentación, comentarios y docstrings en español. Nombres de módulos, funciones y variables en español sin tildes ni eñes (`camino_om`, `accion_minima`, `potencial`), coherentes con los nombres de las carpetas.
- **Docstrings con contenido físico:** cada función pública dice qué calcula, qué significa físicamente y, cuando aplique, a qué sección del plan del taller corresponde (§0 a §6, Bloque A o Bloque B).
- Notación de la sección 2 en todo el código: `D`, `V`, `S_min`, `S0`, `T`. Si una fuente usa otra notación (por ejemplo `σ² = 2D`), la traducción se hace explícitamente en un comentario.
- Funciones pequeñas y verificables por separado. Preferir funciones puras a clases con estado, salvo que el estado sea inevitable (por ejemplo, un modelo de red neuronal).
- Pruebas con `pytest`, organizadas por experimento en `tests/`.

## 9. Entorno de cómputo

- Linux. CPU Intel Core i9-10900KF (10 núcleos, 20 hilos), 96 GB de RAM DDR4, GPU NVIDIA RTX 3060 con 12 GB.
- La máquina se usa a capacidad completa: se permite paralelizar en CPU y GPU cuando la especificación lo indique.
- La memoria de la GPU (12 GB) es el límite práctico para simulaciones grandes: diseña para registrar observables sobre la marcha en lugar de guardar trayectorias completas.
- **La ruta del proyecto contiene espacios.** Usa siempre comillas en los comandos de shell y `pathlib` en Python.

## 10. Decisiones abiertas (no las resuelvas por tu cuenta)

- **Término jacobiano:** el funcional completo de Onsager-Machlup incluye `-½ ∫ V''(x) dt`. Por defecto se usa la forma de ruido débil (sin el término). La variante completa se implementará solo si una especificación lo pide.
- **Definición de "escape" para Kramers:** llegar a la cima (`x = 0`) o caer al otro pozo cambia el prefactor (no el exponente). La fijará la especificación del bloque estocástico.

## 11. Git

- Si el repositorio no está inicializado, la inicialización forma parte del primer hito.
- Un commit por paso lógico dentro de un hito, con mensajes en español que digan qué y por qué.
- Al aprobarse un hito, se etiqueta: `hito-00`, `hito-01`, etc.
- `results/` con datos pesados va en `.gitignore`; los metadatos y resultados pequeños sí se versionan.

## 12. Evolución del entorno

- **El entorno crece con el proyecto.** La base (`sympy`, `numpy`, `scipy`, `matplotlib`) son dependencias del proyecto. Cada experimento agrega las suyas en un grupo de dependencias propio de uv (`estocastico`, `neuronal`), creado en su hito, no antes.
- **Toda dependencia nueva** se agrega con `uv add`, se justifica en una línea y se registra en `docs/decisiones.md` y en la bitácora del hito, con la versión que resolvió uv.
- **`uv.lock`** se actualiza y se versiona en el mismo commit que agrega la dependencia.
- **Después de agregar dependencias se ejecuta la suite completa de pruebas**, no solo la del hito en curso. Si uv cambió la versión de una dependencia existente (por ejemplo, `numpy` al instalar Numba), se reporta antes de continuar: cambia el entorno sobre el que se validaron los hitos anteriores.
- **Las librerías de GPU se verifican con una prueba explícita de que usan la GPU** (por ejemplo, `torch.cuda.is_available()` y el nombre del dispositivo, o la compilación y ejecución de un kernel mínimo con Numba). Que la instalación no dé errores no es suficiente.
- **PyTorch con CUDA** requiere configurar en `pyproject.toml` el índice de paquetes de PyTorch para la versión de CUDA elegida. Esa configuración se decide en el hito neuronal y se documenta.
