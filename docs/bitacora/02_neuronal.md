# Hito 02 — Bloque A reducido: la red variacional

Especificación: `specs/02_neuronal.md` (texto de los revisores, tal cual). Fecha: 2026-10-05. Decisiones: D34-D37 de `docs/decisiones.md`.

## 0. Objetivo y resultado físico que verifica

Minimizar directamente los dos funcionales del taller con una red neuronal (método directo del cálculo de variaciones, tipo Ritz; forma débil), sin pasar por la ecuación de Euler-Lagrange y sin datos, y comparar contra las soluciones cerradas del hito 00:
- escape térmico: `S·D = (1/4)∫(ẋ + V′)² dt` en [−3, 3], con x(−3) = −1 y x(3) = 0; referencias S·D → ΔV = 1 y `x_om`;
- instantón: `S_E = ∫[½ẋ² + V] dτ` en [−4, 4], con x(−4) = −1 y x(4) = +1; referencias S0 = 4√2/3 y `x_kink`.

Verifica las matemáticas del taller: que el minimizador del funcional y su acción son los de §2 y §3, y que la acción no baja de las cotas por completar cuadrados.

Resultado: todos los criterios se cumplen con varios órdenes de margen (sección 3).

## 1. Entorno

Grupo `neuronal` nuevo (D34):

| Paquete | Versión |
|---|---|
| torch | 2.14.1+cpu (índice CPU de PyTorch, explícito y asignado solo a torch) |
| filelock / fsspec / networkx / setuptools | 4.0.12 / 2026.9.0 / 3.7 / 84.0.0 (transitivas de torch) |

- **Versiones existentes:** ninguna cambió; se comprobó comparando la lista completa de paquetes antes y después. numpy sigue en 2.4.6.
- **CPU:** `torch.version.cuda = None` y `torch.cuda.is_available() = False`; no se configuró GPU, como pide la especificación. Prueba mínima de autograd en float64 correcta.
- **Suite completa tras instalar (CLAUDE.md §12):** 185 passed.
- **Hilos:** PyTorch con 10 (`torch.set_num_threads`).

## 2. Qué se implementó

### 2.1 `src/taller/neuronal/` (importa solo de `taller.analitico`)

| Archivo | Contenido |
|---|---|
| `red.py` | `crear_red(capas, semilla, escala_ultima_capa)`: perceptrón 1 → capas → 1 en float64, tanh y salida lineal. Pesos ocultos Xavier; sesgos ocultos uniformes en ±1/√(entradas) (D35); última capa Xavier × 1e-3 y sesgo nulo. Generador propio con semilla. `contar_parametros` |
| `ansatz.py` | `camino`: x = x_a + (x_b − x_a)(t + T)/(2T) + [(t + T)(T − t)/T²]·N(t/T), fronteras exactas. `camino_y_derivada`: ẋ por autograd con `create_graph=True` |
| `accion.py` | `accion_escape` (S·D) y `accion_instanton` (S_E) con `torch.trapezoid`; V y V′ de `referencias`, que operan también con tensores |
| `entrenamiento.py` | `entrenar`: Adam y después L-BFGS (Wolfe fuerte) sobre la malla fija completa; registra la acción en cada evaluación. `evaluar_en_malla_doble`: camino, ẋ y acción en 4001 puntos |
| `guardado.py` | `.npz` con metadatos y escritura atómica; duplica a propósito el del bloque estocástico, que no se puede importar (CLAUDE.md §3) |
| `correr.py` | `python -m taller.neuronal.correr configs/neuronal/entrenamiento.yaml [--hilos N] [--rehacer]`: 2 problemas × 3 arquitecturas, un archivo por combinación |

### 2.2 Configuración

`configs/neuronal/entrenamiento.yaml`, congelada con las pruebas:
- T = 3 y T = 4; malla de 2001 puntos;
- arquitecturas 2 × 32 (por defecto), 3 × 32 y 4 × 64;
- semillas 20261009 (escape) y 20261010 (instantón);
- Adam: 3000 iteraciones con tasa 1e-3;
- L-BFGS: 5000 iteraciones como máximo, historia 50, tolerancias 1e-12 y 1e-15;
- escala de la última capa: 1e-3.

### 2.3 Pruebas (`tests/neuronal/`, congeladas en `4f8d9bd`)

- `oraculo_neuronal.py`: S0, `x_om`, `x_kink`, tolerancias y número de parámetros contados a mano (1153, 2209, 12673).
- `test_configuracion_neuronal.py`: parámetros de la especificación y semillas.
- `test_red_ansatz.py`: fronteras exactas, camino inicial recto, ẋ por autograd frente a diferencias finitas, float64, número de parámetros, semilla reproducible y que la red no impone simetría.
- `test_accion.py`: funcionales contra valores escritos a mano (x_om → 1; recta del escape → 1991/840; reposo → 0; kink → S0; recta del instantón → 1/4 + 64/15).
- `test_criterios_neuronal.py`: criterios de aceptación, recalculados con numpy desde el camino guardado, sin usar `taller.neuronal`.

No se modificó ninguna prueba congelada.

### 2.4 Cuaderno `notebooks/02_neuronal/02_neuronal.py`

Secciones A.0-A.6 con la numeración del plan: método directo (§5′), escape (§2), instantón (§3), acción durante el entrenamiento y cotas, robustez, superposición sobre el tubo reactivo de E7 y resumen. Solo carga resultados. Se ejecuta sin errores desde una sesión limpia (`jupytext --execute`).

### 2.5 Entrenamiento

Commit `b454b38` (árbol limpio), 10 hilos, 4 min 16 s en total para las 6 redes. Después del entrenamiento solo cambió un comentario de `red.py` (el número de decisión), sin efecto en los resultados.

## 3. Pruebas

Suite completa: **222 passed, 0 fallidas, 0 omitidas** (3 min 35 s), incluidas las del hito 02 con los resultados presentes.

### 3.1 Criterios de aceptación (arquitectura por defecto, 2 × 32)

| Criterio | Escape térmico | Instantón | Exigido |
|---|---|---|---|
| Acción | S·D = 1.0000005167 | S_E = 1.8856181255 (S0 = 1.8856180832) | — |
| Error relativo de la acción | +5.17e-7 | +2.24e-8 | < 1e-2 |
| RMS alineado frente a la cerrada | 4.79e-5 en t ∈ [−1, 0.5] | 1.38e-5 en τ ∈ [−2, 2] | < 1e-2 |
| Cota analítica | S·D − 1 = +5.2e-7 | S_E − S0 = +4.2e-8 | ≥ −1e-4 |
| Malla doble (4001 puntos), diferencia relativa | 2.6e-11 | 2.4e-13 | < 1e-4 |
| Cruce de alineación | t = −1.9993 (x = −1/√2) | τ = +0.1734 (x = 0) | — |

En ningún momento del entrenamiento la acción bajó de la cota: el mínimo de toda la historia es 1 + 5.2e-7 y S0 + 4.2e-8.

### 3.2 Robustez ante la arquitectura

| Problema | Arquitectura | Parámetros | Acción | Error relativo | RMS | Cruce | Malla doble | Tiempo (s) | Evaluaciones de L-BFGS |
|---|---|---|---|---|---|---|---|---|---|
| escape | 2 × 32 | 1153 | 1.0000005167 | +5.17e-7 | 4.79e-5 | −1.999 | 2.6e-11 | 23.6 | 6250 |
| escape | 3 × 32 | 2209 | 1.0000001996 | +2.00e-7 | 3.26e-5 | −1.825 | 2.8e-13 | 35.7 | 5847 |
| escape | 4 × 64 | 12673 | 1.0000006030 | +6.03e-7 | 1.42e-5 | −1.848 | 1.9e-12 | 71.2 | 6251 |
| instantón | 2 × 32 | 1153 | 1.8856181255 | +2.24e-8 | 1.38e-5 | +0.173 | 2.4e-13 | 20.2 | 6251 |
| instantón | 3 × 32 | 2209 | 1.8856181108 | +1.47e-8 | 9.30e-6 | −0.226 | 1.9e-13 | 24.1 | 6250 |
| instantón | 4 × 64 | 12673 | 1.8856181529 | +3.70e-8 | 2.05e-5 | −0.028 | 2.9e-15 | 79.3 | 6250 |

Las tres arquitecturas coinciden dentro de la tolerancia: la acción difiere entre ellas en menos de 5e-7 (escape) y 3e-8 (instantón).

**Qué limita la precisión:** al nivel de la tolerancia, ni el horizonte ni la red. Por debajo, la optimización: el error no baja al aumentar 11 veces el número de parámetros, ni es monótono con el tamaño, y L-BFGS se detuvo por el límite de evaluaciones. La dirección difícil es el modo casi plano de traslación: el cruce cambia con la arquitectura. El aporte del horizonte finito no se puede separar sin variar T (estudio 1 del plan, fuera del alcance). Su huella sí se ve en un punto: el borde izquierdo de la ventana alineada del escape, donde la red vale −1 y `x_om(−1) = −1 + 1.7e-4`.

### 3.3 Superposición sobre el tubo reactivo (E7, hito 01)

- RMS(mediana del ruido − red) en t ∈ [−0.5, 0.25]: 0.1197, 0.0833 y 0.0582 para D = 0.25, 0.15 y 0.1. Son los valores del criterio E7, porque la red y `x_om` difieren en 4.0e-5 en esa malla.
- Fracción del camino de la red dentro de la banda 10-90 %: 0.80, 0.89 y 0.93. Antes del origen está siempre dentro. Sale, por debajo del percentil 10, solo en t ∈ (0, 0.15], (0, 0.08] y (0, 0.05]. Es un efecto de la alineación: tras el último cruce por −1/√2, el conjunto está condicionado a subir (hito 01, D33).

## 4. Problemas encontrados y cómo se resolvieron

1. **Índice de PyTorch no explícito** (D34). `uv add --index` sin `explicit` buscaba también las demás dependencias en el índice de PyTorch y la resolución fallaba (`requests`). Se resolvió declarando el índice como explícito y asignándolo solo a torch.
2. **Sesgos nulos imponían la imparidad** (D35). Con tanh y sin sesgos, la red es exactamente impar. La prueba congelada `test_no_impone_simetria` falló en la primera implementación, antes de entrenar. Se corrigió el código (sesgos ocultos no nulos), no la prueba.
3. **L-BFGS terminó por el límite de evaluaciones** en 5 de 6 redes (D37). La precisión alcanzada (1e-7-1e-8) está muy por debajo de la tolerancia; no se cambió la configuración.
4. **Afirmaciones del cuaderno corregidas al revisar las salidas y las figuras**, antes de este commit:
   - "el camino de la red está dentro de la banda del ruido en toda la subida": falso, las fracciones son 0.80-0.93. Se reemplazó por los números y su explicación.
   - "las diferencias con la cerrada aparecen lejos del centro": la figura muestra una oscilación de amplitud ~1e-4 en todo el dominio. Se reemplazó por esa descripción.
   - "el efecto del horizonte está por debajo de la oscilación": no se había comprobado, y es falso en el borde izquierdo del escape (−1.7e-4). Se reemplazó por el cálculo de ese borde.
5. **Cruce del escape cerca del extremo:** quedó en t = −2.00. La ventana alineada [−1, 0.5] empieza 7e-4 dentro del dominio (la prueba verifica que no se salga).

## 5. Figuras generadas (`figures/neuronal/`, PDF y PNG, sin fecha de creación)

Las 6 se regeneran idénticas byte a byte desde `results/`.

| Figura | Qué muestra |
|---|---|
| `caminos_red` | Camino de la red (alineado) frente a `x_om` y frente a `x_kink`, con la diferencia: oscilación de ~1e-4 (escape) y ~5e-5 (instantón) |
| `accion_entrenamiento` | Acción frente a evaluación (Adam y L-BFGS), en escala lineal con la cota marcada y como exceso sobre la cota en escala logarítmica: baja de ~1 a 5e-7 (escape) y 4e-8 (instantón) sin cruzar la cota |
| `red_sobre_tubo_reactivo` | Camino de la red sobre la banda 10-90 % y la mediana de las trayectorias reactivas de E4 para D = 0.25, 0.15 y 0.1 |

## 6. Pendientes y dudas para la revisión

- **Etiqueta `hito-02`:** no se creó, según lo pedido.
- **CLAUDE.md §3 frente a la especificación 02:** la superposición red-ruido está en `notebooks/02_neuronal/` y no en `03_integracion/` (D36). ¿Se deja así o se mueve?
- **Especificación:** se guardó el texto de los revisores tal cual como `specs/02_neuronal.md`, para que el hito tenga su especificación como los anteriores.
- **L-BFGS hasta convergencia estricta:** si se quiere, basta con subir `iteraciones_max`. No cambia ningún criterio.
- **Fuera del alcance, como pide la especificación:** pérdida PINN y estudios de horizonte, modo cero y selección de rama.
