# Hito 02 — Bloque A reducido: la red variacional

Especificación: `specs/02_neuronal.md` (texto de los revisores, tal cual). Fecha: 2026-10-05; revisión de dependencias y reentrenamiento en CPU tras el apagado: 2026-10-06 (sección 7). Decisiones: D34-D41 de `docs/decisiones.md`.

## 0. Objetivo y resultado físico que verifica

Minimizar directamente los dos funcionales del taller con una red neuronal (método directo del cálculo de variaciones, tipo Ritz; forma débil), sin pasar por la ecuación de Euler-Lagrange y sin datos, y comparar contra las soluciones cerradas del hito 00:
- escape térmico: `S·D = (1/4)∫(ẋ + V′)² dt` en [−T_h, T_h] con horizonte T_h = 3 (D40), x(−3) = −1 y x(3) = 0; referencias S·D → ΔV = 1 y `x_om`;
- instantón: `S_E = ∫[½ẋ² + V] dτ` en [−T_h, T_h] con T_h = 4, con x(−4) = −1 y x(4) = +1; referencias S0 = 4√2/3 y `x_kink`.

Verifica las matemáticas del taller: que el minimizador del funcional y su acción son los de §2 y §3, y que la acción no baja de las cotas por completar cuadrados.

Resultado: todos los criterios se cumplen con varios órdenes de margen (sección 3), con los resultados vigentes, entrenados en CPU con 8 hilos en el commit `d8dc92d`.

## 1. Entorno

Grupo `neuronal` nuevo (D34, reemplazado por D38 en la revisión de dependencias):

| Paquete | Versión |
|---|---|
| torch | 2.14.1+cpu (índice CPU de PyTorch, explícito y asignado solo a torch); desde D38, 2.14.1+cu130 (paquete estándar, runtime CUDA 13.0) |
| filelock / fsspec / networkx / setuptools | 4.0.12 / 2026.9.0 / 3.7 / 84.0.0 (transitivas de torch) |

- **Versiones existentes:** ninguna cambió; se comprobó comparando la lista completa de paquetes antes y después. numpy sigue en 2.4.6.
- **CPU:** `torch.version.cuda = None` y `torch.cuda.is_available() = False`; no se configuró GPU, como pide la especificación. Prueba mínima de autograd en float64 correcta.
- **Suite completa tras instalar (CLAUDE.md §12):** 185 passed.
- **Revisión de dependencias (D38):** con torch 2.14.1+cu130 solo cambió torch; numpy sigue en 2.4.6; suite completa 231 passed con la GPU visible (sesión anterior al apagado).
- **Verificación tras el apagado (2026-10-06):** `uv sync` sin cambios (135 paquetes); `sys.prefix` = entorno FMA, numpy 2.4.6, torch 2.14.1+cu130. Se verificaron los hashes de RECORD de los 32 401 archivos instalados: ningún paquete dañado, nada se reinstaló.
- **Hilos y dispositivo:** PyTorch con 10 hilos hasta el apagado; desde D41, solo CPU con 8 hilos (`dispositivo: cpu`, `hilos: 8`; las banderas `--dispositivo` y `--hilos` tienen prioridad).

## 2. Qué se implementó

### 2.1 `src/taller/neuronal/` (importa solo de `taller.analitico`)

| Archivo | Contenido |
|---|---|
| `red.py` | `crear_red(capas, semilla, escala_ultima_capa, dispositivo, dtype)`: perceptrón 1 → capas → 1 en float64, tanh y salida lineal. Pesos ocultos Xavier; sesgos ocultos uniformes en ±1/√(entradas) (D35); última capa Xavier × 1e-3 y sesgo nulo. Generador propio con semilla. `contar_parametros` |
| `ansatz.py` | `camino`: x = x_a + (x_b − x_a)(t + T_h)/(2T_h) + [(t + T_h)(T_h − t)/T_h²]·N(t/T_h), fronteras exactas. `camino_y_derivada`: ẋ por autograd con `create_graph=True` |
| `accion.py` | `accion_escape` (S·D) y `accion_instanton` (S_E) con `torch.trapezoid`; V y V′ de `referencias`, que operan también con tensores |
| `entrenamiento.py` | `entrenar`: Adam y después L-BFGS (Wolfe fuerte) sobre la malla fija completa; registra la acción en cada evaluación. `evaluar_en_malla_doble`: camino, ẋ y acción en 4001 puntos |
| `guardado.py` | `.npz` con metadatos y escritura atómica; duplica a propósito el del bloque estocástico, que no se puede importar (CLAUDE.md §3) |
| `dispositivo.py` | `elegir_dispositivo` (cuda, si no mps, si no cpu, o el forzado), `precision` (float64; float32 con aviso en mps), `nombre_dispositivo` (D39) |
| `correr.py` | `python -m taller.neuronal.correr configs/neuronal/entrenamiento.yaml [--dispositivo {cuda,mps,cpu}] [--hilos N] [--salida DIR] [--rehacer]`: 2 problemas × 3 arquitecturas, un archivo por combinación; dispositivo, nombre y precisión en los metadatos |

### 2.2 Configuración

`configs/neuronal/entrenamiento.yaml`, congelada con las pruebas:
- horizonte T_h = 3 y T_h = 4 (clave `horizonte`, antes `T`; D40); malla de 2001 puntos;
- `hilos: 8` y `dispositivo: cpu` (D41; antes 10 y `null`);
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

- `test_dispositivo.py` (D39): selección automática y forzada, float64 en cpu y cuda, float32 con aviso en mps, bandera, metadatos y `test_el_calculo_corre_en_la_gpu` (se omite sin GPU CUDA).

Única modificación de pruebas congeladas: el renombre autorizado `T` → `horizonte` (D40, commit `ce6b063`), sin cambiar valores ni tolerancias.

### 2.4 Cuaderno `notebooks/02_neuronal/02_neuronal.py`

Secciones A.0-A.6 con la numeración del plan: método directo (§5′), escape (§2), instantón (§3), acción durante el entrenamiento y cotas, robustez, superposición sobre el tubo reactivo de E7 y resumen. Solo carga resultados. Se ejecuta sin errores desde una sesión limpia (`jupytext --execute`).

### 2.5 Entrenamiento

- Primera corrida: commit `b454b38` (árbol limpio), CPU, 10 hilos, 4 min 16 s para las 6 redes. Sus resultados se sobrescribieron en la corrida en GPU (sección 7).
- **Resultados vigentes:** commit `d8dc92d` (árbol limpio), CPU (`--dispositivo cpu --hilos 8`, con `CUDA_VISIBLE_DEVICES=""`), float64; 256.8 s en total para las 6 redes. Una corrida previa idéntica (mismo código y configuración, pero con el `.py` del cuaderno modificado en el árbol) dio resultados **idénticos bit a bit** en los 6 archivos: en CPU con un número fijo de hilos el entrenamiento es determinista.

## 3. Pruebas

Suite completa con los resultados vigentes (CPU, `CUDA_VISIBLE_DEVICES=""`): **230 passed, 1 skipped** (`test_el_calculo_corre_en_la_gpu`, sin GPU visible), 0 fallidas (3 min 27 s). En la primera versión del hito: 222 passed.

### 3.1 Criterios de aceptación (arquitectura por defecto, 2 × 32)

| Criterio | Escape térmico | Instantón | Exigido |
|---|---|---|---|
| Acción | S·D = 1.0000005090 | S_E = 1.8856183008 (S0 = 1.8856180832) | — |
| Error relativo de la acción | +5.09e-7 | +1.15e-7 | < 1e-2 |
| RMS alineado frente a la cerrada | 4.77e-5 en t ∈ [−1, 0.5] | 2.77e-5 en τ ∈ [−2, 2] | < 1e-2 |
| Cota analítica | S·D − 1 = +5.09e-7 | S_E − S0 = +2.18e-7 | ≥ −1e-4 |
| Malla doble (4001 puntos), diferencia relativa | 2.6e-11 | 1.1e-12 | < 1e-4 |
| Cruce de alineación | t = −1.9999 (x = −1/√2) | τ = +0.6018 (x = 0) | — |

En ningún momento del entrenamiento la acción bajó de la cota: el mínimo de toda la historia es 1 + 5.1e-7 y S0 + 2.2e-7.

### 3.2 Robustez ante la arquitectura

| Problema | Arquitectura | Parámetros | Acción | Error relativo | RMS | Cruce | Malla doble | Tiempo (s) | Evaluaciones de L-BFGS |
|---|---|---|---|---|---|---|---|---|---|
| escape | 2 × 32 | 1153 | 1.0000005090 | +5.09e-7 | 4.77e-5 | −2.000 | 2.6e-11 | 23.9 | 6250 |
| escape | 3 × 32 | 2209 | 1.0000001920 | +1.92e-7 | 3.24e-5 | −1.825 | 2.5e-13 | 36.6 | 6090 |
| escape | 4 × 64 | 12673 | 1.0000006032 | +6.03e-7 | 1.42e-5 | −1.847 | 1.9e-12 | 76.6 | 6251 |
| instantón | 2 × 32 | 1153 | 1.8856183008 | +1.15e-7 | 2.77e-5 | +0.602 | 1.1e-12 | 18.9 | 6251 |
| instantón | 3 × 32 | 2209 | 1.8856181170 | +1.79e-8 | 1.16e-5 | −0.211 | 2.1e-13 | 7.5 | 311 |
| instantón | 4 × 64 | 12673 | 1.8856181444 | +3.25e-8 | 1.86e-5 | −0.025 | 3.1e-14 | 93.2 | 6250 |

Tiempos en CPU: i9-10900KF, 8 hilos de PyTorch, float64 (D41). La medición en CUDA queda pendiente (sección 6).

Las tres arquitecturas coinciden dentro de la tolerancia: la acción difiere entre ellas en menos de 5e-7 (escape) y 2e-7 (instantón).

**Qué limita la precisión:** al nivel de la tolerancia, ni el horizonte ni la red. Por debajo, la optimización: el error no baja al aumentar 11 veces el número de parámetros, ni es monótono con el tamaño. La red intermedia (3 × 32) da la acción más baja en ambos problemas y es la única que L-BFGS llevó a su tolerancia (6090 y 311 evaluaciones); las demás se detuvieron por el límite (6250). Con la misma configuración y semilla, pasar de 10 a 8 hilos (otro orden de las sumas) movió el instantón por defecto de S_E − S0 = 4.2e-8 y cruce +0.17 a 2.2e-7 y +0.60. La dirección difícil es el modo casi plano de traslación: el cruce cambia con la arquitectura y con el orden de las sumas. El aporte del horizonte finito no se puede separar sin variar T_h (estudio 1 del plan, fuera del alcance). Su huella sí se ve en los bordes: en el borde izquierdo de la ventana alineada del escape, donde la red vale −1 y `x_om(−1) = −1 + 1.7e-4`, y en la cola derecha del instantón, cuyo borde queda a 3.40 del cruce, donde la red vale +1 y `x_kink(3.40) = 1 − 1.3e-4` (fuera de la ventana del RMS; dentro de ella la diferencia no pasa de 7.5e-5).

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
5. **Cruce del escape cerca del extremo:** en la primera corrida quedó en t = −1.9993 (ventana alineada 7e-4 dentro del dominio); con los resultados vigentes, en t = −1.9999 (1.4e-4 dentro). La prueba verifica que no se salga, y con la corrida en GPU sí se salió (sección 7).

## 5. Figuras generadas (`figures/neuronal/`, PDF y PNG, sin fecha de creación)

Regeneradas desde los resultados vigentes (CPU, `d8dc92d`); dos ejecuciones seguidas del cuaderno dan archivos idénticos byte a byte.

| Figura | Qué muestra |
|---|---|
| `caminos_red` | Camino de la red (alineado) frente a `x_om` y frente a `x_kink`, con la diferencia: oscilación de ~1e-4 (escape); en el instantón ≤ 7.5e-5 en la ventana y hasta 1.9e-4 en la cola derecha (borde del horizonte) |
| `accion_entrenamiento` | Acción frente a evaluación (Adam y L-BFGS), en escala lineal con la cota marcada y como exceso sobre la cota en escala logarítmica: baja de ~1 a 5.1e-7 (escape) y 2.2e-7 (instantón) sin cruzar la cota |
| `red_sobre_tubo_reactivo` | Camino de la red sobre la banda 10-90 % y la mediana de las trayectorias reactivas de E4 para D = 0.25, 0.15 y 0.1 |

## 6. Pendientes y dudas para la revisión

- **Etiqueta `hito-02`:** no se creó, según lo pedido.
- **Medición de tiempos en CUDA: pendiente** (D41). Los tiempos de la sección 3.2 son solo de CPU. Los seis resultados entrenados en la GPU antes del apagado se conservan en `results/neuronal/previos_apagado/cuda/` (39.6-56.7 s por red, con 10 hilos en el anfitrión), pero no hay una comparación en CPU completa hecha en las mismas condiciones, así que no se reportan como medición.
- **Fragilidad del criterio RMS del escape:** con el horizonte T_h = 3, la ventana alineada [−1, 0.5] solo cabe si el cruce cae en t ≥ −2. El cruce lo fija el modo casi plano de traslación y depende del orden de las sumas: en CPU queda en −1.9993/−1.9999 (pasa por 7e-4/1.4e-4), en GPU en −2.0007 (falla). No es un error del camino (RMS y acción están igual de bien), pero el criterio, tal como está, no es robusto al dispositivo ni al número de hilos. No cambié la prueba, la tolerancia, el horizonte ni la semilla; ¿cómo quieren tratarlo?
- **Hilos de Numba:** D41 limita PyTorch a 8 hilos; las pruebas del bloque estocástico siguen con los 10 hilos de Numba de D23. ¿Se limita también Numba?
- **CLAUDE.md §3 frente a la especificación 02:** la superposición red-ruido está en `notebooks/02_neuronal/` y no en `03_integracion/` (D36). ¿Se deja así o se mueve?
- **Especificación:** se guardó el texto de los revisores tal cual como `specs/02_neuronal.md`, para que el hito tenga su especificación como los anteriores.
- **L-BFGS hasta convergencia estricta:** si se quiere, basta con subir `iteraciones_max`. No cambia ningún criterio.
- **Fuera del alcance, como pide la especificación:** pérdida PINN y estudios de horizonte, modo cero y selección de rama.

## 7. Revisión de dependencias, apagado y reentrenamiento en CPU (2026-10-06)

1. **Revisión de dependencias (D38-D40, commits `ce6b063`, `94f9db3`, `c5c4cd7`):** PyTorch estándar con CUDA, selección de dispositivo y renombre T → horizonte. D38-D40 se citaban en los commits pero no se habían escrito en `decisiones.md`; se registraron después del apagado.
2. **Qué corría al apagarse:** tras `c5c4cd7`, (a) se entrenaron las 6 redes en la GPU (00:07-00:11, completo; sobrescribieron los resultados de la primera corrida en `results/neuronal/`); (b) empezó la corrida de comparación de tiempos en CPU con 10 hilos (`--dispositivo cpu --salida results/neuronal/cpu --rehacer`). La máquina se apagó hacia las 00:13, durante la segunda red de esa corrida (quedó un `.tmp` de `escape_tres_capas_32`). El apagado ocurrió bajo carga de CPU, no de GPU.
3. **Entorno:** intacto (sección 1).
4. **Suite con los resultados de la GPU:** 229 passed, 1 skipped, **1 failed**: `test_escape_rms_alineado`, porque el cruce quedó en t = −2.0007 y la ventana alineada salía del dominio (sección 6). Al limitar Numba con `NUMBA_NUM_THREADS=8` fallaron además 4 pruebas estocásticas porque la configuración pide 10 hilos; sin esa variable pasan (90 passed). Fue un error mío al lanzar la suite, no del código.
5. **Cuaderno inconsistente con el código:** leía la clave `T` de los metadatos, que desde `c5c4cd7` se llama `horizonte`: desde una sesión limpia habría fallado con KeyError. Se renombró a T_h en el código y el texto (D40) y se muestra el dispositivo y la precisión (commit `d8dc92d`).
6. **Reentrenamiento en CPU (D41):** los resultados de la GPU y el único completado de la corrida interrumpida se apartaron, sin borrar, a `results/neuronal/previos_apagado/`. Se reentrenó en CPU con 8 hilos. La primera vez, el árbol tenía el cuaderno modificado (lo edité mientras entrenaba); se hizo commit y se reentrenó con el árbol limpio: resultados idénticos bit a bit.
7. **El refactor no cambió los números en CPU:** el resultado completado de la corrida interrumpida (`c5c4cd7`, CPU, 10 hilos) da S·D = 1.0000005166935544, igual a la primera corrida del hito (1.0000005167, antes del renombre y del cambio de dispositivo). Lo que cambia los números es el número de hilos o el dispositivo (orden de las sumas), no el código.
8. **Cuaderno y bitácora:** los números del texto (A.1-A.6) se actualizaron a los resultados vigentes. Cambiaron algunas afirmaciones: L-BFGS se detuvo por el límite en 4 de 6 redes (antes 5); la red más grande ya no da la acción más alta en el instantón (ahora la intermedia da la más baja en ambos problemas); en el instantón aparece una cola de borde en la diferencia. Las conclusiones (criterios, robustez, límite por optimización) no cambian.
