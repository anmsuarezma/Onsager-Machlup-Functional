# Hito 01 — Bloque B: simulación estocástica

Especificación: `specs/01_estocastico.md`, con sus aclaraciones (1)-(15). Fecha: 2026-10-05. Decisiones: D19-D32 de `docs/decisiones.md`.

## 0. Objetivo y resultado físico que verifica

El Bloque B es el experimento físico del taller. Pone a prueba, contra el proceso real `dx = −V′(x) dt + √(2D) dW`, la predicción de la teoría variacional de ruido débil:
- el tiempo de escape escala como `e^{S_min} = e^{ΔV/D}` (ley de Arrhenius, con el exponente que fija la acción de Onsager-Machlup);
- las transiciones siguen el camino `x_om(t) = −1/√(1 + e^{8t})`, en un tubo que se estrecha cuando `D → 0`;
- el escape es un evento raro sin memoria (proceso de Poisson).

La vara es el tiempo medio de primer paso exacto `T(−1 → b)`, sin aproximaciones asintóticas, para dos definiciones de escape: llegar a la cima (`b = 0`) y caer al otro pozo (`b = +1`). Kramers queda como asintótica, y su desviación es un resultado.

Resultado: los criterios E0-E7 se cumplen todos (sección 3).

## 1. Entorno

Mismo sistema y entorno FMA del hito 00. Cambios de este hito (D19-D23, D28):

| Elemento | Valor |
|---|---|
| Python | 3.12.13 (uv) |
| numpy | 2.4.6 (antes 2.5.3; fijado `>=2.4,<2.5`, D20) |
| numba / llvmlite | 0.68.0 / 0.50.0 (grupo `estocastico`) |
| pyyaml | 6.0.3 (grupo `estocastico`; ya estaba como transitiva de Jupyter) |
| scipy / matplotlib | 1.18.1 / 3.11.2 (sin cambios) |
| Cómputo | CPU, Numba `@njit(parallel=True)`, capa de hilos `tbb` (`libtbb.so.12` del sistema); 10 hilos por defecto (D23) |
| GPU | no se usa en este bloque (D19: numba-cuda en modo de mantenimiento e incompatible con numpy 2.5) |

La regresión del hito 00 con numpy 2.4.6 se hizo antes de empezar (D20): 70 pruebas pasan, los valores clave son idénticos bit a bit y las figuras, byte a byte. Se reconfirmó el 2026-10-05, tras un apagado inesperado de la máquina: las seis figuras regeneradas con numpy 2.4.6 son idénticas byte a byte a las versionadas, que se generaron con 2.5.3. Ese mismo día, los 18 079 archivos instalados en FMA se compararon con los hashes de sus `RECORD`, sin archivos dañados.

## 2. Qué se implementó

### 2.1 `src/taller/analitico/referencias.py`

- `tiempo_primer_paso(D, b, x0=-1)`: `T(x0 → b) = (1/D) ∫_{x0}^{b} dy e^{V/D} ∫_{−∞}^{y} dz e^{−V/D}`, con quad anidado. La cola `(−∞, −1]` se calcula una vez y se corta en los puntos fijos. Error relativo máximo frente a mpmath a 40 dígitos: 5.6e-16 (decisión 2 de la especificación).

### 2.2 `src/taller/estocastico/` (importa solo de `taller.analitico`)

| Archivo | Contenido |
|---|---|
| `configuracion.py` | `cargar_config`, `resolver_hilos` (`--hilos` tiene prioridad), `aplicar_hilos` (devuelve hilos y capa), `semillas_trayectorias` (una semilla de 64 bits por trayectoria con `SeedSequence`, con detección de colisiones), `n_trayectorias` (N por D) |
| `integrador.py` | Euler-Maruyama en Numba con un prange sobre trayectorias. Generador xoshiro256** propio por trayectoria, iniciado con splitmix64; normales por el método polar (D28). `paso_em`, `prob_cruce_puente`, `simular_escape` (tiempos a 0 y a +1 con y sin puente browniano, alineación y ventanas de E7 en un búfer circular), `simular_equilibrio` (E2), `simular_browniano_libre` (validación del puente). V′ se compila desde `referencias.dV`, la única fuente |
| `referencia_numpy.py` | `simular_escape_numpy`: implementación vectorizada independiente, con PCG64 (E1) |
| `observables.py` | media y error estándar, CV, pendiente de Arrhenius, probabilidades de Boltzmann por intervalo, L1, alineación de ventanas, mediana y banda, RMS frente a `x_om` |
| `guardado.py` | `.npz` con metadatos en JSON (parámetros, semilla, fecha, commit, `arbol_modificado`, hilos, capa, versiones). Escritura atómica: archivo temporal y `os.replace` (D26) |
| `correr.py` | `python -m taller.estocastico.correr <config> [--hilos N] [--rehacer]`: E1, E2, E3 (un archivo por dt) y E4 (un archivo por D). Salta lo que ya existe; avisa si el árbol tiene cambios sin commit; se niega a correr E4 con `dt: null` |

### 2.3 Configuración (`configs/estocastico/`)

`comun.yaml` (10 hilos), `pruebas.yaml` (semillas de las pruebas, congeladas antes de implementar), `e1_validacion.yaml`, `e2_boltzmann.yaml`, `e3_convergencia_dt.yaml` y `e4_tiempos.yaml` (dt = 1e-3, D30). Nada aleatorio sin semilla desde configuración.

### 2.4 Pruebas (`tests/`, congeladas en el commit `2b3de27`)

- `tests/analitico/test_tiempo_primer_paso.py`: E0.
- `tests/estocastico/`: `oraculo_estocastico.py` (tablas de T exacto de mpmath, escritas a mano), `test_configuracion.py`, `test_integrador.py`, `test_puente_browniano.py`, `test_observables.py`, `test_guardado.py`, `test_validacion_estocastica.py` (E1 y E2, marcadas `lento`) y `test_criterios_produccion.py` (E3-E7 sobre `results/`).
- **Única modificación después de congelar:** `test_e3_dt_de_produccion`, autorizada explícitamente por los revisores (D31; sección 4).

### 2.5 Cuaderno `notebooks/01_estocastico/01_estocastico.py`

Numeración del plan del taller: B.0 (oráculo, §0), B.1 (E1), B.2 (control interno, E2), B.3 (frontera discreta, E3), B.4-B.5 (medición 2: tiempos y Arrhenius, E4-E5), B.6 (Poisson, E6), B.7 (medición 1: tubo reactivo, E7, §2) y B.8 (resumen, §6). Solo carga resultados guardados. Se ejecuta sin errores desde una sesión limpia (`jupytext --execute`, unos 12 s).

### 2.6 Corridas

| Corrida | Quién | Commit | Hilos | Tiempo |
|---|---|---|---|---|
| E3 (4 dt, N = 10⁵, D = 0.25) | revisores | 2f273cf (limpio) | 10 | 55 s |
| E4 (7 D, dt = 1e-3) | revisores | 4a8ce76 (marcado modificado; sección 4) | 14 | 40 min (D = 0.1: 22.9 min) |
| E1 réplica (D = 0.35, 10⁴ + 10⁴) | Claude | 8e29d7f (limpio) | 10 | 23 s |
| E2 réplica (D = 0.5, 10⁵ × 20 muestras) | Claude | 8e29d7f (limpio) | 10 | 21 s |

## 3. Pruebas

Suite completa en `22734bf`: **185 passed, 0 fallidas, 0 omitidas** (3 min 31 s), incluidas las marcadas `lento` y las 34 de producción.

### 3.1 Criterios de la especificación

| Criterio | Obtenido | Esperado |
|---|---|---|
| E0: quad frente a mpmath (7 D × 2 destinos) | error relativo máximo 5.6e-16 | < 1e-8 |
| E0: factor ½ (D ≤ 0.15) | `|T(0)/T(+1) − ½|` ≤ 8.4e-4 | < 5e-3 |
| E0: convergencia a Kramers | `T(+1)/T_K − 1` = 0.250, 0.203, 0.142, 0.107, 0.072, 0.057, 0.043 (D = 0.5 → 0.1), monótona | decreciente |
| E1: KS Numba frente a NumPy, tiempos a +1 (prueba) | p = 0.165 (corregido), 0.145 (sin corregir) | > 0.01 |
| E2: L1 Boltzmann, D = 0.5 (prueba) | 0.0046 | < 0.02 |
| E3: error con puente, D = 0.25 | todos < 2 %; ver tabla 3.2 | < 2 % |
| E3: monotonía entre dt con error > 2 EE | cumple en ambos destinos | decreciente |
| E3: dt de E4 validado por E3 | 1e-3, válido | (aclaración 15) |
| E4: `|T_sim/T_exacto − 1|`, 7 D × 2 destinos | máximo 0.62 %; tabla 3.3 | < 3 % |
| E5: `|pendiente_sim − pendiente_exacta|`, D ≤ 0.2 | 0.0013 (cima: 0.9887 frente a 0.9900), 0.0008 (pozo: 0.9876 frente a 0.9884) | < 0.05 |
| E6: `|CV − 1|`, D ≤ 0.15 | 0.0050 (0.1), 0.0013 (0.125), 0.0014 (0.15) | < 0.05 |
| E7: RMS(mediana − x_om) en t ∈ [−0.5, 0.25] | 0.1197 (D = 0.25), 0.0832 (0.15), 0.0582 (0.1) | decreciente |

Validación del puente browniano (aclaración 7), antes de usarlo:

| Prueba | Corregido | Sin corregir | Criterio |
|---|---|---|---|
| Browniano sin deriva, `P(τ_1 ≤ 0.5)`, exacta 0.15730 (erfc 1) | 0.15850 (+1.04 EE) | 0.11534 (−36 EE) | corregido < 4 EE; sin corregir < exacta − 0.02 |
| Browniano sin deriva, `P(τ_1 ≤ 1)`, exacta 0.31731 | 0.31723 (−0.05 EE) | 0.26124 (−38 EE) | ídem |
| Doble pozo, D = 0.25, dt = 5e-3, cima | −1.25 % ± 0.31 % | +7.6 % | corregido < 2 %; sin corregir > 5 % |
| Doble pozo, ídem, pozo | −1.02 % ± 0.31 % | −0.89 % | corregido < 2 % |

### 3.2 E3 (D = 0.25, N = 10⁵; error relativo en %, ± error estándar)

| dt | cima corregida | pozo corregido | cima sin corregir | predicho (continuidad) | pozo sin corregir |
|---|---|---|---|---|---|
| 1e-2 | −1.29 ± 0.31 | −0.54 ± 0.31 | +11.36 ± 0.35 | +12.51 | −0.39 ± 0.31 |
| 5e-3 | −0.33 ± 0.31 | −0.11 ± 0.31 | +8.74 ± 0.34 | +8.87 | +0.02 ± 0.31 |
| 1e-3 | −0.10 ± 0.31 | +0.08 ± 0.31 | +3.71 ± 0.32 | +3.97 | +0.12 ± 0.31 |
| 5e-4 | −0.13 ± 0.31 | +0.18 ± 0.31 | +2.56 ± 0.32 | +2.81 | +0.23 ± 0.31 |

Predicho: frontera desplazada en `δ = 0.5826·√(2D dt)`, y sesgo `T(−1 → δ)/T(−1 → 0) − 1`.

### 3.3 E4 (dt = 1e-3; error relativo en %, ± error estándar)

| D | N | cima corregida | pozo corregido | cima sin corregir | predicho | T(0)/T(+1) sim / exacto |
|---|---|---|---|---|---|---|
| 0.1 | 20 000 | −0.29 ± 0.70 | −0.19 ± 0.70 | +3.87 ± 0.74 | +4.07 | 0.4994 / 0.5000 |
| 0.125 | 100 000 | −0.59 ± 0.31 | −0.56 ± 0.31 | +3.46 ± 0.32 | +4.04 | 0.4996 / 0.4998 |
| 0.15 | 100 000 | +0.11 ± 0.32 | +0.31 ± 0.32 | +4.23 ± 0.33 | +4.02 | 0.4982 / 0.4992 |
| 0.2 | 100 000 | +0.22 ± 0.31 | +0.03 ± 0.31 | +4.00 ± 0.33 | +3.98 | 0.4970 / 0.4960 |
| 0.25 | 100 000 | +0.28 ± 0.31 | +0.19 ± 0.31 | +4.37 ± 0.33 | +3.97 | 0.4905 / 0.4901 |
| 0.35 | 100 000 | −0.62 ± 0.31 | +0.08 ± 0.31 | +3.41 ± 0.32 | +4.01 | 0.4704 / 0.4738 |
| 0.5 | 100 000 | −0.33 ± 0.30 | −0.32 ± 0.29 | +3.93 ± 0.31 | +4.17 | 0.4490 / 0.4490 |

### 3.4 Resultados físicos sin criterio de aceptación

- **Pendiente de Arrhenius exacta en D ≤ 0.2:** 0.9884 (pozo) y 0.9900 (cima), no ΔV = 1. Las correcciones al prefactor dependen de D, y el ajuste en un intervalo finito las absorbe en la pendiente. La pendiente local exacta `d ln T/d(1/D)` del pozo vale 0.9950 (D = 0.1), 0.9865 (0.15), 0.9742 (0.2) y 0.9496 (0.35): tiende a 1 cuando D → 0.
- **Kramers frente al exacto:** `T/T_K` va de 1.043 (D = 0.1) a 1.250 (D = 0.5). El exponente es correcto; el prefactor no, a D finito.
- **Sesgo de la cima sin corregir con dt = 1e-3:** ≈ +4 % e independiente de D (media +3.90 %, entre +3.41 y +4.37 %; predicho entre +3.97 y +4.17 %).
- **CV fuera del criterio:** 0.991 (D = 0.2), 0.985 (0.25), 0.967 (0.35), 0.931 (0.5). Baja de 1 cuando la barrera deja de ser alta.
- **E7:** la banda 10-90 % en t = 0 se estrecha (0.066, 0.051, 0.041). La mediana de `t_cima − t_alineación` crece (0.228, 0.313, 0.383; ~0.17 por unidad de ln(1/D)). Ventanas con NaN al inicio: 7.50 % (D = 0.25), 0.50 %, 0.03 %.

### 3.5 Réplicas guardadas para el cuaderno (D32)

- **E1** (semillas propias): KS p = 0.111 en el pozo, corregido y sin corregir (cumple); en la cima, 0.052 y 0.023 (informativo).
- **E2:** L1 = 0.0044; el ruido estadístico esperado con muestras independientes es 0.0042 ± 0.0004. Fracción en x > 0: 0.5002.

## 4. Problemas encontrados y cómo se resolvieron

1. **Apagado inesperado de la máquina** antes de empezar. El entorno se verificó con `uv sync` y con los hashes de los 18 079 archivos instalados: nada dañado. Se completaron las tareas que habían quedado a medias: la sección de aclaraciones de la especificación faltaba.
2. **Sesgo de la frontera discreta en la cima** (detectado en el plan, D24). Con una simulación de prueba: +3.4 % con dt = 5e-3 en `T(−1 → 0)` y nada en `T(−1 → +1)`. Ningún dt de la lista habría cumplido E3 en la cima. Se adoptó la corrección de puente browniano, validada con el browniano sin deriva, y se conservan los tiempos sin corregir.
3. **Semillas de 32 bits** (D28). El plan reiniciaba el generador MT19937 de Numba con una semilla por trayectoria, pero solo acepta 32 bits: con 10⁵ trayectorias se esperaba ~1 par de trayectorias idénticas. Se detectó al implementar, antes de usarlo, y se reemplazó por xoshiro256** propio con semillas de 64 bits. Validación: bits idénticos a una implementación en Python puro y al valor publicado de splitmix64(0); 2·10⁶ normales con KS p = 0.70; sin correlación entre corrientes más allá del azar. El rendimiento subió de 1.6·10⁸ a 3.9·10⁸ pasos/s con 10 hilos.
4. **`arbol_modificado` ignoraba los archivos no versionados** (D29). Detectado en la prueba de humo de `correr.py`; corregido antes de cualquier corrida.
5. **Elección del dt de producción** (D30, D31). La regla literal de E3 elegía dt = 1e-2, porque los cuatro dt cumplen el 2 %. Los revisores eligieron 1e-3, como opción conservadora: con la corrección queda un sesgo de orden dt (−1.29 % ± 0.31 % en la cima con 1e-2). La prueba congelada `test_e3_dt_de_produccion` exigía el mayor dt válido; con autorización explícita, ahora exige un dt evaluado y validado en E3 (aclaración 15).
6. **Discrepancia entre la validación del puente y E3 con dt = 5e-3:** −1.02 % frente a −0.11 % en el pozo. La causa son las semillas distintas, nada más: mismo N, mismo generador, y E3 se reproduce bit a bit. Con 12 lotes independientes de 10⁵, el sesgo real con dt = 5e-3 es −0.48 % ± 0.07 % (pozo) y −0.63 % ± 0.11 % (cima); la validación cayó 1.7 EE por debajo y E3, 1.2 EE por encima. La diferencia es de 2.1 errores estándar combinados (√2·0.31 %), no de 3 como se estimó inicialmente. Registrado en D30.
7. **Trazabilidad de E4.** Los siete archivos tienen `arbol_modificado = True` y se corrieron con 14 hilos.
   - **Qué estaba sin commit:** solo `log_E4.log`, el registro de consola de la propia corrida, sin versionar en la raíz. Lo creó la redirección de la salida antes de que arrancara Python, y por eso activó el aviso.
   - **Por qué no afecta a los datos:** ningún archivo versionado se modificó después de `4a8ce76` (12:21:47); la corrida empezó hacia las 12:28. `git diff HEAD -- src configs` está vacío, y lo último que se modificó en `src/` y `configs/` es de las 11:54 y las 12:16, ambos con el contenido del commit.
   - **Verificación directa:** los archivos de D = 0.5, 0.35 y 0.25 (este con ventanas) se regeneraron con el código de HEAD y 10 hilos, y son idénticos bit a bit a los guardados. El número de hilos no cambia el resultado (D27). Los resultados son válidos y E4 no se repitió.
   - **Para evitarlo:** `*.log` se ignora en git (D32).
8. **p bajos de KS en la cima en la réplica de E1** (0.052 y 0.023), y un p = 0.001 en el pozo en 1 de 4 réplicas adicionales de 10⁴ + 10⁴ (investigación en el scratchpad). Se descartó una diferencia sistemática con muestras grandes: 10⁶ trayectorias de Numba frente a 10⁵ de NumPy dan KS p = 0.87 (cima), 0.72 (cima sin corregir), 0.93 (pozo) y 0.95 (pozo sin corregir). Las dos implementaciones coinciden con T exacto dentro de un error estándar (Numba: −0.05 % ± 0.10 % en la cima y −0.09 % ± 0.10 % en el pozo). Los p bajos con muestras de 10⁴ son fluctuaciones.
9. **E1 y E2 no guardaban datos:** las pruebas no escriben en `results/`. Se agregó E1 a `correr.py` y se corrieron réplicas con semillas propias para las figuras (D32).

## 5. Figuras generadas (`figures/estocastico/`, PDF y PNG, sin fecha de creación)

Las 16 se regeneran idénticas byte a byte desde `results/`.

| Figura | Qué muestra |
|---|---|
| `e1_numba_numpy` | Supervivencia de los tiempos a +1, Numba frente a NumPy. Coinciden hasta P ~ 10⁻²; más abajo, las colas fluctúan con ~100 eventos |
| `e2_boltzmann` | Histograma de posiciones con D = 0.5 frente a `e^{−V/D}/Z` (L1 = 0.0044) |
| `e3_convergencia_dt` | Error frente a dt con y sin puente, más el sesgo predicho por la corrección de continuidad. El sesgo sin corregir escala como √dt y sigue la predicción; con el puente queda dentro de ±2 % |
| `e4_sesgo_cima` | Sesgo de la cima sin corregir frente a D: ≈ +4 % constante, como predice la corrección de continuidad; con el puente, compatible con 0 |
| `e5_arrhenius` | ln T frente a 1/D (simulado en ambos destinos, exacto y Kramers) con las pendientes, y `T/T_exacto` frente a D (la simulación en 1; Kramers baja hasta 0.80) |
| `e6_distribucion_tiempos` | Supervivencia de `t/⟨T⟩` para D = 0.5, 0.25 y 0.1 frente a la exponencial: recta cuando la barrera es alta; se desvía con D = 0.5 (CV = 0.931) |
| `e7_tubo_reactivo` | Mediana y banda 10-90 % de las ventanas alineadas frente a `x_om`, para D = 0.25, 0.15 y 0.1, con el intervalo del criterio sombreado. Antes de t = 0 la mediana sigue a `x_om`; después va por delante, porque el ruido alcanza la cima en un tiempo finito |
| `e7_densidad_ventanas` | Densidad 2D (t, x) de las ventanas alineadas: el tubo se estrecha al bajar D |

## 6. Pendientes y dudas para la revisión

- **Etiqueta `hito-01`:** no se creó, según lo pedido; queda para después de la revisión.
- **Mecanismo del sesgo residual de orden dt** con la corrección. E3 lo establece en la cima (−0.63 % ± 0.11 % con dt = 5e-3), pero no su mecanismo ni su dependencia con D. La estimación simple del calentamiento numérico dentro del pozo (D_eff = D/(1 − 4dt)) predice ≈ −8 %, mucho más de lo medido, así que no es la explicación. Con dt = 1e-3, E4 no muestra sesgo en ningún D.
- **E7 y el tramo final:** el criterio usa `t ∈ [−0.5, 0.25]`, donde `x_om ≤ −0.345`. La mediana se adelanta a `x_om` cerca de la cima, porque `x_om` tarda un tiempo infinito en llegar. ¿Conviene cuantificar en la parte teórica la escala logarítmica (~0.17 por unidad de ln(1/D)) del tiempo de llegada del ruido a la cima?
- **Superposición con el camino de la red** (medición 1 del plan): queda para `notebooks/03_integracion/`, cuando exista el Bloque A.
- **Réplica de E1 en el cuaderno:** usa semillas distintas de la prueba, así que sus p (0.111) no son los de la prueba (0.165 y 0.145). Ambos se reportan.
- **Paleta:** igual que en el hito 00, el validador de la guía de visualización requiere `node`, que no está instalado. Se usaron los colores de la paleta validada (D16) y la escala secuencial azul, con codificación secundaria (forma del marcador y estilo de línea).
