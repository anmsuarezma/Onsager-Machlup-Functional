=====================================================
PARTE B — SIMULACIÓN ESTOCÁSTICA (especificación, plan y pruebas)
=====================================================

Guarda esta Parte B, tal cual, como specs/01_estocastico.md.

PROPÓSITO
El Bloque B es el experimento físico del taller: pone a prueba contra el proceso real la predicción de la teoría variacional de ruido débil, es decir, que las transiciones siguen el camino de Onsager-Machlup y que su tiempo escala como e^(ΔV/D). Las redes neuronales (Bloque A) verifican las matemáticas; este bloque verifica la física.

SISTEMA
dx = −V′(x) dt + √(2D) dW, con V(x) = (x² − 1)², integrado con Euler-Maruyama. Condición inicial fija x0 = −1.

DECISIONES DE DISEÑO (aprobadas)
1. Escape: en cada trayectoria se registran dos tiempos de primer paso: por la cima (x = 0) y por el otro pozo (x = +1). La simulación de cada trayectoria termina al llegar a x = +1.
2. Oráculo exacto: el tiempo medio de primer paso exacto se incorpora ahora a src/taller/analitico/referencias.py, como función del destino b ∈ {0, +1}:
   T(x0 → b) = (1/D) ∫_{x0}^{b} dy e^{V(y)/D} ∫_{−∞}^{y} dz e^{−V(z)/D}
   Es la vara principal de este bloque. Kramers queda como asintótica, cuya desviación es un resultado físico.
3. Valores de D: {0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5}.
4. Paso de tiempo: Euler-Maruyama detecta los cruces solo en instantes discretos, así que pierde cruces entre pasos y sesga el tiempo de primer paso hacia arriba. Antes de cualquier resultado se hace un estudio de convergencia en dt ∈ {1e-2, 5e-3, 1e-3, 5e-4} con D = 0.25, y se elige el dt de producción a partir de él.
5. Implementación: Numba CUDA para producción, más una implementación de referencia en NumPy (pocas trayectorias) que sirve para validar el kernel. Cada hilo integra una trayectoria completa y registra sus observables sobre la marcha; no se guardan trayectorias completas.
6. Condición inicial x0 = −1 fija.
7. Trayectorias reactivas: para cada trayectoria se guarda una ventana de duración 3 antes de su primer paso por x = 0, muestreada cada 0.01. Las ventanas se alinean en el último cruce por x = −1/√2 antes de llegar a 0 (la misma convención de origen de tiempo de x_om) y se comparan con x_om(t).

ORGANIZACIÓN
- src/taller/estocastico/: integrador NumPy de referencia, kernel Numba CUDA, funciones de observables y análisis. Importa solo de taller.analitico.
- configs/estocastico/: archivos YAML con D, dt, número de trayectorias y semillas. Nada aleatorio sin semilla desde configuración.
- results/estocastico/: resultados crudos (.npz) con metadatos (parámetros, semilla, fecha, hash del commit, versión de CUDA).
- notebooks/01_estocastico/: cuaderno que carga los resultados guardados, grafica y explica, con la estructura de siempre (objetivo, resultado esperado, cálculo, verificación, interpretación física).
- figures/estocastico/: figuras generadas desde los resultados guardados.

EXPERIMENTOS Y CRITERIOS DE ACEPTACIÓN

E0. Oráculo exacto (en analitico, pruebas nuevas en un archivo nuevo; las pruebas congeladas del hito 00 no se tocan)
- T(−1 → b) evaluado con quad coincide con una evaluación independiente con mpmath en alta precisión: error relativo < 1e-8 para todos los D y ambos destinos.
- Factor ½: |T(−1 → 0)/T(−1 → +1) − 0.5| < 5e-3 para D ≤ 0.15.
- Convergencia a Kramers: |T(−1 → +1)/τ_Kramers − 1| decrece monótonamente al disminuir D en el conjunto de D.

E1. Validación del kernel CUDA contra NumPy
- Con los mismos parámetros (D = 0.35, dt = 1e-3, 10⁴ trayectorias en cada implementación, semillas distintas), las distribuciones de tiempos de primer paso a x = +1 son estadísticamente compatibles: prueba de Kolmogorov-Smirnov de dos muestras con p > 0.01.

E2. Equilibrio de Boltzmann (control físico del integrador)
- Con D = 0.5 (donde hay muchas transiciones y se alcanza el equilibrio en ambos pozos), el histograma de posiciones a tiempos largos coincide con p_s ∝ e^(−V/D) normalizada: distancia L1 entre densidades < 0.02 con dt = 1e-3. Ojo: con D pequeño la partícula solo equilibra dentro de un pozo; por eso esta prueba se hace con D = 0.5.

E3. Convergencia en dt (D = 0.25)
- El error relativo |T_sim/T_exacto − 1| decrece al disminuir dt. Se grafica contra dt.
- El dt de producción es el mayor dt cuyo error relativo es < 0.02. Si ninguno lo cumple, se reporta y se detiene: no se amplía la lista de dt sin aprobación.

E4. Tiempos de escape contra el oráculo exacto (todos los D, ambos destinos, dt de producción)
- 10⁵ trayectorias por D. Se permite reducir N para D = 0.1 si el costo lo exige, justificándolo y reportando el error estándar resultante.
- Criterio: |T_sim/T_exacto − 1| < 0.03 para todos los D y ambos destinos, reportando también el error estándar de cada media.

E5. Ley de Arrhenius
- Ajuste lineal de ln T contra 1/D para D ≤ 0.2, tanto con los datos simulados como con el oráculo exacto en los mismos D.
- Criterio: |pendiente_sim − pendiente_exacta| < 0.05.
- Reportar (sin criterio de aceptación, es un resultado físico) cuánto se aleja la pendiente exacta de ΔV = 1 en ese rango y por qué: las correcciones al prefactor dependen de D.

E6. Escape como evento de Poisson
- Para D ≤ 0.15, el coeficiente de variación (desviación estándar/media) de los tiempos de primer paso a x = +1 cumple |CV − 1| < 0.05: la distribución es aproximadamente exponencial, como corresponde a un evento raro sin memoria.

E7. Trayectorias reactivas contra el camino óptimo
- Para D ∈ {0.1, 0.15, 0.25}, se alinean las ventanas según la decisión 7 y se calcula la mediana punto a punto.
- Criterio: la desviación cuadrática media entre la mediana y x_om(t) en t ∈ [−0.5, 0.5] decrece al disminuir D.
- Figuras: mediana y banda del 10–90 % frente a x_om(t) para cada D, y la densidad 2D de las ventanas alineadas. Muestran el estrechamiento del tubo reactivo al disminuir D.

FIGURAS (en figures/estocastico/, desde los resultados guardados)
E2: histograma frente a Boltzmann. E3: error frente a dt. E4–E5: ln T frente a 1/D (simulado, exacto y Kramers). E6: distribución de tiempos frente a la exponencial. E7: tubo reactivo frente a x_om para cada D.

QUÉ PRESENTAR AHORA (sin implementar)
- Archivos de src/taller/estocastico/ con sus funciones (nombre y propósito).
- Diseño del kernel: cómo genera números aleatorios por hilo (generador, semillas), cómo registra los dos tiempos de primer paso y cómo guarda las ventanas sin agotar los 12 GB de la GPU. Incluye una estimación de memoria y de tiempo de cómputo por experimento.
- Estructura de los archivos de configuración y de los resultados .npz.
- La lista de pruebas mapeada a E0–E7, y cuáles de los experimentos son pruebas de pytest y cuáles se ejecutan como corridas de producción (por su costo).
- Cualquier ambigüedad o criterio que consideres mal planteado. Dilo; no lo corrijas por tu cuenta.

Después escribe las pruebas, con valores esperados escritos explícitamente (oráculo independiente: nada calculado con el mismo código que se prueba), muéstramelas y detente. No implementes hasta que las apruebe.

=====================================================
ACLARACIONES APROBADAS (2026-10-05)
=====================================================

Decisiones de los revisores posteriores a la especificación (registradas en docs/decisiones.md, D19-D22). Complementan el texto original, que queda sin modificar.

(1) Cómputo en CPU, no en GPU (D19). La decisión de diseño 5 se sustituye: la producción usa Numba en CPU (@njit(parallel=True), con prange sobre trayectorias) y se mantiene la implementación de referencia en NumPy. Cada iteración del prange integra una trayectoria completa y registra sus observables sobre la marcha. Donde el texto dice "kernel CUDA" (decisión 5, E1, QUÉ PRESENTAR) debe leerse "integrador Numba en CPU". El límite de 12 GB de la GPU no aplica; la estimación de memoria se hace contra la RAM.

(2) Metadatos (D19, D22). En lugar de la versión de CUDA, cada resultado .npz guarda: número de hilos, capa de hilos de Numba y versiones de numpy, numba y llvmlite, además de lo ya pedido (parámetros, semilla, fecha y hash del commit).

(3) numpy fijado en >=2.4,<2.5 (D20). La regresión del hito 00 con numpy 2.4.6 se hizo antes de empezar este hito: 70 pruebas pasan, los valores clave del cuaderno son idénticos bit a bit y las figuras, byte a byte.

(4) Hilos (D22). 16 hilos por defecto (8 núcleos con hyperthreading), no los 20 de la máquina. El valor va en la configuración; la bandera --hilos de los scripts tiene prioridad. Se aplica con numba.set_num_threads.

=====================================================
ACLARACIONES APROBADAS (2026-10-05, revisión del plan)
=====================================================

Respuestas de los revisores al plan del hito y a sus preguntas. Complementan el texto original y las aclaraciones (1)-(4); donde una de ellas se reemplaza, se indica.

(5) División del trabajo. Claude escribe el código, las pruebas y los scripts, y ejecuta solo pruebas y validaciones pequeñas. Los revisores ejecutan las corridas de producción (E3 y E4). Para cada corrida, Claude entrega una estimación de tiempo con 10 hilos.

(6) Hilos: 10 por defecto. Reemplaza el valor de la aclaración (4). Razón: la máquina se ha apagado varias veces bajo carga y no se quiere exigirla de más. La bandera --hilos sigue teniendo prioridad.

(7) Corrección de puente browniano en los tiempos de primer paso. Euler-Maruyama solo ve la trayectoria en instantes discretos y pierde los cruces que ocurren entre pasos; en la cima x = 0 la deriva se anula y el sesgo de T(−1 → 0) es de varios por ciento con los dt de la decisión 4.
- Si en un paso x_n < b y x_{n+1} < b, se considera que la trayectoria cruzó b entre ambos instantes con probabilidad exp(−(b − x_n)(b − x_{n+1})/(D dt)) (puente browniano con σ² = 2D), y el cruce se asigna al final del paso.
- Se aplica igual a ambos destinos (cima y pozo), por uniformidad.
- Se valida por separado antes de usarla en E3 y E4, con un caso de solución conocida: movimiento browniano sin deriva, cuya probabilidad de alcanzar un nivel antes de un tiempo dado es exacta (principio de reflexión).
- Se guardan y reportan también los tiempos sin corregir: el sesgo y su corrección son un resultado del taller.
- Los criterios de E3 (2 %) y E4 (3 %) se aplican a los tiempos corregidos de ambos destinos, tal como están.

(8) E3. N = 10⁵ trayectorias por dt. La monotonía del error se exige solo entre los dt cuyo error supera 2 errores estándar (por debajo, el error está dominado por el ruido estadístico). El dt de producción es el mayor dt cuyo error es < 0.02 en ambos destinos.

(9) E4. Para D = 0.1 se usan 2·10⁴ trayectorias (error estadístico relativo ≈ 0.7 %); para los demás D, 10⁵.

(10) E5. El ajuste se hace con ambos destinos. Las pendientes exactas en D ≤ 0.2 (0.9884 para el pozo, 0.9900 para la cima) se reportan como resultado físico: en ese rango de D, las correcciones al prefactor desvían la pendiente de Arrhenius de ΔV = 1.

(11) E7. La ventana se extiende hasta t_cima + 0.5: muestras en t_cima − 3 + 0.01·k, k = 0, ..., 350. El criterio de la desviación cuadrática media se evalúa en t ∈ [−0.5, 0.25], donde x_om ≤ x_om(0.25) = −1/√(1 + e²) ≈ −0.345 (corregido en la revisión de las pruebas: no −0.35, valor que x_om alcanza en t ≈ 0.246). Razón: x_om solo describe la subida y nunca alcanza la cima; cerca de la cima la deriva se anula, el ruido domina y el tubo es más ancho. Las figuras muestran la ventana completa.

(12) Las muestras de una ventana anteriores a t = 0 (cuando t_cima < 3) se guardan como NaN y la mediana se calcula con nanmedian.

(13) E2. 10⁵ trayectorias, equilibrado hasta t = 50 y 20 muestras por trayectoria separadas 5.0, con intervalos de ancho 0.05 en [−2, 2]. Razón: con D = 0.5 el tiempo de correlación es del orden del tiempo de transición (~10); muestras separadas 1.0 estarían muy correlacionadas y el ruido estadístico de la distancia L1 quedaría cerca del umbral.

(14) Reanudación. La máquina se ha apagado varias veces. El script de producción guarda un archivo por D (E4) y por dt (E3) y, si el resultado ya existe, lo salta, salvo con la bandera --rehacer. Así un apagado solo cuesta el D en curso.

(15) dt de producción (precisa la regla de E3 y la aclaración 8; 2026-10-05, después de E3). E4 debe usar un dt validado por E3: uno de los dt evaluados en E3 cuyo error relativo corregido es < 0.02 en ambos destinos. No tiene que ser el mayor: elegir un dt menor que el máximo válido es una elección conservadora y está permitida. Para E4 se eligió dt = 1e-3 (D30), aunque los cuatro dt cumplen el 2 %. La prueba test_e3_dt_de_produccion se modificó con autorización explícita para verificar esta regla (D31).
