Iniciamos el hito 02: Bloque A reducido, la red variacional. Tenemos unas dos horas, así que el alcance está cerrado.

OBJETIVO
Minimizar directamente los dos funcionales del taller con una red neuronal (método directo del cálculo de variaciones, tipo Ritz), sin pasar por Euler-Lagrange, y comparar contra las soluciones cerradas del hito 00. Solo la red variacional: nada de PINN ni de estudios de horizonte, modo cero o selección de rama.

DEPENDENCIAS
- Grupo neuronal en uv con PyTorch SOLO CPU (índice de paquetes CPU de PyTorch configurado en pyproject.toml). No configures CUDA ni GPU en este hito.
- Aplica la sección 12 del CLAUDE.md: suite completa después de instalar; si cambia numpy (debe seguir en 2.4.x), detente y repórtalo.

PROBLEMAS
1. Escape térmico (Onsager-Machlup, forma Freidlin-Wentzell). Como D solo multiplica la acción, se minimiza S·D = (1/4)∫(ẋ + V′)² dt en t ∈ [−T, T], con x(−T) = −1 y x(T) = 0. Referencias: S·D → 1 y x_om(t) = −1/√(1 + e^(8t)).
2. Instantón. S_E = ∫[½ẋ² + V] dτ en τ ∈ [−T, T], con x(−T) = −1 y x(T) = +1. Referencias: S_E → S0 = 4√2/3 y x_kink(τ) = tanh(√2 τ).
Una red independiente para cada problema.

ARQUITECTURA
- Perceptrón multicapa: entrada s = t/T (normalizada a [−1, 1]), capas ocultas con tanh, salida lineal escalar N(s). Por defecto: 2 capas ocultas de 32 neuronas.
- Inicialización Xavier en las capas ocultas; la última capa con pesos casi nulos, de modo que el camino inicial sea la recta entre los extremos.
- Ansatz que impone las fronteras exactamente: x(t) = x_a + (x_b − x_a)(t + T)/(2T) + [(t + T)(T − t)/T²]·N(t/T). No uses penalizaciones de frontera: con la acción como pérdida, el optimizador podría reducir la acción sin llegar a la frontera.
- No impongas simetrías (por ejemplo, que N sea impar en el instantón): la red debe encontrarlas a partir del funcional.

ACCIÓN Y ENTRENAMIENTO
- Malla fija de integración (2001 puntos equiespaciados en [−T, T]), evaluada completa en cada iteración, sin lotes, para que la pérdida sea determinista.
- ẋ con autograd; la acción se integra con la regla del trapecio. Todo en float64.
- Adam (unas miles de iteraciones, tasa del orden de 1e-3) y después L-BFGS hasta convergencia. Registra la acción en cada iteración.
- T = 3 para el escape térmico y T = 4 para el instantón. Si no se cumplen los criterios, repórtalo con los números antes de cambiar T.
- Semillas y todos los parámetros (T, malla, arquitectura, iteraciones, tasas) en configs/neuronal/.

SIN DATOS
Las soluciones analíticas y los resultados de la simulación NO se usan en el entrenamiento: solo en la validación y en las figuras. La red debe encontrar el camino solo a partir del funcional.

VERIFICACIONES NUMÉRICAS
- Malla: reevalúa la acción del camino final con una malla del doble de puntos y reporta la diferencia relativa (debe ser < 1e-4).
- Alineación: antes de comparar con la referencia, alinea el camino de la red en su cruce por x = −1/√2 (escape térmico) o x = 0 (instantón), la misma convención de origen del hito 00 y de E7.

CRITERIOS DE ACEPTACIÓN (pruebas en tests/neuronal/, con los valores esperados escritos explícitamente)
- Escape térmico: |S·D − 1| < 0.01, y RMS entre el camino alineado y x_om en t ∈ [−1, 0.5] < 0.01.
- Instantón: |S_E − 4√2/3|/(4√2/3) < 0.01, y RMS entre el camino alineado y x_kink en τ ∈ [−2, 2] < 0.01.
- Cota analítica: S·D ≥ 1 − 1e-4 y S_E ≥ S0 − 1e-4, como exigen las cotas de completar cuadrados del hito 00 y el carácter de cota superior del método de Ritz.

ROBUSTEZ ANTE LA ARQUITECTURA
Además de la arquitectura por defecto, entrena ambos problemas con 3 capas ocultas de 32 neuronas y con 4 capas ocultas de 64. Los criterios de aceptación se aplican a la arquitectura por defecto. Para las otras dos, reporta en una tabla: acción final, error relativo frente a la referencia, RMS frente al camino de referencia, número de parámetros y tiempo de entrenamiento. Se espera que los tres resultados coincidan dentro de la tolerancia. Si no coinciden, repórtalo; no cambies la arquitectura por defecto sin aprobación.

PROCESO (sin esperar revisión intermedia, por el tiempo)
1. Escribe las pruebas con estos criterios y haz su commit: quedan congeladas.
2. Implementa src/taller/neuronal/ (importa solo de taller.analitico) y un script que entrene ambos problemas y guarde los resultados en results/neuronal/ con metadatos.
3. Entrena, corre la suite completa y repórtame los números de cada criterio, la verificación de la malla, la tabla de robustez y los tiempos de entrenamiento.
4. Si algún criterio falla, NO lo cambies: repórtalo y detente.
5. Si todo pasa, escribe el cuaderno notebooks/02_neuronal/ (estructura de siempre, numeración del plan) con estas figuras en figures/neuronal/ (PDF y PNG, sin fecha de creación):
   - camino de la red frente a x_om, y frente a x_kink;
   - acción frente a iteración de entrenamiento, con la cota analítica marcada;
   - superposición del camino de la red sobre el tubo reactivo de E7 (cargando los resultados de E4 ya guardados).
   La superposición es la única parte que combina los dos bloques: va en el cuaderno, no en src/.
   Incluye la tabla de robustez con una frase de interpretación: qué limita la precisión, el horizonte finito T o la red.
6. Completa docs/bitacora/02_neuronal.md, haz los commits y detente. No crees la etiqueta.
