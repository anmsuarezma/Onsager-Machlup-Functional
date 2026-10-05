=====================================================
PARTE B — CUADERNO ANALÍTICO (solo plan, no ejecutes)
=====================================================

Primero guarda esta Parte B, tal cual, como specs/00_analitico.md.

PROPÓSITO
Construir la referencia simbólica contra la que se contrastarán los experimentos numéricos, y que además sirva de apoyo a la persona que desarrolla la teoría. Cada resultado debe derivarse con sympy, no copiarse de este texto: los resultados esperados de abajo son criterios de aceptación, no datos de entrada.

ORGANIZACIÓN
- La lógica vive en src/taller/analitico/: funciones que construyen y devuelven las expresiones simbólicas, y un módulo referencias.py con versiones numéricas (numpy) de las cantidades de referencia, que importarán los demás experimentos.
- El cuaderno notebooks/00_analitico/ (emparejado .py percent) llama a esas funciones y explica. Cada sección sigue la estructura del taller: objetivo de la sección, resultado esperado, cálculo, verificación e interpretación física. Las explicaciones en markdown deben ser útiles para alguien que hace la teoría a mano: claras, con las ecuaciones en LaTeX y sin saltar pasos importantes.
- Cada resultado simbólico se contrasta además con una verificación numérica independiente con scipy.

CONTENIDO

§0 El potencial y el equilibrio
- V(x) = (x²−1)², V′, V″; puntos fijos de ẋ = −V′ y su estabilidad por linealización; curvaturas V″(±1) = 8 y V″(0) = −4; ΔV = 1.
- Verificar que p_s ∝ e^(−V/D) anula la corriente de probabilidad de Fokker-Planck, J = −V′p − D p′.

§1–§2 Escape térmico (Onsager-Machlup, forma Freidlin-Wentzell)
- Lagrangiano L = (ẋ + V′)²/(4D). Derivar la ecuación de Euler-Lagrange automáticamente con sympy (euler_equations). Esperado: ẍ = V′V″.
- Integral primera por invariancia temporal, H = ẋ ∂L/∂ẋ − L; verificar que se conserva sobre las soluciones. Esperado: proporcional a ẋ² − V′².
- Con E = 0, las ramas ẋ = ±V′; identificar cuál conecta x = −1 (t → −∞) con x = 0 (t → +∞).
- Resolver la ecuación separable. Esperado: x_om(t) = −1/√(1 + e^(8t)), con el origen de tiempo en el cruce por x = −1/√2.
- Verificar con residuo simbólico nulo que x_om cumple la ecuación de primer orden y la de Euler-Lagrange, y sus límites.
- Tasas asintóticas: alejamiento de −1 como e^(8t) y llegada a 0 como e^(−4t); relacionarlas con V″(−1) y |V″(0)|.
- S_min: evaluar la acción sobre x_om. Esperado: S_min = ΔV/D = 1/D, usando que el término cruzado 2ẋV′ es una derivada total.
- Cota global: verificar la identidad (ẋ+V′)² = (ẋ−V′)² + 4ẋV′ y mostrar que implica S ≥ ΔV/D para toda trayectoria admisible.

§3 Instantón
- Lagrangiano euclídeo L_E = ½ẋ² + V. Euler-Lagrange automático. Esperado: ẍ = V′.
- Integral primera: ½ẋ² − V = 0; rama ẋ = √(2V) = √2(1 − x²) en |x| < 1.
- Resolver. Esperado: x_kink(τ) = tanh(√2 τ). Verificar con residuo nulo.
- S0 = ∫ √(2V) dx en [−1, 1]. Esperado: 4√2/3.
- Cota de Bogomolny: verificar ½ẋ² + V = ½(ẋ − √(2V))² + ẋ√(2V), de donde S_E ≥ S0.
- Tasa asintótica 2√2 = √V″(±1), la frecuencia del oscilador en cada pozo.

Potenciales efectivos
- U_OM = −½V′² (cimas en −1, 0 y +1) y U_E = −V (cimas solo en ±1). Explicar por qué el camino térmico termina en la cima y el instantón cruza de pozo a pozo.

§4 Puente Fokker-Planck → Schrödinger
- Con p = e^(−V/2D) ψ, derivar el operador de tipo Schrödinger. Esperado: H = −D∂² + V′²/(4D) − V″/2.
- Verificar que ψ0 = e^(−V/2D) cumple Hψ0 = 0 (estado de energía cero, raíz de Boltzmann).

Predicción de Kramers (referencia para el bloque estocástico)
- ⟨τ⟩ ≈ (2π/√(V″(−1)|V″(0)|)) e^(ΔV/D). Tabla para D = 0.1, 0.15, 0.25, 0.35, 0.5.
- Dejar escrito que el prefactor depende de la definición de "escape" (llegar a la cima o caer al otro pozo); esa decisión no se toma en este hito.

Variante: Onsager-Machlup completo (solo derivación)
- Lagrangiano con el término jacobiano: L = (ẋ + V′)²/(4D) − ½V″. Derivar su Euler-Lagrange. Esperado: ẍ = V′V″ − D V‴. Explicar que la corrección es proporcional a D y por eso el camino óptimo depende del ruido. No se usa en otros hitos salvo que lo decidamos.

FIGURAS (en figures/analitico/, PDF y PNG, generadas desde las funciones del módulo)
- El potencial V(x).
- x_om(t) y x_kink(τ) en la misma gráfica.
- Los potenciales efectivos U_OM y U_E.

MÓDULO referencias.py (exporta, en numpy)
V, dV, d2V, x_om(t), x_kink(tau), S_min(D), S0, tau_kramers(D).

PRUEBAS (tests/analitico/; criterios de aceptación definidos por nosotros, no los modifiques)
1. Las ecuaciones de Euler-Lagrange obtenidas con sympy coinciden simbólicamente con ẍ = V′V″, ẍ = V′ y ẍ = V′V″ − D V‴.
2. Residuos simbólicos exactamente cero: x_om en ẋ = V′ y en su Euler-Lagrange; x_kink en ẋ = √(2V) y en su Euler-Lagrange.
3. La integral primera de cada lagrangiano tiene derivada temporal nula sobre las soluciones de Euler-Lagrange.
4. S_min·D = 1 y S0 = 4√2/3 exactos en sympy; y con scipy.quad, error relativo < 1e-10 en ambos.
5. Las identidades de completar cuadrados (OM y Bogomolny) se simplifican a cero.
6. Hψ0 = 0 simbólicamente, y el operador H coincide con la expresión esperada.
7. p_s anula la corriente J simbólicamente.
8. solve_ivp aplicado a ẋ = V′ y a ẋ = √(2V), desde condiciones iniciales tomadas de las soluciones cerradas, reproduce x_om y x_kink con error absoluto máximo < 1e-8 en el intervalo integrado.
9. Las funciones de referencias.py coinciden con las expresiones de sympy evaluadas (lambdify) con error < 1e-12 en una malla.

QUÉ PRESENTAR AHORA (sin implementar)
- La lista de archivos de src/taller/analitico/ con las funciones de cada uno (nombre y una línea de propósito).
- El índice de secciones del cuaderno.
- La lista de pruebas, con nombre y qué verifica cada una, mapeada a los 9 criterios.
- Cualquier ambigüedad o cosa de esta especificación que consideres incorrecta o mal planteada. Dila; no la corrijas por tu cuenta.

=====================================================
ACLARACIONES APROBADAS (2026-10-04)
=====================================================

Respuestas de los revisores a las preguntas del plan. Complementan el texto original, que queda sin modificar.

(1) Las ecuaciones de Euler-Lagrange se comparan despejando ẍ.

(2) Para el kink se usa la forma explícita √(2V) = √2(1 − x²), válida en |x| < 1, tanto en el residuo como en S0. Se justifica en el cuaderno: el kink vive en (−1, 1), donde 1 − x² > 0, y la rama positiva es la que va de −1 a +1. La prueba correspondiente verifica además que √(2V) y √2(1 − x²) coinciden en (−1, 1).

(3) solve_ivp parte de t = 0 e integra hacia adelante y hacia atrás: t ∈ [−2, 2] para x_om y τ ∈ [−2, 2] para x_kink, con DOP853, rtol = 1e-12, atol = 1e-14. Si el criterio de 1e-8 falla así, se reporta con los números; no se amplía ni se reduce el intervalo para que pase.

(4) Criterio 9: error absoluto < 1e-12 para V, dV, d2V, x_om, x_kink y S0; error RELATIVO < 1e-12 para S_min y tau_kramers. Malla: x ∈ [−1.5, 1.5] con 301 puntos; t y τ ∈ [−2, 2] con 401 puntos; D ∈ {0.1, 0.15, 0.25, 0.35, 0.5}.

(5) referencias.py se escribe a mano en numpy, NO con lambdify, para que el criterio 9 compare dos implementaciones independientes (sympy y numpy a mano).

(6) Las pruebas contienen escritos los valores esperados de la especificación: son el oráculo independiente. La regla del CLAUDE.md de no duplicar constantes se refiere al código fuente (src/).

(7) En el cuaderno queda escrito que Kramers es una asintótica para D → 0 y que en D = 0.35 y 0.5 se esperan desviaciones en el bloque estocástico.

(8) En el cuaderno se explica que el tiempo para llegar a x = 0 es asintóticamente la mitad del tiempo de transición, porque desde la cima la partícula cae a cada lado con probabilidad ½. La definición que usará el bloque estocástico sigue sin decidirse.

(9) Las identidades y Hψ0 se verifican además numéricamente, como verificación complementaria, en una malla DETERMINISTA (sin aleatoriedad ni semilla) de espaciado irregular que NO pasa por los puntos fijos −1, 0 y +1, donde V′ = 0 y las identidades se cumplirían trivialmente; Hψ0 se evalúa por diferencias finitas. La malla se documenta en la prueba. La verificación simbólica sigue siendo la principal. [Corregida el 2026-10-05: la versión anterior decía "puntos aleatorios con semilla fija".]

(10) Se agrega tests/analitico/test_potencial.py con estos criterios:
- puntos fijos exactamente {−1, 0, 1}; V″(±1) = 8; V″(0) = −4; ΔV = 1;
- tasas asintóticas del camino OM: 8 al alejarse de −1 y 4 al llegar a 0, obtenidas como límites con sympy;
- tasa del kink: 2√2 = √V″(±1);
- tau_kramers(D) = (2π/√32)·e^(1/D), con error relativo < 1e-12, y prefactor 2π/√32 ≈ 1.1107.

(11) En el cuaderno, τ se reserva para el tiempo imaginario del instantón; el tiempo de escape se escribe τ_esc. La función tau_kramers conserva su nombre, y su docstring aclara la convención.

(12) Las pruebas se escriben primero, a partir de los criterios, y se presentan para revisión. Una vez aprobadas quedan congeladas (CLAUDE.md §5, regla 1); solo entonces se implementan el módulo y el cuaderno.
