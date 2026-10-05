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
