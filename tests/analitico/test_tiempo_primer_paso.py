"""E0 del hito 01: tiempo medio de primer paso exacto T(x0 → b), referencia del Bloque B.

T(x0 → b) = (1/D) ∫_{x0}^{b} dy e^{V(y)/D} ∫_{−∞}^{y} dz e^{−V(z)/D}, con x0 = −1 y b ∈ {0, +1}.

Archivo nuevo: las pruebas congeladas del hito 00 no se tocan. Los valores esperados se
calcularon con mpmath (40 dígitos, cuadratura tanh-sinh con −∞ exacto) y están escritos a
mano abajo; `_T_mpmath` permite regenerarlos y la prueba marcada `lento` lo hace.
Kramers se escribe a mano: 2π/√32 · e^{1/D}.
"""

import mpmath as mp
import numpy as np
import pytest

from taller.analitico.referencias import tiempo_primer_paso

VALORES_D = [0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.5]

# T(−1 → 0) y T(−1 → +1), mpmath con mp.dps = 40, redondeados a 17 cifras.
T_CIMA = {
    0.1: 12762.675627150007,
    0.125: 1749.1908845846764,
    0.15: 467.17537566755423,
    0.2: 90.475939875231091,
    0.25: 33.945744287679815,
    0.35: 11.019610799182002,
    0.5: 4.6065396131071025,
}
T_POZO = {
    0.1: 25527.091364292459,
    0.125: 3500.0348111867892,
    0.15: 935.93210455964312,
    0.2: 182.4176729855807,
    0.25: 69.263644760836284,
    0.35: 23.259751015597875,
    0.5: 10.258570067184178,
}

TOL_MPMATH = 1e-8
TOL_FACTOR_MEDIO = 5e-3


def _T_mpmath(D: str, b: int) -> mp.mpf:
    """Evaluación independiente con mpmath (solo para regenerar las tablas)."""
    mp.mp.dps = 40
    Dm = mp.mpf(D)
    V = lambda y: (y**2 - 1) ** 2
    interior = lambda y: mp.quad(lambda z: mp.exp(-V(z) / Dm), [-mp.inf, -1, y])
    nodos = [-1, -0.5, 0] if b == 0 else [-1, -0.5, 0, 0.5, 1]
    return mp.quad(lambda y: mp.exp(V(y) / Dm) * interior(y), nodos) / Dm


def _tau_kramers(D: float) -> float:
    return 2 * np.pi / np.sqrt(32) * np.exp(1 / D)


@pytest.mark.parametrize("D", VALORES_D)
@pytest.mark.parametrize("b, tabla", [(0.0, T_CIMA), (1.0, T_POZO)], ids=["cima", "pozo"])
def test_quad_contra_mpmath(D: float, b: float, tabla: dict) -> None:
    """E0: T(−1 → b) con quad coincide con mpmath, error relativo < 1e-8."""
    error = abs(tiempo_primer_paso(D, b) / tabla[D] - 1)
    assert error < TOL_MPMATH, f"D={D}, b={b}: {error:.3e}"


@pytest.mark.parametrize("D", [d for d in VALORES_D if d <= 0.15])
def test_factor_medio(D: float) -> None:
    """E0: |T(−1 → 0)/T(−1 → +1) − ½| < 5e-3 para D ≤ 0.15 (desde la cima se cae a cada lado con probabilidad ½)."""
    cociente = tiempo_primer_paso(D, 0.0) / tiempo_primer_paso(D, 1.0)
    assert abs(cociente - 0.5) < TOL_FACTOR_MEDIO, f"D={D}: {cociente:.6f}"


def test_convergencia_a_kramers() -> None:
    """E0: |T(−1 → +1)/τ_Kramers − 1| decrece monótonamente al disminuir D."""
    D_decreciente = sorted(VALORES_D, reverse=True)
    desviacion = [abs(tiempo_primer_paso(d, 1.0) / _tau_kramers(d) - 1) for d in D_decreciente]
    assert np.all(np.diff(desviacion) < 0), desviacion


@pytest.mark.lento
@pytest.mark.parametrize("D", VALORES_D)
def test_tablas_regenerables_con_mpmath(D: float) -> None:
    """Las tablas escritas a mano coinciden con mpmath recalculado (consistencia del oráculo)."""
    for b, tabla in ((0, T_CIMA), (1, T_POZO)):
        assert abs(float(_T_mpmath(repr(D), b)) / tabla[D] - 1) < 1e-15
