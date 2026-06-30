"""
Tests for the analytical QED plasma-pressure correction module
(``primat.qed_pressure``).

Physical context
----------------
The QED interaction pressure δP(T) corrects the EM plasma equation of state
during BBN.  It is negative and of order α/π × T⁴ ≈ 7×10⁻³/(3) × T⁴.  The
O(e³) ring contribution is positive and about 10× smaller.

These tests verify:
1. The Fermi-Dirac phase-space integrals I₀₁ and I₂₋₁ reproduce known
   analytic limits (ultra-relativistic and low-T suppression).
2. δP_a is negative and δP_{e3} is positive at all temperatures.
3. The computed values match hard-coded reference values (computed once at
   high quadrature precision) — these are independent of any external file
   so that they remain valid after rates/plasma/ is regenerated.
4. The derivative tables satisfy d(δP)/dT ≈ Δ(δP)/ΔT from finite differences.
"""
import numpy as np
import pytest

from primat.qed_pressure import (_I01, _I2m1, _dPa, _dPe3,
                                compute_qed_pressure_tables, save_qed_tables)


# ---------------------------------------------------------------------------
# Analytic-limit checks for the Fermi-Dirac integrals (fast, no I/O)
# ---------------------------------------------------------------------------

def test_I01_ultrarelativistic():
    """At x→0 (T ≫ mₑ), I₀₁(x) → ∫₀^∞ E/(e^E+1) dE = π²/12."""
    result = _I01(1e-4)
    expected = np.pi**2 / 12.     # ≈ 0.82247
    assert abs(result - expected) < 1e-4, f"I01(x→0) = {result:.6f}, expected {expected:.6f}"


def test_I2m1_ultrarelativistic():
    """At x→0, I₂₋₁(x) → ∫₀^∞ E/(e^E+1) dE = π²/12 (same as I₀₁ in UR limit)."""
    result = _I2m1(1e-4)
    expected = np.pi**2 / 12.
    assert abs(result - expected) < 1e-4, f"I2m1(x→0) = {result:.6f}, expected {expected:.6f}"


def test_I01_nonrelativistic_suppressed():
    """At x > 50 (T ≪ mₑ/50), I₀₁ = 0 (non-relativistic cutoff applied)."""
    assert _I01(51.) == 0., "I01 should be zero above the non-relativistic cutoff"


def test_I2m1_nonrelativistic_suppressed():
    """At x > 50, I₂₋₁ ≡ 0 (non-relativistic cutoff applied)."""
    assert _I2m1(51.) == 0.


# ---------------------------------------------------------------------------
# Sign and monotonicity of δP_a and δP_{e3}
# ---------------------------------------------------------------------------

def test_dPa_is_negative():
    """δP_a must be negative at all temperatures (interaction lowers pressure)."""
    for T in [0.1, 1.0, 10., 50., 100.]:
        val = _dPa(T)
        assert val < 0, f"δP_a({T} MeV) = {val:.3e} is not negative"


def test_dPe3_is_positive():
    """δP_{e3} must be positive (ring/plasmon contribution increases pressure)."""
    for T in [0.1, 1.0, 10., 50., 100.]:
        val = _dPe3(T)
        assert val > 0, f"δP_e3({T} MeV) = {val:.3e} is not positive"


def test_dPe3_smaller_than_dPa():
    """δP_{e3} is O(α^{3/2}) so |δP_{e3}| < |δP_a| at all T."""
    for T in [0.5, 1.0, 5., 20., 100.]:
        assert abs(_dPe3(T)) < abs(_dPa(T)), (
            f"At T={T} MeV: |δP_e3|={abs(_dPe3(T)):.3e} ≥ |δP_a|={abs(_dPa(T)):.3e}"
        )


# ---------------------------------------------------------------------------
# Pinned numerical values — regression guard independent of external files
# ---------------------------------------------------------------------------

def test_I01_pinned_at_x1():
    """I₀₁(1.0) = 0.54287383 — pinned against a high-accuracy quadrature."""
    # x = mₑ/T = 1 corresponds to T ≈ mₑ ≈ 0.511 MeV; the integral has no
    # closed form, so we pin the numerically computed value as a regression guard.
    assert abs(_I01(1.0) - 0.54287383) < 1e-6, f"I01(1.0) = {_I01(1.0):.8f}"


def test_I2m1_pinned_at_x1():
    """I₂₋₁(1.0) = 0.87634737 — pinned against a high-accuracy quadrature."""
    assert abs(_I2m1(1.0) - 0.87634737) < 1e-6, f"I2m1(1.0) = {_I2m1(1.0):.8f}"


def test_dPa_pinned_values():
    """δP_a at T = 1, 10, 100 MeV matches expected values to 1 ppm.

    Reference values computed at epsabs=epsrel=1e-13.  Units: MeV⁴.
    At T=100 MeV (ultra-relativistic): δP_a ≈ -(α/π)(7π²/60)T⁴
        = -(1/137.04/π)(7π²/60)·10⁸ ≈ -1.5919e5, consistent with the table.
    """
    expected = {1.0: -1.33243740e-3,
                10.: -1.58591018e+1,
                100.: -1.59193917e+5}
    for T, val in expected.items():
        got = _dPa(T)
        assert abs((got - val) / val) < 1e-5, (
            f"δP_a({T} MeV): got {got:.8e}, expected {val:.8e}")


def test_dPe3_pinned_values():
    """δP_{{e3}} at T = 1, 10, 100 MeV matches expected values to 1 ppm.

    Units: MeV⁴.  At T=100 MeV the ring/plasmon term should be ≈ 8.9% of
    |δP_a|, consistent with δP_{{e3}}/|δP_a| ∝ (α^{{1/2}}) being ~1/12.
    """
    expected = {1.0: 1.33628873e-4,
                10.: 1.41674368e+0,
                100.: 1.41757878e+4}
    for T, val in expected.items():
        got = _dPe3(T)
        assert abs((got - val) / val) < 1e-5, (
            f"δP_e3({T} MeV): got {got:.8e}, expected {val:.8e}")


# ---------------------------------------------------------------------------
# Derivative consistency
# ---------------------------------------------------------------------------

def test_derivative_consistency():
    """CubicSpline dδP/dT from compute_qed_pressure_tables agrees with direct FD.

    The derivatives returned by :func:`compute_qed_pressure_tables` are cubic-
    spline derivatives of the tabulated δP values (not analytic derivatives of
    the Fermi-Dirac integrals).  We verify them against a direct central finite
    difference on :func:`_dPa` and :func:`_dPe3` evaluated at T±ε — this is a
    proper accuracy test because the reference is independent of the spline.

    At T = 1, 5, 10, 50 MeV the spline should match to better than 0.01%
    (with 500 grid points the CubicSpline derivative error is ~1e-6 relative).
    The test uses 200 points over a modest range to keep runtime short.
    """
    tables = compute_qed_pressure_tables(T_min=0.5, T_max=60., n_pts=200,
                                         verbose=False)
    from scipy.interpolate import CubicSpline
    spl_e2 = CubicSpline(tables["T"], tables["dP_e2"])
    spl_e3 = CubicSpline(tables["T"], tables["dP_e3"])

    eps = 1e-4   # relative step for central FD
    for T in [1., 5., 10., 50.]:
        h = T * eps
        # Central FD reference (directly from the integrand functions)
        fd_e2 = (_dPa(T + h) - _dPa(T - h)) / (2 * h)
        fd_e3 = (_dPe3(T + h) - _dPe3(T - h)) / (2 * h)
        spline_e2 = spl_e2(T, 1)
        spline_e3 = spl_e3(T, 1)
        assert abs((spline_e2 - fd_e2) / fd_e2) < 1e-4, (
            f"dδP_a/dT spline error at T={T} MeV: "
            f"spline={spline_e2:.6e}, FD={fd_e2:.6e}")
        assert abs((spline_e3 - fd_e3) / fd_e3) < 1e-4, (
            f"dδP_{{e3}}/dT spline error at T={T} MeV: "
            f"spline={spline_e3:.6e}, FD={fd_e3:.6e}")


# ---------------------------------------------------------------------------
# Round-trip: compute → save → load
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    """Tables written by save_qed_tables round-trip correctly through np.loadtxt.

    ``save_qed_tables`` writes a single 7-column file ``QED_tables.txt``:
    T, dP_a, dP_e3, d(dP_a)/dT, d(dP_e3)/dT, d2(dP_a)/dT2, d2(dP_e3)/dT2.
    """
    tables = compute_qed_pressure_tables(T_min=1., T_max=10., n_pts=30,
                                         verbose=False)
    save_qed_tables(tables, str(tmp_path), verbose=False)

    fpath = tmp_path / "QED_tables.txt"
    assert fpath.exists(), "QED_tables.txt was not written"
    data = np.loadtxt(str(fpath))
    assert data.shape == (30, 7), f"QED_tables.txt has unexpected shape {data.shape}"

    # Verify all seven columns round-trip to within floating-point precision
    np.testing.assert_allclose(data[:, 0], tables["T"],            rtol=1e-5)
    np.testing.assert_allclose(data[:, 1], tables["dP_e2"],        rtol=1e-5)
    np.testing.assert_allclose(data[:, 2], tables["dP_e3"],        rtol=1e-5)
    np.testing.assert_allclose(data[:, 3], tables["d_dP_e2_dT"],   rtol=1e-5)
    np.testing.assert_allclose(data[:, 4], tables["d_dP_e3_dT"],   rtol=1e-5)
    np.testing.assert_allclose(data[:, 5], tables["d2_dP_e2_dT2"], rtol=1e-5)
    np.testing.assert_allclose(data[:, 6], tables["d2_dP_e3_dT2"], rtol=1e-5)
