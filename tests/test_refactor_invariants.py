"""
Invariants introduced by the performance/cleanup refactor.

These pin down the behaviour that the refactor relies on for correctness:

* MC results are independent of ``n_jobs`` (the per-worker reuse of the
  background + weak rates must not change the numbers).
* ``eta0b`` is recomputed whenever ``Omegabh2`` is reassigned.
* The electron-thermo tabulation reproduces the exact integrals.
* The fast ``_LinearRate`` evaluator matches ``interp1d(kind='linear')``.
"""
import numpy as np
import pytest

from pyprimat.config import PyPRConfig


# ---------------------------------------------------------------------------
# 1a — MC reuse must be independent of n_jobs
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.solve
def test_mc_njobs_independence():
    """Same seeds must give identical samples for serial vs parallel runs."""
    from pyprimat.main import mc_uncertainty
    base = {"network": "small"}
    mc1 = mc_uncertainty(6, ["YPBBN", "DoH"], params=base, n_jobs=1, seed=0)
    mcP = mc_uncertainty(6, ["YPBBN", "DoH"], params=base, n_jobs=3, seed=0)
    np.testing.assert_array_equal(mc1["YPBBN"].values, mcP["YPBBN"].values)
    np.testing.assert_array_equal(mc1["DoH"].values,   mcP["DoH"].values)


# ---------------------------------------------------------------------------
# 2e — eta0b tracks Omegabh2
# ---------------------------------------------------------------------------

def test_eta0b_tracks_omegabh2_attribute():
    cfg = PyPRConfig({"Omegabh2": 0.022425})
    e0 = cfg.eta0b
    cfg.Omegabh2 = 0.024
    assert cfg.eta0b == pytest.approx(e0 * 0.024 / 0.022425, rel=1e-12)


def test_eta0b_tracks_omegabh2_setitem():
    cfg = PyPRConfig({"Omegabh2": 0.022425})
    e0 = cfg.eta0b
    cfg["Omegabh2"] = 0.024
    assert cfg.eta0b == pytest.approx(e0 * 0.024 / 0.022425, rel=1e-12)


def test_gn_and_taun_come_from_defaults():
    """GN and tau_n must be present and overridable (single source of truth)."""
    cfg = PyPRConfig({"GN": 1.234e-44, "tau_n": 880.0})
    assert cfg.GN == 1.234e-44
    assert cfg.tau_n == 880.0
    assert cfg.Mpl == pytest.approx(1.0 / np.sqrt(1.234e-44), rel=1e-12)


# ---------------------------------------------------------------------------
# §6.1 — pluggable extra energy density (extra_rho)
# ---------------------------------------------------------------------------

def test_extra_rho_is_additive_in_hubble():
    """Each ``extra_rho`` callable adds ``rho(Tg)`` to ``rho_tot`` in ``_Hubble``.

    ``_Hubble`` returns ``H = MeV_to_secm1 * sqrt(rho_tot * 8*pi/(3*Mpl^2))``,
    so adding a constant extra energy density ``extra`` [MeV^4] through the
    ``extra_rho`` plug-in must increase ``H^2`` by exactly
    ``extra * 8*pi/(3*Mpl^2)``, independently of everything else `_Hubble`
    computes.
    """
    from pyprimat.main import PyPR
    base = {"network": "small", "verbose": False}
    p0 = PyPR(base)
    extra = 1.e-2  # MeV^4, an arbitrary but sizeable extra radiation density
    p1 = PyPR(base, extra_rho=[lambda Tg: extra])

    Tg = 1.0  # MeV
    H0 = p0._Hubble(Tg, Tg, Tg, Tg)
    H1 = p1._Hubble(Tg, Tg, Tg, Tg)
    assert H1 > H0

    dH2 = (H1 / p0.cfg.MeV_to_secm1)**2 - (H0 / p0.cfg.MeV_to_secm1)**2
    assert dH2 == pytest.approx(extra * 8. * np.pi / (3. * p0.cfg.Mpl**2), rel=1e-12)


def test_ede_is_appended_to_extra_rho():
    """``fEDE > 0`` appends exactly one ``rho_EDE`` callable to
    ``self._extra_rho``, via the same generic plug-in mechanism that
    ``extra_rho=`` callers use."""
    from pyprimat.main import PyPR
    p_no_ede = PyPR({"network": "small", "verbose": False})
    assert p_no_ede._extra_rho == []

    p_ede = PyPR({"network": "small", "verbose": False, "fEDE": 0.05})
    assert len(p_ede._extra_rho) == 1


# ---------------------------------------------------------------------------
# 1b — electron-thermo tabulation reproduces the exact integrals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [0.05, 0.2, 0.5, 1.0, 5.0])
def test_tabulated_electron_thermo_matches_exact(T):
    """The cubic-interpolant table (always used) reproduces the exact quad
    integrals (``_*_exact``) to within the interpolation tolerance."""
    import pyprimat.plasma as thermo
    thermo.initialise(PyPRConfig())
    p = thermo._default
    tab   = (p.rho_e(T), p.p_e(T), p.drho_e_dT(T), p.dp_e_dT(T))
    exact = (p._rho_e_exact(T), p._p_e_exact(T),
             p._drho_e_dT_exact(T), p._dp_e_dT_exact(T))
    for e, t in zip(exact, tab):
        assert t == pytest.approx(e, rel=1e-5)


# ---------------------------------------------------------------------------
# speedup — _LinearRate matches interp1d(kind='linear', fill_value='extrapolate')
# ---------------------------------------------------------------------------

def test_linear_rate_matches_interp1d():
    from scipy.interpolate import interp1d
    from pyprimat.nuclear import _LinearRate
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.001, 10.0, 50))
    y = rng.uniform(1e-3, 1e3, 50)
    fast = _LinearRate(x, y)
    ref = interp1d(x, y, kind="linear", bounds_error=False,
                   fill_value="extrapolate")
    # over the queried T9 range, including extrapolation above the grid (->11.6)
    xq = np.linspace(0.02, 11.6, 2000)
    np.testing.assert_allclose(fast(xq), ref(xq), rtol=1e-12, atol=0.0)


def test_linear_rate_scalar_and_array():
    from pyprimat.nuclear import _LinearRate
    f = _LinearRate(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    assert f(2.0) == pytest.approx(20.0)
    np.testing.assert_allclose(f(np.array([1.5, 2.5])), [15.0, 25.0])
