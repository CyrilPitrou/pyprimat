"""Tests for weak_rates: Fn integral, Fermi-Coulomb, rate functions."""
import os

import pytest
import numpy as np
from pyprimat.config import PyPRConfig
import pyprimat.weak_rates as wr


@pytest.fixture(scope="module")
def cfg():
    return PyPRConfig({"numba_installed": False})


# ---------------------------------------------------------------------------
# Fermi-Dirac helpers
# ---------------------------------------------------------------------------

def test_FD2_between_zero_and_one():
    """FD2 is a Fermi-Dirac occupation: must lie in (0, 1)."""
    for E, x in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]:
        val = wr.FD2(E, x)
        assert 0.0 < val < 1.0


def test_FD2_large_argument_vanishes():
    """FD2 must return 0 when x*E exceeds the cutoff."""
    assert wr.FD2(1.0, 1e4) == 0.0


def test_FD_nu3_zero_phi_equals_FD2():
    """With phi=0, FD_nu3 must reduce to FD2."""
    E, x = 1.5, 1.0
    assert wr.FD_nu3(E, 0.0, x) == pytest.approx(wr.FD2(E, x), rel=1e-10)


# ---------------------------------------------------------------------------
# FermiCoulomb correction
# ---------------------------------------------------------------------------

def test_FermiCoulomb_positive(cfg):
    for b in [0.1, 0.5, 0.9]:
        assert wr.FermiCoulomb(b, cfg) > 0


def test_FermiCoulomb_close_to_one_at_small_alpha(cfg):
    """In the limit α → 0, the Fermi-Coulomb factor approaches 1."""
    val = wr.FermiCoulomb(0.5, cfg)
    assert val == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# ComputeFn — neutron-decay phase-space factor
# ---------------------------------------------------------------------------

def test_ComputeFn_positive(cfg):
    Fn = wr.ComputeFn(cfg)
    assert Fn > 0


def test_ComputeFn_Born_smaller_than_full(cfg):
    """Born-level Fn (no radiative corrections) should be smaller than the full Fn."""
    cfg_born = PyPRConfig({"numba_installed": False, "nTOp_Born_approximation": True})
    Fn_full = wr.ComputeFn(cfg)
    Fn_born = wr.ComputeFn(cfg_born)
    assert Fn_born < Fn_full


def test_ComputeFn_order_of_magnitude(cfg):
    """Fn should be ~ 1.6 in natural units (standard textbook value)."""
    Fn = wr.ComputeFn(cfg)
    assert 1.0 < Fn < 3.0


# ---------------------------------------------------------------------------
# RecomputeWeakRates — interpolated rate functions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rate_interpolants(cfg):
    """Pre-tabulated weak rate interpolants loaded from disk (correct T_ν history)."""
    import pyprimat.plasma as thermo
    thermo.initialise(cfg)
    return wr.InterpolateWeakRates(cfg)


def test_returns_two_interpolants(rate_interpolants):
    assert len(rate_interpolants) == 2          # forward (n->p), backward (p->n)


def test_all_rates_positive(rate_interpolants):
    """Both rate interpolants should return positive values in their range."""
    from pyprimat.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    T_K = 3.0 * MeV_to_K
    for interp in rate_interpolants:
        assert interp(T_K) > 0


def test_forward_greater_than_backward(rate_interpolants):
    """n→p forward rate > p→n backward rate at all T.

    Neutron decay (n→p) is energetically favored (mn > mp), so the
    forward rate always exceeds the backward rate: ratio = exp(+Q/T) > 1.
    """
    nTOp_frwrd_HT = rate_interpolants[0]
    nTOp_bkwrd_HT = rate_interpolants[1]
    from pyprimat.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    for T_MeV in [1.0, 3.0, 10.0]:
        T_K = T_MeV * MeV_to_K
        assert nTOp_frwrd_HT(T_K) > nTOp_bkwrd_HT(T_K), (
            f"frwrd should be > bkwrd at T={T_MeV} MeV"
        )


def test_ratio_decreases_toward_one_at_high_T(rate_interpolants):
    """frwrd/bkwrd should decrease toward 1 as T increases (detailed balance).

    At low T, ratio ≈ exp(Q/T) >> 1. At high T, ratio → 1.
    """
    nTOp_frwrd_HT = rate_interpolants[0]
    nTOp_bkwrd_HT = rate_interpolants[1]
    from pyprimat.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    ratio_low  = nTOp_frwrd_HT(1.0  * MeV_to_K) / nTOp_bkwrd_HT(1.0  * MeV_to_K)
    ratio_high = nTOp_frwrd_HT(10.0 * MeV_to_K) / nTOp_bkwrd_HT(10.0 * MeV_to_K)
    assert ratio_high < ratio_low   # ratio decreases toward 1
    assert ratio_high > 1.0         # but stays above 1


def test_forward_rate_increases_with_T(rate_interpolants):
    """n→p rate should increase with T in the HT era (well above freeze-out)."""
    nTOp_frwrd_HT = rate_interpolants[0]
    from pyprimat.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    rate_low  = nTOp_frwrd_HT(1.0  * MeV_to_K)
    rate_high = nTOp_frwrd_HT(10.0 * MeV_to_K)
    assert rate_high > rate_low


# ---------------------------------------------------------------------------
# Vectorised fixed-order quadrature convergence (IDEAS 5.1)
# ---------------------------------------------------------------------------

def test_gauss_legendre_converged():
    """The fixed-order Gauss-Legendre rate quadrature is converged (IDEAS §5.1).

    ComputeWeakRates replaced the per-grid-point adaptive scipy.quad with a
    single fixed-order Gauss-Legendre rule (``_N_GL`` nodes) vectorised over the
    whole temperature grid.  The physics-numerics gate for that change is that
    the rates are *converged* in the node count: re-evaluating them with twice
    as many nodes must not move Gamma_{n->p}/Gamma_{p->n} by more than 1e-5,
    i.e. far below the 1e-4 level at which the weak physics flags move Neff/YP.
    This pins ``_N_GL`` against an accidental reduction.

    The residual ~1e-6 wiggle sits entirely in the low-temperature
    free-neutron-decay regime, where the integrand develops a near-sharp Fermi
    step at the decay endpoint E = Q/m_e (slow for any Gauss rule) -- and where
    the n<->p rates no longer matter to BBN (neutrons are already frozen).  At
    the freeze-out temperatures that *do* matter the integrand is smooth and the
    convergence is far tighter, which is why the standard-run YP/D-H shift only
    by ~2e-7 / ~3e-11 (see the WEAK_RATE_FORMAT_VERSION notes in weak_rates.py).

    The test calls ComputeWeakRates directly (which never writes to the
    rates/weak/ cache -- only RecomputeWeakRates does), with
    include_nTOp_thermal=False so it exercises purely the vectorised CCR+FMCCR
    quadrature and needs no on-disk thermal table.
    """
    import numpy as np
    import pyprimat.plasma as plasma

    cfg = PyPRConfig({"include_nTOp_thermal": False})
    plasma.initialise(cfg)

    # Representative photon-temperature grid [MeV] over the BBN range, with a
    # post-decoupling neutrino ratio (the exact ratio is irrelevant to a
    # convergence check; any smooth T_nu(T_gamma) exercises the same integrand).
    MeV_to_K = cfg.MeV_to_Kelvin
    Tg  = np.logspace(np.log10(cfg.T_end / MeV_to_K),
                      np.log10(cfg.T_start / MeV_to_K), 60)
    Tnu = Tg * (4. / 11.) ** (1. / 3.)

    _, f0, b0 = wr.ComputeWeakRates([Tg, Tnu], cfg)

    # Refine the Gauss-Legendre rule to 2*_N_GL nodes and recompute.
    nodes0, weights0 = wr._GL_NODES, wr._GL_WEIGHTS
    try:
        wr._GL_NODES, wr._GL_WEIGHTS = np.polynomial.legendre.leggauss(2 * wr._N_GL)
        _, f1, b1 = wr.ComputeWeakRates([Tg, Tnu], cfg)
    finally:
        wr._GL_NODES, wr._GL_WEIGHTS = nodes0, weights0

    # Forward rate is strictly positive everywhere: pure relative tolerance.
    assert np.max(np.abs(f1 - f0) / np.abs(f0)) < 1e-5
    # Backward rate passes through ~0 at low T; only compare where it is an
    # appreciable fraction of the forward rate.
    mask = np.abs(b0) > 1e-6 * np.abs(f0).max()
    assert np.max(np.abs(b1[mask] - b0[mask]) / np.abs(b0[mask])) < 1e-5


# ---------------------------------------------------------------------------
# Weak-rate cache fingerprint — chemical-potential / distortion sensitivity
# ---------------------------------------------------------------------------

def test_fingerprint_changes_with_munuOverTnu():
    """A change in munuOverTnu must invalidate the weak-rate cache.

    munuOverTnu shifts the neutrino Fermi-Dirac occupation that enters every
    n<->p rate integral, so a cache built for one value must not be silently
    reused for another.  This pins ``_BACKGROUND_FINGERPRINT_FIELDS``
    (weak_rates.py) to keep including ``munuOverTnu``.
    """
    cfg0 = PyPRConfig({"munuOverTnu": 0.0})
    cfg1 = PyPRConfig({"munuOverTnu": 0.1})

    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 != fp1


def test_fingerprint_changes_with_delta_xi_nu():
    """A change in delta_xi_nu (analytic spectral-distortion amplitude) must
    invalidate the weak-rate cache, for the same reason as munuOverTnu above.
    """
    common = {"spectral_distortions": True, "analytic_distortions": True,
              "incomplete_decoupling": False}
    cfg0 = PyPRConfig({**common, "delta_xi_nu": 0.0})
    cfg1 = PyPRConfig({**common, "delta_xi_nu": 0.05})

    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 != fp1


# ---------------------------------------------------------------------------
# Custom NEVO table overrides (Item 1: nevo_file/nevo_spectral_file/nevo_grid_file)
# ---------------------------------------------------------------------------

def test_fingerprint_changes_with_nevo_file():
    """Pointing nevo_file at a different (even identical-content) filename
    must invalidate the weak-rate cache, since the cached rates were
    integrated against whatever neutrino-temperature history that file
    encodes -- the cache cannot know two filenames happen to agree."""
    cfg0 = PyPRConfig({"network": "small"})
    cfg1 = PyPRConfig({"network": "small", "nevo_file": "NEVOPRIMAT_col_1_7.csv"})

    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 != fp1


def test_nevo_file_missing_raises_value_error():
    """A nevo_file override that doesn't exist under rates/NEVO/ raises a
    clear ValueError at PyPRConfig construction time, not a confusing error
    deep inside neutrino_history when the table is first read."""
    with pytest.raises(ValueError, match="nevo_file.*not found"):
        PyPRConfig({"nevo_file": "does_not_exist.csv"})


@pytest.mark.slow
@pytest.mark.solve
def test_nevo_file_with_custom_copy_reproduces_default(tmp_path):
    """A copy of the default NEVO thermo table under a different filename,
    selected via nevo_file, must give identical results to the default (same
    content, different path) -- while still registering as a fingerprint
    cache miss (test_fingerprint_changes_with_nevo_file above)."""
    import shutil
    from pyprimat.main import PyPR

    cfg_default = PyPRConfig({"network": "small"})
    src = os.path.join(cfg_default.data_dir, "rates", "NEVO",
                        "NEVOPRIMAT_col_1_7.csv")
    dst = os.path.join(cfg_default.data_dir, "rates", "NEVO",
                        "NEVOPRIMAT_col_1_7_test_copy.csv")
    shutil.copy(src, dst)
    try:
        # weak_rate_cache=False on *both* runs: the shipped rates/weak/*.txt
        # cache (which "nevo_file=...test_copy.csv" deliberately misses, see
        # test_fingerprint_changes_with_nevo_file) stores rates on its own T
        # grid and re-interpolates them, which differs from a fresh
        # ComputeWeakRates integration at the ~1e-3 level
        # (test_recomputed_rates_match_cached) -- comparing a cache hit to a
        # cache miss would spuriously fail at rel=1e-12 even for identical
        # physics. Forcing both through ComputeWeakRates with the same
        # [T_gamma_vec, T_nue_vec] and dFDneu_func (built from the
        # nevo_spectral_file/nevo_grid_file defaults, untouched by nevo_file)
        # makes them bit-identical.
        r_default = PyPR({"network": "small", "verbose": False,
                           "weak_rate_cache": False}).PyPRresults()
        r_custom = PyPR({"network": "small", "verbose": False,
                          "weak_rate_cache": False,
                          "nevo_file": "NEVOPRIMAT_col_1_7_test_copy.csv"}).PyPRresults()
    finally:
        os.remove(dst)

    assert r_custom["Neff"] == pytest.approx(r_default["Neff"], rel=1e-12)
    assert r_custom["YPBBN"] == pytest.approx(r_default["YPBBN"], rel=1e-12)
    assert r_custom["DoH"] == pytest.approx(r_default["DoH"], rel=1e-12)


# ---------------------------------------------------------------------------
# RecomputeWeakRates — recompute path vs the fingerprinted cache (IDEAS 7.1)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_recomputed_rates_match_cached():
    """The recompute path (weak_rate_cache=False) must agree with the cache.

    conftest.py's session-scoped ``solved_small``/``solved_large`` fixtures
    deliberately use the *default* config (``weak_rate_cache=True``), so they
    load the shipped, fingerprinted ``rates/weak/nTOp_*.txt`` tables instead
    of paying the ~1.8 s ``ComputeWeakRates`` integration on every test
    session -- that is what keeps the default test tier cheap.

    This is the one dedicated test that exercises the recompute path
    (``RecomputeWeakRates`` falling through to ``ComputeWeakRates`` when
    ``weak_rate_cache=False``) and checks it reproduces the cached tables:
    both describe the same physics for the same ``[T_gamma_vec, T_nue_vec]``,
    just one read from disk (quadratic interpolation of the saved grid) and
    the other freshly integrated (quadratic interpolation of a freshly built
    grid) -- so they should agree to well within a percent.
    """
    from pyprimat.main import PyPR
    from pyprimat.config import PyPRConfig

    r_cached = PyPR({"network": "small"})                        # loads rates/weak/*.txt
    r_fresh  = PyPR({"network": "small", "weak_rate_cache": False})  # forces ComputeWeakRates

    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    for T_MeV in [0.5, 1.0, 3.0, 10.0]:
        T_K = T_MeV * MeV_to_K
        for cached, fresh in ((r_cached._nTOp_frwrd, r_fresh._nTOp_frwrd),
                              (r_cached._nTOp_bkwrd, r_fresh._nTOp_bkwrd)):
            assert fresh(T_K) == pytest.approx(cached(T_K), rel=2e-3)
