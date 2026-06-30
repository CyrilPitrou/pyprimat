"""Tests for weak_rates: Fn integral, Fermi-Coulomb, rate functions."""
import os

import pytest
import numpy as np
from primat.config import PRIMATConfig
from primat.plasma import Plasma
from primat.neutrino_history import InstantaneousDecoupling, AnalyticDistortion
import primat.weak_rates as wr
import primat.weak_rates.corrections as corrections


def test_corrections_all_names_exist():
    """Every name in corrections.__all__ must resolve to a real attribute.

    ``from .corrections import *`` (primat/weak_rates/__init__.py) raises
    ``AttributeError`` at import time -- i.e. ``import primat`` itself
    fails -- if ``__all__`` lists a name that was renamed/removed without
    updating ``__all__`` to match. Catches that class of mistake directly
    rather than relying on every refactor remembering to grep for it.
    """
    missing = [name for name in corrections.__all__ if not hasattr(corrections, name)]
    assert not missing, f"corrections.__all__ lists missing names: {missing}"


@pytest.fixture(scope="module")
def cfg():
    return PRIMATConfig({"numba_installed": False})


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
    """Born-level Fn (no radiative/finite-mass corrections) should be smaller than the full Fn."""
    cfg_born = PRIMATConfig({"numba_installed": False,
                           "radiative_corrections": False,
                           "finite_mass_corrections": False})
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
    return wr.InterpolateWeakRates(cfg)


def test_returns_two_interpolants(rate_interpolants):
    assert len(rate_interpolants) == 2          # forward (n->p), backward (p->n)


def test_all_rates_positive(rate_interpolants):
    """Both rate interpolants should return positive values in their range."""
    from primat.config import PRIMATConfig
    MeV_to_K = PRIMATConfig().MeV_to_Kelvin
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
    from primat.config import PRIMATConfig
    MeV_to_K = PRIMATConfig().MeV_to_Kelvin
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
    from primat.config import PRIMATConfig
    MeV_to_K = PRIMATConfig().MeV_to_Kelvin
    ratio_low  = nTOp_frwrd_HT(1.0  * MeV_to_K) / nTOp_bkwrd_HT(1.0  * MeV_to_K)
    ratio_high = nTOp_frwrd_HT(10.0 * MeV_to_K) / nTOp_bkwrd_HT(10.0 * MeV_to_K)
    assert ratio_high < ratio_low   # ratio decreases toward 1
    assert ratio_high > 1.0         # but stays above 1


def test_forward_rate_increases_with_T(rate_interpolants):
    """n→p rate should increase with T in the HT era (well above freeze-out)."""
    nTOp_frwrd_HT = rate_interpolants[0]
    from primat.config import PRIMATConfig
    MeV_to_K = PRIMATConfig().MeV_to_Kelvin
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
    thermal_corrections=False so it exercises purely the vectorised CCR+FMCCR
    quadrature and needs no on-disk thermal table.
    """
    import numpy as np

    cfg = PRIMATConfig({"thermal_corrections": False})

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
    cfg0 = PRIMATConfig({"munuOverTnu": 0.0})
    cfg1 = PRIMATConfig({"munuOverTnu": 0.1})

    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 != fp1


def test_fingerprint_changes_with_y_SZ():
    """A change in y_SZ (analytic spectral-distortion amplitude) must
    invalidate the weak-rate cache, for the same reason as munuOverTnu above.
    """
    common = {"spectral_distortions": True, "analytic_distortions": True,
              "incomplete_decoupling": False}
    cfg0 = PRIMATConfig({**common, "y_SZ": 0.0})
    cfg1 = PRIMATConfig({**common, "y_SZ": 0.05})

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
    cfg0 = PRIMATConfig({"network": "small"})
    cfg1 = PRIMATConfig({"network": "small", "nevo_file": "NEVOPRIMAT_col_1_7.csv"})

    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 != fp1


def test_nevo_file_missing_raises_value_error():
    """A nevo_file override that doesn't exist under rates/NEVO/ raises a
    clear ValueError at PRIMATConfig construction time, not a confusing error
    deep inside neutrino_history when the table is first read."""
    with pytest.raises(ValueError, match="nevo_file.*not found"):
        PRIMATConfig({"nevo_file": "does_not_exist.csv"})


@pytest.mark.slow
@pytest.mark.solve
def test_nevo_file_with_custom_copy_reproduces_default(tmp_path):
    """A copy of the default NEVO thermo table under a different filename,
    selected via nevo_file, must give identical results to the default (same
    content, different path) -- while still registering as a fingerprint
    cache miss (test_fingerprint_changes_with_nevo_file above)."""
    import shutil
    from primat.main import PRIMAT

    cfg_default = PRIMATConfig({"network": "small"})
    src = cfg_default.resolve_rates_path("NEVO", "NEVOPRIMAT_col_1_7.csv")
    # Use tmp_path (auto-cleaned by pytest) and an absolute path for nevo_file,
    # to avoid writing into the package data tree (which can fail on network
    # or cloud-synced volumes when the file is evicted between creation and
    # removal, causing a spurious FileNotFoundError in os.remove).
    dst = tmp_path / "NEVOPRIMAT_col_1_7_test_copy.csv"
    shutil.copy(src, str(dst))
    # weak_rate_cache=False on *both* runs: the shipped rates/weak/*.txt
    # cache (which "nevo_file=...test_copy.csv" deliberately misses, see
    # test_fingerprint_changes_with_nevo_file) stores rates on its own T
    # grid and re-interpolates them, which differs from a fresh
    # ComputeWeakRates integration at the ~1e-3 level
    # (test_recomputed_rates_match_cached) -- comparing a cache hit to a
    # cache miss would spuriously fail even for identical physics.
    # Forcing both through ComputeWeakRates with the same
    # [T_gamma_vec, T_nue_vec] and dFDneu_func (built from the
    # nevo_spectral_file/nevo_grid_file defaults, untouched by nevo_file)
    # makes the *non-thermal* part bit-identical. The *thermal* part
    # (CCRTh, see _thermal_fingerprint) is cached on disk separately and
    # keyed by its own fingerprint, which nevo_file also enters; the
    # custom-copy run therefore cannot hit the shipped
    # nTOp_thermal_<hash>.txt cache and must re-run the vegas Monte Carlo
    # integration from scratch, which is not bit-reproducible run to run
    # (no fixed RNG seed) -- so the two YPBBN/DoH/Neff values agree only
    # to vegas's intrinsic noise level (~1e-6 relative), not to full
    # precision. rel=1e-4 is generous against that noise while still
    # catching a real regression (e.g. accidentally loading the wrong
    # file content, which would shift results at the percent level).
    # save_nTOp=False/save_nTOp_thermal=False: weak_rate_cache=False alone
    # forces a recompute but does NOT stop the result being written back
    # to the tracked rates/weak/ cache files -- since vegas has no fixed
    # seed, that recompute-and-overwrite would dirty the committed
    # nTOp_thermal_<hash>.txt for the *default* fingerprint on every test
    # run, and (for the custom-fingerprint run) leave a stray new cache
    # file behind. Disable both so this test only reads, never writes.
    r_default = PRIMAT({"network": "small", "verbose": False,
                       "weak_rate_cache": False,
                       "save_nTOp": False, "save_nTOp_thermal": False}).primat_results()
    r_custom = PRIMAT({"network": "small", "verbose": False,
                      "weak_rate_cache": False,
                      "save_nTOp": False, "save_nTOp_thermal": False,
                      "nevo_file": str(dst)}).primat_results()

    assert r_custom["Neff"] == pytest.approx(r_default["Neff"], rel=1e-4)
    assert r_custom["YPBBN"] == pytest.approx(r_default["YPBBN"], rel=1e-4)
    assert r_custom["DoH"] == pytest.approx(r_default["DoH"], rel=1e-4)


# ---------------------------------------------------------------------------
# nevo_file_prefix
# ---------------------------------------------------------------------------

def test_nevo_file_prefix_missing_raises_value_error():
    """A nevo_file_prefix whose derived default files don't exist under
    rates/NEVO/ raises a clear ValueError at PRIMATConfig construction time."""
    with pytest.raises(ValueError, match="nevo_file_prefix.*not found"):
        PRIMATConfig({"nevo_file_prefix": "DOES_NOT_EXIST"})


@pytest.mark.slow
@pytest.mark.solve
def test_nevo_file_prefix_reproduces_default(tmp_path):
    """A renamed copy of the shipped NEVO thermo + spectral tables under a
    custom nevo_file_prefix reproduces the default results up to vegas's
    intrinsic Monte Carlo noise (same content, different path), while
    changing the fingerprint (cache miss). The shipped thermal-correction
    cache (nTOp_thermal_<hash>.txt) is keyed on a fingerprint that includes
    nevo_file_prefix, so the custom-prefix run misses it and re-runs the
    vegas integration from scratch instead of loading the bit-identical
    cached table the default run uses -- see
    test_nevo_file_with_custom_copy_reproduces_default for the same
    reasoning and the resulting rel=1e-4 tolerance (~1e-6 vegas noise level,
    generous margin)."""
    import shutil
    from primat.main import PRIMAT

    cfg_default = PRIMATConfig({"network": "small"})
    nevo_dir = cfg_default.resolve_rates_path("NEVO")
    pairs = [
        ("NEVOPRIMAT_col_1_7.csv", "MYPREFIX_col_1_7.csv"),
        ("NEVOPRIMAT.csv",         "MYPREFIX.csv"),
    ]
    for src_name, dst_name in pairs:
        shutil.copy(os.path.join(nevo_dir, src_name),
                     os.path.join(nevo_dir, dst_name))
    try:
        cfg0 = PRIMATConfig({"network": "small"})
        cfg1 = PRIMATConfig({"network": "small", "nevo_file_prefix": "MYPREFIX"})
        fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
        fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
        assert fp0 != fp1

        # See the matching comment in test_nevo_file_with_custom_copy_reproduces_default:
        # save_nTOp=False/save_nTOp_thermal=False keep this test read-only
        # w.r.t. the tracked rates/weak/ cache files.
        r_default = PRIMAT({"network": "small", "verbose": False,
                           "weak_rate_cache": False,
                           "save_nTOp": False, "save_nTOp_thermal": False}).primat_results()
        r_custom  = PRIMAT({"network": "small", "verbose": False,
                           "weak_rate_cache": False,
                           "save_nTOp": False, "save_nTOp_thermal": False,
                           "nevo_file_prefix": "MYPREFIX"}).primat_results()
    finally:
        for _, dst_name in pairs:
            try:
                os.remove(os.path.join(nevo_dir, dst_name))
            except FileNotFoundError:
                pass  # file may have been evicted on cloud-synced volumes

    assert r_custom["Neff"] == pytest.approx(r_default["Neff"], rel=1e-4)
    assert r_custom["YPBBN"] == pytest.approx(r_default["YPBBN"], rel=1e-4)
    assert r_custom["DoH"] == pytest.approx(r_default["DoH"], rel=1e-4)


# ---------------------------------------------------------------------------
# external_scale_factor
# ---------------------------------------------------------------------------

def test_fingerprint_unaffected_by_external_scale_factor():
    """external_scale_factor only changes how a(T_gamma) is obtained (entropy
    ODE vs. reading the NEVO table's x column, see CLAUDE.md "Advanced:
    custom NEVO tables") -- it does not change the rate(T) integrand itself
    (ComputeWeakRates takes a T grid directly, with no dependence on a(T)).
    _WEAK_RATE_BG_FIELDS (weak_rates.py) deliberately excludes
    external_scale_factor for exactly this reason (see the module's "v1"
    format-version note), so the weak-rate cache fingerprint -- unlike the
    physical a(T)/t(T) history -- must NOT change."""
    cfg0 = PRIMATConfig({"network": "small"})
    cfg1 = PRIMATConfig({"network": "small", "external_scale_factor": True})
    fp0 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg0))
    fp1 = wr.fingerprint_hash(wr._weak_rate_fingerprint(cfg1))
    assert fp0 == fp1


def test_external_scale_factor_requires_incomplete_decoupling():
    """external_scale_factor reads a(T) from NEVOTable.x_of_Tg, which is only
    built when incomplete_decoupling=True."""
    with pytest.raises(ValueError, match="external_scale_factor.*incomplete_decoupling"):
        PRIMATConfig({"external_scale_factor": True, "incomplete_decoupling": False,
                     "spectral_distortions": False})


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
    from primat.main import PRIMAT
    from primat.config import PRIMATConfig

    r_cached = PRIMAT({"network": "small"})                        # loads rates/weak/*.txt
    r_fresh  = PRIMAT({"network": "small", "weak_rate_cache": False})  # forces ComputeWeakRates

    MeV_to_K = PRIMATConfig().MeV_to_Kelvin
    for T_MeV in [0.5, 1.0, 3.0, 10.0]:
        T_K = T_MeV * MeV_to_K
        for cached, fresh in ((r_cached.background.weak_nTOp_frwrd_raw, r_fresh.background.weak_nTOp_frwrd_raw),
                              (r_cached.background.weak_nTOp_bkwrd_raw, r_fresh.background.weak_nTOp_bkwrd_raw)):
            assert fresh(T_K) == pytest.approx(cached(T_K), rel=2e-3)


@pytest.mark.slow
def test_analytic_distortion_weak_rates_stay_fast():
    """Guard against the SD-FM np.vectorize regression that made
    analytic_distortions=True PRIMAT builds take ~17 s instead of ~1-2 s.

    Root cause (now fixed): the SD-FM correction
    (weak_rates.corrections._L_SD_FMCCR/_NoCCR, active whenever
    spectral_distortions=True, analytic_distortions=True,
    finite_mass_corrections=True) evaluated
    neutrino_history.AnalyticDistortion's `_fd`/`dFDneu_func`/energy-moment
    closures through `np.vectorize`, even though every one of them is built
    from plain elementwise numpy arithmetic and only needed a scalar `if`
    replaced by `np.where` to become natively array-vectorised -- np.vectorize
    instead fell back to ~3.7e6 individual Python calls per build. This
    combination is *never* cached (weak_rates.api.ComputeWeakRates's
    `forced_recompute` bypasses the rates/weak/ cache whenever
    spectral_distortions and analytic_distortions are both True, since
    y_SZ/y_gray are continuous MCMC-scanned knobs), so every single
    "Run BBN" with this flag combination pays this cost -- it must stay fast.

    Builds a small-network PRIMAT once per case (first call absorbs any
    one-time numba JIT compilation, so it isn't charged to the timing below)
    then times a second, JIT-warm build; asserts it stays within a generous
    multiple of the plain-default build time -- loose enough to tolerate
    ordinary hardware/load variance, tight enough to catch a return of the
    ~10x np.vectorize blow-up.
    """
    import time
    from primat import PRIMAT

    baseline_params = {"network": "small"}
    slow_cases = [
        {"network": "small", "analytic_distortions": True, "incomplete_decoupling": False},
        {"network": "small", "analytic_distortions": True, "incomplete_decoupling": False,
         "y_SZ": 0.1, "y_gray": 0.05},
        {"network": "small", "analytic_distortions": True, "incomplete_decoupling": False,
         "finite_mass_corrections": True, "radiative_corrections": False},
    ]

    def _timed_build(params):
        PRIMAT(params=dict(params))  # warm-up: absorb one-time numba JIT compilation
        t0 = time.perf_counter()
        PRIMAT(params=dict(params))
        return time.perf_counter() - t0

    baseline = _timed_build(baseline_params)
    for params in slow_cases:
        elapsed = _timed_build(params)
        budget = max(5.0, 8. * baseline)
        assert elapsed < budget, (
            f"weak-rate build for {params} took {elapsed:.2f}s "
            f"(baseline {baseline:.2f}s, budget {budget:.2f}s) -- likely a "
            "return of the np.vectorize SD-FM performance regression")


def test_setup_fd_impls_rewraps_on_numba_installed_change():
    """Regression test: _setup_fd_impls must re-wrap when numba_installed flips.

    It used to latch on a one-shot boolean (``_fd_impls_initialized``), so a
    second call -- e.g. from a second PRIMATConfig with the opposite
    numba_installed value -- was a silent no-op: whichever variant (jitted or
    plain Python) got set up *first* in the process stuck around forever,
    regardless of what later callers asked for. _setup_fd_impls now tracks
    the actual last-applied value (``_fd_impls_numba``) and always rebuilds
    from the pristine pure-Python implementations, so this is idempotent in
    either direction. Numerical results are identical either way (see
    test_FD2_between_zero_and_one etc.); this test only checks *which*
    implementation (jitted vs plain) is installed, via the wrapped function's
    type -- a numba CPUDispatcher vs an ordinary Python function.
    """
    pytest.importorskip("numba")
    from numba.core.registry import CPUDispatcher

    # Start from a known state, then flip numba_installed back and forth and
    # check the module-level FD_* names track it each time (not just once).
    wr._setup_fd_impls(False)
    assert not isinstance(wr.FD_nu3, CPUDispatcher)

    wr._setup_fd_impls(True)
    assert isinstance(wr.FD_nu3, CPUDispatcher)
    assert isinstance(wr.FD2, CPUDispatcher)

    wr._setup_fd_impls(False)
    assert not isinstance(wr.FD_nu3, CPUDispatcher)
    assert not isinstance(wr.FD2, CPUDispatcher)

    # Restore numba=True so later tests in this session (most fixtures use
    # the real numba_installed autodetection) see the JIT-compiled versions.
    wr._setup_fd_impls(True)


# ---------------------------------------------------------------------------
# SD-FM: finite-nucleon-mass correction to the spectral-distortion channel
# (generate_rates/PRIMAT-Main-gray.m's deltaChiFM; analytic-distortion mode
# only -- see _chi_func_sd_fm_v / _L_SD_FMCCR / _L_SD_FMNoCCR docstrings).
# ---------------------------------------------------------------------------

def _analytic_history(params):
    """Build an AnalyticDistortion neutrino history for the SD-FM tests."""
    cfg = PRIMATConfig(dict(params, incomplete_decoupling=False,
                           spectral_distortions=True, analytic_distortions=True))
    base = InstantaneousDecoupling(cfg, Plasma(cfg))
    return cfg, AnalyticDistortion(base)


def test_dFDneu_moments_keys_present_only_in_analytic_mode():
    """dFDneu_moments is the dict consumed by the SD-FM term; only
    AnalyticDistortion sets it (NEVO-table mode has no closed-form
    en-derivative of the tabulated distortion)."""
    _, hist = _analytic_history({"network": "small", "y_SZ": 0.05,
                                  "y_gray": 0.03})
    expected_keys = {"e2p0", "e3p0", "e2p1", "e3p1", "e4p1",
                      "e2p2", "e3p2", "e4p2"}
    assert set(hist.dFDneu_moments) == expected_keys
    for fn in hist.dFDneu_moments.values():
        assert np.isfinite(fn(1.5, 1.0, 1.3, +1))


def test_dFDneu_moments_match_finite_difference():
    """Each moment M[n,k](en) = d^k/den^k[en^n*dFDneu_func(en)] must agree
    with a finite difference of en^n*dFDneu_func, both for en >= 0 (the
    "raw" formula) and en < 0 (the antisymmetric-dispatch branch) -- this is
    the live-code analogue of scratch/derive_sd_fm_distortions.py's
    self-check (which only validates the symbolic derivation, not its
    transcription into neutrino_history.py).  h is tuned per-order to
    balance truncation error against floating-point round-off in the second
    derivative (h=1e-3 keeps roundoff/h^2 << truncation ~ h^2, unlike the
    naive h=1e-6 that works for first derivatives only).
    """
    _, hist = _analytic_history({"network": "small", "y_SZ": 0.11,
                                  "y_gray": 0.17})
    znu = 1.3
    for en in (-3.5, -0.4, 0.4, 3.5):
        for sgnq in (+1.0, -1.0):
            for n, order in ((2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)):
                h = 1e-5 if order == 1 else 1e-3
                f = lambda e: e**n * hist.dFDneu_func(e, None, znu, sgnq)
                if order == 1:
                    fd_est = (f(en + h) - f(en - h)) / (2 * h)
                else:
                    fd_est = (f(en + h) - 2 * f(en) + f(en - h)) / h**2
                closed = hist.dFDneu_moments[f"e{n}p{order}"](en, None, znu, sgnq)
                assert closed == pytest.approx(fd_est, abs=2e-4)


def test_sd_fm_vanishes_at_zero_distortion_amplitude():
    """With y_SZ=y_gray=0, dFDneu_func (hence its moments) is
    identically zero, so the SD-FM correction (built entirely from those
    moments) must vanish too."""
    cfg, hist = _analytic_history({"network": "small", "y_SZ": 0.,
                                    "y_gray": 0.,
                                    "munuOverTnu": 0.})
    ctx = wr._build_rate_context([np.array([1e9, 2e9]), np.array([0.7e9, 1.4e9])], cfg)
    T_arr = np.array([1e9, 5e9])
    for sgnq in (+1, -1):
        for L_func in (wr._L_SD_FMCCR, wr._L_SD_FMNoCCR):
            val = L_func(ctx, T_arr, sgnq, hist.dFDneu_moments)
            assert np.allclose(val, 0., atol=1e-15)


def test_sd_fm_nonzero_for_nonzero_distortion():
    """A nonzero distortion amplitude must make the SD-FM correction nonzero
    (sanity check that the term is actually wired up, not a silent no-op)."""
    cfg, hist = _analytic_history({"network": "small", "y_SZ": 0.05,
                                    "y_gray": 0.03})
    ctx = wr._build_rate_context([np.array([1e9, 2e9]), np.array([0.7e9, 1.4e9])], cfg)
    T_arr = np.array([1e9, 5e9])
    val = wr._L_SD_FMCCR(ctx, T_arr, +1, hist.dFDneu_moments)
    assert np.all(np.abs(val) > 0.)


def test_correction_terms_includes_sd_fm_only_with_finite_mass_corrections():
    """_correction_terms must add an "SD_FM" term iff dFDneu_moments is
    supplied AND cfg.finite_mass_corrections is True -- mirroring
    generate_rates/PRIMAT-Main-gray.m's IPENdpSDFM/IPENdpSDFMCCR, which are
    gated by $SpectralDistortions && $AnalyticDistortions only (finite-mass
    corrections being on is implicit there since deltaChiFM is the only
    place that function is used)."""
    cfg, hist = _analytic_history({"network": "small", "y_SZ": 0.05,
                                    "y_gray": 0.03,
                                    "finite_mass_corrections": True})
    ctx = wr._build_rate_context([np.array([1e9, 2e9]), np.array([0.7e9, 1.4e9])], cfg)
    T_arr = np.array([1e9, 5e9])

    names_with_fm = [name for name, _ in
                      wr._correction_terms(ctx, T_arr, +1, hist.dFDneu_func, hist.dFDneu_moments)]
    assert "SD_FM" in names_with_fm

    names_without_moments = [name for name, _ in
                      wr._correction_terms(ctx, T_arr, +1, hist.dFDneu_func, None)]
    assert "SD_FM" not in names_without_moments
