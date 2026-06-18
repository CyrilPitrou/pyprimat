"""
Regression tests against the reference values in CLAUDE.md.

Two layers:

* Default-precision sanity checks (via the ``solved_small`` / ``solved_large``
  fixtures) with loose tolerances — cheap, catch gross regressions.
* High-precision *reference* checks (``reference`` marker) that rerun at the
  exact settings used to produce the published numbers
  (numerical_precision=1e-10, sampling_temperature_per_decade=2000, sampling_nTOp_per_decade=125,
  T_start_cosmo=100 MeV) and pin them to the tight CLAUDE.md tolerances
  (YP +/-1e-5, D/H +/-3e-9).  These take ~60 s total and are the real guard
  for changes to the nuclear network.
"""
import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Default-precision sanity checks (loose)
# ---------------------------------------------------------------------------


@pytest.mark.solve
def test_small_network_YPBBN(solved_small):
    assert solved_small.results["YPBBN"] == pytest.approx(0.2469983, abs=1e-4)


@pytest.mark.solve
def test_small_network_DoH(solved_small):
    assert solved_small.results["DoH"] == pytest.approx(2.43490e-5, rel=2e-3)


@pytest.mark.solve
def test_large_network_YPBBN(solved_large):
    assert solved_large.results["YPBBN"] == pytest.approx(0.2470017, abs=1e-4)


@pytest.mark.solve
def test_large_network_DoH(solved_large):
    assert solved_large.results["DoH"] == pytest.approx(2.43561e-5, rel=2e-3)


@pytest.mark.solve
def test_Neff_close_to_standard(solved_small):
    """Neff should be close to 3.044 for the standard model."""
    assert solved_small.results["Neff"] == pytest.approx(3.044, abs=0.005)


@pytest.mark.solve
def test_Born_mode_lowers_YP(solved_small):
    """Born-only n<->p rates (radiative/finite-mass corrections off) give lower YP."""
    from pyprimat.main import PyPR
    r_born = PyPR({"radiative_corrections": False,
                   "finite_mass_corrections": False,
                   "network": "small"})
    r_born.solve()
    assert r_born.results["YPBBN"] < solved_small.results["YPBBN"] - 0.001


@pytest.mark.solve
def test_Li7oH_order_of_magnitude(solved_small):
    """Li7/H should be in the range 1e-10 to 1e-9."""
    Li7 = solved_small.results["Li7oH"]
    assert 1e-10 < Li7 < 1e-9


@pytest.mark.solve
def test_He3oH_order_of_magnitude(solved_small):
    """He3/H should be in the range 1e-6 to 1e-4."""
    He3 = solved_small.results["He3oH"]
    assert 1e-6 < He3 < 1e-4


# ---------------------------------------------------------------------------
# High-precision reference checks (tight) — reproduce the CLAUDE.md numbers
# ---------------------------------------------------------------------------
# Settings used to produce the published reference values.
_REF_PARAMS = dict(numerical_precision=1e-10, sampling_temperature_per_decade=2000,
                   sampling_nTOp_per_decade=125, T_start_cosmo_MeV=100.0,
                   Omegabh2=0.022425, verbose=False, debug=False)


@pytest.fixture(scope="session")
def ref_small():
    from pyprimat.main import PyPR
    return PyPR({**_REF_PARAMS, "network": "small"}).PyPRresults()


@pytest.fixture(scope="session")
def ref_large():
    from pyprimat.main import PyPR
    return PyPR({**_REF_PARAMS, "network": "large", "amax": 8}).PyPRresults()


@pytest.mark.reference
def test_reference_small_YPBBN(ref_small):
    assert ref_small["YPBBN"] == pytest.approx(0.2469983, abs=1e-5)


@pytest.mark.reference
def test_reference_small_DoH(ref_small):
    assert ref_small["DoH"] == pytest.approx(2.43490e-5, abs=3e-9)


@pytest.mark.reference
def test_reference_large_YPBBN(ref_large):
    assert ref_large["YPBBN"] == pytest.approx(0.2470017, abs=1e-5)


@pytest.mark.reference
def test_reference_large_DoH(ref_large):
    assert ref_large["DoH"] == pytest.approx(2.43561e-5, abs=3e-9)


# ---------------------------------------------------------------------------
# No-numba full solve: pure-Python kernels must agree with the JIT path
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_no_numba_small_matches_numba(solved_small):
    """Pure-Python (numba_installed=False) must agree with the JIT path to 1e-4."""
    from pyprimat.main import PyPR
    r_nn = PyPR({"numba_installed": False, "network": "small"}).PyPRresults()
    assert r_nn["YPBBN"] == pytest.approx(solved_small.results["YPBBN"], rel=1e-4)
    assert r_nn["DoH"]   == pytest.approx(solved_small.results["DoH"],   rel=1e-4)


@pytest.mark.solve
def test_no_numba_large_amax8_smoke():
    """Pure-Python large/amax=8 network solve completes and YP is physically
    reasonable (the old "medium" network's exact 68-reaction equivalent)."""
    from pyprimat.main import PyPR
    r = PyPR({"numba_installed": False, "network": "large", "amax": 8}).PyPRresults()
    assert 0.24 < r["YPBBN"] < 0.25
    assert 2.0e-5 < r["DoH"] < 3.0e-5


# ---------------------------------------------------------------------------
# amax cutoff: large network filtered to A <= 20 matches the full large
# network to ~1e-3
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_amax_filter_light_elements_match_large(solved_large):
    """With amax=20, heavy reactions (A>20) are dropped; light elements match
    the full large network."""
    from pyprimat.main import PyPR
    r = PyPR({"network": "large", "amax": 20}).PyPRresults()
    # Light elements should still match the full large-network result to
    # ~1e-3 relative.
    assert r["YPBBN"] == pytest.approx(solved_large.results["YPBBN"], rel=1e-3)
    assert r["DoH"]   == pytest.approx(solved_large.results["DoH"],   rel=1e-3)


@pytest.mark.solve
def test_small_amax2_collapses_to_deuterium_channel():
    """``network="small", amax=2`` must collapse both MT and LT to just the
    n<->p weak rate + n_p__d_g (CUSTOMPOPUP.md §1.2's MT-branch amax-ordering
    fix): previously the MT-era intersection used the *unfiltered* bare names,
    so an amax-violating reaction could still run in the MT era."""
    from pyprimat.main import PyPR
    from pyprimat.config import PyPRConfig
    from pyprimat.network_data import load_network
    cfg = PyPRConfig({"network": "small", "amax": 2})
    mt_names = load_network(cfg, era="MT").names
    lt_names = load_network(cfg, era="LT").names
    assert mt_names == ["n__p", "n_p__d_g"]
    assert lt_names == ["n__p", "n_p__d_g"]

    r = PyPR({"network": "small", "amax": 2}).PyPRresults()
    assert r["YPBBN"] == 0.0
    assert r["DoH"] > 0.0
