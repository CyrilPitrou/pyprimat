"""
Regression tests against the reference values in CLAUDE.md.

Two layers:

* Default-precision sanity checks (via the ``solved_small`` / ``solved_large``
  fixtures) with loose tolerances — cheap, catch gross regressions.
* High-precision *reference* checks (``reference`` marker) that rerun at the
  exact settings used to produce the published numbers
  (numerical_precision=1e-10, n_temperature_table=10000, sampling_nTOp=500,
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
    assert solved_small._results["YPBBN"] == pytest.approx(0.2469156, abs=1e-4)


@pytest.mark.solve
def test_small_network_DoH(solved_small):
    assert solved_small._results["DoH"] == pytest.approx(2.43647e-5, rel=2e-3)


@pytest.mark.solve
def test_large_network_YPBBN(solved_large):
    assert solved_large._results["YPBBN"] == pytest.approx(0.2469190, abs=1e-4)


@pytest.mark.solve
def test_large_network_DoH(solved_large):
    assert solved_large._results["DoH"] == pytest.approx(2.43718e-5, rel=2e-3)


@pytest.mark.solve
def test_Neff_close_to_standard(solved_small):
    """Neff should be close to 3.044 for the standard model."""
    assert solved_small._results["Neff"] == pytest.approx(3.044, abs=0.005)


@pytest.mark.solve
def test_nTOp_Born_approximation_lowers_YP(solved_small):
    """Born-level n<->p rates (no radiative corrections) give lower YP."""
    from pyprimat.main import PyPR
    r_born = PyPR({"nTOp_Born_approximation": True,
                        "network": "small"})
    r_born.solve()
    assert r_born._results["YPBBN"] < solved_small._results["YPBBN"] - 0.001


@pytest.mark.solve
def test_Li7oH_order_of_magnitude(solved_small):
    """Li7/H should be in the range 1e-10 to 1e-9."""
    Li7 = solved_small._results["Li7oH"]
    assert 1e-10 < Li7 < 1e-9


@pytest.mark.solve
def test_He3oH_order_of_magnitude(solved_small):
    """He3/H should be in the range 1e-6 to 1e-4."""
    He3 = solved_small._results["He3oH"]
    assert 1e-6 < He3 < 1e-4


# ---------------------------------------------------------------------------
# High-precision reference checks (tight) — reproduce the CLAUDE.md numbers
# ---------------------------------------------------------------------------
# Settings used to produce the published reference values.
_REF_PARAMS = dict(numerical_precision=1e-10, n_temperature_table=10000,
                   sampling_nTOp=500, T_start_cosmo_MeV=100.0,
                   Omegabh2=0.022425, verbose=False, debug=False)


@pytest.fixture(scope="session")
def ref_small():
    from pyprimat.main import PyPR
    return PyPR({**_REF_PARAMS, "network": "small"}).PyPRresults()


@pytest.fixture(scope="session")
def ref_large():
    from pyprimat.main import PyPR
    return PyPR({**_REF_PARAMS, "network": "medium"}).PyPRresults()


@pytest.mark.reference
def test_reference_small_YPBBN(ref_small):
    assert ref_small["YPBBN"] == pytest.approx(0.2469156, abs=1e-5)


@pytest.mark.reference
def test_reference_small_DoH(ref_small):
    assert ref_small["DoH"] == pytest.approx(2.43647e-5, abs=3e-9)


@pytest.mark.reference
def test_reference_large_YPBBN(ref_large):
    assert ref_large["YPBBN"] == pytest.approx(0.2469190, abs=1e-5)


@pytest.mark.reference
def test_reference_large_DoH(ref_large):
    assert ref_large["DoH"] == pytest.approx(2.43721e-5, abs=3e-9)


# ---------------------------------------------------------------------------
# No-numba full solve: pure-Python kernels must agree with the JIT path
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_no_numba_small_matches_numba(solved_small):
    """Pure-Python (numba_installed=False) must agree with the JIT path to 1e-4."""
    from pyprimat.main import PyPR
    r_nn = PyPR({"numba_installed": False, "network": "small"}).PyPRresults()
    assert r_nn["YPBBN"] == pytest.approx(solved_small._results["YPBBN"], rel=1e-4)
    assert r_nn["DoH"]   == pytest.approx(solved_small._results["DoH"],   rel=1e-4)


@pytest.mark.solve
def test_no_numba_medium_smoke():
    """Pure-Python medium network solve completes and YP is physically reasonable."""
    from pyprimat.main import PyPR
    r = PyPR({"numba_installed": False, "network": "medium"}).PyPRresults()
    assert 0.24 < r["YPBBN"] < 0.25
    assert 2.0e-5 < r["DoH"] < 3.0e-5


# ---------------------------------------------------------------------------
# amax cutoff: large network filtered to A <= 20 matches medium to ~1e-3
# ---------------------------------------------------------------------------

@pytest.mark.solve
def test_amax_filter_light_elements_match_medium(solved_large):
    """With amax=20, heavy reactions (A>20) are dropped; light elements match medium."""
    from pyprimat.main import PyPR
    r = PyPR({"network": "large", "amax": 20}).PyPRresults()
    # Light elements should still match the medium result to ~1e-3 relative
    assert r["YPBBN"] == pytest.approx(solved_large._results["YPBBN"], rel=1e-3)
    assert r["DoH"]   == pytest.approx(solved_large._results["DoH"],   rel=1e-3)
