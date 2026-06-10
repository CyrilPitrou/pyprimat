"""Tests for the PyPR public API."""
import pytest
import numpy as np
from pyprimat.main import PyPR


def test_A_N_Z_dicts():
    r = PyPR()
    assert r.A["He4"] == 4
    assert r.Z["He4"] == 2
    assert r.N["He4"] == 2
    assert r.A["H2"] == 2
    assert r.A["Li7"] == 7
    assert r.A["n"] == 1
    assert r.Z["n"] == 0


def test_getitem_returns_callable(solved_small):
    fn = solved_small["He4"]
    assert callable(fn)


def test_getitem_returns_positive_values(solved_small):
    t = np.logspace(0, 5, 20)
    vals = solved_small["He4"](t)
    assert vals.shape == (20,)
    # Before He4 forms its abundance is physically ~0; the stiff BDF solver
    # (and the linear interpolation between its output points) can leave
    # machine-noise-level negative excursions there.  Require non-negativity
    # only above that noise floor (final He4 ~ 0.06).
    assert np.all(vals >= -1e-12)


def test_getitem_scalar_input(solved_small):
    val = solved_small["He4"](100.0)
    assert isinstance(val, float)
    assert val > 0


def test_getitem_unknown_species_raises(solved_small):
    with pytest.raises(KeyError):
        solved_small["Unobtainium"]


def test_getitem_all_small_network_species(solved_small):
    for sp in ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"]:
        fn = solved_small[sp]
        assert callable(fn)
        assert fn(100.0) >= 0


def test_T_of_t_and_t_of_T_are_inverses(solved_small):
    T_test = 0.5   # MeV
    t_val = float(solved_small.t_of_T(T_test))
    T_back = float(solved_small.T_of_t(t_val))
    assert T_back == pytest.approx(T_test, rel=1e-4)


def test_get_quantity_result_key(solved_small):
    assert solved_small.get_quantity("YPBBN") == pytest.approx(
        solved_small._results["YPBBN"], rel=1e-12)


def test_get_quantity_nuclide_name(solved_small):
    val = solved_small.get_quantity("He4")
    assert val > 0
    assert val == pytest.approx(solved_small._Y_final["He4"], rel=1e-12)


def test_get_quantity_unknown_raises(solved_small):
    with pytest.raises(ValueError):
        solved_small.get_quantity("not_a_thing")


def test_lazy_solve_triggers_on_accessor():
    """Accessing a result without calling solve() should auto-trigger it."""
    r = PyPR({"network": "small"})
    assert r._results is None
    yp = r.YPBBN()
    assert r._results is not None
    assert yp > 0


def test_solve_cached():
    """Calling solve() twice returns identical results (no re-computation)."""
    r = PyPR({"network": "small"})
    res1 = r.solve()
    res2 = r.solve()
    assert res1["YPBBN"] == res2["YPBBN"]


def test_PyPRresults_returns_dict(solved_small):
    res = solved_small.PyPRresults()
    assert isinstance(res, dict)
    for key in ("YPBBN", "YPCMB", "DoH", "He3oH", "Li7oH", "Neff"):
        assert key in res


def test_result_values_physical(solved_small):
    res = solved_small._results
    assert 0.20 < res["YPBBN"] < 0.30
    assert 1e-5 < res["DoH"]   < 5e-5
    assert 1e-6 < res["He3oH"] < 1e-4
    assert 1e-10 < res["Li7oH"] < 1e-9
    assert 2.5 < res["Neff"] < 3.5
