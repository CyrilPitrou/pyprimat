"""Tests for mc_uncertainty, MCResult, and MCQuantityResult."""
import os
import pytest
import numpy as np
from primat.main import mc_uncertainty, MCResult, MCQuantityResult, _mc_run_batch

# Every test in this module runs at least one mc_uncertainty() loop, i.e.
# several full PRIMAT().solve() calls -- squarely in the "solve" tier.
pytestmark = [pytest.mark.slow, pytest.mark.solve]

_BASE = {"network": "small"}
_NUM_MC = 8


@pytest.fixture(scope="module")
def mc_single():
    return mc_uncertainty(_NUM_MC, "YPBBN", params=_BASE, n_jobs=2, seed=0)


@pytest.fixture(scope="module")
def mc_multi():
    return mc_uncertainty(_NUM_MC, ["YPBBN", "DoH", "Li7oH"],
                          params=_BASE, n_jobs=2, seed=0)


# --- MCResult structure ---

def test_mc_single_returns_MCResult(mc_single):
    assert isinstance(mc_single, MCResult)


def test_mc_multi_returns_MCResult(mc_multi):
    assert isinstance(mc_multi, MCResult)


def test_mc_single_has_expected_key(mc_single):
    assert "YPBBN" in list(mc_single)


def test_mc_multi_has_all_keys(mc_multi):
    for key in ("YPBBN", "DoH", "Li7oH"):
        assert key in list(mc_multi)


# --- MCQuantityResult attributes ---

def test_central_is_float(mc_single):
    assert isinstance(mc_single["YPBBN"].central, float)


def test_mean_is_float(mc_single):
    assert isinstance(mc_single["YPBBN"].mean, float)


def test_std_is_float(mc_single):
    assert isinstance(mc_single["YPBBN"].std, float)


def test_values_shape(mc_single):
    assert mc_single["YPBBN"].values.shape == (_NUM_MC,)


def test_mean_consistent_with_values(mc_single):
    q = mc_single["YPBBN"]
    assert q.mean == pytest.approx(np.mean(q.values), rel=1e-10)


def test_std_consistent_with_values(mc_single):
    q = mc_single["YPBBN"]
    assert q.std == pytest.approx(np.std(q.values), rel=1e-10)


def test_central_close_to_nominal(mc_single):
    """Central value should match a plain solve at nominal rates."""
    assert mc_single["YPBBN"].central == pytest.approx(0.2469, abs=1e-3)


# --- std > 0 (rates actually vary) ---

def test_std_positive(mc_single):
    assert mc_single["YPBBN"].std > 0


def test_std_positive_multi(mc_multi):
    for key in ("YPBBN", "DoH", "Li7oH"):
        assert mc_multi[key].std > 0


# --- Reproducibility ---

def test_same_seed_same_result():
    mc_a = mc_uncertainty(4, "YPBBN", params=_BASE, n_jobs=1, seed=42)
    mc_b = mc_uncertainty(4, "YPBBN", params=_BASE, n_jobs=1, seed=42)
    np.testing.assert_array_equal(mc_a["YPBBN"].values, mc_b["YPBBN"].values)


def test_different_seed_different_result():
    mc_a = mc_uncertainty(4, "YPBBN", params=_BASE, n_jobs=1, seed=0)
    mc_b = mc_uncertainty(4, "YPBBN", params=_BASE, n_jobs=1, seed=99)
    assert not np.allclose(mc_a["YPBBN"].values, mc_b["YPBBN"].values)


# --- Incremental reuse (prev=) ---

def test_extend_matches_full_run():
    """Extending an N-sample result to M>N must give *exactly* the same M
    samples as computing M from scratch -- the whole point of the ``prev``
    reuse is that the first N samples are seed-deterministic and untouched."""
    full = mc_uncertainty(6, ["YPBBN", "DoH"], params=_BASE, n_jobs=1, seed=0)
    part = mc_uncertainty(3, ["YPBBN", "DoH"], params=_BASE, n_jobs=1, seed=0)
    ext  = mc_uncertainty(6, ["YPBBN", "DoH"], params=_BASE, n_jobs=1, seed=0,
                          prev=part)
    for q in ("YPBBN", "DoH"):
        np.testing.assert_array_equal(full[q].values, ext[q].values)
        assert full[q].central == ext[q].central


def test_extend_truncates_when_fewer_requested():
    """Requesting fewer samples than ``prev`` truncates without solving."""
    big   = mc_uncertainty(6, "YPBBN", params=_BASE, n_jobs=1, seed=0)
    small = mc_uncertainty(4, "YPBBN", params=_BASE, n_jobs=1, seed=0, prev=big)
    np.testing.assert_array_equal(big["YPBBN"].values[:4], small["YPBBN"].values)


def test_prev_ignored_when_seed_differs():
    """An incompatible ``prev`` (different seed) is silently ignored, giving a
    full recompute at the requested seed rather than reusing stale samples."""
    prev = mc_uncertainty(3, "YPBBN", params=_BASE, n_jobs=1, seed=0)
    ref  = mc_uncertainty(3, "YPBBN", params=_BASE, n_jobs=1, seed=5)
    got  = mc_uncertainty(3, "YPBBN", params=_BASE, n_jobs=1, seed=5, prev=prev)
    np.testing.assert_array_equal(ref["YPBBN"].values, got["YPBBN"].values)


def test_result_records_seed():
    """MCResult.seed is stored so callers (e.g. the GUI) can decide whether a
    cached result is reusable as ``prev``."""
    mc = mc_uncertainty(2, "YPBBN", params=_BASE, n_jobs=1, seed=7)
    assert mc.seed == 7


# --- nuclide name as quantity ---

def test_nuclide_quantity_works():
    mc = mc_uncertainty(4, "He4", params=_BASE, n_jobs=2, seed=0)
    assert isinstance(mc, MCResult)
    assert mc["He4"].central > 0
    assert mc["He4"].std > 0


# --- Large network variation ---

def test_mc_large_network_varies_heavy_elements():
    """Verify that MC on the large network varies species only present there."""
    # We choose B10, which is only produced in the large network (or at least
    # its variation depends on large-network-only reactions).
    # Using a tiny sample size for speed.
    mc = mc_uncertainty(4, ["DoH", "B10"], params={"network": "large"}, n_jobs=2, seed=0)
    assert mc["DoH"].std > 0
    assert mc["B10"].std > 0


# ---------------------------------------------------------------------------
# tau_n variation (Item 14)
# ---------------------------------------------------------------------------

def test_tau_n_alone_gives_nonzero_spread_in_YPBBN():
    """With no nuclear-rate offsets (rate_keys=[]), the only randomness left
    is tau_n_sample = tau_n_central + std_tau_n * randn() (one extra draw per
    sample, see _mc_run_batch).  Since YPBBN depends on the n<->p weak-rate
    normalisation 1/(Fn*tau_n), its spread across samples must be non-zero and
    of plausible magnitude (a fraction of a percent, comparable to the
    rate-driven spread in test_std_positive)."""
    res = np.array(_mc_run_batch({"network": "small", "verbose": False},
                                  rate_keys=[], quantities=["YPBBN"],
                                  seeds=list(range(8))))
    std = res[:, 0].std()
    assert 0 < std < 1e-3


def test_tau_n_normalization_false_disables_tau_n_effect():
    """With cfg.tau_n_normalization=False, tau_n does not enter background.NormWeakRates
    (see StandardBackground._setup_weak_rates), so the extra per-sample tau_n
    draw must be a no-op:
    with no rate offsets either, every sample reproduces the central value."""
    res = np.array(_mc_run_batch(
        {"network": "small", "verbose": False, "tau_n_normalization": False},
        rate_keys=[], quantities=["YPBBN"], seeds=list(range(8))))
    assert np.all(res[:, 0] == res[0, 0])


# ---------------------------------------------------------------------------
# custom_network support in mc_uncertainty / _mc_run_batch
# ---------------------------------------------------------------------------

import primat
_TABLES_DIR = os.path.join(os.path.dirname(primat.__file__),
                            "rates", "nuclear", "tables", "d_d__He3_n")


def _table_text(T9, rate, err):
    """Build a 3-column rate-table text buffer (T9, rate, err), one row per
    sample point -- the format expected by custom_network["replaced"]."""
    lines = [f"{t:.6e} {r:.6e} {e:.6e}" for t, r, e in zip(T9, rate, err)]
    return "\n".join(lines) + "\n"


def test_removed_reaction_changes_central_value():
    """Removing a reaction alters the solved network, so DoH's central value
    (computed with custom_network) must differ from the default-network one."""
    default = mc_uncertainty(2, "DoH", params=_BASE, n_jobs=1, seed=0)
    removed = mc_uncertainty(2, "DoH", params=_BASE, n_jobs=1, seed=0,
                              custom_network={"removed": ["d_d__t_p"]})
    assert removed["DoH"].central != pytest.approx(default["DoH"].central)


def test_custom_error_column_drives_spread():
    """The core of the user's question: a reaction's MC spread should track
    its *custom* error column, not the shipped default. Same median rate,
    two different uncertainty factors -- low spread vs high spread."""
    T9, rate, _err = np.loadtxt(
        os.path.join(_TABLES_DIR, "d_d__He3_n_primat.txt"), unpack=True)

    noerr_table  = _table_text(T9, rate, np.full_like(rate, 1.0))
    bigerr_table = _table_text(T9, rate, np.full_like(rate, 3.0))

    base_params = {"network": "small", "verbose": False, "debug": False}
    seeds = list(range(8))

    res_noerr = np.array(_mc_run_batch(
        base_params, rate_keys=["p_d_d__He3_n"], quantities=["DoH"], seeds=seeds,
        custom_network={"replaced": {"d_d__He3_n": noerr_table}}))
    res_bigerr = np.array(_mc_run_batch(
        base_params, rate_keys=["p_d_d__He3_n"], quantities=["DoH"], seeds=seeds,
        custom_network={"replaced": {"d_d__He3_n": bigerr_table}}))

    std_noerr  = res_noerr[:, 0].std()
    std_bigerr = res_bigerr[:, 0].std()
    # expsigma=1 means p_d_d__He3_n no longer perturbs the rate (median *
    # exp(p*log(1)) = median); the residual std_noerr is from the unrelated
    # per-sample tau_n draw (_mc_run_batch), so it should be tiny compared to
    # the big-error case rather than exactly zero.
    assert std_bigerr > 100 * std_noerr


def test_replaced_table_std_via_public_api():
    """Same as above, but through the public mc_uncertainty() entry point,
    proving the custom_network plumbing works end-to-end (not just via the
    internal _mc_run_batch worker)."""
    T9, rate, _err = np.loadtxt(
        os.path.join(_TABLES_DIR, "d_d__He3_n_primat.txt"), unpack=True)
    bigerr_table = _table_text(T9, rate, np.full_like(rate, 5.0))

    default = mc_uncertainty(8, "DoH", params=_BASE, n_jobs=1, seed=0)
    replaced = mc_uncertainty(
        8, "DoH", params=_BASE, n_jobs=1, seed=0,
        custom_network={"replaced": {"d_d__He3_n": bigerr_table}})

    assert replaced["DoH"].std > default["DoH"].std


def test_prev_ignored_when_custom_network_differs():
    """A prev computed under one custom_network must not be silently reused
    for a different one -- mirrors test_prev_ignored_when_seed_differs."""
    T9, rate, _err = np.loadtxt(
        os.path.join(_TABLES_DIR, "d_d__He3_n_primat.txt"), unpack=True)
    bigerr_table = _table_text(T9, rate, np.full_like(rate, 5.0))
    cn = {"replaced": {"d_d__He3_n": bigerr_table}}

    prev = mc_uncertainty(3, "DoH", params=_BASE, n_jobs=1, seed=0)
    ref  = mc_uncertainty(3, "DoH", params=_BASE, n_jobs=1, seed=0, custom_network=cn)
    got  = mc_uncertainty(3, "DoH", params=_BASE, n_jobs=1, seed=0, custom_network=cn,
                          prev=prev)
    np.testing.assert_array_equal(ref["DoH"].values, got["DoH"].values)


def test_prev_ignored_when_params_differ():
    """A prev computed under different params (here: network) must not be
    silently reused -- closes the pre-existing blind spot in the reuse guard."""
    prev = mc_uncertainty(3, "DoH", params={"network": "small"}, n_jobs=1, seed=0)
    large_amax8 = {"network": "large", "amax": 8}
    ref  = mc_uncertainty(3, "DoH", params=large_amax8, n_jobs=1, seed=0)
    got  = mc_uncertainty(3, "DoH", params=large_amax8, n_jobs=1, seed=0,
                          prev=prev)
    np.testing.assert_array_equal(ref["DoH"].values, got["DoH"].values)
