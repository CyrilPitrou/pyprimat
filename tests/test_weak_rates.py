"""Tests for weak_rates: Fn integral, Fermi-Coulomb, rate functions."""
import pytest
import numpy as np
from pypr.config import PyPRConfig
import pypr.weak_rates as wr


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
    import pypr.plasma as thermo
    thermo.initialise(cfg)
    return wr.InterpolateWeakRates(cfg)


def test_returns_two_interpolants(rate_interpolants):
    assert len(rate_interpolants) == 2          # forward (n->p), backward (p->n)


def test_all_rates_positive(rate_interpolants):
    """Both rate interpolants should return positive values in their range."""
    from pypr.config import PyPRConfig
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
    from pypr.config import PyPRConfig
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
    from pypr.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    ratio_low  = nTOp_frwrd_HT(1.0  * MeV_to_K) / nTOp_bkwrd_HT(1.0  * MeV_to_K)
    ratio_high = nTOp_frwrd_HT(10.0 * MeV_to_K) / nTOp_bkwrd_HT(10.0 * MeV_to_K)
    assert ratio_high < ratio_low   # ratio decreases toward 1
    assert ratio_high > 1.0         # but stays above 1


def test_forward_rate_increases_with_T(rate_interpolants):
    """n→p rate should increase with T in the HT era (well above freeze-out)."""
    nTOp_frwrd_HT = rate_interpolants[0]
    from pypr.config import PyPRConfig
    MeV_to_K = PyPRConfig().MeV_to_Kelvin
    rate_low  = nTOp_frwrd_HT(1.0  * MeV_to_K)
    rate_high = nTOp_frwrd_HT(10.0 * MeV_to_K)
    assert rate_high > rate_low
