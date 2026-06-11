"""Tests for plasma thermodynamics functions."""
import pytest
import numpy as np
from pyprimat.config import PyPRConfig
import pyprimat.plasma as thermo


@pytest.fixture(scope="module", autouse=True)
def initialise():
    thermo.initialise(PyPRConfig())


@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
def test_rho_g_positive_and_scales(T):
    assert thermo.rho_g(T) > 0
    assert thermo.rho_g(2 * T) == pytest.approx(thermo.rho_g(T) * 16, rel=1e-6)


@pytest.mark.parametrize("T", [0.5, 1.0, 5.0])
def test_rho_e_positive(T):
    assert thermo.rho_e(T) > 0


def test_rho_e_vanishes_at_low_T():
    assert thermo.rho_e(1e-5) == 0.0


@pytest.mark.parametrize("T", [0.5, 1.0, 5.0])
def test_p_e_positive(T):
    assert thermo.p_e(T) > 0


def test_spl_and_dspl_dT_consistent_with_standalone():
    """spl_and_dspl_dT must return the same values as spl and dspl_dT separately."""
    for T in [0.2, 0.5, 1.0, 5.0]:
        s_combined, ds_combined = thermo.spl_and_dspl_dT(T)
        assert s_combined  == pytest.approx(thermo.spl(T),     rel=1e-10)
        assert ds_combined == pytest.approx(thermo.dspl_dT(T), rel=1e-10)


def test_dspl_dT_finite_difference():
    """dspl_dT should agree with a finite-difference estimate of d(spl)/dT."""
    T = 1.0
    dT = 1e-4
    fd = (thermo.spl(T + dT) - thermo.spl(T - dT)) / (2 * dT)
    assert thermo.dspl_dT(T) == pytest.approx(fd, rel=1e-4)


def test_T_nu_decoupling_high_T_limit():
    """At high T >> me, entropy is dominated by photons+e±, so T_nu → T_γ."""
    T = 100.0
    assert thermo.T_nu_decoupling(T) == pytest.approx(T, rel=1e-3)


def test_T_nu_decoupling_low_T_limit():
    """At low T << me, only photon entropy survives, so T_nu → T_γ*(4/11)^(1/3)."""
    T = 0.001
    expected = T * (4.0 / 11.0) ** (1.0 / 3.0)
    assert thermo.T_nu_decoupling(T) == pytest.approx(expected, rel=1e-3)


def test_rho_nu_scaling():
    """rho_nu should scale as T^4."""
    T = 2.0
    assert thermo.rho_nu(2 * T) == pytest.approx(thermo.rho_nu(T) * 16, rel=1e-6)


def test_spl_positive():
    for T in [0.1, 1.0, 10.0]:
        assert thermo.spl(T) > 0


def test_electron_thermo_cache_not_clobbered_by_nondefault_fingerprint():
    """A non-default fingerprint must not overwrite the shipped electron cache.

    The shipped rates/plasma/electron_thermo_cache.txt is fingerprinted on
    (n_electron_table, T_start_cosmo_MeV) with the defaults.  Building a Plasma
    with a different T_start_cosmo_MeV recomputes the table in memory, but with
    save_electron_thermo=False (the default) it must NOT rewrite the tracked
    file -- otherwise e.g. the T_start_cosmo_MeV=100 reference run would churn
    the shipped table and perturb later default runs (IDEAS.md §8.2).
    """
    import os
    from pyprimat.plasma import Plasma

    cfg = PyPRConfig()
    cache_path = os.path.join(cfg.data_dir, "rates", "plasma",
                              "electron_thermo_cache.txt")
    before = open(cache_path, "rb").read()

    # Different fingerprint -> guaranteed recompute path inside Plasma.__init__.
    Plasma(PyPRConfig({"T_start_cosmo_MeV": 100.0}))

    assert open(cache_path, "rb").read() == before, (
        "electron_thermo_cache.txt was modified by a non-default run "
        "(save_electron_thermo should gate the write)"
    )
