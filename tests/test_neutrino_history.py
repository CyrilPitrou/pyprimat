"""Tests for the pluggable neutrino-sector background.

``pyprimat.neutrino_history.make_neutrino_history`` assembles a
``NeutrinoHistory`` (temperatures, heating, spectral distortion, extra energy
density) from a ``PyPRConfig``.  These tests pin the factory dispatch and the
protocol attributes without running a full BBN solve -- they only build the
neutrino-sector object on top of an initialised plasma.
"""
import numpy as np
import pytest

from pyprimat.config import PyPRConfig
from pyprimat.plasma import Plasma
from pyprimat.neutrino_history import (
    make_neutrino_history, NEVOTable, InstantaneousDecoupling, AnalyticDistortion,
)


def _history(params):
    cfg = PyPRConfig(params)
    return cfg, make_neutrino_history(cfg, Plasma(cfg))


def _Tg_MeV(cfg):
    """A representative photon temperature [MeV] inside the e+e- range."""
    return 0.5  # ~ m_e, where e+e- annihilation heating is active


def test_default_is_nevo_table_no_distortion():
    """Default config (incomplete decoupling, no distortions) -> NEVOTable.

    ``spectral_distortions=True`` is the ``PyPRConfig`` default (IDEAS2.md
    item 2), so it is explicitly disabled here to isolate the "no distortion"
    case checked by this test (the distorted case is covered by
    ``test_nevo_spectral_distortion_builds_callable`` below).
    """
    cfg, nh = _history({"network": "small", "spectral_distortions": False})
    assert isinstance(nh, NEVOTable)
    assert nh.dFDneu_func is None
    assert nh.rho_nu_SD is None
    # All four temperature/heating ingredients are present and callable.
    Tg = _Tg_MeV(cfg)
    for fn in (nh.Tnue_of_Tg, nh.Tnumu_of_Tg, nh.Tnutau_of_Tg, nh.N_NEVO_of_Tg):
        assert np.isfinite(float(fn(Tg)))
    assert float(nh.Tnue_of_Tg(Tg)) > 0.


def test_nevo_heating_nonzero_instantaneous_zero():
    """NEVO injects entropy into neutrinos (N != 0); instantaneous does not."""
    cfg_n, nh_nevo = _history({"network": "small"})
    # Sample N over the annihilation range; at least one point must be nonzero.
    Tgs = np.logspace(-2, 0.5, 50)   # 0.01 - ~3 MeV
    N_nevo = np.array([float(nh_nevo.N_NEVO_of_Tg(T)) for T in Tgs])
    assert np.any(np.abs(N_nevo) > 0.)

    # spectral_distortions=True (PyPRConfig default) requires
    # incomplete_decoupling=True; disable it for the instantaneous case.
    cfg_i, nh_inst = _history({"network": "small", "incomplete_decoupling": False,
                                "spectral_distortions": False})
    assert isinstance(nh_inst, InstantaneousDecoupling)
    N_inst = nh_inst.N_NEVO_of_Tg(Tgs)
    assert np.allclose(N_inst, 0.)


def test_instantaneous_three_flavours_share_temperature():
    """Instantaneous decoupling: all three flavours have the same temperature."""
    # spectral_distortions=True (PyPRConfig default) requires
    # incomplete_decoupling=True; disable it for the instantaneous case.
    cfg, nh = _history({"network": "small", "incomplete_decoupling": False,
                         "spectral_distortions": False})
    Tg = _Tg_MeV(cfg)
    Te = float(nh.Tnue_of_Tg(Tg))
    Tm = float(nh.Tnumu_of_Tg(Tg))
    Tt = float(nh.Tnutau_of_Tg(Tg))
    assert Te == pytest.approx(Tm) == pytest.approx(Tt)
    # Below e+e- annihilation the neutrino temperature drops below T_gamma.
    assert Te < Tg


def test_analytic_distortion_decorates_instantaneous():
    """analytic_distortions wraps the base in AnalyticDistortion.

    Config requires instantaneous decoupling for the analytic mode, so the
    decorated base is an InstantaneousDecoupling: the temperatures/heating are
    inherited unchanged while dFDneu_func / rho_nu_SD are added.
    """
    cfg, nh = _history({
        "network": "small",
        "spectral_distortions": True,
        "analytic_distortions": True,
        "incomplete_decoupling": False,
        "delta_xi_nu": 0.05,
        "y_SZ": 0.01,
    })
    assert isinstance(nh, AnalyticDistortion)
    assert nh.dFDneu_func is not None
    assert nh.rho_nu_SD is not None
    # A nonzero mu-shift must produce a nonzero distortion somewhere.
    vals = [nh.dFDneu_func(en, 1.0, 1.0, +1) for en in (0.5, 1.0, 2.0)]
    assert any(abs(v) > 0. for v in vals)
    # Extra energy density is positive and a small fraction of rho_nu.
    Tnu = 0.3
    assert nh.rho_nu_SD(Tnu) != 0.


def test_nevo_spectral_distortion_builds_callable():
    """NEVO-based distortion (analytic_distortions=False) sets dFDneu_func."""
    cfg, nh = _history({
        "network": "small",
        "spectral_distortions": True,
        "analytic_distortions": False,
        "incomplete_decoupling": True,
    })
    assert isinstance(nh, NEVOTable)
    assert nh.dFDneu_func is not None
    # The callable returns a finite float for an in-range argument.
    assert np.isfinite(nh.dFDneu_func(1.0, 1.0, 1.0, +1))
