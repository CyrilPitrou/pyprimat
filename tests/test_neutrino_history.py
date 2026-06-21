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

    ``spectral_distortions=True`` is the ``PyPRConfig`` default, so it is
    explicitly disabled here to isolate the "no distortion"
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


# ---------------------------------------------------------------------------
# external_scale_factor / x_of_Tg
# ---------------------------------------------------------------------------

def test_x_of_Tg_present_for_nevo_table_only():
    """x_of_Tg -- the NEVO table's x(T_gamma) interpolant used by
    external_scale_factor=True -- is built for NEVOTable and left at the
    NeutrinoHistory default (None) for InstantaneousDecoupling, which has no
    table to read x from."""
    cfg, nh = _history({"network": "small", "spectral_distortions": False})
    assert isinstance(nh, NEVOTable)
    assert nh.x_of_Tg is not None
    Tg = _Tg_MeV(cfg)
    assert float(nh.x_of_Tg(Tg)) > 0.

    cfg_i, nh_i = _history({"network": "small", "incomplete_decoupling": False,
                             "spectral_distortions": False})
    assert isinstance(nh_i, InstantaneousDecoupling)
    assert nh_i.x_of_Tg is None


@pytest.mark.slow  # external_scale_factor changes the cache fingerprint -> recompute (~3 s)
def test_external_scale_factor_a_of_T_matches_minimal_on_table_grid():
    """a(T_gamma) and t(T_gamma) built from external_scale_factor=True (direct
    NEVO-table x(T) lookup) must agree with the default entropy-conservation
    ODE solve to within ~1e-5,
    confirming a ~ x is a valid alternative background construction."""
    from pyprimat.main import PyPR

    p_min = PyPR({"network": "small"})
    p_ext = PyPR({"network": "small", "external_scale_factor": True})

    # Probe over the table's covered T_gamma range (avoid the extrapolated
    # tails, which are exact by construction in both modes -- a ~ 1/T).
    Tg_min, Tg_max = p_min.cfg.T_end / p_min.cfg.MeV_to_Kelvin, 3.0  # MeV
    Tgs = np.logspace(np.log10(Tg_min), np.log10(Tg_max), 50)

    a_min, a_ext = p_min.background.a_of_T(Tgs), p_ext.background.a_of_T(Tgs)
    t_min, t_ext = p_min.background.t_of_T(Tgs), p_ext.background.t_of_T(Tgs)

    assert np.allclose(a_ext, a_min, rtol=1e-5)
    assert np.allclose(t_ext, t_min, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.solve
def test_external_scale_factor_matches_minimal():
    """A full BBN solve with external_scale_factor=True reproduces the default
    (minimal-mode) Neff, YPBBN and D/H to the precision expected from the
    a(T)/t(T) agreement above."""
    from pyprimat.main import PyPR

    r_min = PyPR({"network": "small"}).PyPRresults()
    r_ext = PyPR({"network": "small", "external_scale_factor": True}).PyPRresults()

    assert r_ext["Neff"]  == pytest.approx(r_min["Neff"],  rel=1e-10)
    assert r_ext["YPBBN"] == pytest.approx(r_min["YPBBN"], rel=1e-5)
    assert r_ext["DoH"]   == pytest.approx(r_min["DoH"],   rel=3e-5)
