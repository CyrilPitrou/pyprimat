
import pytest
import numpy as np
from pyprimat.main import PyPR

# Each test below runs two full PyPR().solve() calls with
# spectral_distortions on/off (a fingerprint mismatch against the shipped
# weak-rate cache also triggers a recompute) -- "solve" tier.
pytestmark = [pytest.mark.slow, pytest.mark.solve]


def test_spectral_distortions_effect():
    """Verify that spectral distortions have a small but non-zero effect on D/H."""
    params_base = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": True,
        "QED_corrections": True,
        "spectral_distortions": False,
        "verbose": False,
    }
    pr_base = PyPR(params_base)
    res_base = pr_base.solve()
    
    params_spec = params_base.copy()
    params_spec["spectral_distortions"] = True
    pr_spec = PyPR(params_spec)
    res_spec = pr_spec.solve()
    
    # Relative difference should be around 0.02% with the current implementation
    diff = (res_spec['DoH'] - res_base['DoH']) / res_base['DoH']
    
    # Check that it's positive and in the expected ballpark
    assert diff > 0
    assert 1e-4 < diff < 1e-3  # 0.01% to 0.1%

def test_spectral_distortions_Neff():
    """In NEVO, the energy density of distortions is 0 by construction
    relative to the defined neutrino temperatures. Neff should stay the same."""
    params_base = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": True,
        "spectral_distortions": False,
    }
    pr_base = PyPR(params_base)
    res_base = pr_base.solve()

    params_spec = params_base.copy()
    params_spec["spectral_distortions"] = True
    pr_spec = PyPR(params_spec)
    res_spec = pr_spec.solve()

    # Neff is determined by the background solve, which is the same
    # because rho_nu_SD is 0 for NEVO distortions.
    assert res_spec['Neff'] == pytest.approx(res_base['Neff'], rel=1e-8)


def test_analytic_distortion_shifts_Neff():
    """Analytic mu-type spectral distortions (delta_xi_nu != 0, Item 6) must
    shift Neff via the extra neutrino energy density rho_nu_SD, by exactly
    the amount that _Hubble already adds to the expansion rate.

    Unlike the NEVO case above (where rho_nu_SD is 0 by construction), the
    analytic-distortion path (incomplete_decoupling=False,
    analytic_distortions=True) has a non-zero rho_nu_SD(T_nu)
    (neutrino_history.AnalyticDistortion._rho_nu_SD). _Hubble adds the single
    aggregate term self._rho_nu_SD(Tnu_avg) to rho_tot for the expansion
    rate; _setup_derived_cosmo.N_eff now adds the *same* term (same Tnu_avg
    convention) to rho_rad, so

        Neff_spec - Neff_base = rho_nu_SD(Tnu_last) / rho_g(Tg_last)
                                 / ((7/8)(4/11)^(4/3))

    -- this pins that consistency to the ~1e-8 precision required by
    CLAUDE.md (FINAL.md Item 6, step 3).

    delta_xi_nu=0.05 gives Inty3_mu = (dxi/4)*dxi*(dxi^2 + 2*pi^2) > 0 (with
    munuOverTnu=0, the default), so rho_nu_SD > 0 and Neff increases.
    """
    params_base = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": False,
        "spectral_distortions": False,
        "verbose": False,
    }
    res_base = PyPR(params_base).solve()

    params_spec = dict(params_base, spectral_distortions=True,
                        analytic_distortions=True, delta_xi_nu=0.05)
    pr_spec = PyPR(params_spec)
    res_spec = pr_spec.solve()

    diff = res_spec['Neff'] - res_base['Neff']
    assert 1e-3 < diff < 1e-2

    Tg_last  = pr_spec._Tg_vec[-1]
    Tnu_last = pr_spec._Tnu_vec[-1]
    rho_g    = pr_spec.plasma.rho_g(Tg_last)
    rho_SD   = pr_spec._rho_nu_SD(Tnu_last)
    diff_expected = rho_SD / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

    assert diff == pytest.approx(diff_expected, rel=1e-8)
