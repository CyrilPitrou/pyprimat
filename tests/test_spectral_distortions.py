
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
