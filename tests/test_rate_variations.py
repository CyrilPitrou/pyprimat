"""
Tests for nuclear rate variation and MC uncertainty propagation.

The ``p_<reaction>`` mechanism shifts a reaction rate by ``exp(p × σ)``
relative to its median value, enabling MCMC sampling of nuclear-rate
uncertainties.  These tests verify that:
1. varying a rate actually changes the predicted abundances;
2. restoring p=0 reproduces the baseline to floating-point precision;
3. the MC runner propagates rate uncertainty to non-zero spread in observables.

The ``test_config_dynamic_attr`` test (attribute routing for p_* / delta_*)
lives in ``test_config.py`` where it logically belongs.
"""
import numpy as np
import pytest

from primat import PRIMAT, mc_uncertainty


@pytest.mark.slow
@pytest.mark.solve
def test_solve_variation():
    """Varying p_n_p__d_g shifts D/H; reverting p=0 restores the baseline."""
    inst = PRIMAT(params={"network": "small", "verbose": False})
    res0 = inst.solve()
    dh0  = res0["DoH"]

    # Shift n_p__d_g by +1σ and re-solve
    inst.cfg.p_n_p__d_g = 1.0
    res1 = inst.solve()
    dh1  = res1["DoH"]
    assert dh1 != dh0, "Changing p_n_p__d_g should affect D/H"

    # Restore and verify exact match (deterministic ODE)
    inst.cfg.p_n_p__d_g = 0.0
    res2 = inst.solve()
    dh2  = res2["DoH"]
    assert np.isclose(dh2, dh0, rtol=1e-10), (
        f"Reverting p_n_p__d_g should match baseline: {dh2:.8e} vs {dh0:.8e}"
    )


@pytest.mark.slow
@pytest.mark.solve
def test_mc_large_network():
    """MC uncertainty spread is positive for D/H and B10 in the large network."""
    mc = mc_uncertainty(5, ["DoH", "B10"],
                        params={"network": "large"}, n_jobs=-1)
    assert mc["DoH"].std > 0, "D/H should have non-zero uncertainty"
    assert mc["B10"].std > 0, "B10 should have non-zero uncertainty in large network"


if __name__ == "__main__":
    test_solve_variation()
    test_mc_large_network()
