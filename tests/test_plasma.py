"""Tests for plasma thermodynamics functions."""
import pytest
import numpy as np
from primat.config import PRIMATConfig
from primat.plasma import Plasma, rho_g, rho_nu


@pytest.fixture(scope="module")
def thermo():
    return Plasma(PRIMATConfig())


@pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
def test_rho_g_positive_and_scales(T):
    assert rho_g(T) > 0
    assert rho_g(2 * T) == pytest.approx(rho_g(T) * 16, rel=1e-6)


@pytest.mark.parametrize("T", [0.5, 1.0, 5.0])
def test_rho_e_positive(thermo, T):
    assert thermo.rho_e(T) > 0


def test_rho_e_vanishes_at_low_T(thermo):
    assert thermo.rho_e(1e-5) == 0.0


@pytest.mark.parametrize("T", [0.5, 1.0, 5.0])
def test_p_e_positive(thermo, T):
    assert thermo.p_e(T) > 0


def test_spl_and_dspl_dT_consistent_with_standalone(thermo):
    """spl_and_dspl_dT must return the same values as spl and dspl_dT separately."""
    for T in [0.2, 0.5, 1.0, 5.0]:
        s_combined, ds_combined = thermo.spl_and_dspl_dT(T)
        assert s_combined  == pytest.approx(thermo.spl(T),     rel=1e-10)
        assert ds_combined == pytest.approx(thermo.dspl_dT(T), rel=1e-10)


def test_dspl_dT_finite_difference(thermo):
    """dspl_dT should agree with a finite-difference estimate of d(spl)/dT."""
    T = 1.0
    dT = 1e-4
    fd = (thermo.spl(T + dT) - thermo.spl(T - dT)) / (2 * dT)
    assert thermo.dspl_dT(T) == pytest.approx(fd, rel=1e-4)


def test_T_nu_decoupling_high_T_limit(thermo):
    """At high T >> me, entropy is dominated by photons+e±, so T_nu → T_γ."""
    T = 100.0
    assert thermo.T_nu_decoupling(T) == pytest.approx(T, rel=1e-3)


def test_T_nu_decoupling_low_T_limit(thermo):
    """At low T << me, only photon entropy survives, so T_nu → T_γ*(4/11)^(1/3)."""
    T = 0.001
    expected = T * (4.0 / 11.0) ** (1.0 / 3.0)
    assert thermo.T_nu_decoupling(T) == pytest.approx(expected, rel=1e-3)


def test_rho_nu_scaling():
    """rho_nu should scale as T^4."""
    T = 2.0
    assert rho_nu(2 * T) == pytest.approx(rho_nu(T) * 16, rel=1e-6)


def test_spl_positive(thermo):
    for T in [0.1, 1.0, 10.0]:
        assert thermo.spl(T) > 0


def test_electron_thermo_cache_refreshed_on_fingerprint_mismatch():
    """A fingerprint mismatch triggers a recompute that overwrites the cache
    with the new configuration's fingerprint (electron-thermo recompute is
    cheap, ~0.7 s, so the cache is always kept consistent with the last run --
    unlike the more expensive weak-rate cache, see weak_rates.RecomputeWeakRates).

    The shipped data/plasma/electron_thermo_cache.txt is restored afterwards
    so this test does not leave the working tree dirty.
    """
    import os
    from primat.cache_utils import fingerprint_hash, read_cache_fingerprint_hash
    from primat.plasma import Plasma, ELECTRON_THERMO_FORMAT_VERSION

    cfg = PRIMATConfig()
    cache_path = os.path.join(cfg._resolved_data_dir, "plasma",
                              "electron_thermo_cache.txt")
    before = open(cache_path, "rb").read()

    try:
        # Different fingerprint -> guaranteed recompute path inside Plasma.__init__.
        cfg_alt = PRIMATConfig({"T_start_cosmo_MeV": 100.0})
        Plasma(cfg_alt)

        expected_hash = fingerprint_hash({
            "format_version":   ELECTRON_THERMO_FORMAT_VERSION,
            "n_electron_table": cfg_alt.n_electron_table,
            "T_start_cosmo_MeV": cfg_alt.T_start_cosmo_MeV,
        })
        assert read_cache_fingerprint_hash(cache_path) == expected_hash
    finally:
        with open(cache_path, "wb") as f:
            f.write(before)


# ---------------------------------------------------------------------------
# C backend plasma tests (require primat._primat_c to be built)
# ---------------------------------------------------------------------------

from primat.backend import HAS_C_BACKEND, run_bbn

requires_c_backend = pytest.mark.skipif(
    not HAS_C_BACKEND,
    reason="primat._primat_c C extension is not built"
)


@requires_c_backend
@pytest.mark.slow
@pytest.mark.backend
def test_c_backend_plasma_without_cache(tmp_path):
    """C backend can compute electron-thermo tables from scratch (no cache file).
    
    This verifies the fix for the NaN issue where the C backend's electron
    integrands could return NaN when the adaptive quadrature evaluated them
    slightly below E=x (the lower integration bound), causing sqrt(negative).
    """
    import os
    import shutil
    
    # Save the original cache file and remove it
    cfg = PRIMATConfig()
    cache_path = os.path.join(cfg._resolved_data_dir, "plasma",
                              "electron_thermo_cache.txt")
    
    # Remove the cache file to force C backend to compute from scratch
    cache_backup = tmp_path / "electron_thermo_cache_backup.txt"
    if os.path.exists(cache_path):
        shutil.copy(cache_path, cache_backup)
        os.remove(cache_path)
    
    try:
        # This should succeed (not fail with "cpr_ode_rk45: max_steps exceeded")
        result = run_bbn({"network": "small"}, force_backend="c")
        
        # Verify we got reasonable results (not NaN or obviously wrong)
        assert np.isfinite(result["YPBBN"])
        assert np.isfinite(result["DoH"])
        assert result["YPBBN"] > 0.24
        assert result["YPBBN"] < 0.25
        assert result["DoH"] > 2e-5
        assert result["DoH"] < 3e-5
    finally:
        # Restore the cache file
        if cache_backup.exists():
            shutil.copy(cache_backup, cache_path)


@requires_c_backend
@pytest.mark.slow
@pytest.mark.backend
def test_c_backend_plasma_with_cache():
    """C backend can read electron-thermo cache written by Python backend."""
    # Ensure cache exists (should be there from normal usage)
    result = run_bbn({"network": "small"}, force_backend="c")
    
    # Verify we got reasonable results
    assert np.isfinite(result["YPBBN"])
    assert np.isfinite(result["DoH"])
    assert result["YPBBN"] > 0.24
    assert result["YPBBN"] < 0.25
    assert result["DoH"] > 2e-5
    assert result["DoH"] < 3e-5


@requires_c_backend
@pytest.mark.slow
@pytest.mark.backend
def test_c_backend_plasma_recompute():
    """C backend can recompute electron-thermo cache when forced."""
    result = run_bbn({"network": "small", "recompute_electron_thermo": True}, 
                    force_backend="c")
    
    # Verify we got reasonable results
    assert np.isfinite(result["YPBBN"])
    assert np.isfinite(result["DoH"])
    assert result["YPBBN"] > 0.24
    assert result["YPBBN"] < 0.25
    assert result["DoH"] > 2e-5
    assert result["DoH"] < 3e-5
