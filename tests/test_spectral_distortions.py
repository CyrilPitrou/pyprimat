
import pytest
import numpy as np
from primat.main import PRIMAT

# Each test below runs two full PRIMAT().solve() calls with
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
    pr_base = PRIMAT(params_base)
    res_base = pr_base.solve()
    
    params_spec = params_base.copy()
    params_spec["spectral_distortions"] = True
    pr_spec = PRIMAT(params_spec)
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
    pr_base = PRIMAT(params_base)
    res_base = pr_base.solve()

    params_spec = params_base.copy()
    params_spec["spectral_distortions"] = True
    pr_spec = PRIMAT(params_spec)
    res_spec = pr_spec.solve()

    # Neff is determined by the background solve, which is the same
    # because rho_nu_SD is 0 for NEVO distortions.
    assert res_spec['Neff'] == pytest.approx(res_base['Neff'], rel=1e-8)


def test_ytype_distortion_shifts_Neff():
    """Analytic y-type spectral distortion (y_SZ != 0) must shift Neff via the
    extra neutrino energy density rho_nu_SD, by exactly the amount
    ``background.Hubble`` adds to the expansion rate.

    Unlike the NEVO case above (where rho_nu_SD is 0 by construction), the
    analytic-distortion path (incomplete_decoupling=False,
    analytic_distortions=True) has a non-zero rho_nu_SD(T_nu)
    (neutrino_history.AnalyticDistortion._rho_nu_SD).  ``background.Hubble``
    and ``N_eff`` add the same term, so

        Neff_spec - Neff_base = rho_nu_SD(Tnu_last) / rho_g(Tg_last)
                                 / ((7/8)(4/11)^(4/3))

    pinned to ~1e-8 (CLAUDE.md). y_SZ=0.01 with mu(none)/gray off isolates the
    y-type term so the absolute-normalisation check below pins it exactly.
    """
    params_base = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": False,
        "spectral_distortions": False,
        "verbose": False,
    }
    res_base = PRIMAT(params_base).solve()

    y_sz = 0.01
    params_spec = dict(params_base, spectral_distortions=True,
                        analytic_distortions=True, y_SZ=y_sz)
    pr_spec = PRIMAT(params_spec)
    res_spec = pr_spec.solve()

    diff = res_spec['Neff'] - res_base['Neff']
    assert diff > 0   # Inty3SZ(0) = 7 pi^4/15 > 0, so rho_nu_SD > 0

    Tg_last  = pr_spec.background.Tg_vec[-1]
    Tnu_last = pr_spec.background.Tnu_vec[-1]
    rho_g    = pr_spec.plasma.rho_g(Tg_last)
    rho_SD   = pr_spec.background.rho_nu_SD(Tnu_last)
    diff_expected = rho_SD / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

    assert diff == pytest.approx(diff_expected, rel=1e-8)

    # Absolute normalisation of rho_nu_SD (locks the N_nu=3 single-particle
    # prefactor of PRIMAT-Main-gray.m line 832 against the historical
    # factor-2/3 bug that effectively used N_nu=2). Computed here from raw
    # numbers, independent of neutrino_history's helper:
    #   rho_nu_SD = N_nu * (Tnu^4 / 2pi^2) * y_SZ * Inty3SZ(0),  Inty3SZ(0)=7pi^4/15
    rho_SD_expected = 3.0 * (Tnu_last ** 4 / (2. * np.pi ** 2)) * y_sz * (7. * np.pi ** 4 / 15.)
    assert rho_SD == pytest.approx(rho_SD_expected, rel=1e-9)


def test_gray_distortion_shifts_Neff():
    """The gray-type analytic distortion (``cfg.y_gray``, from
    generate_rates/PRIMAT-Main-gray.m) must shift Neff the same way the
    y-type distortion does (test_ytype_distortion_shifts_Neff above), via
    ``background.rho_nu_SD``'s exact
    ``Inty3Gray(y_gray) = y_gray * 7*pi^4/120`` contribution (see
    neutrino_history.AnalyticDistortion._build_analytic_distortion's
    ``_rho_nu_SD``).  y_gray=0.05 with SZ off isolates the gray term so the
    diff_expected and absolute-normalisation checks below pin it exactly.
    """
    params_base = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": False,
        "spectral_distortions": False,
        "verbose": False,
    }
    res_base = PRIMAT(params_base).solve()

    y_gray = 0.05
    params_spec = dict(params_base, spectral_distortions=True,
                        analytic_distortions=True, y_gray=y_gray)
    pr_spec = PRIMAT(params_spec)
    res_spec = pr_spec.solve()

    diff = res_spec['Neff'] - res_base['Neff']
    assert diff > 0

    Tg_last  = pr_spec.background.Tg_vec[-1]
    Tnu_last = pr_spec.background.Tnu_vec[-1]
    rho_g    = pr_spec.plasma.rho_g(Tg_last)
    rho_SD   = pr_spec.background.rho_nu_SD(Tnu_last)
    diff_expected = rho_SD / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

    assert diff == pytest.approx(diff_expected, rel=1e-8)

    # Absolute normalisation (N_nu=3 single-particle prefactor):
    #   rho_nu_SD = N_nu * (Tnu^4/2pi^2) * 2*y_gray*Inty3_FD,  Inty3_FD = 7pi^4/120
    rho_SD_expected = 3.0 * (Tnu_last ** 4 / (2. * np.pi ** 2)) * 2. * y_gray * (7. * np.pi ** 4 / 120.)
    assert rho_SD == pytest.approx(rho_SD_expected, rel=1e-9)


def test_genuine_chemical_potential_shifts_rates_and_energy():
    """A genuine neutrino chemical potential (``munuOverTnu``) must do TWO
    things, neither of them a spectral distortion:

    1. shift the n<->p weak rates (through the FD_nu3 integrand), and hence
       YP/D/H -- a nu_e degeneracy changes the equilibrium n/p ratio. For a
       small positive xi this lowers YP (more protons), so the effect on YP is
       sizeable and of order -xi (here ~-0.24*xi).
    2. raise the neutrino energy density, and hence Neff, by EXACTLY the
       per-flavour excess T^4 (xi^2/4 + xi^4/(8 pi^2)) summed over the three
       flavours (plasma.rho_nu_chempot_excess). This pins the absolute Neff
       shift to ~1e-7, and is O(xi^2) (positive for either sign of xi).
    """
    import numpy as np
    from primat.plasma import rho_nu_chempot_excess

    xi = 0.05
    common = {
        "network": "small",
        "numerical_precision": 1e-8,
        "incomplete_decoupling": False,
        # Finite-temperature radiative corrections assume an equilibrium FD
        # plasma and are inconsistent with a chemical potential -> off here.
        "thermal_corrections": False,
        "spectral_distortions": False,
        "verbose": False,
    }
    res_base = PRIMAT(dict(common)).solve()
    pr_gen = PRIMAT(dict(common, munuOverTnu=xi))
    res_gen = pr_gen.solve()

    # (1) Weak-rate effect on YP: large, of order -xi (n/p equilibrium shift).
    yp_eff = (res_gen['YPBBN'] - res_base['YPBBN']) / res_base['YPBBN']
    assert yp_eff < 0.0                      # positive xi -> fewer neutrons
    # dYP/dxi ~ -0.24, so the relative effect (dYP/YP)/xi ~ 0.24/0.247 ~ 0.97.
    assert 0.5 < abs(yp_eff) / xi < 1.5      # sizeable, ~ -xi; comfortably bracketed

    # (2) Energy-density effect on Neff: exact, positive, O(xi^2).
    dNeff = res_gen['Neff'] - res_base['Neff']
    assert dNeff > 0.0
    Tg_last  = pr_gen.background.Tg_vec[-1]
    Tnu_last = pr_gen.background.Tnu_vec[-1]
    rho_g    = pr_gen.plasma.rho_g(Tg_last)
    # 3 flavours each carrying the chemical-potential energy excess.
    rho_excess = 3.0 * rho_nu_chempot_excess(Tnu_last, xi)
    dNeff_expected = rho_excess / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))
    assert dNeff == pytest.approx(dNeff_expected, rel=1e-6)


def test_chemical_potential_neff_is_even_in_sign():
    """The chemical-potential energy density is even in xi, so +xi and -xi give
    the SAME Neff shift (the antineutrino carries the opposite sign and ρ is
    even). The YP shift, by contrast, flips sign with xi (n/p equilibrium)."""
    common = {
        "network": "small", "numerical_precision": 1e-8,
        "incomplete_decoupling": False, "thermal_corrections": False,
        "spectral_distortions": False, "verbose": False,
    }
    res_p = PRIMAT(dict(common, munuOverTnu=0.03)).solve()
    res_m = PRIMAT(dict(common, munuOverTnu=-0.03)).solve()
    res_0 = PRIMAT(dict(common)).solve()
    # Neff shift identical for +/- xi (even in xi):
    assert (res_p['Neff'] - res_0['Neff']) == pytest.approx(
        res_m['Neff'] - res_0['Neff'], rel=1e-6)
    # YP shift opposite in sign (odd in xi at leading order):
    assert (res_p['YPBBN'] - res_0['YPBBN']) * (res_m['YPBBN'] - res_0['YPBBN']) < 0.0


def test_finite_mass_corrections_change_SD_FM_contribution():
    """Regression/sanity check that SD-FM (the finite-mass correction to
    the spectral-distortion n<->p rate channel, only active in
    analytic-distortion mode -- see weak_rates._L_SD_FMCCR/_L_SD_FMNoCCR) is
    actually wired into a full BBN solve: toggling
    ``finite_mass_corrections`` while analytic distortions are on must move
    D/H by a small but non-zero amount, mirroring how the existing
    plain-FD finite-mass correction (FMCCR) is known to be a small
    correction on top of CCR.
    """
    params_fm_on = {
        "network": "small",
        "numerical_precision": 1e-6,
        "incomplete_decoupling": False,
        "spectral_distortions": True,
        "analytic_distortions": True,
        "y_SZ": 0.05,
        "y_gray": 0.03,
        "finite_mass_corrections": True,
        "verbose": False,
    }
    res_fm_on = PRIMAT(params_fm_on).solve()

    params_fm_off = dict(params_fm_on, finite_mass_corrections=False)
    res_fm_off = PRIMAT(params_fm_off).solve()

    diff = abs(res_fm_on['DoH'] - res_fm_off['DoH']) / res_fm_off['DoH']
    # Toggling finite_mass_corrections here also flips the plain-FD FMCCR
    # term (always on alongside SD-FM, both gated by the same flag); the
    # combined effect on D/H is ~2.7e-3 with this distortion amplitude. A
    # generous bound catches both "not wired up at all" (diff == 0) and
    # "wildly wrong magnitude" regressions.
    assert 0. < diff < 1e-2
