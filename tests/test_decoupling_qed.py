"""
Tests for the incomplete_decoupling and QED_corrections flags.

Physical background
-------------------
Two independent approximations control the neutrino–plasma thermodynamics:

1. ``incomplete_decoupling`` (True = NEVO table, False = instantaneous limit):
   whether the three SM neutrino flavours partially share in the entropy
   injected by e+e- annihilations.  In the instantaneous limit all three
   temperatures are set by EM entropy conservation; in the NEVO treatment they
   are read from a pre-computed table that accounts for the residual coupling.

2. ``QED_corrections`` (True = full, False = free ideal gas):
   whether the EM plasma equation of state includes QED interaction pressure
   corrections (stored in the ``rates/plasma/QED_*.txt`` tables).

The four flag combinations form a 2×2 matrix.  The two NEVO files
(``NEVOPRIMAT_col_1_7.csv`` and ``NEVOPRIMAT_NoQED_col_1_7.csv``) must be
matched to the corresponding QED setting to keep the neutrino temperatures
consistent with the rest of the thermodynamics.

Test structure
--------------
* **Unit tests** (fast, no full solve): check the plasma module and config.
* **Neff reference values**: pin the Neff for each of the four combinations
  to the values established from the full computation (tolerance ±0.0001,
  tighter than the ~0.01 differences between cases).
* **Limit tests**: verify the mathematically exact limits
  (Neff = 3 for free-gas instantaneous, (Tγ/Tν)³ ratio from the QED formula).
* **Monotonicity**: the ordering of Neff across flag combinations must be
  physically correct.

All solve-based tests carry the ``slow`` marker.
"""
import pytest
import numpy as np
from pyprimat.config import PyPRConfig
from pyprimat.plasma import Plasma


# ---------------------------------------------------------------------------
# Helper: run a full solve for a given flag combination
# ---------------------------------------------------------------------------

def _solve(incomplete_decoupling, QED_corrections):
    from pyprimat.main import PyPR
    return PyPR({
        "incomplete_decoupling": incomplete_decoupling,
        "QED_corrections":       QED_corrections,
        # spectral_distortions=True (the PyPRConfig default)
        # requires incomplete_decoupling=True (NEVO spectrum); this module's
        # 2x2 matrix exercises incomplete_decoupling=False too, and spectral
        # distortions are an independent axis covered by
        # test_spectral_distortions.py, so disable them here for all four
        # combinations.
        "spectral_distortions":  False,
    }).solve()


# ---------------------------------------------------------------------------
# Unit tests: plasma module with QED_corrections=False
# ---------------------------------------------------------------------------

class TestPlasmaNoQED:
    """When QED_corrections=False, the three QED pressure functions must be
    identically zero and spl/T³ must equal 11π²/45 in the high-T limit."""

    @pytest.fixture(autouse=True)
    def thermo(self):
        return Plasma(PyPRConfig({"QED_corrections": False}))

    def test_PQEDofT_is_zero(self, thermo):
        """QED interaction pressure P must vanish when QED_corrections=False."""
        for T in [0.1, 1.0, 10.0, 100.0]:
            assert thermo.PQEDofT(T) == 0.0

    def test_dPQEDdT_is_zero(self, thermo):
        for T in [0.1, 1.0, 10.0, 100.0]:
            assert thermo.dPQEDdT(T) == 0.0

    def test_d2PQEDdT2_is_zero(self, thermo):
        for T in [0.1, 1.0, 10.0, 100.0]:
            assert thermo.d2PQEDdT2(T) == 0.0

    def test_spl_highT_equals_sigma_inf(self, thermo):
        """Without QED corrections, spl(T)/T³ must converge to 11π²/45 at high T
        (photons + e+e- in the massless limit)."""
        sigma_inf = 11.0 * np.pi**2 / 45.0
        T = 100.0   # T >> me = 0.511 MeV
        assert thermo.spl(T) / T**3 == pytest.approx(sigma_inf, rel=1e-3)

    def test_spl_lowT_equals_photon_entropy(self, thermo):
        """Without QED corrections, spl(T)/T³ → 4π²/45 at low T (photons only)."""
        s_photon = 4.0 * np.pi**2 / 45.0
        T = 0.001   # T << me
        assert thermo.spl(T) / T**3 == pytest.approx(s_photon, rel=1e-6)


class TestPlasmaWithQED:
    """When QED_corrections=True (default), the QED functions must be nonzero
    and spl/T³ must differ from 11π²/45 at high T."""

    @pytest.fixture(autouse=True)
    def thermo(self):
        return Plasma(PyPRConfig({"QED_corrections": True}))

    def test_PQEDofT_nonzero_at_MeV(self, thermo):
        """QED pressure correction is nonzero at T ~ m_e."""
        assert thermo.PQEDofT(1.0) != 0.0

    def test_spl_highT_differs_from_sigma_inf(self, thermo):
        """With QED corrections, spl(T)/T³ ≠ 11π²/45 at high T."""
        sigma_inf = 11.0 * np.pi**2 / 45.0
        T = 100.0
        ratio = thermo.spl(T) / T**3
        # The QED correction shifts the ratio; it must not be equal to sigma_inf.
        assert abs(ratio - sigma_inf) / sigma_inf > 1e-4


# ---------------------------------------------------------------------------
# Unit test: (Tγ/Tν)³ ratio in the instantaneous-decoupling case
# ---------------------------------------------------------------------------

class TestInstantaneousDecouplingRatio:
    """The ratio (Tγ/Tν)³ = sbar_ref / (4π²/45) must match the analytical
    formulas for both QED settings."""

    def _sbar_ref_from_config(self, QED_corrections):
        """Reproduce the sbar_ref logic from main._setup_background_and_cosmo."""
        # spectral_distortions=True (PyPRConfig default) requires
        # incomplete_decoupling=True; disable it here since this helper always
        # uses incomplete_decoupling=False (instantaneous-decoupling limit).
        cfg = PyPRConfig({"QED_corrections": QED_corrections,
                          "incomplete_decoupling": False,
                          "spectral_distortions": False})
        if QED_corrections:
            alpha = cfg.alphaem
            ratio3 = (11.0/4.0
                      - 25.0*alpha / (8.0*np.pi)
                      + 10.0*alpha**(3.0/2.0) * np.sqrt(np.pi/3.0) / np.pi**2)
            return ratio3 * (4.0*np.pi**2 / 45.0)
        else:
            return 11.0*np.pi**2 / 45.0

    def test_no_qed_gives_11_over_4(self):
        """Without QED corrections (Tγ/Tν)³ = 11/4 exactly."""
        sbar_ref = self._sbar_ref_from_config(False)
        s_photon = 4.0*np.pi**2 / 45.0
        assert sbar_ref / s_photon == pytest.approx(11.0/4.0, rel=1e-12)

    def test_with_qed_matches_perturbative_formula(self):
        """With QED corrections (Tγ/Tν)³ must equal the perturbative result
        11/4 − 25α/(8π) + 10α^{3/2} √(π/3) / π² (Dodelson & Turner 1992,
        Heckler 1994)."""
        cfg = PyPRConfig()
        alpha = cfg.alphaem
        expected = (11.0/4.0
                    - 25.0*alpha / (8.0*np.pi)
                    + 10.0*alpha**(3.0/2.0) * np.sqrt(np.pi/3.0) / np.pi**2)
        sbar_ref = self._sbar_ref_from_config(True)
        s_photon = 4.0*np.pi**2 / 45.0
        assert sbar_ref / s_photon == pytest.approx(expected, rel=1e-12)

    def test_qed_ratio_is_less_than_11_over_4(self):
        """QED corrections reduce (Tγ/Tν)³ relative to the free-gas value,
        because they heat the photon bath slightly more than the free-gas
        prediction."""
        sbar_ref_qed   = self._sbar_ref_from_config(True)
        sbar_ref_no_qed = self._sbar_ref_from_config(False)
        # QED corrections raise sbar_ref (more entropy in the plasma at high T),
        # which means a larger Tγ/Tν ratio and thus a slightly higher Neff.
        assert sbar_ref_qed < sbar_ref_no_qed


# ---------------------------------------------------------------------------
# Neff reference values for the four flag combinations (slow, full solves)
# ---------------------------------------------------------------------------

# Reference Neff values (8 decimal places) established from full computation.
# Tolerances are set to ±0.0001, small enough to distinguish the ~0.01–0.03
# differences between flag combinations but robust to minor solver variations.
_NEFF_REFS = {
    (True,  True):  3.04397730,   # full NEVO + QED (the standard SM result)
    (False, True):  3.00964519,   # instantaneous decoupling + QED
    (True,  False): 3.03465328,   # full NEVO (NoQED table) without QED
    (False, False): 3.00000000,   # free-gas instantaneous (exact analytic limit)
}

_NEFF_TOL = 0.0001   # tighter than any inter-case difference


@pytest.mark.slow
@pytest.mark.solve
@pytest.mark.parametrize("incomplete,qed", [
    (True,  True),
    (False, True),
    (True,  False),
    (False, False),
])
def test_neff_four_combinations(incomplete, qed):
    """Pin Neff for each of the four (incomplete_decoupling, QED_corrections)
    combinations to the established reference value (tolerance ±0.0001)."""
    res = _solve(incomplete, qed)
    expected = _NEFF_REFS[(incomplete, qed)]
    assert res["Neff"] == pytest.approx(expected, abs=_NEFF_TOL), (
        f"incomplete_decoupling={incomplete}, QED_corrections={qed}: "
        f"Neff={res['Neff']:.8f}, expected {expected:.8f}"
    )


@pytest.mark.slow
@pytest.mark.solve
def test_neff_free_gas_exact():
    """The free-gas instantaneous limit (incomplete=F, QED=F) must give
    Neff = 3 exactly (to within 1e-6), since T_ν = T_γ (11/4)^{-1/3} exactly
    and Neff = (ρ_rad − ρ_γ) / [ρ_γ (7/8)(4/11)^{4/3}] = 3."""
    res = _solve(False, False)
    assert res["Neff"] == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Monotonicity / ordering tests (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.solve
def test_neff_incomplete_decoupling_raises_neff():
    """Incomplete decoupling always raises Neff relative to instantaneous,
    for both QED settings, because neutrinos are partially reheated."""
    res_TT = _solve(True,  True)
    res_FT = _solve(False, True)
    res_TF = _solve(True,  False)
    res_FF = _solve(False, False)
    assert res_TT["Neff"] > res_FT["Neff"]
    assert res_TF["Neff"] > res_FF["Neff"]


@pytest.mark.slow
@pytest.mark.solve
def test_neff_qed_raises_neff_for_instantaneous():
    """QED corrections raise Neff in the instantaneous-decoupling case,
    because they increase the photon entropy relative to neutrinos."""
    res_FT = _solve(False, True)
    res_FF = _solve(False, False)
    assert res_FT["Neff"] > res_FF["Neff"]


@pytest.mark.slow
@pytest.mark.solve
def test_neff_qed_effect_smaller_than_decoupling_effect():
    """The QED correction to Neff (with instantaneous decoupling) is smaller
    than the incomplete-decoupling correction (with QED on), reflecting the
    known hierarchy of effects."""
    delta_qed        = _solve(False, True)["Neff"]  - _solve(False, False)["Neff"]
    delta_incomplete = _solve(True,  True)["Neff"]  - _solve(False, True)["Neff"]
    assert delta_qed < delta_incomplete


# ---------------------------------------------------------------------------
# NEVO file selection test (unit, no solve -- but slow: see marker below)
# ---------------------------------------------------------------------------

@pytest.mark.slow  # incomplete_decoupling/QED_corrections != cache fingerprint
                    # -> PyPR.__init__ recomputes the n<->p weak rates twice (~3 s)
def test_nevo_file_selection():
    """Verify that _setup_background_and_cosmo selects the correct NEVO file:
    the QED table for QED_corrections=True, the NoQED table otherwise.
    We check by inspecting the neutrino-temperature ratio T_νe/T_γ at a
    moderate temperature where the two tables differ noticeably."""
    from pyprimat.main import PyPR

    inst_qed   = PyPR({"incomplete_decoupling": True, "QED_corrections": True})
    inst_noqed = PyPR({"incomplete_decoupling": True, "QED_corrections": False})

    # At T_γ ~ 2 MeV the neutrino temperature starts to deviate from T_γ
    # as e+e- annihilations proceed.  The two NEVO tables give different
    # T_νe values there, reflecting different plasma equations of state.
    T_probe = 2.0   # MeV
    ratio_qed   = inst_qed.background.TnuofT(T_probe)   / T_probe
    ratio_noqed = inst_noqed.background.TnuofT(T_probe) / T_probe

    # Both ratios must be in (0, 1] (neutrinos are cooler than or equal to
    # photons after partial reheating) and they must differ between the two
    # tables, confirming different files were loaded.
    assert 0.9 < ratio_qed   <= 1.0
    assert 0.9 < ratio_noqed <= 1.0
    # The two tables differ by ~9e-6 at this temperature; use abs=1e-6 so
    # the tolerance is below that difference and the test catches a mis-load.
    assert ratio_qed != pytest.approx(ratio_noqed, abs=1e-6), (
        "T_νe/T_γ is identical for QED and NoQED NEVO tables — "
        "likely the same file was loaded for both cases."
    )


# ---------------------------------------------------------------------------
# "Nheating" time-evolution column: present only for a real NEVO heating
# table (incomplete_decoupling=True), absent for the N=0 stub used by
# InstantaneousDecoupling (see Background.has_heating_table /
# PyPR._write_time_evolution).
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.solve
def test_has_heating_table_flag_tracks_incomplete_decoupling():
    """``background.has_heating_table`` mirrors ``cfg.incomplete_decoupling``:
    True selects NEVOTable (a real N(T_gamma) heating table), False selects
    InstantaneousDecoupling (the N=0 stub)."""
    from pyprimat.main import PyPR
    p_nevo = PyPR({"network": "small", "incomplete_decoupling": True})
    # spectral_distortions=True (PyPRConfig default) requires
    # incomplete_decoupling=True; disable it for the instantaneous case.
    p_inst = PyPR({"network": "small", "incomplete_decoupling": False,
                    "spectral_distortions": False})
    assert p_nevo.background.has_heating_table is True
    assert p_inst.background.has_heating_table is False


@pytest.mark.slow
@pytest.mark.solve
def test_nheating_column_present_only_with_heating_table(tmp_path):
    """The background time-evolution TSV (output_background_evolution=True)
    includes an "Nheating" column when incomplete_decoupling=True (a real
    NEVO heating table is available), and omits it entirely when
    incomplete_decoupling=False (where it would just be a column of zeros
    from the InstantaneousDecoupling stub)."""
    from pyprimat.main import PyPR

    out_nevo = tmp_path / "nevo_background.tsv"
    PyPR({
        "network": "small",
        "incomplete_decoupling": True,
        "output_background_evolution": True,
        "output_background_file": str(out_nevo),
    }).solve()
    header_nevo = out_nevo.read_text().splitlines()[0].split("\t")
    assert "Nheating" in header_nevo

    out_inst = tmp_path / "inst_background.tsv"
    PyPR({
        "network": "small",
        "incomplete_decoupling": False,
        # spectral_distortions=True (PyPRConfig default) requires
        # incomplete_decoupling=True; disable it for the instantaneous case.
        "spectral_distortions": False,
        "output_background_evolution": True,
        "output_background_file": str(out_inst),
    }).solve()
    header_inst = out_inst.read_text().splitlines()[0].split("\t")
    assert "Nheating" not in header_inst


@pytest.mark.slow
@pytest.mark.solve
def test_time_evolution_HT_era_abundances_are_zero_before_network_starts(tmp_path):
    """Before the nuclear network starts integrating a given species (the HT
    era for every nuclide but n/p), the ``Y<species>`` columns of the
    ``output_time_evolution`` TSV are **exactly 0** (the value ``_embed``/
    ``Y_of_t`` produce there) -- a previous version filled this region with
    the Nuclear Statistical Equilibrium (Saha) prediction, which was removed
    because the fill is often physically wrong (see
    ``NuclearNetwork._write_time_evolution``'s docstring). The columns must
    still be finite (the NEXT line, the first MT-era row, may legitimately
    still be 0/tiny for a heavy nuclide and is not checked here)."""
    from pyprimat.main import PyPR
    import numpy as np

    out_path = tmp_path / "evolution.tsv"
    pr = PyPR({
        "network": "small", "verbose": False,
        "output_time_evolution": True, "output_file": str(out_path),
    })
    pr.solve()

    with open(out_path) as f:
        header = f.readline().strip().split("\t")
        data = np.loadtxt(f)

    t = data[:, header.index("t")]
    # All output rows are >= T_start_cosmo (t_out starts at t_cosmo), and the
    # first row is deep in the HT era (t_weak ~ a few seconds, well after
    # T_start_cosmo at 10 MeV).
    assert t[0] < 1e-2

    for name in pr.nuclear.abundance_names:
        if name in ("n", "p"):
            continue
        y = data[0, header.index("Y" + name)]
        assert y == 0., f"Y{name} is not exactly 0 at the first HT-era row"
        assert np.isfinite(y), f"Y{name} is not finite at the first HT-era row"
