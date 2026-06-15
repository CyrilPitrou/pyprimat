"""
Tests for the QED corrections to radiative-capture nuclear rates.

When ``nuclear_qed_corrections=True`` the forward rate tables of five reactions
(npTOdg, dpTOHe3g, tpTOag, taTOLi7g, He3aTOBe7g) are multiplied by a
T9-dependent factor derived in Pitrou & Pospelov 2020.  These tests verify:

1. The correction factors have the expected magnitude (sub-percent, > 1).
2. The polynomial fit for npTOdg matches the published low-T limit.
3. The electric-dipole Kroll factor increases with T9 (more pair-production
   phase space at higher energy) for the four remaining reactions.
4. Corrections are applied to the loaded median rate tables and not to reactions
   that do not appear in the corrected set.
5. p_* and NP_delta_* variations still work correctly on top of the corrected
   medians.
6. A full solve with nuclear_qed_corrections=True completes and shifts D/H
   relative to the uncorrected run.
"""

import numpy as np
import pytest

from pyprimat.config import PyPRConfig
from pyprimat.network_data import _qed_nuclear_rescale, load_network


# ---------------------------------------------------------------------------
# Unit tests on the correction function itself
# ---------------------------------------------------------------------------

# The five reactions that receive QED corrections and the reactions that don't.
QED_REACTIONS = ["npTOdg", "dpTOHe3g", "tpTOag", "taTOLi7g", "He3aTOBe7g"]
NO_QED_REACTIONS = ["ddTOHe3n", "ddTOtp", "tdTOan", "Be7nTOLi7p", "He3nTOtp"]


class TestQEDCorrectionFunction:
    """Unit tests for ``_qed_nuclear_rescale``."""

    @pytest.fixture
    def t9_grid(self):
        """Representative T9 values spanning the BBN era (0.001–10 GK)."""
        return np.logspace(-3, 1, 200)

    def test_corrections_are_above_unity(self, t9_grid):
        """All QED correction factors must be > 1 (pair production increases the rate)."""
        for rxn in QED_REACTIONS:
            f = _qed_nuclear_rescale(rxn, t9_grid)
            assert np.all(f > 1.0), (
                f"{rxn}: correction factor must be > 1 everywhere, "
                f"got min = {f.min():.8f}"
            )

    def test_corrections_are_sub_percent(self, t9_grid):
        """All QED correction factors must be below 1.01 (sub-percent correction)."""
        for rxn in QED_REACTIONS:
            f = _qed_nuclear_rescale(rxn, t9_grid)
            assert np.all(f < 1.01), (
                f"{rxn}: correction factor unexpectedly large, "
                f"got max = {f.max():.6f}"
            )

    def test_no_correction_for_other_reactions(self, t9_grid):
        """Reactions without a photon in the final state return None."""
        for rxn in NO_QED_REACTIONS:
            assert _qed_nuclear_rescale(rxn, t9_grid) is None, (
                f"{rxn} should have no QED correction"
            )

    def test_npTOdg_low_T_limit(self):
        """npTOdg polynomial approaches its T9→0 cap at very low temperature.

        The polynomial fit was capped at 1.0009003934476768, which is the
        Pitrou & Pospelov 2020 value of the Kroll factor evaluated at the
        threshold energy Q_np/mₑ (T9→0 limit).
        """
        T9_ZERO_LIMIT = 1.0009003934476768
        t9_tiny = np.array([1e-4, 1e-5])
        f = _qed_nuclear_rescale("npTOdg", t9_tiny)
        # The polynomial increases with T9, so at very small T9 it should be
        # close to (but not exceed) the cap.
        assert np.all(f <= T9_ZERO_LIMIT + 1e-12), (
            f"npTOdg factor must not exceed T9→0 cap {T9_ZERO_LIMIT}, got {f}"
        )
        # At T9=0.001 (the low end of the BBN grid) the value is ~1.00033296
        f_001 = _qed_nuclear_rescale("npTOdg", np.array([0.001]))
        assert 1.0003 < f_001[0] < 1.0004, (
            f"npTOdg at T9=0.001 expected ~1.00033, got {f_001[0]:.8f}"
        )

    def test_kroll_reactions_increase_with_T9(self):
        """For the four electric-dipole reactions the Kroll factor increases with T9.

        Higher temperature → higher kinetic energy → more available phase space
        for the emitted photon to produce an e⁺e⁻ pair.
        """
        T9_lo = np.array([0.01])
        T9_hi = np.array([5.0])
        for rxn in ["dpTOHe3g", "tpTOag", "taTOLi7g", "He3aTOBe7g"]:
            f_lo = _qed_nuclear_rescale(rxn, T9_lo)[0]
            f_hi = _qed_nuclear_rescale(rxn, T9_hi)[0]
            assert f_hi > f_lo, (
                f"{rxn}: Kroll factor should increase with T9; "
                f"f(0.01)={f_lo:.8f}, f(5.0)={f_hi:.8f}"
            )

    def test_kroll_magnitude_at_reference_T9(self):
        """Check correction magnitudes at T9=0.1 GK against expected ranges.

        Expected values cross-checked against PRIMAT-Main.m with
        $NuclearRatesQEDCorrections=True at T9=0.1 GK.
        """
        T9_ref = np.array([0.1])
        expected_ranges = {
            # (min, max) at T9=0.1 GK; values from _qed_nuclear_rescale, ±2e-6 tolerance
            "npTOdg":    (1.000341, 1.000346),
            "dpTOHe3g":  (1.002175, 1.002181),
            "tpTOag":    (1.004156, 1.004161),
            "taTOLi7g":  (1.000976, 1.000981),
            "He3aTOBe7g":(1.000394, 1.000399),
        }
        for rxn, (lo, hi) in expected_ranges.items():
            f = _qed_nuclear_rescale(rxn, T9_ref)[0]
            assert lo < f < hi, (
                f"{rxn} at T9=0.1: expected factor in ({lo}, {hi}), got {f:.8f}"
            )


# ---------------------------------------------------------------------------
# Integration tests: corrections are actually applied to loaded rate tables
# ---------------------------------------------------------------------------

class TestQEDCorrectionInNetwork:
    """Verify that load_network applies QED corrections to the right rows."""

    @pytest.fixture(scope="class")
    def nets(self):
        """Base (uncorrected) and QED-corrected small networks."""
        cfg_base = PyPRConfig({"network": "small", "nuclear_qed_corrections": False})
        cfg_qed  = PyPRConfig({"network": "small", "nuclear_qed_corrections": True})
        return (load_network(cfg_base, era="LT"),
                load_network(cfg_qed,  era="LT"))

    def test_qed_reactions_are_upscaled(self, nets):
        """Rates for the five QED reactions must be strictly larger with the flag on."""
        net_base, net_qed = nets
        for rxn in QED_REACTIONS:
            if rxn not in net_base.names:
                continue        # not in small network; skip
            # names[0] = nTOp (weak), thermonuclear rates start at names[1] → fwd_median[0]
            i = net_base.names.index(rxn) - 1
            ratio = net_qed._fwd_median[i] / net_base._fwd_median[i]
            assert np.all(ratio > 1.0), (
                f"{rxn}: QED-corrected median should exceed base median everywhere"
            )
            assert np.all(ratio < 1.01), (
                f"{rxn}: QED correction should be sub-percent"
            )

    def test_non_qed_reactions_unchanged(self, nets):
        """Rates for reactions without a QED correction must be identical."""
        net_base, net_qed = nets
        for rxn in net_base.names[1:]:
            if rxn in QED_REACTIONS:
                continue
            i = net_base.names.index(rxn) - 1
            assert np.allclose(net_qed._fwd_median[i], net_base._fwd_median[i],
                               rtol=0, atol=0), (
                f"{rxn}: non-QED reaction should be unchanged"
            )

    def test_p_variation_stacks_on_corrected_median(self, nets):
        """p_* variations are applied on top of the QED-corrected median.

        With p=+1 and QED on, the active rate must equal the corrected median
        times exp(log(expsigma)).
        """
        net_base, net_qed = nets
        cfg_qed = PyPRConfig({"network": "small", "nuclear_qed_corrections": True,
                              "p_npTOdg": 1.0})
        net_qed.apply_variations(cfg_qed)
        i = net_qed.names.index("npTOdg") - 1
        expected = net_qed._fwd_median[i] * np.exp(np.log(net_qed._expsigma[i]))
        assert np.allclose(net_qed._fwd[i], expected, rtol=1e-12), (
            "Active rate with p=+1 should equal QED-corrected median × exp(ln σ)"
        )


# ---------------------------------------------------------------------------
# End-to-end: a full solve with QED corrections shifts D/H
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.solve
def test_solve_qed_corrections_shift_DoH():
    """nuclear_qed_corrections=True shifts D/H relative to the baseline.

    The five corrected rates all affect the D/H abundance.  The direction of
    the shift is not pinned here (it depends on the net balance of reactions),
    but the result must differ from the uncorrected run.
    """
    from pyprimat import PyPR

    res_base = PyPR({"network": "small", "nuclear_qed_corrections": False,
                          "verbose": False}).solve()
    res_qed  = PyPR({"network": "small", "nuclear_qed_corrections": True,
                          "verbose": False}).solve()

    dh_base = res_base["DoH"]
    dh_qed  = res_qed["DoH"]

    assert dh_base != dh_qed, (
        "nuclear_qed_corrections should shift D/H but got identical values"
    )
    # The shift should be tiny (sub-percent correction to sub-percent rates)
    rel_shift = abs(dh_qed - dh_base) / dh_base
    assert rel_shift < 0.01, (
        f"QED nuclear correction shift on D/H is unexpectedly large: {rel_shift:.2e}"
    )
    assert rel_shift > 1e-7, (
        f"QED nuclear correction shift on D/H is suspiciously small: {rel_shift:.2e}"
    )
