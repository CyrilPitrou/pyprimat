"""Pin a few of primat.constants.CONST's *derived* values.

CONST.alphaem, CONST.GF and CONST.mZ are primary inputs (PDG values, set
verbatim); CONST.sW2 (sin^2(theta_W)) and the effective electron/muon
couplings derived from it are computed from those three via the on-shell
relation. A typo in that formula (a stray factor of 2, a swapped GF/mZ) would
not raise any error -- it would just silently shift every weak rate that uses
sW2 by a few percent. These tests pin the derived numbers against an
independent hand-computation of the same formula, so such a typo fails loudly
here instead of showing up as an unexplained drift in Neff/YP.
"""
import numpy as np
import pytest

from primat.constants import CONST


def test_sW2_matches_onshell_relation():
    """sin^2(theta_W) = 1/2 * (1 - sqrt(1 - 2*sqrt(2)*pi*alphaem/(GF*mZ^2)))."""
    expected = 0.5 * (1. - np.sqrt(1. - 2. * np.sqrt(2.) * np.pi * CONST.alphaem
                                    / (CONST.GF * CONST.mZ**2)))
    assert CONST.sW2 == pytest.approx(expected, rel=1e-12)
    # Sanity check against the well-known PDG ballpark value (~0.223 in the
    # MSbar scheme; the on-shell scheme used here is close but not identical).
    assert 0.20 < CONST.sW2 < 0.24


def test_effective_couplings_consistent_with_sW2():
    """geL/geR/gmuL are simple offsets of sW2 (electron/muon neutral-current couplings)."""
    assert CONST.geL == pytest.approx(0.5 + CONST.sW2, rel=1e-12)
    assert CONST.geR == pytest.approx(CONST.sW2, rel=1e-12)
    assert CONST.gmuL == pytest.approx(-0.5 + CONST.sW2, rel=1e-12)


def test_MeV_to_Kelvin_round_trips_T_weak_and_T_nucl():
    """T_weak/T_nucl are MeV_to_Kelvin scaled by their defining MeV values."""
    assert CONST.T_weak == pytest.approx(1.0 * CONST.MeV_to_Kelvin, rel=1e-12)
    assert CONST.T_nucl == pytest.approx(0.11 * CONST.MeV_to_Kelvin, rel=1e-12)
