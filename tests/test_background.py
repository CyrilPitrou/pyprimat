"""
Direct unit tests for ``primat.background.StandardBackground``.

Physical background
--------------------
``StandardBackground`` builds the ``a <-> t <-> T`` relations and the
n<->p weak rates that ``NuclearNetwork`` (Class 2) consumes. It is normally
only exercised indirectly, through a full ``PRIMAT(...).solve()`` (see
``tests/test_api.py``'s ``T_of_t``/``t_of_T`` inverse checks on
``solved_small``, and ``tests/test_decoupling_qed.py``'s Neff pins for the
four ``incomplete_decoupling`` x ``QED_corrections`` combinations). This
module instantiates ``StandardBackground`` on its own (``cfg`` + ``Plasma``,
no nuclear network / no full solve) to check two claims documented in
CLAUDE.md that no existing test pins directly:

* ``external_scale_factor`` (read ``a(T)`` straight from the NEVO table's
  ``x`` column) agrees with the default entropy-conservation ODE integration
  to ~1e-5 in both ``a(T)`` and ``t(T)``.
* Outside the NEVO table's tabulated temperature range, both modes
  extrapolate assuming radiation domination: ``a ~ 1/T``, ``t ~ 1/T**2``.

Building a bare ``StandardBackground`` (rather than a full ``PRIMAT``) skips
the nuclear-network integration entirely, making these checks a few hundred
ms each instead of a full BBN solve.
"""
import numpy as np
import pytest

from primat.config import PRIMATConfig
from primat.plasma import Plasma
from primat.background import StandardBackground

pytestmark = pytest.mark.slow


def _build_background(**overrides):
    """Instantiate a ``StandardBackground`` directly (no nuclear network).

    Args:
        **overrides: PRIMATConfig keys to override; ``incomplete_decoupling``
            defaults to True (required by ``external_scale_factor``).

    Returns:
        A ready-to-query ``StandardBackground`` instance.
    """
    params = {"network": "small", "incomplete_decoupling": True}
    params.update(overrides)
    cfg = PRIMATConfig(params)
    plasma = Plasma(cfg)
    return StandardBackground(cfg, plasma)


# ---------------------------------------------------------------------------
# external_scale_factor: table lookup vs entropy-conservation ODE
# ---------------------------------------------------------------------------

# Photon temperatures [MeV] spanning the bulk of the NEVO table's range
# (~0.013-50.5 MeV, see primat/data/NEVO/NEVOPRIMAT_col_1_7.csv), well away
# from either edge so both modes are interpolating, not extrapolating.
_T_MID_TABLE_MEV = [5.0, 1.0, 0.5, 0.1, 0.05]


def test_external_scale_factor_agrees_with_ode_integration():
    """a(T) and t(T) from external_scale_factor=True/False must agree to
    ~1e-5, per CLAUDE.md's "Advanced: custom NEVO tables" claim."""
    bg_ode = _build_background(external_scale_factor=False)
    bg_ext = _build_background(external_scale_factor=True)

    for T in _T_MID_TABLE_MEV:
        a_ode, a_ext = bg_ode.a_of_T(T), bg_ext.a_of_T(T)
        t_ode, t_ext = bg_ode.t_of_T(T), bg_ext.t_of_T(T)
        assert a_ext == pytest.approx(a_ode, rel=2e-5), f"a(T={T}) mismatch"
        assert t_ext == pytest.approx(t_ode, rel=2e-5), f"t(T={T}) mismatch"


@pytest.mark.parametrize("external_scale_factor", [False, True])
def test_extrapolation_beyond_nevo_table_is_radiation_dominated(
        external_scale_factor):
    """Below the NEVO table's minimum tabulated T (~0.013 MeV -- routinely
    reached near the end of a standard BBN run, T_end=0.001 MeV), both
    a(T) and t(T) must extrapolate as pure radiation domination:
    a*T = const, t*T**2 = const."""
    bg = _build_background(external_scale_factor=external_scale_factor)

    T_lo, T_hi = 2e-3, 4e-3   # MeV, both below the table's ~0.013 MeV floor
    a_lo, a_hi = bg.a_of_T(T_lo), bg.a_of_T(T_hi)
    t_lo, t_hi = bg.t_of_T(T_lo), bg.t_of_T(T_hi)

    assert a_lo * T_lo == pytest.approx(a_hi * T_hi, rel=1e-3)
    assert t_lo * T_lo ** 2 == pytest.approx(t_hi * T_hi ** 2, rel=1e-3)


# ---------------------------------------------------------------------------
# Neutrino-sector getters (rho_nu_total_final, Omeganuh2_*)
# ---------------------------------------------------------------------------

def test_neutrino_sector_getters_are_positive_finite():
    """rho_nu_total_final (returns (Tg_final, rho_nu_tot)) and the two
    Omeganuh2_* getters are only exercised today as intermediate steps
    inside PRIMAT.solve()'s Neff/Omega_nu computation; check them directly
    for finiteness and sign."""
    bg = _build_background()

    Tg_f, rho_nu = bg.rho_nu_total_final()
    assert np.isfinite(Tg_f) and Tg_f > 0
    assert np.isfinite(rho_nu) and rho_nu > 0

    om_rel = bg.Omeganuh2_relnu()
    om_nr = bg.Omeganuh2_nrnu()
    assert np.isfinite(om_rel) and om_rel > 0
    assert np.isfinite(om_nr) and om_nr >= 0
