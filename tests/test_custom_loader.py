"""
Tests for the custom network file loader.

The ``small_parthenope`` network (``rates/nuclear/networks/small_parthenope.txt``)
exercises the ``bare_name, filename.txt`` syntax that routes individual reactions
to non-default rate tables.  For example:

    d_d__He3_n, ddTOHe3n_parthenope.txt

causes the d + d → He3 + n rate to be read from the Parthenope (Gariazzo 2021)
table instead of the default AC2024 one.

These tests verify:
1. The network loads without error and has the expected species / reaction count.
2. Reactions with a custom filename actually use a different rate table
   (not just silently fall back to the default).
3. The loaded network passes the formal N/Z/Q conservation check, confirming
   the stoichiometry is valid even when rates come from different sources.
4. A full BBN solve with the parthenope rates completes and gives
   physically reasonable primordial abundances.
"""
import numpy as np
import pytest

from primat.network_data import load_network
from primat.config import PRIMATConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_both():
    """Load the standard small network and the parthenope custom network."""
    cfg_std = PRIMATConfig({"network": "small", "verbose": False})
    cfg_ph  = PRIMATConfig({"network": "small_parthenope", "verbose": False})
    net_std = load_network(cfg_std, era="LT")
    net_ph  = load_network(cfg_ph,  era="LT")
    return net_std, net_ph


# ---------------------------------------------------------------------------
# Structure checks (fast — no solve)
# ---------------------------------------------------------------------------

def test_small_parthenope_loads():
    """The parthenope network file parses without error."""
    cfg = PRIMATConfig({"network": "small_parthenope", "verbose": False})
    net = load_network(cfg, era="LT")
    assert len(net.names) > 0


def test_small_parthenope_same_reaction_set():
    """small_parthenope has the same 12 nuclear reactions as small."""
    net_std, net_ph = _load_both()
    # Both should have n__p + 12 nuclear reactions = 13 entries
    assert len(net_ph.names) == len(net_std.names), (
        f"Expected {len(net_std.names)} reactions, got {len(net_ph.names)}"
    )
    assert set(net_ph.names) == set(net_std.names), (
        "Reaction names differ between small and small_parthenope"
    )


def test_small_parthenope_same_species():
    """small_parthenope carries the same 8 nuclides as the standard small network."""
    net_std, net_ph = _load_both()
    assert net_ph.species == net_std.species


def test_parthenope_rates_differ_from_standard():
    """Reactions with custom filenames must use different rate tables.

    ``d_d__He3_n`` is routed to ``ddTOHe3n_parthenope.txt`` (Gariazzo 2021)
    instead of ``d_d__He3_n.txt`` (Gom17).  The two rate files have different
    values, so the loaded forward-rate arrays must differ.
    """
    net_std, net_ph = _load_both()

    idx_std = net_std.names.index("d_d__He3_n")
    idx_ph  = net_ph.names.index("d_d__He3_n")

    # _fwd[i-1] because index 0 is n__p (not in _fwd array)
    rates_std = net_std._fwd[idx_std - 1]
    rates_ph  = net_ph._fwd[idx_ph - 1]

    assert not np.allclose(rates_std, rates_ph, rtol=1e-6), (
        "d_d__He3_n rate arrays are identical: custom filename was not loaded"
    )


def test_parthenope_conservation():
    """The custom-filename network must still conserve N, Z, and Q.

    Using a different rate table does not change the stoichiometry; the
    formal conservation check must pass for all reactions.
    """
    from primat.network_builder import check_conservation, compile_network

    cfg = PRIMATConfig({"network": "small_parthenope", "verbose": False})
    net = load_network(cfg, era="LT")
    cnet = compile_network(net.network, len(net.species))

    # Raises ValueError if any reaction violates baryon/charge conservation.
    check_conservation(cnet, net.N, net.Z,
                       weak_indices=set(net.weak_indices),
                       lepton_dZ=net.lepton_dZ)


# ---------------------------------------------------------------------------
# Full solve (slow — requires a complete BBN integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.solve
def test_parthenope_solve_is_physical():
    """A full solve with parthenope rates gives physically reasonable YP and D/H."""
    from primat.main import PRIMAT
    r = PRIMAT({"network": "small_parthenope",
                   "verbose": False}).primat_results()
    # Standard BBN values: YP ≈ 0.247, D/H ≈ 2.4e-5.  Allow wide tolerance
    # since the parthenope rates are legitimately different.
    assert 0.23 < r["YPBBN"] < 0.26, f"YP = {r['YPBBN']:.5f} is outside physical range"
    assert 1e-5 < r["DoH"]   < 5e-5, f"D/H = {r['DoH']:.3e} is outside physical range"


# ---------------------------------------------------------------------------
# Adding a brand-new reaction (GUI "Add a new reaction" feature)
# ---------------------------------------------------------------------------
#
# The GUI's "Customise Reactions" panel can inject a reaction that need not be
# in the selected network -- or in the shipped catalog at all -- by passing
# ``custom_network={"added": {name: raw_table_text}}``.  Its stoichiometry is
# derived from the name's "a_b__c_d" syntax and its reverse-rate coefficients
# from nuclide data, all in-memory.  These tests cover that path end to end.

def _flat_table_text(rate=1.0e2):
    """A 3-column (T9, rate, err) rate table spanning the master grid."""
    T9 = np.logspace(-3, 1, 200)
    r = rate * np.ones_like(T9)
    e = np.ones_like(T9)
    return "# test added reaction\n" + "\n".join(
        f"{t:.6e} {rr:.6e} {ee:.6e}" for t, rr, ee in zip(T9, r, e)
    )


def test_added_reaction_enters_network():
    """A new reaction absent from the catalog is added with derived stoichiometry.

    ``t_t__He4_n_n`` (t + t -> He4 + 2n) is a real fusion reaction not present
    in primat's shipped ``reactions_large.csv``; adding it to the *small*
    network must extend the LT reaction list and give it a non-trivial
    detailed-balance reverse rate (it is purely nuclear, so abg != 0).
    """
    from primat.network_data import UpdateNuclearRates
    cfg = PRIMATConfig({"network": "small", "verbose": False})
    cn = {"added": {"t_t__He4_n_n": _flat_table_text()}}
    upd = UpdateNuclearRates(cfg, custom_network=cn)
    assert "t_t__He4_n_n" in upd._order_LT
    i = upd._order_LT.index("t_t__He4_n_n")
    # _abg excludes the prepended weak n__p (index 0), so it is offset by one.
    abg = upd._lt_net._abg[i - 1]
    assert not np.allclose(abg, 0.0), "expected a detailed-balance reverse rate"


def test_added_weak_reaction_is_forward_only():
    """A new *weak* reaction (emitted lepton) is forward-only (abg = 0).

    ``p_p__d_Bp`` (p + p -> d + e+ + nu) carries a Bp lepton, so it is flagged
    weak and -- like the shipped beta-decays -- left without a reverse rate,
    while still being recorded in ``weak_indices``.
    """
    from primat.network_data import UpdateNuclearRates
    cfg = PRIMATConfig({"network": "small", "verbose": False})
    cn = {"added": {"p_p__d_Bp": _flat_table_text()}}
    upd = UpdateNuclearRates(cfg, custom_network=cn)
    i = upd._order_LT.index("p_p__d_Bp")
    assert i in upd._lt_net.weak_indices
    assert np.allclose(upd._lt_net._abg[i - 1], 0.0)


def test_added_reaction_non_conserving_rejected():
    """A reaction that violates baryon/charge conservation raises ValueError."""
    from primat.network_data import UpdateNuclearRates
    cfg = PRIMATConfig({"network": "small", "verbose": False})
    # p + p -> He4 conserves neither A (2 != 4) nor Z; must be rejected.
    cn = {"added": {"p_p__He4": _flat_table_text()}}
    with pytest.raises(ValueError):
        UpdateNuclearRates(cfg, custom_network=cn)


@pytest.mark.slow
@pytest.mark.solve
def test_added_reaction_solve_completes():
    """A full BBN solve with an added reaction completes and stays physical."""
    from primat.main import PRIMAT
    cn = {"added": {"t_t__He4_n_n": _flat_table_text()}}
    r = PRIMAT({"network": "small", "verbose": False},
             custom_network=cn).primat_results()
    assert 0.23 < r["YPBBN"] < 0.26
    assert 1e-5 < r["DoH"] < 5e-5
