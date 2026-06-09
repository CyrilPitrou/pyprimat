"""
Tests for the custom network file loader.

The ``small_parthenope`` network (``rates/nuclear/networks/small_parthenope.txt``)
exercises the ``bare_name, filename.txt`` syntax that routes individual reactions
to non-default rate tables.  For example:

    ddTOHe3n, ddTOHe3n_parthenope.txt

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

from pypr.nuclear import load_network
from pypr.config import PyPRConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_both():
    """Load the standard small network and the parthenope custom network."""
    cfg_std = PyPRConfig({"network": "small", "verbose": False})
    cfg_ph  = PyPRConfig({"network": "small_parthenope", "verbose": False})
    net_std = load_network(cfg_std, era="LT")
    net_ph  = load_network(cfg_ph,  era="LT")
    return net_std, net_ph


# ---------------------------------------------------------------------------
# Structure checks (fast — no solve)
# ---------------------------------------------------------------------------

def test_small_parthenope_loads():
    """The parthenope network file parses without error."""
    cfg = PyPRConfig({"network": "small_parthenope", "verbose": False})
    net = load_network(cfg, era="LT")
    assert len(net.names) > 0


def test_small_parthenope_same_reaction_set():
    """small_parthenope has the same 12 nuclear reactions as small."""
    net_std, net_ph = _load_both()
    # Both should have nTOp + 12 nuclear reactions = 13 entries
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

    ``ddTOHe3n`` is routed to ``ddTOHe3n_parthenope.txt`` (Gariazzo 2021)
    instead of ``ddTOHe3n.txt`` (Gom17).  The two rate files have different
    values, so the loaded forward-rate arrays must differ.
    """
    net_std, net_ph = _load_both()

    idx_std = net_std.names.index("ddTOHe3n")
    idx_ph  = net_ph.names.index("ddTOHe3n")

    # _fwd[i-1] because index 0 is nTOp (not in _fwd array)
    rates_std = net_std._fwd[idx_std - 1]
    rates_ph  = net_ph._fwd[idx_ph - 1]

    assert not np.allclose(rates_std, rates_ph, rtol=1e-6), (
        "ddTOHe3n rate arrays are identical: custom filename was not loaded"
    )


def test_parthenope_conservation():
    """The custom-filename network must still conserve N, Z, and Q.

    Using a different rate table does not change the stoichiometry; the
    formal conservation check must pass for all reactions.
    """
    from pypr.network_builder import check_conservation, compile_network

    cfg = PyPRConfig({"network": "small_parthenope", "verbose": False})
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
def test_parthenope_solve_is_physical():
    """A full solve with parthenope rates gives physically reasonable YP and D/H."""
    from pypr.main import PyPR
    r = PyPR({"network": "small_parthenope",
                   "compute_nTOp": False,
                   "verbose": False}).PyPRresults()
    # Standard BBN values: YP ≈ 0.247, D/H ≈ 2.4e-5.  Allow wide tolerance
    # since the parthenope rates are legitimately different.
    assert 0.23 < r["YPBBN"] < 0.26, f"YP = {r['YPBBN']:.5f} is outside physical range"
    assert 1e-5 < r["DoH"]   < 5e-5, f"D/H = {r['DoH']:.3e} is outside physical range"
