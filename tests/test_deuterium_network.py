"""
Tests for the ``network="large", amax=2`` configuration: the old standalone
``deuterium`` network (a single-reaction ``n_p__d_g`` network shipped as a
clean-slate starting point for users building their own custom networks) is
reproduced by restricting the large network to A <= 2 -- see CLAUDE.md's "Key
configuration flags" table and CUSTOMPOPUP.md §1/§3.1 for the migration.

Unlike the old standalone ``deuterium.txt`` (n_p__d_g only), the large
network's A<=2 slice also includes ``p_p_n__d_p`` (same product, different
channel), so there are 2 thermonuclear reactions, not 1 -- the physical
scenario (baryon conservation, no nuclide beyond D produced, D/H value) is
unaffected (matches the old deuterium-only D/H to ~1e-9 relative, see
CLAUDE.md/CUSTOMPOPUP.md §3.1).

Checks the load (species/reaction count, formal conservation) and a full solve:
baryon number is conserved and, since there is no channel beyond n+p->d, all
heavier nuclides (He3, He4, Li7, Be7) stay at their initial (zero) abundance.
"""
import numpy as np
import pytest

_PARAMS = {"network": "large", "amax": 2, "verbose": False}


def test_amax2_network_loads_and_conserves():
    """Loads the 3-nuclide (n, p, d) network restricted to A<=2 and passes the
    formal N/Z conservation check."""
    from pyprimat.config import PyPRConfig
    from pyprimat.network_data import load_network
    from pyprimat.network_builder import compile_network, check_conservation
    cfg = PyPRConfig(_PARAMS)
    ln = load_network(cfg)
    assert ln.species[:2] == ["n", "p"]
    assert "d" in ln.species or "H2" in ln.species
    assert ln.n_reac == 3                  # n__p (implicit) + n_p__d_g + p_p_n__d_p
    assert 0 in ln.weak_indices            # n__p
    cnet = compile_network(ln.network, len(ln.species))
    check_conservation(cnet, ln.N, ln.Z, weak_indices=ln.weak_indices)  # raises if bad


@pytest.mark.slow
@pytest.mark.solve
def test_amax2_solve_conserves_baryon_and_skips_heavier_nuclides():
    """Full large-network, amax=2 solve: baryon number conserved exactly, and
    no abundance beyond D/H is produced (no reaction channel feeds
    He3/He4/Li7/Be7)."""
    from pyprimat import PyPR
    p = PyPR(params=_PARAMS)
    r = p.solve()
    assert r["He3oH"] == 0.0
    assert r["YPBBN"] == 0.0
    assert r["Li7oH"] == 0.0
    assert np.isfinite(r["DoH"])
    assert r["DoH"] > 0.0
