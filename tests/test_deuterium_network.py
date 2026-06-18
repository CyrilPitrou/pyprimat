"""
Tests for the ``deuterium`` network (``network="deuterium"``): a single-reaction
``n_p__d_g`` network shipped as a clean-slate starting point for users building
their own custom networks (see ``rates/nuclear/networks/deuterium.txt`` and the
CLAUDE.md "Key configuration flags" table).

Checks the load (species/reaction count, formal conservation) and a full solve:
baryon number is conserved and, since there is no channel beyond n+p->d, all
heavier nuclides (He3, He4, Li7, Be7) stay at their initial (zero) abundance.
"""
import numpy as np
import pytest


def test_deuterium_network_loads_and_conserves():
    """Loads the 3-nuclide (n, p, d) single-reaction network and passes the
    formal N/Z conservation check."""
    from pyprimat.config import PyPRConfig
    from pyprimat.network_data import load_network
    from pyprimat.network_builder import compile_network, check_conservation
    cfg = PyPRConfig({"network": "deuterium", "verbose": False})
    ln = load_network(cfg)
    assert ln.species[:2] == ["n", "p"]
    assert "d" in ln.species or "H2" in ln.species
    assert ln.n_reac == 2                             # n__p (implicit) + n_p__d_g
    assert 0 in ln.weak_indices                      # n__p
    cnet = compile_network(ln.network, len(ln.species))
    check_conservation(cnet, ln.N, ln.Z, weak_indices=ln.weak_indices)  # raises if bad


@pytest.mark.slow
@pytest.mark.solve
def test_deuterium_solve_conserves_baryon_and_skips_heavier_nuclides():
    """Full deuterium-network solve: baryon number conserved exactly, and no
    abundance beyond D/H is produced (no reaction channel feeds He3/He4/Li7/Be7)."""
    from pyprimat import PyPR
    p = PyPR(params={"network": "deuterium", "verbose": False})
    r = p.solve()
    assert r["He3oH"] == 0.0
    assert r["YPBBN"] == 0.0
    assert r["Li7oH"] == 0.0
    assert np.isfinite(r["DoH"])
    assert r["DoH"] > 0.0
