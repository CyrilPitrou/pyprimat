"""
Tests for the large BBN network (``network="large"``): ~433 reactions over ~59
nuclides, loaded from the generated CSVs and integrated in the LT era only.

These check the load (species/reaction counts, formal conservation), the
vectorised rate buffer (finite, bounded), and a full solve: that baryon number
is conserved exactly and that the light-element abundances the small network
predicts (n, p, d, t, He4, Li7, Be7) agree with the medium network -- they must,
since the extra heavy-nuclide channels are tiny corrections.  The heavy-nuclide
tail itself (B, C, N, O, ...) is approximate (limited by the AC2024 rate floors)
and is not asserted here.

Skips if the generated ``rates/nuclear/AC2024`` folder is absent.
"""
import os

import numpy as np
import pytest

_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "rates", "nuclear", "data")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated",
)


@_needs_ac2024
def test_large_network_loads_and_conserves():
    """Loads 59 nuclides / ~424 reactions and passes the formal N/Z check."""
    from pyprimat.config import PyPRConfig
    from pyprimat.nuclear import load_network
    from pyprimat.network_builder import compile_network, check_conservation
    cfg = PyPRConfig({"network": "large", "verbose": False})
    ln = load_network(cfg)
    assert ln.species[:2] == ["n", "p"]
    assert len(ln.species) >= 55
    assert ln.n_reac >= 400
    assert 0 in ln.weak_indices                      # nTOp
    assert len(ln.weak_indices) >= 30                # nTOp + beta decays
    cnet = compile_network(ln.network, len(ln.species))
    check_conservation(cnet, ln.N, ln.Z, weak_indices=ln.weak_indices)  # raises if bad


@_needs_ac2024
def test_large_rate_buffer_is_finite_and_bounded():
    """fill_buffer must return finite rates across the LT temperature range,
    despite the exp(gamma/T9) detailed-balance factors of endothermic reactions."""
    from pyprimat.config import PyPRConfig
    from pyprimat.nuclear import load_network
    ln = load_network(PyPRConfig({"network": "large", "verbose": False}))
    for T9 in (0.08, 0.05, 0.02, 0.012):
        r = ln.fill_buffer(T9 / 1e-9, lambda T: 1.0, lambda T: 0.5)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)
        assert r.max() < 1e301


@_needs_ac2024
@pytest.mark.slow
def test_large_solve_conserves_baryon_and_matches_medium():
    """Full large-network solve: baryon number conserved, and the light-element
    finals agree with the medium network (the heavy channels are tiny)."""
    from pyprimat import PyPR
    med = PyPR(params={"network": "medium", "verbose": False})
    med.solve()
    big = PyPR(params={"network": "large", "verbose": False})
    big.solve()

    # Baryon number: sum_s A_s Y_s = 1 to high precision.
    from pyprimat.config import PyPRConfig
    A = {s: sum(PyPRConfig.Nuclides.get(s, [0, 0])) for s in big._Y_final}
    # Build A for every large-network species from its (N,Z) in nuclides.csv.
    from pyprimat.nuclear import load_network
    ln = load_network(PyPRConfig({"network": "large", "verbose": False}))
    Avec = {s: int(n) + int(z) for s, n, z in zip(ln.species, ln.N, ln.Z)}
    baryon = sum(Avec[s] * y for s, y in big._Y_final.items())
    assert abs(baryon - 1.0) < 1e-6

    # Light-element finals agree with medium (relative, for the non-tiny ones).
    for s in ("p", "H2", "H3", "He4", "Li7", "Be7"):
        assert abs(big._Y_final[s] - med._Y_final[s]) / abs(med._Y_final[s]) < 2e-3
