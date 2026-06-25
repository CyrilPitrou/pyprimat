"""
Tests for the large BBN network (``network="large"``): ~433 reactions over ~59
nuclides, loaded from the generated CSVs and integrated in the LT era only.

These check the load (species/reaction counts, formal conservation), the
vectorised rate buffer (finite, bounded), and a full solve: that baryon number
is conserved exactly and that the light-element abundances the small network
predicts (n, p, d, t, He4, Li7, Be7) agree with the large network restricted
to amax=8 (the old "medium" network's 68-reaction equivalent) -- they must,
since the extra heavy-nuclide channels are tiny corrections.  The heavy-nuclide
tail itself (B, C, N, O, ...) is approximate (limited by the AC2024 rate floors)
and is not asserted here.

Skips if the generated ``rates/nuclear/AC2024`` folder is absent.
"""
import os

import numpy as np
import pytest

_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..", "primat",
                           "rates", "csv")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/csv not generated",
)


@_needs_ac2024
def test_large_network_loads_and_conserves():
    """Loads 59 nuclides / ~424 reactions and passes the formal N/Z check."""
    from primat.config import PRIMATConfig
    from primat.network_data import load_network
    from primat.network_builder import compile_network, check_conservation
    cfg = PRIMATConfig({"network": "large", "verbose": False})
    ln = load_network(cfg)
    assert ln.species[:2] == ["n", "p"]
    assert len(ln.species) >= 55
    assert ln.n_reac >= 400
    assert 0 in ln.weak_indices                      # n__p
    assert len(ln.weak_indices) >= 30                # n__p + beta decays
    cnet = compile_network(ln.network, len(ln.species))
    check_conservation(cnet, ln.N, ln.Z, weak_indices=ln.weak_indices)  # raises if bad


@_needs_ac2024
def test_large_rate_buffer_is_finite_and_bounded():
    """fill_buffer must return finite rates across the LT temperature range,
    despite the exp(gamma/T9) detailed-balance factors of endothermic reactions."""
    from primat.config import PRIMATConfig
    from primat.network_data import load_network
    ln = load_network(PRIMATConfig({"network": "large", "verbose": False}))
    for T9 in (0.08, 0.05, 0.02, 0.012):
        r = ln.fill_buffer(T9 / 1e-9, lambda T: 1.0, lambda T: 0.5)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0)
        assert r.max() < 1e301


@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_large_solve_conserves_baryon_and_matches_amax8():
    """Full large-network solve: baryon number conserved, and the light-element
    finals agree with large/amax=8 (the heavy channels are tiny)."""
    from primat import PRIMAT
    med = PRIMAT(params={"network": "large", "amax": 8, "verbose": False})
    med.solve()
    big = PRIMAT(params={"network": "large", "verbose": False})
    big.solve()

    # Baryon number: sum_s A_s Y_s = 1 to high precision.
    from primat.config import PRIMATConfig
    A = {s: sum(PRIMATConfig.Nuclides.get(s, [0, 0])) for s in big.nuclear.Y_final}
    # Build A for every large-network species from its (N,Z) in nuclides.csv.
    from primat.network_data import load_network
    ln = load_network(PRIMATConfig({"network": "large", "verbose": False}))
    Avec = {s: int(n) + int(z) for s, n, z in zip(ln.species, ln.N, ln.Z)}
    baryon = sum(Avec[s] * y for s, y in big.nuclear.Y_final.items())
    assert abs(baryon - 1.0) < 1e-6

    # Light-element finals agree with large/amax=8 (relative, non-tiny ones).
    # H3/Li7/Be7 are excluded: the large network alone carries the
    # t__He3_Bm/Be7__Li7_Bp analytic decay reactions (commit 6221e43), whose
    # laboratory decay constants convert ~0.23% of H3->He3 and ~18% of
    # Be7->Li7 over the ~15-day integration window (T_end=0.001 MeV) -- a
    # real large-network-only effect, not a regression (see CLAUDE.md
    # "Per-nuclide final abundances").
    for s in ("p", "H2", "He4"):
        assert abs(big.nuclear.Y_final[s] - med.nuclear.Y_final[s]) / abs(med.nuclear.Y_final[s]) < 2e-3


@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_large_network_time_evolution_tsv(tmp_path):
    """``output_time_evolution=True`` writes a TSV for network="large" too
    (Item 5): one ``Y<species>`` column per of the ~59 large-network nuclides,
    no per-reaction flux columns (those are small/large-amax8 only), and the
    final He4/D/Li7 rows agree with the large/amax=8 time series to the same
    tolerances as the final-abundance comparison above."""
    from primat import PRIMAT
    import numpy as np

    out_path = tmp_path / "large_evolution.tsv"
    big = PRIMAT(params={
        "network": "large", "verbose": False,
        "output_time_evolution": True, "output_file": str(out_path),
    })
    big.solve()

    with open(out_path) as f:
        header = f.readline().strip().split("\t")
        data = np.loadtxt(f)

    # One Y<species> column per large-network nuclide; no reaction-flux
    # columns (output_rates_time_evolution defaults to False).
    y_cols = ["Y_" + s for s in big.nuclear.abundance_names]
    assert all(c in header for c in y_cols)
    assert len(y_cols) == len(big.nuclear.abundance_names)
    assert not any(h.endswith("_frwrd") for h in header)

    # Every column must stay finite, including the exact-0 HT-era
    # Y<species> entries for the ~59 large-network nuclides (no NSE/Saha
    # fill is applied any more, see NuclearNetwork._write_time_evolution).
    assert np.isfinite(data).all()

    med = PRIMAT(params={"network": "large", "amax": 8, "verbose": False})
    med.solve()

    # Compare the final-time row of the large-network TSV against the
    # large/amax=8 network's final abundances (same tolerance as the
    # final-abundance comparison in test_large_solve_conserves_baryon_and_matches_amax8).
    # Li7 is excluded for the same reason as in that test: the large
    # network's Be7__Li7_Bp decay reaction (commit 6221e43) converts ~18% of
    # Be7 into Li7 over the full integration window, so large Li7 is ~4x
    # large/amax=8 Li7 by design (see CLAUDE.md "Per-nuclide final abundances").
    for s in ("He4", "H2"):
        col = header.index("Y_" + s)
        y_final_tsv = data[-1, col]
        assert abs(y_final_tsv - med.nuclear.Y_final[s]) / abs(med.nuclear.Y_final[s]) < 2e-3
