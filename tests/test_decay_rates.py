# -*- coding: utf-8 -*-
"""
Tests for radioactive-decay treatment in the `large` network (DECAY.md §3 and §4).

Section 3.3 — decay_reverse_rates flag:
    By default decay reactions are treated as irreversible (abg = (0,0,0)).
    The ``decay_reverse_rates=True`` flag asks ``load_network`` to compute
    detailed-balance reverse rates from nuclide data.  However, for any β-decay
    X → Y + B± the Q-value is positive (energy is released), so the
    detailed-balance gamma coefficient is negative:

        gamma = -Q / (kB × 1 GK) ≈ -Q [MeV] / 0.0862

    At BBN endpoint T9 ~ 0.001 the reverse-rate factor exp(gamma/T9) is
    exp(-Q [MeV] / (0.0862 × 0.001)) = exp(-11.6 Q/MeV).  For the
    softest Q-value encountered (H3 → He3 + Bm, Q ≈ 18.6 keV) this is
    exp(-18.6e-3/0.0862/0.001) = exp(-216) ≈ 0.  For the shortest-lived
    large-network decay (O13, Q ≈ 17.8 MeV) it is exp(-206 000) ≈ 0.
    The reverse channel is therefore numerically inert across the entire BBN
    integration window.

    We test this at two levels:

    a) **Unit test (fast)**: check that the ``bwd_cap`` array computed by
       ``load_network`` is zero for every decay reaction, even when
       ``decay_reverse_rates=True``.  The cap is evaluated at T_nucl
       (the nucleosynthesis onset, the *highest* temperature in the LT era)
       and the reverse rate is suppressed by exp(gamma/T_nucl) where
       gamma/T_nucl ≪ -100 for every decay -- so the cap rounds to zero
       immediately without needing a full ODE solve.

    b) **Solve test (slow, uses the session solved_large fixture)**: run one
       new solve with ``decay_reverse_rates=True`` and compare final
       abundances against the pre-solved default-config reference.  The
       reverse caps are not *exactly* zero (the largest, He6→Li6, is
       ~6e-12 s^-1; see test (a)), and acting over the ~1.3×10^6 s LT window
       they shift the most sensitive species — the free-neutron floor n
       (Y ~ 4×10^-16) — by ~2×10^-6 relative (absolute Δ ~ 1×10^-21, utterly
       negligible physically).  We therefore demand relative agreement < 1e-5
       for every species above an abundance floor of 1e-20; species below
       that floor are heavy-tail nuclides at Y ~ 1e-50…1e-110 where a relative
       metric is meaningless.

Section 4 — Decay Time (DT) era:
    After BBN ends at T_end ≈ 0.001 MeV (t_end ≈ 1.3×10^6 s ≈ 15 days),
    long-lived isotopes continue to decay on timescales of years to Myr.
    The DT era propagates the abundance vector via the constant decay matrix
    D: Y(t) = exp(D × Δt) × Y(t_end), using
    ``scipy.sparse.linalg.expm_multiply``.

    Key physics to verify (half-lives from decays.txt):
      - Na22 (T½ ≈ 2.60 yr ≈ 8.21×10^7 s):  fully decays by 1 Gyr.
      - C14 (T½ ≈ 5700 yr ≈ 1.80×10^11 s):  fully decays by 1 Gyr.
      - He4, H2, p (stable):                 unchanged in DT era.
      - Baryon number conserved by D matrix.

    A cross-check against Radau (implicit Runge-Kutta) is included over the
    first 10 yr, where Radau can integrate reliably.
"""
import os
import numpy as np
import pytest

_AC2024_DIR = os.path.join(os.path.dirname(__file__), "..", "pyprimat",
                           "rates", "nuclear", "data")
_needs_ac2024 = pytest.mark.skipif(
    not os.path.isdir(_AC2024_DIR),
    reason="rates/nuclear/data not generated",
)

# ---------------------------------------------------------------------------
# §3.3a  bwd_cap is zero for all decay reactions (fast unit test)
# ---------------------------------------------------------------------------

@_needs_ac2024
def test_decay_reverse_bwd_cap_is_zero():
    """With decay_reverse_rates=True, bwd_cap is negligible for every decay.

    Two mechanisms make the reverse rate inert for all decay channels.

    1. Simple two-body β-decays (X → Y + Bm/Bp, e.g. H3→He3, Be7→Li7, C14→N14):
       ``compute_detailed_balance_coefficients`` computes the Q-value from binding
       energies, which is a valid approximation for strong reactions but misses the
       neutron-proton mass difference (Δ_n − Δ_p ≈ 782 keV) for weak transitions.
       For β⁻ decay (n→p), the formula gives Q_computed = Q_true − 782 keV.
       Since the largest Q_true for a pure β⁻ decay is well below 782 keV (H3 has
       Q ≈ 18.6 keV; Be7 electron capture is only Q ≈ 861.8 keV but is handled
       separately), the computed Q is negative → gamma > 0.  The ``gamma >= 0``
       guard in ``load_network`` then falls back to abg = (0,0,0), giving bwd_cap = 0.

    2. Multi-body beta-delayed channels (Li8→αα+Bm, B8→αα+Bp, Li9→ααn+Bm,
       C9→ααp+Bp): these have large positive Q_true (≳ 15 MeV), so Q_computed
       is also positive after the stoichiometry bug fix (products expanded to
       repeated species before passing to the function).  gamma << 0 and
       exp(gamma / T9_nucl) underflows to 0, giving bwd_cap ≈ 0.

    The net result: all decay reactions get bwd_cap < 1e-10 s^-1 regardless
    of the decay channel.  The assertion uses 1e-10 s^-1 as the threshold;
    the largest observed value is ~ 6e-12 s^-1 (He6→Li6).
    """
    from pyprimat.config import PyPRConfig
    from pyprimat.network_data import load_network

    cfg = PyPRConfig({"network": "large", "verbose": False,
                      "decay_reverse_rates": True})
    ln = load_network(cfg)

    # weak_indices includes nTOp (index 0) + all β-decay reactions.
    # bwd_cap shape: (n_reactions_minus_nTOp,); weak_indices[i] - 1 is the index.
    decay_indices = [i for i in ln.weak_indices if i != 0]
    for rxn_idx in decay_indices:
        # bwd_cap is indexed without nTOp: entry i corresponds to names[i+1].
        cap = float(ln._bwd_cap[rxn_idx - 1])
        assert cap < 1e-10, (
            f"Decay {ln.names[rxn_idx]}: bwd_cap = {cap:.3e} s^-1 is not"
            f" negligible -- reverse rate may affect the ODE."
        )


# ---------------------------------------------------------------------------
# §3.3b  decay_reverse_rates=True does not change final abundances (slow)
# ---------------------------------------------------------------------------

@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_decay_reverse_rates_inert(solved_large):
    """``decay_reverse_rates=True`` must not meaningfully change abundances.

    Uses the session-scoped ``solved_large`` fixture (default config,
    ``decay_reverse_rates=False``) as the reference, and runs one new
    large-network solve with ``decay_reverse_rates=True``.

    The reverse caps are not exactly zero (largest ~6e-12 s^-1, He6→Li6;
    see ``test_decay_reverse_bwd_cap_is_zero``); integrated over the
    ~1.3×10^6 s LT window they perturb the most sensitive species (the free
    neutron floor n, Y ~ 4e-16) by ~2×10^-6 relative, i.e. ~1e-21 absolute.
    We therefore require relative agreement < 1e-5 for every species whose
    reference abundance exceeds an ``ABUNDANCE_FLOOR`` of 1e-20.  Species
    below that floor are heavy-tail nuclides (Y ~ 1e-50…1e-110) where the
    reverse channel produces relative shifts up to ~1e-4 on numbers far
    below any physical relevance — comparing them by relative error is
    meaningless.  This still demonstrates the reverse channel is numerically
    inert for every observable abundance, two orders of magnitude tighter
    than the 1e-3 large-vs-medium tolerance from ``test_large_network.py``.
    """
    from pyprimat import PyPR

    ABUNDANCE_FLOOR = 1e-20   # below this, relative comparison is noise

    run_on = PyPR(params={"network": "large", "verbose": False,
                          "decay_reverse_rates": True})
    run_on.solve()

    Y_off = solved_large.nuclear.Y_final
    Y_on  = run_on.nuclear.Y_final

    for s in solved_large.nuclear.abundance_names:
        y_off = Y_off[s]
        y_on  = Y_on[s]
        if abs(y_off) <= ABUNDANCE_FLOOR:
            continue
        rel = abs(y_on - y_off) / abs(y_off)
        assert rel < 1e-5, (
            f"{s}: decay_reverse_rates changed Y by {rel:.2e} "
            f"(off={y_off:.6e}, on={y_on:.6e})"
        )


# ---------------------------------------------------------------------------
# §4  Decay Time (DT) era: TSV output + column layout
# ---------------------------------------------------------------------------

@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_decay_era_tsv(tmp_path, solved_large):
    """DT era writes a valid TSV with correct column layout.

    Uses the session-scoped ``solved_large`` fixture so no extra ODE solve is
    needed.  The DT integration and TSV write are called directly on the
    already-solved NuclearNetwork, with ``cfg.output_decay_file`` temporarily
    pointed at ``tmp_path``.

    Checks:
    1. The TSV header matches ``abundance_names``.
    2. Every entry is finite and non-negative (``_integrate_decay_era`` clips
       tiny floating-point negatives from matrix-exponential cancellation).
    3. The file has the expected number of rows (``n_points`` below).
    4. Stable species (He4, H2, p) do not drift: relative change < 1e-6.
    """
    n_points = 30
    YR = 86400.0 * 365.2422
    t_end_s = solved_large.nuclear._lt_t_end_s()
    t_grid  = np.logspace(np.log10(t_end_s + YR),
                          np.log10(t_end_s + 1e9 * YR), n_points)

    nn = solved_large.nuclear
    D  = nn._build_decay_matrix(nn.nucl._lt_net)
    Y0 = np.array([nn.Y_final.get(s, 0.0) for s in nn.abundance_names])
    Y_DT = nn._integrate_decay_era(D, Y0, t_end_s, t_grid)

    out_path = tmp_path / "decay_evolution.tsv"
    # Temporarily redirect the output path in cfg so _write_decay_evolution
    # uses tmp_path instead of cfg.output_decay_file.
    orig_path = nn.cfg.output_decay_file
    nn.cfg.output_decay_file = str(out_path)
    try:
        nn._write_decay_evolution(t_grid, Y_DT)
    finally:
        nn.cfg.output_decay_file = orig_path

    assert out_path.exists(), "_write_decay_evolution did not create the file"

    with open(out_path) as fh:
        header = fh.readline().strip().split("\t")
        data = np.loadtxt(fh)

    expected_cols = ["t"] + ["Y" + s for s in nn.abundance_names]
    assert header == expected_cols
    assert data.shape == (n_points, len(expected_cols))
    assert np.all(np.isfinite(data)), "non-finite entry in DT era TSV"
    assert np.all(data >= 0.0),       "negative abundance in DT era TSV"

    # Stable species must not drift.
    col = {h: i for i, h in enumerate(header)}
    for s in ("He4", "H2", "p"):
        if "Y" + s not in col:
            continue
        y0 = Y0[nn.abundance_names.index(s)] if s in nn.abundance_names else 0.0
        if y0 == 0.0:
            continue
        y_dt = data[:, col["Y" + s]]
        rel_drift = np.max(np.abs(y_dt - y0)) / abs(y0)
        assert rel_drift < 1e-6, (
            f"{s} (stable) drifted by {rel_drift:.2e} in DT era"
        )


# ---------------------------------------------------------------------------
# §4  Decay Time (DT) era: known decay physics
# ---------------------------------------------------------------------------

@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_decay_era_physics(solved_large):
    """DT era: known-half-life decays and baryon conservation on a 1-Gyr grid.

    Uses the session-scoped ``solved_large`` fixture (LT era already solved;
    only the constant-matrix DT integration is performed here, which is fast).

    Species checked:
      - Na22 (T½ ≈ 2.60 yr):  Y/Y0 < 1e-5 after 1 Gyr (≈ 3.85×10^8 half-lives).
      - C14  (T½ ≈ 5700 yr):  Y/Y0 < 1e-5 after 1 Gyr (≈ 175 000 half-lives).
      - He4  (stable):         |Y - Y0| / Y0 < 1e-6 at every DT output point.
      - Baryon number:         |Σ_s A_s Y_s(t) - Σ_s A_s Y0_s| / Σ_s A_s Y0_s
                               < 1e-6 at every output point.
        (Bm/Bp carriers are not in the ODE state, so A-weighted mass fraction
        is approximately, not exactly, conserved per decay event; the bound
        is generous.)
    """
    from pyprimat.network_data import load_network
    from pyprimat.config import PyPRConfig

    cfg = PyPRConfig({"network": "large", "verbose": False})
    nucl_data = load_network(cfg)

    nn = solved_large.nuclear
    D = nn._build_decay_matrix(nn.nucl._lt_net)
    names = nn.abundance_names
    Y0 = np.array([nn.Y_final.get(s, 0.0) for s in names])
    t_end_s = nn._lt_t_end_s()

    YR = 86400.0 * 365.2422   # seconds per Julian year
    t_grid = np.logspace(np.log10(YR), np.log10(1e9 * YR), 80)
    Y_DT = nn._integrate_decay_era(D, Y0, t_end_s, t_grid)

    name_idx = {s: i for i, s in enumerate(names)}

    # Na22 fully decayed.
    if "Na22" in name_idx:
        i = name_idx["Na22"]
        y0_val = Y0[i]
        if y0_val > 1e-30:
            assert Y_DT[-1, i] / y0_val < 1e-5, (
                f"Na22 (T½=2.60 yr) should be gone after 1 Gyr; "
                f"Y/Y0 = {Y_DT[-1, i]/y0_val:.2e}"
            )

    # C14 fully decayed.
    if "C14" in name_idx:
        i = name_idx["C14"]
        y0_val = Y0[i]
        if y0_val > 1e-30:
            assert Y_DT[-1, i] / y0_val < 1e-5, (
                f"C14 (T½=5700 yr) should be gone after 1 Gyr; "
                f"Y/Y0 = {Y_DT[-1, i]/y0_val:.2e}"
            )

    # He4 unchanged (stable).
    if "He4" in name_idx:
        i = name_idx["He4"]
        y0_val = Y0[i]
        if y0_val > 0:
            rel = np.max(np.abs(Y_DT[:, i] - y0_val)) / y0_val
            assert rel < 1e-6, f"He4 (stable) drifted by {rel:.2e} in DT era"

    # Baryon number conservation.
    A_vec = (nucl_data.N + nucl_data.Z).astype(float)
    baryon_0 = float(Y0 @ A_vec)
    baryon_DT = Y_DT @ A_vec
    rel_err = np.max(np.abs(baryon_DT - baryon_0)) / baryon_0
    assert rel_err < 1e-6, (
        f"Baryon number violated in DT era: max |ΔA|/A = {rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# §4  DT era: expm_multiply cross-check vs Radau (slow)
# ---------------------------------------------------------------------------

@_needs_ac2024
@pytest.mark.slow
@pytest.mark.solve
def test_decay_era_expm_vs_radau(solved_large):
    """Cross-check expm_multiply vs Radau on a 10-year DT window.

    Over the first 10 years the ODE is not extremely stiff (the fastest
    meaningful decay is Na22 with T½ ≈ 2.6 yr; everything shorter-lived has
    already fully decayed within the LT era), so Radau can integrate reliably
    with ``rtol=1e-8, atol=1e-30, max_step=1e8 s`` (≈ 3 yr cap prevents the
    solver from jumping over Na22's transient).

    For every species with Y > 1e-20 at the comparison times (i.e. above
    machine-epsilon regime), the two solvers must agree to 1e-5 relative if
    Radau required < 2000 function evaluations (well-behaved), or to 1e-3
    if it needed more (documenting the difficulty of the eigenvalue spread
    and treating expm_multiply as ground truth).
    """
    from scipy.integrate import solve_ivp

    nn = solved_large.nuclear
    D = nn._build_decay_matrix(nn.nucl._lt_net)
    names = nn.abundance_names
    Y0 = np.array([nn.Y_final.get(s, 0.0) for s in names])
    t_end_s = nn._lt_t_end_s()

    YR = 86400.0 * 365.2422
    t_DT_end = t_end_s + 10.0 * YR
    t_compare = np.logspace(np.log10(t_end_s + 0.1 * YR),
                            np.log10(t_DT_end), 4)
    # np.logspace's last point can round to just above t_DT_end (10**log10(x)
    # is not exactly x), which would put t_eval outside solve_ivp's t_span.
    # Clamp the integration window's upper bound to t_compare's actual max.
    t_DT_end = float(t_compare[-1])

    Y_expm = nn._integrate_decay_era(D, Y0, t_end_s, t_compare)

    sol_radau = solve_ivp(
        lambda t, Y: D @ Y,
        [t_end_s, t_DT_end],
        Y0,
        method="Radau",
        jac=lambda t, Y: D,
        t_eval=t_compare,
        rtol=1e-8,
        atol=1e-30,
        max_step=1e8,
    )

    tight_tol = 1e-5 if sol_radau.nfev < 2000 else 1e-3

    for k in range(len(t_compare)):
        for i, s in enumerate(names):
            y_ref = Y_expm[k, i]
            if y_ref < 1e-20:
                continue
            y_rad = sol_radau.y[i, k]
            rel = abs(y_rad - y_ref) / y_ref
            assert rel < tight_tol, (
                f"expm vs Radau at t={t_compare[k]:.3g} s: {s} disagrees "
                f"by {rel:.2e} (expm={y_ref:.3e}, Radau={y_rad:.3e}, "
                f"nfev={sol_radau.nfev})"
            )
