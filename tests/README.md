# PyPRIMAT Test Suite

This test suite was written by [Claude](https://claude.ai) (Anthropic) during an
interactive development session with Cyril Pitrou. It is meant to be read as
documentation as much as run as a check: every test group below states the
physics or software property it pins down.

## Why these tests exist

PyPRIMAT is a numerical physics code whose output (primordial abundances) is a
single set of numbers that is easy to get *subtly* wrong — a mistyped Jacobian
entry, a rate read from the wrong column, a refactor that shifts a result by a
few parts in 1e5. The suite is built in layers so that:

1. **fast unit tests** catch gross breakage in seconds (config, plasma
   thermodynamics, the public API, the Monte-Carlo machinery, weak-rate
   helpers);
2. **structural cross-checks** prove the compiled, stoichiometry-driven network
   kernels agree with the declarative reaction table to machine precision (this
   is what caught four latent Jacobian bugs — see `test_network_builder.py`);
3. **regression tests** pin the final abundances, both loosely at default
   precision and tightly (the `reference` marker) at the exact settings that
   produced the published CLAUDE.md values.

## Running the tests

From the repository root:

```bash
pytest Tests/                      # everything (~3 min)
pytest Tests/ -m "not slow"        # fast unit + structural tests (~1 min)
pytest Tests/ -m "not reference"   # skip only the ~1 min high-precision runs
pytest Tests/ -m reference         # only the tight CLAUDE.md regression
pytest Tests/test_plasma.py -v     # a single file, verbose
```

## Markers

| Marker | Meaning |
|--------|---------|
| `slow` | runs at least one full PyPRIMAT solve; deselect with `-m "not slow"` |
| `reference` | high-precision runs (numerical_precision=1e-10, n_temperature_table=10000, sampling_nTOp=500, T_start_cosmo=100 MeV) that reproduce the CLAUDE.md values to YP ±1e-5, D/H ±3e-9; ~60 s total |

## Structure

| File | What it checks |
|------|----------------|
| `conftest.py` | Session-scoped fixtures: pre-solved small- and large-network `PyPR` instances reused across tests (built once, not per test). |
| `test_config.py` | `PyPRConfig`: defaults, user overrides, unknown-key warnings, the `Nuclides` table, that `eta0b` tracks `Omegabh2`, and that there is exactly one MCMC weight per network reaction. |
| `test_plasma.py` | Plasma/neutrino thermodynamics: `rho_g`, `rho_e`/`p_e` positivity and the e± cutoff, `spl`/`dspl_dT` self-consistency (combined vs separate evaluation, vs finite differences), `T_nu_decoupling` high- and low-T limits. |
| `test_decoupling_qed.py` | The `incomplete_decoupling` × `QED_corrections` 2×2 flag matrix: that `PofT`/`dPdT`/`d2PdT2` vanish when `QED_corrections=False`; that `spl/T³` equals `11π²/45` (free-gas) or differs from it (QED) at high T; that the instantaneous-decoupling $(T_\gamma/T_\nu)^3$ ratio equals `11/4` without QED and the Dodelson–Turner–Heckler perturbative formula with QED; that the correct NEVO file is loaded for each combination; and Neff reference values pinned for all four combinations. |
| `test_api.py` | Public API: `A/N/Z` dicts, `__getitem__` abundance interpolators (scalar and array input, non-negativity), `get_quantity`, lazy `solve()`, `T_of_t`/`t_of_T`. |
| `test_mc.py` | Monte-Carlo machinery: `MCResult`/`MCQuantityResult` shapes and attributes, mean/std consistency, reproducibility for a fixed seed, that varying rates gives `std > 0`. |
| `test_weak_rates.py` | n↔p weak rates: Fermi-Dirac helpers, the Fermi-Coulomb correction, the neutron-decay phase-space integral `ComputeFn`, and that the two loaded rate interpolants (forward/backward) are positive and obey detailed balance (ratio → 1 at high T). |
| `test_refactor_invariants.py` | Properties the performance refactor relies on: MC results independent of `n_jobs`, `eta0b` recomputed on reassignment, GN/tau_n overridable, electron-thermo tabulation ≈ exact integrals, `_LinearRate` ≈ `interp1d(kind='linear')`. |
| `test_custom_loader.py` | The `small_parthenope` custom network file: verifies the reaction set and species match the standard small network; that reactions routed to non-default files (e.g. `ddTOHe3n_parthenope.txt`) actually use different rate values; that the loaded network passes N/Z/Q conservation; and that a full BBN solve gives physically reasonable YP and D/H. |
| `test_qed_pressure.py` | The analytical QED plasma-pressure module (`pypr.qed_pressure`): Fermi-Dirac integral analytic limits (UR limit → π²/12, non-relativistic cutoff), sign conventions (δP_a < 0, δP_{e3} > 0), agreement with the PRIMAT-generated tables to 0.5% at T ≥ 2 MeV, numerical derivative consistency, and a save/load round-trip check. |
| `test_network_generation.py` | The offline generation layer (`generate_from_primat/`): token resolution, the formal baryon/charge conservation check, that `nuclides.csv` agrees with the hard-coded table, that the deduced reaction list is a superset of the 12-/62-reaction networks, and that the computed detailed-balance coefficients reproduce the published values. |
| `test_network_builder.py` | The generic stoichiometry-driven kernels (`network_builder`), the single network path: compiled RHS/Jacobian equal the `reactions` reference to machine precision; the formal N/Z conservation check (passes for real nets, fires on a broken one); numerical baryon-number conservation; the full `UpdateNuclearRates` driver methods (rhs/rhsMT/rhsLT + Jacobians); era-independent table invariants (buffer-order lengths, per-reaction A/Z conservation); and the `amax` mass-cutoff filter (correct count, nuclide bound, conservation, and invalid-value rejection). |
| `test_large_network.py` | The large network: it loads (~59 nuclides, ~433 reactions) and passes the formal conservation check; the vectorised rate buffer stays finite/bounded across the LT range; and a full solve conserves baryon number exactly while matching the medium network on the light elements. |
| `test_nuclear_qed.py` | QED corrections to radiative-capture rates (Pitrou & Pospelov 2020): correction factors are > 1 and sub-percent; the npTOdg polynomial matches its T9→0 cap; the four Kroll-formula reactions increase monotonically with T9; reference magnitudes at T9=0.1 GK are pinned to ±2e-6; non-QED reactions are unchanged; p_* variations stack correctly on the corrected median; and a full solve with the flag on shifts D/H by a detectable but sub-percent amount. |
| `test_regression.py` | Final abundances: loose default-precision sanity checks, tight `reference`-marked checks against the published CLAUDE.md values, no-numba full solve checks (pure-Python kernels must match JIT to 1e-4), and the `amax` cutoff verification (large network filtered to A ≤ 20 matches medium light elements to ~1e-3). |
