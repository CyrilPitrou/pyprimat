# FINAL.md — Plan for the next round of PyPRIMAT improvements

This plan covers 15 requested changes/investigations. Each item states the
**current state** (with file:line references gathered by code investigation),
the **proposed change**, and **which model should do it**.

## Ordering and model assignment

Following the convention already established in `IDEAS.md` §10 and `FIX.md`:
mechanical, low-risk, well-pinned changes go to **Sonnet**; changes that
require physics-numerics judgment, literature verification, or
interface/architecture design go to **Opus** (or Fable), and run **last**, on
top of a codebase that the Sonnet phases have already made easier to validate.

| Phase | Items | Model | Risk |
|-------|-------|-------|------|
| 1 | 15, 10, 9, 8, 7 | Sonnet | none–low: mechanical, test-pinned |
| 2 | 5, 4, 1, 14, 6 | Sonnet | low–medium: mechanical extensions, new config knobs |
| 3 | 11+12+13, 2, 3 | **Opus** (or Fable for §3) | medium–high: physics literature, formula derivation, interface design |

**Validation gate for every phase** (unchanged from `IDEAS.md`/`CLAUDE.md`):
`python runfiles/PyPRIMAT_run.py` must reproduce the CLAUDE.md reference
values, and `pytest -m reference` must pass after every item.

---

# Phase 1 — Sonnet (mechanical, low risk)

## Item 15 — Are weak rates recomputed when the chemical potential changes?

**Current state**: Already correct. `_BACKGROUND_FINGERPRINT_FIELDS`
(`pyprimat/weak_rates.py:117-128`) includes both `munuOverTnu` and
`delta_xi_nu`. `_weak_rate_fingerprint(cfg)` (`weak_rates.py:155-186`) folds
these into the cache fingerprint hash, so any change to either parameter
invalidates `rates/weak/nTOp_*.txt` / `pTOn_*.txt` and forces a recompute.
`tau_n`/`tau_n_flag` are deliberately excluded (correctly — they are a
post-hoc normalisation, see `main.py:431-436`).

**Proposed change**: No code change needed. Add one small regression test
(e.g. `tests/test_weak_rate_cache.py`) that:
1. Computes the fingerprint hash for two configs differing only in
   `munuOverTnu` (or `delta_xi_nu`) and asserts the hashes differ.
2. Optionally, runs `RecomputeWeakRates` once with `munuOverTnu=0` (writing a
   cache), then again with `munuOverTnu=0.01` and confirms the rates differ
   and a recompute happened (no false cache hit).

This pins the behaviour so a future refactor of the fingerprint fields list
can't silently regress it.

**Model**: Sonnet (verification + one pinning test).

---

## Item 10 — Config.py comments: developer notes vs user-facing docs

**Current state**: `pyprimat/config.py` contains several long
implementation-rationale blocks aimed at developers, including explicit
references to `IDEAS.md` (which CLAUDE.md says must not be referenced in a
config file or anywhere else):

- Lines ~37–41: incomplete-decoupling vs instantaneous-decoupling explanation
  with `(4/11)^(1/3)` math and a "physically inconsistent" warning that
  belongs in validation-error text, not a comment.
- Lines ~51–73: 23-line spectral-distortion block with δf^SZ / μ-type / SZ
  math — too low-level for a config comment.
- Lines ~83–102: weak-rate fingerprinting block, **explicitly cites
  "IDEAS.md §1.2", "§5.1", "§8.2"** and discusses internal cache-file sharing
  rationale (mentions `generate_table_CLASS_CAMB.py`).
- Lines ~106–116: thermal-correction cache block — discusses vegas/dblquad
  internals and multi-minute MC integration cost.
- Lines ~120–129: finite-T radiative-correction accuracy knobs — implementation
  detail (vegas vs scipy fallback).

**Proposed change**:
- Remove every `IDEAS.md` reference from `config.py` (and grep the whole repo
  for stray `IDEAS.md`/`IDEAS` mentions outside `IDEAS.md` itself and fix
  those too).
- For each of the long blocks above: keep a **1-3 line user-facing comment**
  describing *what the flag does and when to change it* (e.g. "Use
  `analytic_distortions=True` for a fast parametric distortion model;
  `False` to use the tabulated NEVO spectral-distortion data").
- Move the removed implementation rationale (cache file layout, vegas vs
  dblquad trade-offs, fingerprint field lists, "physically inconsistent
  combination raises ValueError because…") into the docstrings of the
  modules that actually implement it: `weak_rates.py` (fingerprint /
  thermal-cache docstrings), `plasma.py` (electron-thermo cache docstring),
  and the `__init__` validation `raise ValueError(...)` messages themselves
  (so a user who hits the error sees the *why* right there).
- Keep CLAUDE.md's "comment heavily, cite physics" requirement in mind: the
  *physics* meaning of each flag stays, only the *developer-process*
  rationale (IDEAS.md section numbers, "this used to be a bug", caching
  internals) moves out.

**Model**: Sonnet (pure documentation/comment relocation, no logic change;
diff should touch only comments/docstrings).

---

## Item 9 — Simplify `tabulate_electron_thermo` / `recompute_electron_thermo`

**Current state** (`pyprimat/config.py:45,47`, `pyprimat/plasma.py:513-570`):
Two flags control electron-thermo caching:
- `tabulate_electron_thermo` (bool): on/off for tabulation+interpolation vs.
  exact `quad` integration at every call (the slow path).
- `recompute_electron_thermo` (bool): only meaningful when the first is
  `True`; forces recomputation even if a fingerprint-valid cache file exists.
- A third flag `save_electron_thermo` controls write-back.

The "slow, no tabulation at all" path (`tabulate_electron_thermo=False`) is a
footgun: nothing in the physics requires it, it exists only as a fallback,
and a casual user who flips it gets a silent ~10-100x slowdown with no
accuracy benefit (the tabulation is interpolation on a fine grid, not a
physics approximation).

**Proposed change**: per the user's request — "tabulation should always be
used; we should just allow recompute if the cache is missing or to force a
recompute, but that's it":
1. Remove `tabulate_electron_thermo` entirely (tabulation is now always on).
2. Keep a single flag, e.g. `recompute_electron_thermo` (bool, default
   `False`): if `False`, load the cache when the fingerprint matches, else
   recompute; if `True`, always recompute. In both recompute cases, write the
   new cache back automatically (fold `save_electron_thermo` into this — i.e.
   recomputation always saves, since the fingerprinted cache is
   self-validating and cheap to keep fresh). Drop `save_electron_thermo` as a
   separate user-facing flag (or keep as an internal kw default `True` if
   some test needs to suppress writes — check `tests/` usage first).
3. Update `plasma.py:513-570` accordingly: the branch that fully skips
   tabulation (`if not tabulate_electron_thermo: ...`) is deleted.
4. Grep `tests/`, `notebooks/`, `runfiles/`, `pyprimat/gui/params_form.py` for
   `tabulate_electron_thermo` / `save_electron_thermo` and update/remove.
5. Since `PyPRConfig` already warns on unknown keys (per IDEAS.md §1.7's
   discussion), removing these keys is safe for old configs (they'll just
   warn, not crash) — but update the notebooks anyway for cleanliness.

**Validation**: `pytest -m reference` (electron-thermo numbers feed directly
into the background and must reproduce CLAUDE.md values bit-for-bit since the
default tabulated path is unchanged, only the never-used "no tabulation"
branch is removed).

**Model**: Sonnet.

---

## Item 8 — Public API: `a_of_t`, `t_of_a`, `a_of_T`, `T_of_a`

**Current state** (`pyprimat/main.py`): `_setup_background_and_cosmo`
(lines ~189-379) already builds all four interpolants internally:
- `_a_of_T` (line ~305-306, from the a(T) ODE solution)
- `_T_of_a` (line ~309-318, `interp1d(a_grid, T_grid, ...)`)
- `_a_of_t` (interp1d, from the t(a) integration)
- `_T_of_t`, `_t_of_T` (already exposed publicly via properties at
  `main.py:903-911`)

Only `T_of_t` and `t_of_T` are exposed as public `@property`.

**Proposed change**: Add four `@property` accessors next to the existing
`T_of_t`/`t_of_T` (main.py:903-911), following the exact same pattern:
```python
@property
def a_of_T(self):
    """Scale factor a as a function of photon temperature T [MeV]."""
    return self._a_of_T

@property
def T_of_a(self):
    """Photon temperature T [MeV] as a function of scale factor a."""
    return self._T_of_a

@property
def a_of_t(self):
    """Scale factor a as a function of cosmic time t [s]."""
    return self._a_of_t

@property
def t_of_a(self):
    """Cosmic time t [s] as a function of scale factor a."""
    return self._t_of_a
```
Check each interpolant is normalised consistently (same `a` normalisation
convention, e.g. `a=1` at some reference T) — document the convention in each
docstring with units and an example call. Add a short test
(`tests/test_public_api.py` or extend an existing background test) checking
round-trips, e.g. `a_of_T(T_of_a(a)) ≈ a` and `t_of_a(a_of_t(t)) ≈ t` on a few
sample points after `solve()`.

**Model**: Sonnet.

---

## Item 7 — `Nheating_out` should be optional; review output robustness

**Current state**:
- `Nheating_out = self._N_NEVO_of_Tg(T_out)` (`main.py:880`) is **always**
  computed and added to the results dict. This does *not* currently fail when
  `incomplete_decoupling=False`, because
  `InstantaneousDecoupling.N_NEVO_of_Tg` is set to `lambda Tg:
  np.zeros_like(...)` (`neutrino_history.py:338-340`) — so today it returns an
  array of zeros rather than erroring or being absent.
- The results dict is a single literal with ~8 hard-coded keys
  (`main.py:751-762`), built unconditionally — no incremental/conditional
  assembly.

**Proposed change**: The user's request is "must be added in output only if
available; in general is this output function robust?" Two parts:
1. **Make `Nheating_out` conditional on availability rather than
   always-zero-filled**: when `incomplete_decoupling=False`, `N_NEVO_of_Tg` is
   not a *real* heating rate (it's a stub returning zeros) — including it as
   if it were real data is misleading. Change `InstantaneousDecoupling` to not
   provide `N_NEVO_of_Tg` (or have `make_neutrino_history`/`PyPR` track an
   `_has_heating_table` bool), and in the results-dict assembly only add the
   `Nheating_out` key `if self._has_heating_table:` (or equivalent). Document
   in the results-dict docstring/README which keys are conditional and under
   what flag.
2. **General robustness review** of the function assembling the results dict
   (`main.py:751-762` and surrounding `solve()` code, ~lines 700-900): check
   for any other place where an output array/key depends on a flag
   (`spectral_distortions`, `is_large`, `output_time_evolution`,
   `incomplete_decoupling`) but is currently included unconditionally with a
   placeholder/zero value, vs. genuinely omitted. Make the same
   "include only if meaningful, document which keys are conditional"
   treatment consistent across all such keys. Add a short docstring at the
   top of the results-assembly section listing all possible keys and their
   conditions.

**Validation**: add a test that runs with `incomplete_decoupling=False` and
asserts `'Nheating_out' not in results` (or whatever the chosen sentinel is),
and one with `incomplete_decoupling=True` asserting it *is* present and
non-trivial. `pytest -m reference` must still pass (default config has
`incomplete_decoupling=True`, so the key continues to be present by default).

**Model**: Sonnet.

---

# Phase 2 — Sonnet (mechanical extensions, new config knobs)

## Item 5 — Time-evolution output for the large network

**Current state** (`main.py:710-726`): when `output_time_evolution=True` and
`cfg.is_large`, the code prints a message saying the TSV is "not written for
the large network (unsupported)" and tells the user to use
`run[species](t)` instead. The per-nuclide abundance time series **is**
computed during the LT-era solve (`sol_LT.y`) and embedded in the `_Y_of_t`
interpolator (lines ~704-705); it's just not written to the TSV. The current
TSV writer (`_write_time_evolution`, used for small/medium) additionally
recomputes per-reaction *fluxes* for the 8/12-species networks, which is what
gates `is_large` out.

**Proposed change**:
1. Extend `_write_time_evolution` (or add a sibling
   `_write_time_evolution_large`) to handle the ~59-species large network:
   write columns `t, a, T, Y_<nuclide>` for all nuclides in
   `lt_net.names` (from `rates/nuclear/data/nuclides.csv`), using `sol_LT`
   directly (same `t`/`a`/`T` grid as small/medium).
2. The per-reaction flux columns (only meaningful for small/medium, ~12-62
   reactions) should be **omitted** for the large network rather than
   attempted for ~433 reactions — document this explicitly in the TSV header
   comment ("flux columns omitted for network='large'; use
   `run[species](t)` interpolators for fluxes if needed").
3. Remove the "unsupported" print and the `if cfg.is_large:` early-return;
   replace with the new writer call.
4. Update `CLAUDE.md`/README wherever `output_time_evolution` is documented
   as "small/medium only".

**Validation**: add a `slow`-tier test that runs `network="large"`,
`output_time_evolution=True`, checks the TSV is written, has the expected
number of `Y_<nuclide>` columns (~59), and that `He4`/`D`/`Li7` columns match
the corresponding medium-network time series to the existing ≲1e-3/≲1e-4
tolerances from CLAUDE.md's per-nuclide table.

**Model**: Sonnet.

---

## Item 4 — Auto-derive stoichiometry for new reactions + duplicate check

**Current state**:
- `reaction_stoichiometry(name)` (`pyprimat/nuclear.py:433-480`) raises
  `KeyError` (line ~467) if a reaction is not found in
  `rates/nuclear/data/detailed_balance.csv`.
- `load_network()` raises `KeyError(f"reaction {name!r} is not present in
  reactions_large.csv")` (around `nuclear.py:1001`) for unknown reactions in a
  network file.
- `_tokenise(name)` (`nuclear.py:419-430`) already parses a compact reaction
  name like `"ddTOHe3n"` into a token list (`["d","d","TO","He3","n"]` →
  `["H2","H2","TO","He3","n"]` via the alias map at `nuclear.py:183-195`),
  using a greedy left-to-right match against the known-nuclide token list.
  The `"TO"` token **already marks the reactant/product split** in the name
  itself — `reaction_stoichiometry`'s current use of `beta` (detailed-balance
  exponent) to find the split point is needed only for the *existing*
  catalog-driven path, not for parsing a fresh name.
- `generate_rates/nuclide_table.py` has `build_nuclide_table` and
  `conservation_residual` (and `is_decay`, `resolve_token`) which check A/Z
  conservation given reactant/product nuclide dicts — these are exactly the
  functions to reuse for validating an auto-derived stoichiometry.

**Proposed change**:
1. In `nuclear.py`, add a fallback path in `reaction_stoichiometry(name)`:
   if `name` is not found in the `detailed_balance.csv` catalog, call
   `_tokenise(name)`, split the token list at the `"TO"` token into
   reactants/products, count multiplicities on each side
   (`collections.Counter`), and return the two dicts — exactly like the
   catalog path's final step, but driven by the `"TO"` position instead of
   `beta`.
2. Validate the auto-derived stoichiometry with a conservation check: sum
   `A_s` and `Z_s` (from `nuclides.csv`, already loaded by
   `_reaction_catalog()`) over reactants must equal the sum over products
   (mirroring `conservation_residual` in `generate_rates/nuclide_table.py`
   — import/reuse it, or port the small residual computation directly into
   `nuclear.py` to avoid a `pyprimat` ↔ `generate_rates` import). On mismatch,
   raise a clear `ValueError` naming the reaction and the A/Z imbalance,
   instead of a cryptic `KeyError`.
3. **Duplicate-reaction check**: when adding a reaction to a custom network
   (in `load_network()` / wherever network-file lines are parsed, around
   `nuclear.py:1001` and the network-file reading loop), check the reaction
   name (and/or its derived stoichiometry, to also catch a reaction written
   with a different but equivalent token order) against reactions already
   present in the network being built; raise a clear `ValueError` ("reaction
   X is already present in network Y") rather than silently double-counting.
4. Update the CLAUDE.md "extension recipe" (IDEAS.md §6.4 already flagged this
   as undocumented): "(1) add `rates/nuclear/tables/<name>.txt`, (2) add the
   line to a network file, (3) stoichiometry is now auto-derived from the
   reaction name and validated for A/Z conservation; if the name can't be
   tokenised, add a row to `reactions_large.csv`/`detailed_balance.csv`
   manually."

**Validation**: add tests in `tests/test_nuclear.py` (or similar): (a) a
reaction name not in `detailed_balance.csv` but composed of known tokens
(e.g. a synthetic `"He3He3TOHe4pp"`-style name) auto-derives correct
stoichiometry and passes conservation; (b) an unbalanced synthetic name
raises `ValueError` with a clear message; (c) adding a duplicate reaction to
a custom network file raises `ValueError`.

**Model**: Sonnet.

---

## Item 1 — Configurable NEVO file names and y-grid; fingerprint update

**Current state** (`pyprimat/neutrino_history.py`):
- The 6/7-column NEVO table path is **hard-coded**:
  `cfg.data_dir + "/rates/NEVO/" + ("NEVOPRIMAT_col_1_7.csv" |
  "NEVOPRIMAT_NoQED_col_1_7.csv")` (lines ~118-120), selected only by
  `cfg.QED_corrections`.
- The 86-column spectral-distortion file (`"NEVOPRIMAT.csv"` /
  `"NEVOPRIMAT_NoQED.csv"`, lines ~187-189) and the y-grid file
  (`"NEVOGrid.csv"`, line ~190, an 80-node grid) are likewise hard-coded
  package paths.
- None of these are user-configurable via `PyPRConfig`.
- The weak-rate fingerprint (`_BACKGROUND_FINGERPRINT_FIELDS`,
  `weak_rates.py:117-128`) does not reference any NEVO file identity — only
  flags like `incomplete_decoupling`, `QED_corrections`.

**Proposed change**:
1. Add new `PyPRConfig` parameters (with the current hard-coded filenames as
   defaults, so existing behaviour is unchanged):
   - `nevo_file: str | None = None` — override for the 6/7-col NEVO table
     filename (relative to `rates/NEVO/`); `None` → current
     `QED_corrections`-based default selection.
   - `nevo_spectral_file: str | None = None` — override for the 86-col file.
   - `nevo_grid_file: str | None = None` — override for the y-grid
     (`NEVOGrid.csv`).
2. Thread these through `neutrino_history.py`'s loading functions (replace
   the hard-coded literals with `cfg.nevo_file or <default-based-on-QED>`,
   etc.).
3. **Fingerprint**: add the *resolved* filenames (or, better, a content hash
   of the resolved files — reuse `fingerprint_hash` from `cache_utils.py` on
   the file bytes/mtime) to `_BACKGROUND_FINGERPRINT_FIELDS` so that pointing
   at a custom NEVO file correctly invalidates the weak-rate cache. Since
   these are new optional fields with `None`/default-derived values for
   existing configs, this is backward compatible (existing cached files get a
   new fingerprint on first load with the new code and recompute once —
   acceptable, document it).
4. Validate in `PyPRConfig.__init__`: if a custom file is given, check it
   exists under `rates/NEVO/` (or an absolute path) and has the expected
   number of columns (6/7 for `nevo_file`, 86 for `nevo_spectral_file`,
   matching the y-grid length for `nevo_grid_file`), raising a clear error
   early rather than a confusing shape-mismatch deep in `neutrino_history.py`.
5. Document the three new parameters in `config.py` (per Item 10's "brief,
   user-facing" style) and in CLAUDE.md/README under "advanced: custom NEVO
   tables".

**Validation**: `pytest -m reference` unchanged (defaults preserved). New
test: construct `PyPR` with `nevo_file` pointing at a copy of the default
file under a different name, confirm identical results to the default, and
confirm the fingerprint differs from the default-file run (cache miss is
correctly triggered).

**Model**: Sonnet.

---

## Item 14 — MC uncertainty: vary `tau_n`; quick 30-point MC in the GUI

**Current state**:
- `config.py:116-118` already has `tau_n=878.4` and `std_tau_n=0.5` (1σ, in
  seconds) — the latter is currently **unused**.
- `mc_uncertainty(num_mc, quantity, params=None, n_jobs=-1, seed=0)`
  (`main.py:1068-1146`) only varies `p_<reaction>` rate-offset parameters,
  each drawn `~ N(0,1)` per sample (line ~1060), then run via
  `_mc_run_batch`.
- `tau_n` enters via `_NormWeakRates = 1/(Fn * cfg.tau_n)` (`main.py:431-436`)
  — a simple multiplicative normalisation, cheap to vary per-sample (does
  **not** require recomputing the weak-rate integrals/cache, since `tau_n` is
  excluded from the weak-rate fingerprint by design — see Item 15).
- GUI (`pyprimat/gui/params_form.py`): sidebar form built from
  `_FORM_METADATA` (lines ~49-127) + `_CONDITIONAL` dict (~151-156); results
  shown via `panels.render_results_panel()` (point values only, no
  uncertainty).

**Proposed change**:
1. In `mc_uncertainty`/`_mc_run_batch`, in addition to sampling `p_<reaction>
   ~ N(0,1)`, sample `tau_n_sample = cfg.tau_n + std_tau_n * randn()` per MC
   draw and pass `tau_n=tau_n_sample` in that sample's `params` dict. Use the
   same `seed`-derived RNG stream (don't add a second independent RNG that
   could desync reproducibility — draw `tau_n` from the same per-sample
   `Generator` as the rate offsets, e.g. one extra `randn()` call per sample,
   documented order).
2. GUI: add a boolean `quick_mc_uncertainty` (default `False`) to
   `_FORM_METADATA`, with help text like "Run a quick 30-sample Monte Carlo to
   estimate ±1σ uncertainty on the displayed observables (rates + τ_n
   varied)." When enabled, after the main `_solve()`, call
   `mc_uncertainty(30, ["YPBBN","DoH","Li7oH","Neff",...], params=current_params,
   seed=0)` (30 points per the user's request — fast enough for interactive
   use) and display mean±std next to each observable in
   `render_results_panel()` (e.g. an extra "±1σ (30-pt MC)" column).
3. Document that 30 points gives a rough/noisy estimate (compared to e.g. 500
   in `MonteCarloRates.ipynb`) — label it "quick estimate" in the UI.

**Validation**: unit test that `mc_uncertainty` output's `tau_n`-driven spread
in, e.g., `Neff` or the n/p freeze-out abundance is non-zero and of plausible
magnitude (sanity bound, not a tight reference). GUI: manual check via
`/run` skill that the toggle works and doesn't crash for small/medium/large
networks.

**Model**: Sonnet.

---

## Item 6 — Neff must include spectral-distortion energy density

**Current state** (`main.py`):
- `N_eff(Tg, Tnue, Tnumu, Tnutau)` (`main.py:391-394`):
  ```python
  rho_rad = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau) \
            + rho_g + thermo.rho_nu_extra(Tg)
  return (rho_rad - rho_g) / rho_g / ((7/8)*(4/11)**(4/3))
  ```
  This uses only the **baseline Fermi-Dirac** neutrino energy densities
  `rho_nu(Tnu_alpha)`.
- Separately, `_Hubble` (lines ~178-186) **does** add `self._rho_nu_SD` (the
  extra energy density from analytic spectral distortions, computed in
  `AnalyticDistortion._rho_nu_SD(Tnu)`, `neutrino_history.py:465-477`) to
  `rho_tot` when `self._rho_nu_SD is not None`.
- **Gap**: spectral-distortion energy density correctly affects the expansion
  rate (via `_Hubble`) but is *not* reflected in the reported `Neff`.

**Proposed change**:
1. In `N_eff(...)` (or wherever it's evaluated for the output, `main.py`
   around the results assembly), add the same `self._rho_nu_SD(Tnu)` term (if
   not `None`) to `rho_rad`, mirroring `_Hubble`'s treatment:
   ```python
   rho_rad = rho_g + sum(thermo.rho_nu(Tnu_a) for Tnu_a in (Tnue,Tnumu,Tnutau)) \
             + thermo.rho_nu_extra(Tg)
   if self._rho_nu_SD is not None:
       rho_rad += sum(self._rho_nu_SD(Tnu_a) for Tnu_a in (Tnue,Tnumu,Tnutau))
   ```
   — **check carefully** whether `_rho_nu_SD` in `_Hubble` is already summed
   over the three flavours or is a single aggregate; match exactly so Neff and
   the Hubble rate use the *same* total extra energy density (consistency is
   the main correctness requirement here — an inconsistency between the
   energy density that drives expansion and the one reported as Neff would be
   a worse bug than the current omission).
2. This only changes results when `spectral_distortions=True` (and
   `analytic_distortions=True`, since `_rho_nu_SD` is currently only set by
   `AnalyticDistortion`) — the default-flags reference run
   (`spectral_distortions=False`) is **unaffected**, so `pytest -m reference`
   and the CLAUDE.md table stay valid as the regression gate.
3. Add a new test: with `spectral_distortions=True, analytic_distortions=True,
   y_SZ=<nonzero>`, assert `Neff` differs from the `y_SZ=0` case by an amount
   consistent with `_rho_nu_SD`'s contribution (compute the expected ΔNeff
   analytically from `_rho_nu_SD` and `rho_g` and compare to the simulated
   ΔNeff to the precision required by CLAUDE.md, ~1e-8).

**Model**: Sonnet (the formula addition is small and localised; the existing
`_rho_nu_SD` machinery from `AnalyticDistortion` already does the hard physics
— this item just plugs an existing, tested quantity into a second formula).
Flag for a quick Opus/physicist sanity check of the *consistency* point in
step 1 if any ambiguity remains after reading `_Hubble`'s exact usage.

---

# Phase 3 — Opus (physics judgment, literature, interface design) — run last

## Items 11+12+13 — weak-rate physics: citations, missing CCR term, thermal-correction documentation

### 11. Citation check: "Frenkel–Galitskii–Migdal" / "Blaizot–Zinn-Justin"

**Current state**: These names do **not** appear in `weak_rates.py` (where
the user expected them). They appear in
**`pyprimat/qed_pressure.py:23` and `:26`**, attached to the QED
plasma-pressure corrections:
- Line 23: `δP_a(T) = (α/π) T⁴ [...]` — "Leading O(α) one-loop correction
  (Frenkel–Galitskii–Migdal)."
- Line 26: `δP_{e3}(T) = α^{3/2} (4/3)√(2π) T⁴ [...]` — "O(α^{3/2})
  ring/plasmon contribution (Blaizot–Zinn-Justin)."

**Task**: Verify whether these attributions are correct, or hallucinated.
- "Galitskii–Migdal" is a real, well-known result in many-body theory (the
  Galitskii–Migdal energy/sum-rule relation); whether "Frenkel" belongs in
  the name for *this specific* O(α) QED-plasma pressure term needs checking
  against the literature (e.g. Frenkel's name is associated with some QED
  finite-T plasma works, but confirm).
- "Blaizot–Zinn-Justin" — both are real thermal-field-theory authors; check
  whether the O(α^{3/2}) ring/plasmon (Debye-screening) result for the QED
  pressure is conventionally attributed to them, or to e.g. Shuryak / a more
  standard "ring diagram resummation" reference.
- Cross-check against `doc/Pitrou_etal_PhysReptArxivVersion.pdf` §II.E and
  `generate_rates/PRIMAT-Main.m` — do either cite a primary source for δP_a /
  δP_e3? If PRIMAT-Main.m or the Phys. Rep. paper cite specific papers, use
  those as the primary reference (per CLAUDE.md's citation policy) and either
  confirm or replace the author-name shorthand in `qed_pressure.py:23,26`
  with a proper citation (paper + equation number).
- If after checking the names are correct, simply add the proper
  paper reference (consistent with CLAUDE.md's citation style) next to them;
  if incorrect/unverifiable, replace with whatever the Phys. Rep. paper /
  PRIMAT-Main.m actually cites, or remove the attribution and just keep the
  equation description.

### 12. Add missing CCR term to `_L_SD`

**Current state**: `_L_SD` (`weak_rates.py:850-876`) computes only the
**Born-level** spectral-distortion correction (`δχ` from the deviation
`dFDneu_func`, no CCR multiplier). `PRIMAT-Main.m` defines two related
quantities:
- `IPENdpSD` (line ~1472): SD without CCR — what `_L_SD` currently implements.
- `IPENdpSDCCR` (line ~1473): SD **with** the CCR (charged-current radiative)
  correction multiplier applied — i.e. `IPENdpFrom\[Chi]CCR` rather than
  `IPENdpFrom\[Chi]NoCCR`.

So `_L_SD` is missing the CCR-weighted version that PRIMAT-Main.m uses for the
full spectral-distortion contribution.

**Task**:
1. Read the `IPENdpSDCCR` definition in `PRIMAT-Main.m` (and its dependencies
   `IPENdpFrom\[Chi]CCR` / `IPENdpFrom\[Chi]NoCCR`) and the corresponding
   Phys. Rep. equations (likely near Eq. 81 and the CCR equations around
   Eqs. ~70-80, §III/IV — cross-check `doc/PhysReptRevised.tex`).
2. Determine the correct way to combine the CCR multiplier with the
   spectral-distortion `δχ` term — i.e. should `_L_SD` multiply `delta_chi`
   by the same CCR factor `(1 + α·...)` that the Born-level rate uses
   elsewhere in `weak_rates.py` (search for where the CCR factor is applied to
   the main Born rate, to reuse the identical expression/interpolant)?
3. Implement `_L_SD` (or add a parallel `_L_SD_CCR` and decide which is used
   by `ComputeWeakRates` when `spectral_distortions=True`) so that the
   spectral-distortion contribution matches `IPENdpSDCCR`, with full
   docstring/equation citations per CLAUDE.md.
4. **Precision validation**: per CLAUDE.md, this changes results at the
   `spectral_distortions=True` level only (default reference run unaffected).
   Add/extend a test comparing `Neff`/`YPBBN`/`D/H` with
   `spectral_distortions=True, analytic_distortions=True, delta_xi_nu=<val>,
   y_SZ=<val>` before/after the fix, and (if feasible) cross-check against a
   PRIMAT-Main.m numeric evaluation of `IPENdpSDCCR` at a few `(T, y_SZ,
   delta_xi_nu)` points to confirm the new term's magnitude is correct (not
   just "non-zero").

### 13. Document `_L_CCRTh_interpolants` with LaTeX equations; column titles

**Current state**: `_L_CCRTh_interpolants` (`weak_rates.py:884-1202`) computes
finite-temperature radiative corrections (Brown & Sawyer 2001-style), citing
"Physics Reports §III.H, Eqs. 107-113" at a high level, but the nested
integrand functions `IPENCCRT`, `IPENCCRDiffBremsstrahlung` and helper
`A`, `B`, `Chitilde_vec`, `FD2_vec` lack per-formula LaTeX/equation
cross-references. `doc/PhysReptRevised.tex` has labelled equations
`\label{CCRn}` (~line 2144-2148) and `\label{CCRp}` (~line 2188-2192) for the
T=0 CCR, plus a thermal section (`DefGammaT`, `BSnpFormal`) — these need to be
matched to the corresponding code blocks.

**Task**:
1. For each nested function/integrand in `_L_CCRTh_interpolants`, add a
   docstring block transcribing the corresponding formula in LaTeX (as a
   raw-string comment, e.g. `r"""  L^{CCRTh}_{n\to p}(T) = ... """`), with the
   exact equation number/label from `doc/PhysReptRevised.tex` (use the `.tex`
   source — it's available precisely so equation numbers can be checked
   without OCR/PDF guesswork) and/or `doc/Pitrou_etal_PhysReptArxivVersion.pdf`
   §III.H.
2. **`nTOp_thermal_corrections` / `pTOn_thermal_corrections` column titles**
   (`weak_rates.py:1173-1202`, files `rates/weak/{nTOp,pTOn}_thermal_corrections.txt`):
   the current header is just `"T[K] L_nTOpCCRTh"` / `"T[K] L_pTOnCCRTh"`.
   Expand to clarify units and meaning, e.g.
   `"T[K]  L_nTOpCCRTh[dimensionless, additive correction to chi_n->p]"`
   (confirm the actual units/normalisation of `L_*CCRTh` from the formula
   derived in step 1 before finalising the header text). *(This sub-part is
   mechanical and could be done by Sonnet once the equation/units are pinned
   down by the rest of item 13 — but bundling it here avoids a second
   round-trip since the units must be confirmed from the same formula
   analysis.)*

**Model for 11+12+13**: **Opus**. All three require reading
`doc/PhysReptRevised.tex`/the Phys. Rep. PDF and `PRIMAT-Main.m` carefully,
matching formulas to code, and (for #12) a numerically-sensitive physics
change at the `spectral_distortions=True` precision level — exactly the kind
of "physics-numerics judgment" IDEAS.md §10 reserves for Opus/Fable.
Recommend doing #11 (citation check) and #13 (documentation) first since they
are read-only/docs-only and will surface the formulas needed for #12's
implementation.

---

## Item 2 — Analytic-distortion config + gray-body distortions notebook

**Current state**:
- Distortion parameters are individual scalar flags in `config.py:52-73`:
  `spectral_distortions`, `analytic_distortions`, `delta_xi_nu`, `y_SZ`. No
  dict-based "distortion model" mechanism exists.
- `AnalyticDistortion` (`neutrino_history.py:345-477`) is a decorator wrapping
  the base `NeutrinoHistory`, overriding `dFDneu_func` (weak-rate correction,
  used by `_L_SD`/Item 12) and `rho_nu_SD` (extra energy density, used by
  `_Hubble` and now also `Neff`, Item 6). `_dFDneu_analytic`
  (lines ~396-427) implements μ-type (`delta_xi_nu`) and y-type/SZ (`y_SZ`)
  distortions; `_rho_nu_SD` (~465-477) gives closed-form energy-density
  integrals (`Inty3Mu`, `Inty3SZ` from PRIMAT-Main.m).
- `doc/GrayBody_2504.07178v2.pdf` (Barenboim, Froustey, Pitrou, Sanchis,
  arXiv:2504.07178) defines a "gray-body" distortion (Eq. 8):
  `f_ν(x, y_g) = (y_g+1)^{-3} \hat f_ν(x/(y_g+1))`, with `\hat f_ν(x) =
  1/(e^x+1)`, valid for `y_g > -1` (generalising the SZ-type distortion to
  negative amplitudes), and energy-density change `δρ_ν/ρ̂_ν = y_g` (Eq. 9).
- `notebooks/AbundancesXi.ipynb` is the closest existing precedent (parameter
  scan over `munuOverTnu`).

**Task** (two parts):

1. **Design question — should distortions be config'd via a dict?**
   Recommendation: yes, but additively, not as a breaking change. Add an
   optional `distortion_model: str = "mu_y"` config key (`"mu_y"` = current
   μ-type + y_SZ behaviour, default; `"gray"` = new gray-body model using a
   single amplitude `y_gray` per Eq. 8/9 above). Keep `delta_xi_nu`/`y_SZ` as
   the parameters for `"mu_y"`; add `y_gray: float = 0.0` for `"gray"`. This
   avoids breaking `delta_xi_nu`/`y_SZ`-based notebooks/configs while giving a
   clean switch for the new model — a small, additive interface change (unlike
   Item 3, which is a bigger interface redesign). `AnalyticDistortion` gains a
   `model` parameter dispatching `_dFDneu_analytic`/`_rho_nu_SD` to either the
   existing μ/y formulas or new gray-body formulas (Eq. 8/9, derived in part 2
   below). Validate in `__init__`: `distortion_model in {"mu_y","gray"}`, and
   `y_gray > -1` per the paper's validity bound.

2. **Notebook**: write `notebooks/GrayBodyDistortions.ipynb` (modeled on
   `AbundancesXi.ipynb`'s parameter-scan structure) that:
   - Summarises the gray-body model from `doc/GrayBody_2504.07178v2.pdf`
     (Eq. 8/9), with the derivation of `_dFDneu` (the deviation from FD at
     fixed `(x, znu)`, needed by `_L_SD`/`ComputeWeakRates`) and the closed-form
     `_rho_nu_SD_gray(Tnu)` (energy-density integral giving exactly
     `δρ_ν/ρ̂_ν = y_g`) — derive these symbolically/numerically in the
     notebook (e.g. with `sympy` or numeric integration) as a worked
     reference *before* they're hardened into `neutrino_history.py`.
   - Demonstrates a parameter scan over `y_gray` (including negative values,
     the paper's key new regime) showing `Neff`, `YPBBN`, `D/H` vs `y_gray`,
     using `distortion_model="gray"` from part 1.
   - Cross-checks the closed-form `_rho_nu_SD_gray` against the formula
     `δρ_ν/ρ̂_ν = y_g` numerically (sanity check before relying on it in
     `_Hubble`/`Neff`).
   - This notebook can double as the **specification** for the
     `neutrino_history.py` implementation of part 1 — implement the gray-body
     branch of `AnalyticDistortion` directly from the notebook's derived
     expressions, then add it to `_BACKGROUND_FINGERPRINT_FIELDS` (`y_gray`,
     `distortion_model`) per the existing pattern (Item 1's fingerprint
     update is a good template).

**Validation**: default config (`distortion_model="mu_y"`, the implicit
current default) must reproduce `pytest -m reference` exactly (no behaviour
change for existing users). New tests for `distortion_model="gray"`: (a)
`y_gray=0` reproduces the undistorted (`spectral_distortions=False`) result;
(b) `_rho_nu_SD_gray` matches `δρ_ν/ρ̂_ν = y_g` to high precision; (c) Neff
shift (via Item 6's fix) matches the analytic `y_g`-driven ΔNeff.

**Model**: **Opus** — requires deriving new physics formulas from a paper and
designing the config/interface extension correctly (a wrong choice here, e.g.
overloading `y_SZ` instead of adding `distortion_model`, would create the kind
of "two flags doing similar things" confusion Item 9 is fixing elsewhere).
The notebook-as-specification approach lets Opus do the physics derivation in
a reviewable, runnable form before any package code changes.

---

## Item 3 — NEVO "minimal" vs "full external" background mode + extrapolation

**Current state**:
- `neutrino_history.py:118-121` loads `NEVOPRIMAT_col_1_7.csv` with columns
  `x=me/(kB Tg)`, `z=a*Tg`, `T_νe/T_νμ/T_ντ` ratios, and `N_NEVO` (heating).
  Note `z=a*Tg` already implicitly encodes `a` (given `Tg` from `x`); whether
  a `t` (cosmic time) column exists in this or the fuller 86-column
  `NEVOPRIMAT.csv` needs to be checked as part of this task — the user's
  premise ("NEVO files give a, Tg, t, heating, T_nu, spectra") should be
  verified against the actual CSV column count/headers before designing the
  "full external" mode.
- `_setup_background_and_cosmo` (`main.py:189-379`) implements exactly the
  "minimal setup" the user describes: it does **not** use any `a`/`t` columns
  directly; it recomputes `a(T)` from the NEVO heating `N(Tg)` + plasma
  entropy via an ODE (`d(ln a)/d(ln T) = -[3 s̄ + T ds̄/dT]/[N + 3s̄]`,
  lines ~250-307), then `T_of_a` by inversion (~309-318), then `t(a)` by
  Hubble-integration (~321-349) using `T_νe/μ/τ(Tg)` from NEVO.
- **Extrapolation**: `T_να/Tγ` ratios use constant extrapolation at table
  edges (`fill_value=(_ratio[-1], _ratio[0])`, correct physics — both
  asymptotic regimes are radiation-like with constant ratios). `N_NEVO(Tg)`
  uses **`fill_value=(0.,0.)`** outside the table — i.e. heating is assumed
  zero outside the tabulated range, with no radiation-domination correction.
- This item directly corresponds to **`IDEAS.md` §6.2**, which is *explicitly
  reserved for Fable/Opus, last phase*: "the value of this item is in choosing
  the interface correctly... a wrong cut here propagates into every future
  extension."

**Task**:
1. **Verify the NEVO file contents**: inspect `rates/NEVO/NEVOPRIMAT_col_1_7.csv`
   and `rates/NEVO/NEVOPRIMAT.csv` (86-col) headers/column count to determine
   whether `a`, `t` are actually present as columns (directly or derivable,
   e.g. `a` from `z=a*Tg`). Document the actual column layout (this
   contradicts or confirms the user's "in principle" premise).
2. **Design a `NeutrinoHistory` mode switch** (per IDEAS §6.2's recommended
   `NeutrinoHistory` protocol with `NEVOTable`/`InstantaneousDecoupling`
   implementations):
   - `background_mode: "minimal"` (default, current behaviour): recompute
     `a(T)`, `t(a)` from `N_NEVO(Tg)` + entropy + Hubble integration, using
     only `T_να(Tg)` and `N_NEVO(Tg)` from the NEVO file — exactly today's
     code. Document this as "the least-wrong fallback for users who have
     `T_ν(T_γ)` and heating `N` from their own exotic-physics code but no
     full background solver."
   - `background_mode: "full_external"`: directly use the NEVO-provided `a`,
     `Tg`, `t`, `T_να` (and spectral distortions if `spectral_distortions=True`
     and the table provides them) via interpolation — **no** ODE re-solve for
     `a(T)`/`t(a)`. Only viable if step 1 confirms these columns exist (if not,
     this mode may need the 86-column file or a different/extended NEVO file
     format — flag this as a possible blocker/scope reduction).
3. **Out-of-range extrapolation rule** (applies to both modes, replacing the
   current `N_NEVO -> 0` fill): outside the NEVO table's T-range, assume a
   purely radiation-dominated universe, `rho_tot ∝ a^{-4}` (equivalently
   `T ∝ 1/a` exactly, `N_NEVO -> const` rather than `0`, since `N` represents
   a fractional heating rate that should asymptote, not vanish — derive the
   correct asymptotic value/slope from the entropy-conservation ODE itself:
   what value of `N` makes `d(ln a)/d(ln T) = -1` exactly, the standard
   radiation-domination relation?). Implement as an extrapolation branch in
   the `N_NEVO_of_Tg` interpolant (`neutrino_history.py:170-171`) and
   correspondingly in `full_external` mode's `a(T)`/`t(a)` interpolants
   (extend with `a ∝ 1/T`, `t ∝ a^2` analytically beyond the table).
4. Add `background_mode` to `_BACKGROUND_FINGERPRINT_FIELDS` (Item 1's
   fingerprint pattern) since it changes the weak-rate integration's `Tg`
   history in `full_external` mode.

**Validation**: `background_mode="minimal"` with default NEVO files must be
**bit-identical** to current behaviour (it *is* current behaviour — this is a
refactor + extrapolation-rule addition, not a behaviour change for the
default path, except possibly at the extreme edges of the T-grid where the
old `N->0` vs new `N->radiation-domination-value` differ — check whether
`T_start_cosmo_MeV`/`T_end` are within the NEVO table range by default; if so,
the extrapolation branch is never exercised by the reference run and
`pytest -m reference` is unaffected). `background_mode="full_external"` (if
feasible per step 1) should reproduce `minimal` to within the tolerances of
CLAUDE.md's table when using the *same* NEVO file (cross-validation that the
two computation paths agree).

**Model**: **Opus or Fable**, last item, per IDEAS.md §6.2's explicit
reservation — interface design (the `NeutrinoHistory` protocol/mode split)
and the radiation-domination extrapolation derivation both require physics +
architecture judgment where a wrong choice "propagates into every future
extension."
