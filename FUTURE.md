# FUTURE.md — improvement plan for PyPRIMAT

A prioritised, actionable plan to make the code **more robust, readable,
flexible, simple, modular and fast** — while keeping the physics transparent
and easy to extend. Each item cites concrete evidence (`file:line`) and states
the *why*, the *change*, and the *payoff*.

## Baseline (what is already good — keep it)

- Clean **Class 1 / Class 2 split**: `background.py` (cosmology) drives
  `nuclear_network.py` (ODE network) through a *minimal* documented interface
  (`T_of_t`, `t_of_T`, `rhoB_BBN`, `weak_nTOp_*`). This is the single best
  structural decision in the code — preserve it.
- **Data-driven, stoichiometry-compiled network** (`network_builder.py`):
  one generic RHS/Jacobian kernel serves small/large/amax networks; new
  reactions are added by dropping a rate table + a network-file line.
- **Fingerprinted caches** for weak rates / electron thermo / QED tables.
- **Heavily commented**, physics-first docstrings; a tiered test suite
  (178 fast tests pass) with `slow`/`solve`/`reference`/`gui`/`notebook` markers.

The items below are improvements, not rescue work.

---

## P0 — Correctness & hygiene (do first; small, high-value)

### 0.1 Three tests are currently RED from stale hard-coded pins
`tests/test_cli.py::test_cli_default_summary`, `::test_cli_json_matches_default_summary`
and `tests/test_gui.py::test_default_run_matches_cli_reference` assert **exact
8-decimal** observables (`YP=0.24699914`, `DoH≈2.4349992e-5`, `rel=1e-8`). After
commit `e00f062` bumped `rate_grid_npts` 500→1000 and
`sampling_temperature_per_decade` 400→600, the actual values shifted to
`0.24699899…` — a **~1.5e-7 drift, far inside the documented ±1e-5 / ±3e-9
tolerances**, yet the suite is red.

- **Why it matters:** "does it run as intended?" currently answers *no* on a
  clean checkout, purely because of test brittleness, not physics. Every
  default tweak forces a manual pin refresh.
- **Change:** assert against the **documented tolerances** with
  `pytest.approx(expected, abs=1e-5)` (YP/Neff), `abs=3e-9` (DoH), instead of
  string equality / `rel=1e-8`. Centralise the reference numbers in **one**
  fixture (e.g. `tests/reference_values.py`) consumed by CLI + GUI + reference
  tests, so there is a single place to update.
- **Bonus:** rewrite the GUI "parity" test to **compute the CLI result in-process
  and compare GUI == CLI**, not GUI == literal. Then it tests parity (its stated
  purpose) and can never go stale.

### 0.2 Writable cache pollutes the package source tree
44 untracked `pyprimat/rates/weak/nTOp_*.txt` / `nTOp_thermal_*.txt` cache files
accumulate **inside the package** and are not in `.gitignore`. Each new config
fingerprint drops another file.

- **Why it matters:** untracked-file noise, accidental commits, package bloat,
  and a non-reproducible source tree.
- **Change (pick one):**
  1. Move the writable cache to a per-user dir (`platformdirs.user_cache_dir`)
     and ship only a tiny canonical set read-only with the package; **or**
  2. Keep in-tree but add `pyprimat/rates/weak/nTOp_*` to `.gitignore` (ship a
     small committed seed set) and add a `pyprimat cache --clear` command + an
     LRU cap.
- Also remove/relocate root scratch scripts (`grid_npts_timing.py`,
  `precision_study.py`, `generate_rates/PRIMAT-Main-gray.m`) into a
  `scratch/`/`studies/` dir (gitignored) or delete them.

### 0.3 Defensive `getattr(cfg, …, default)` on guaranteed fields
~20 sites use e.g. `getattr(cfg, "decay_era", False)`,
`getattr(cfg, "t_decay_end", 3.156e16)` (`nuclear_network.py:365-369`) for keys
that **always exist** in `DEFAULT_PARAMS`.

- **Why it matters:** it hides the config contract and silently swallows typos
  / renames (a renamed key keeps returning the default forever).
- **Change:** access `cfg.decay_era` directly. Reserve `getattr` for genuinely
  optional, non-`DEFAULT_PARAMS` attributes.

---

## P1 — Modularity & readability (the two giant modules)

### 1.1 Split `weak_rates.py` (1834 lines) and kill the scalar/`_v` duplication
There are ~20 Fermi–Dirac integrand functions, each present **twice** — a
scalar `FD_nu3` *and* a vectorised `_FD_nu3_v` (and `RadCorrResum` /
`_RadCorrResum_v`, etc.). Two hand-maintained copies of every kernel must be
kept bit-identical.

- **Change:** write each integrand **once** as an array-aware function and
  derive the scalar path from it (or JIT both from one source with numba). Then
  split the module into `weak_rates/` subpackage:
  `integrands.py` (FD kernels), `corrections.py` (Born/CCR/FM/SD/thermal
  `_L_*` terms), `cache.py` (fingerprint + `RecomputeWeakRates`), `api.py`.
- **Payoff:** removes the single biggest "two-things-in-sync" hazard; each
  physical correction becomes independently readable and testable.

### 1.2 Decompose `network_data.load_network` (~400 lines, 1255-1655)
One function parses the network file, resolves alternate tables, applies QED
rescaling, resamples onto the master grid, and compiles.

- **Change:** extract named steps — `parse_network_file` → `resolve_tables` →
  `apply_nuclear_qed` → `resample_to_master_grid` → `compile`. Each gets a
  docstring and a focused unit test.
- **Payoff:** the "add a reaction / new rate source" path becomes followable
  end-to-end — directly serving the goal of easy physics extension.

### 1.3 Mark the reference RHS/Jacobian as test-only
`network_data.network_rhs` / `network_jacobian` duplicate the production kernels
in `network_builder.py` and are used **only** by `test_network_builder.py` as an
independent oracle (good practice).

- **Change:** move them into `tests/` (or a `tests/_oracles.py`) and drop them
  from `network_data.__all__`, so the public API has one obvious kernel path.

---

## P2 — Robustness of the GUI workflow

The GUI runs (GUI tests pass), but `params_form.py` (1329 lines, ~40 functions,
**104** `st.session_state` references, manual `_bump_dialog_gen` / `_tabs_gen`
remount hacks) is the most fragile area, concentrated in the custom-network
dialog.

- **Change:**
  - Centralise every session-state key in one `SESSION_KEYS` namespace/dataclass
    (no more scattered string literals).
  - Extract the custom-network dialog into a small **state controller** object
    (build/import/edit/export) that the view renders, separating "what the
    network is" from "how Streamlit shows it".
  - Document the Streamlit key-remount pattern **once** (it currently recurs as
    inline lore in several docstrings).
- **Add coverage** for the under-tested workflow branches: import a `.zip` →
  edit → export round-trip; `amax` filter applied to a custom base network;
  invalid uploaded rate table → clean `st.error`.

---

## P3 — Flexibility for new physics (the stated end-goal)

### 3.1 Formalise a `PyPR(..., background=…)` injection hook
`background.py` already documents a *future* `PyPR(..., background=...)` seam,
and `Background` is a clean base class (NotImplementedError stubs, not `abc`).
Today only `extra_rho` and a file-based `custom_background` are wired in.

- **Change:** let `PyPR.__init__` accept a `background=` instance (falling back
  to `StandardBackground`). This turns "non-standard expansion history" into a
  ~20-line subclass instead of an internal edit.

### 3.2 Retire the module-level mutable plasma singleton
`plasma.initialise()` sets module globals and is explicitly documented as unsafe
under concurrency; `main.py` already uses a **per-instance** `Plasma`. The global
is now legacy.

- **Change:** remove the module-level state once nothing depends on it (grep
  shows the per-instance path is the live one). Removes a real footgun for MC /
  parallel runs.

### 3.3 Name the physics magic numbers
`Neff = 3.044` is hard-coded in `_setup_EDE` and the `_replace_LCDM_with_exact`
sanity check (`background.py:507,546`). Promote to a single
`CONST.Neff_SM = 3.0440` (or `cfg`) with the citation, so a "what if Neff_SM
changes" study is a one-line edit.

### 3.4 Add a short "Adding new physics" guide
The erased design docs left a gap. Add a concise `doc/EXTENDING.md` (≤2 pages):
(a) add a reaction, (b) add a background/dark-sector component via `extra_rho` or
`background=`, (c) add a neutrino-history variant via `make_neutrino_history`.
Cite the **published paper** for provenance — not volatile internal planning docs
(the recently-removed `GUI.md`/`NEUTRINOS.md` references showed how those rot).

---

## P4 — Speed (only after the above; measure first)

Current default solve ≈ 2.6 s (large, amax=8); not a pain point. Before
optimising, add a tiny `studies/profile_solve.py` to attribute time across
HT/MT/LT + background + weak-rate setup. Likely candidates once measured:

- **Weak-rate thermal correction** (`vegas`/`dblquad`) dominates first-run
  cost; ensure the cache is the default hot path and the integral accuracy
  knobs (`vegas_n_eval`, `epsrel_thermal`) are tuned to the needed precision.
- **Jacobian sparsity:** the large-network dense `(n_sp, n_sp)` Jacobian
  (`_jac_kernel`) could move to a sparse layout for the ~59-nuclide network.
- Confirm numba `cache=True` kernels are actually being reused across runs.

---

## Suggested sequencing

1. **P0.1 + P0.2** (make the suite green, de-clutter the tree) — an afternoon.
2. **P0.3 + P1.3** (config hygiene, oracle relocation) — small.
3. **P1.1 / P1.2** (split the two giants) — the big readability win; do as
   separate, test-guarded PRs with no numerical change (assert bit-identical
   observables before/after).
4. **P2** (GUI controller) and **P3** (extensibility hooks) — feature-level.
5. **P4** (speed) — last, measurement-driven.

**Invariant for every refactor:** the CLAUDE.md reference values
(`YP`, `D/H`, per-nuclide `Y`) must stay within tolerance — wrap each PR with a
before/after `PyPRIMAT_run.py` diff.
