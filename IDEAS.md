# IDEAS.md — Critical review and improvement plan for PyPRIMAT

Date: 2026-06-10.  Produced from a full read of `pyprimat/`, the test suite,
`runfiles/`, `notebooks/`, the packaging metadata, and spot-checks against
`generate_rates/PRIMAT-Main.m` and `doc/Pitrou_etal_PhysReptArxivVersion.pdf`.
Items are ordered by priority inside each section.  Items marked **[BUG]** were
verified by actually running the code; items marked **[FIXED in this pass]**
have already been applied.

---

## 1. Verified bugs and regressions

### 1.1 [BUG][FIXED in this pass] `rates/` path broken by the package reorganisation
`pyprimat/nuclear.py::_network_dir_from_cwd` still pointed to `../rates`
(the old repo-root location) after commit 726f755 moved `rates/` inside the
package.  Consequence: `_REACTIONS_MEDIUM` silently fell back to `[]`, so
`ORDER_LT == ["nTOp"]`, the per-reaction `p_*` dictionary in `PyPRConfig` was
initialised empty, and the dynamically attached `<rxn>_frwrd` logging methods
were never created.  Two tests (`test_p_rate_keys_count`,
`test_buffer_orders_have_expected_lengths`) caught it.  Fixed; both pass now.

**Lesson to act on:** the `try/except OSError → []` fallback at import time
(`nuclear.py:216-220`) is exactly the kind of silent degradation that turned a
rename into a latent bug.  Replace it with a hard failure (or at minimum a
loud warning).  A missing data directory is never a state in which the package
should "work".

### 1.2 [BUG] `spectral_distortions=True` is silently a no-op when `compute_nTOp=False`
With `compute_nTOp=False`, `tests/test_spectral_distortions.py::
test_spectral_distortions_effect` fails (diff == 0).  The shipped default is
`compute_nTOp=True`, which masks the trap — but any user who sets the flag to
False for speed walks straight into it.  Cause: `RecomputeWeakRates`
(`weak_rates.py:942`) ignores `dFDneu_func` entirely when loading
pre-tabulated rates from disk.  The same trap applies to *every* parameter
that affects the weak rates: `munuOverTnu`, `delta_xi_nu`, `y_SZ`,
`nTOp_Born_approximation`, `incomplete_decoupling`, `sampling_nTOp`, even
`DeltaNeff` through the background temperature history — all silently ignored
when rates are read from `rates/weak/*.txt`.

**Proposal (agreed design): a fingerprinted, self-validating cache.**

*Fingerprint contents.*  Collect every configuration entry that changes the
computed rates into one dict, serialise it as canonical JSON, and hash it
(sha256, truncated).  The relevant fields are:

| Field | Why it changes the rates |
|-------|--------------------------|
| `format_version` (int, bump on any change to the file layout or rate physics) | invalidates old caches after code changes |
| `sampling_nTOp` | grid density of the tabulation |
| `nTOp_Born_approximation` | Born vs fully corrected rates |
| `include_nTOp_thermal` (+ the identity/mtime of the thermal table used) | thermal radiative corrections on/off |
| `incomplete_decoupling`, `QED_corrections` | select the NEVO table → the Tν(Tγ) relation the rates are integrated over |
| `munuOverTnu` | reduced neutrino chemical potential enters the FD occupations |
| `spectral_distortions`, `analytic_distortions` | distortion mode: off / NEVO / analytic |
| `delta_xi_nu`, `y_SZ` | analytic-distortion amplitudes |
| `T_start_cosmo_MeV`, `n_temperature_table` | background grid the (Tg, Tν) vectors are sampled on |
| `tau_n_flag` is *not* included | τ_n only rescales the normalisation, applied after interpolation |

*File format.*  Keep the two human-readable text files, but write the
fingerprint and the full JSON dict as `#`-comment header lines of
`rates/weak/nTOp_frwrd.txt` / `nTOp_bkwrd.txt`.  The JSON line is for humans
("with which flags was this produced?"), the hash line is what the loader
compares.  Files without a header (the current ones) are treated as stale.

*Loader logic* (replaces the `compute_nTOp` branch in `RecomputeWeakRates`):

1. Compute the fingerprint from the current config.
2. If the cache file exists *and* its fingerprint matches → load.
3. Otherwise → recompute; if saving is enabled, overwrite the cache with the
   new header.  Print one line saying *why* it recomputed (no cache /
   fingerprint mismatch, with the differing keys listed).
4. **Forced recompute:** when `spectral_distortions=True` and
   `analytic_distortions=True`, skip the cache entirely (never load, never
   save).  Analytic distortions are by construction user-customised
   (`delta_xi_nu`, `y_SZ` are continuous knobs being scanned), so caching
   them invites stale hits and pollutes `rates/weak/` with one file per
   parameter point.  The same forced-recompute rule should apply to any
   future user-supplied `dFDneu_func` callable, which cannot be fingerprinted.

*Flag retirement.*  `compute_nTOp` and `save_nTOp` then disappear from the
user's mental load: the cache is always consistent, so loading is always
safe.  Keep only `save_nTOp` (default True for the standard SM
configuration, so the shipped tables stay fresh) and a
`weak_rate_cache=False` escape hatch for debugging.  Apply the identical
header/fingerprint scheme to the thermal-correction tables
(`compute/save_nTOp_thermal`) and the electron-thermo cache
(`recompute_electron_thermo`), which have the same staleness problem.

*Migration.*  Regenerate the shipped `rates/weak/*.txt` once with the
default-flag fingerprint, so a fresh clone/install hits the cache on the
default configuration and pays the ~2 s recomputation only when it changes
physics flags — which is exactly when it must.

### 1.3 [BUG] `mc_uncertainty` requires `joblib`, which is not declared anywhere
`main.py:1396` does `from joblib import Parallel, ...` but `pyproject.toml`
lists only `numpy`/`scipy` (+ optional `numba`, `vegas`).  A
fresh `pip install PyPRIMAT` user following `MonteCarloRates.ipynb` gets an
`ImportError`.  Add `joblib` to the dependencies (it is light) or to
`[recommended]` with a graceful serial fallback.

### 1.4 [BUG] README quick-start does not run
`README.md:31` says `from PyPR import PyPR` — the package is `pyprimat`.
Also `print(f"D/H = {result['DoH']:.5f}")  # ~2.43647` would print `0.00002`
(`DoH` is ~2.4e-5, not 2.4).  The root `__init__.py` docstring references a
`DoHx1e5` result key that does not exist.  The first thing a new user copies
must work verbatim.

### 1.5 [BUG] Default outputs are written *inside the installed package*
Relative `output_file` / `output_final_file` paths are resolved against
`cfg.working_dir` = the `pyprimat/` package directory (`main.py:1081-1083`,
`1169-1170`).  Verified: a run writes `pyprimat/results/output_tables.tsv`.
For a non-editable pip install this means writing into `site-packages`
(read-only on many systems, and certainly the wrong place).  Resolve relative
output paths against the **current working directory** (the universal
convention); keep `working_dir` only for *reading* package data.  Also rename
`working_dir` to `data_dir` — it is the package-data root, not a working
directory.

### 1.6 [BUG] Stray `__init__.py` at the repository root
Untracked `./__init__.py` makes the repo root look like a package and
shadows/duplicates the real `pyprimat/__init__.py` (with the wrong `DoHx1e5`
docstring).  Delete it.  Same for the committed-or-not `PyPRIMAT.egg-info/`
and `.pytest_cache/` — add a `.gitignore` covering `*.egg-info`, `__pycache__`,
`.pytest_cache`, `results/`, `pyprimat/results/`, `graphify-out/` (if it is
meant to stay local).

### 1.7 [BUG] Notebooks use retired flag names — silently ignored
All seven notebooks pass `compute_nTOp_flag` / `save_nTOp_flag`, which are no
longer parameter keys (they are now `compute_nTOp` / `save_nTOp`).
`PyPRConfig` warns "unknown parameter keys ignored" and runs with the
*defaults* instead — so e.g. `Sensitivity.ipynb`, which believes it computes
rates once and reloads them afterwards, actually does whatever the current
default says.  Update all notebooks to the current key names (and, once the
fingerprinted cache of §1.2 lands, simply delete these keys from the
notebooks).  This is also an argument for making unknown keys an *error*
rather than a warning: in notebook output a warning scrolls by unseen.

### 1.8 Broken example in `PyPRConfig` docstring
`config.py:173` shows `PyPRConfig({"Omegabh2": 0.022, "is_small": False})` as
canonical usage — but `is_small` is a read-only property, not a parameter key,
so this exact call emits an "unknown parameter keys ignored" warning and does
nothing.  CLAUDE.md likewise documents `is_small` as a flag.  Either support
it as a write-alias for `network`, or (better) drop the alias everywhere and
document only `network=`.

---

## 2. Packaging and the "pip install → run" experience

The stated goal — *somebody does `pip install` and runs it directly* — is
close but not met yet:

1. **Data files**: `rates/` (12 MB) is now correctly inside the package and
   declared as package-data, so wheels should work.  Add a CI check that
   builds a wheel, installs it in a clean venv, and runs a smoke solve — this
   is the only reliable way to catch path regressions like §1.1 (which the
   editable install masked for module paths but not for the `..` bug).
2. **Console entry point**: add `[project.scripts] pyprimat = pyprimat.cli:main`
   with a tiny CLI (`pyprimat --Omegabh2 0.02242 --network medium`), so a
   pip user can run BBN without writing any Python.  The body of
   `runfiles/PyPRIMAT_run.py` is essentially this CLI already.
3. **Trim the wheel**: `rates/nuclear/tables/` holds ~430 per-reaction text
   files.  Fine for now (12 MB total), but consider one `.npz` per network era
   (see §5.2) which also shrinks load time.
4. **Versioning**: `__version__` is hard-coded in three places (`pyproject.toml`,
   root `__init__.py`, `main.py`).  Single-source it
   (`importlib.metadata.version("PyPRIMAT")`).
5. **Python floor**: declared `>=3.10`; code uses `dataclass`, `|` unions —
   consistent.  Add classifiers and a `Repository`/`Documentation` URL.
6. **`runfiles/` sys.path hacks**: each script prepends the repo root to
   `sys.path`.  Once the package is pip-installed this is dead weight; keep a
   one-line comment ("only needed when running from a git checkout") or drop
   the scripts in favour of the CLI + notebooks.

---

## 3. Configuration: shrink the surface, remove the magic

`PyPRConfig` is the part newcomers meet first and currently the hardest to
trust.

1. **Split constants from run-time flags.**  Physical constants (PDG values,
   unit conversions) and run-time parameters live in one class.  Move the
   constants to a frozen module-level `Constants` dataclass (`pyprimat/constants.py`)
   that physics code imports directly; keep `PyPRConfig` as *only* the ~50
   user-settable knobs.  Benefits: the config becomes printable/serialisable
   (one `asdict()` for the weak-rate cache fingerprint of §1.2), and the
   CGS-units scaffolding (`Kelvin = 1.`, `cached_property erg`, …) stops
   suggesting that `kB` is a tunable.
2. **Drop the `__getattr__`/`__setattr__` magic for `p_*`/`NP_delta_*`.**
   Today *any* attribute starting with `p_` is silently accepted
   (`cfg.p_typoTOdg = 1.0` does nothing detectable), and a typo like
   `cfg.networkk` would be caught but `cfg.p_networkk` wouldn't.  Replace with
   an explicit dict parameter: `PyPR({"rate_offsets": {"npTOdg": 1.0}})`,
   validated against the loaded network's reaction names.  This also removes
   the constructor's import of `.nuclear` (a config↔nuclear circular
   dependency) — validation can happen in `load_network`, which knows the
   actual reactions.
3. **Group flags into sub-configs** mirroring the physics:
   `weak=WeakRateOptions(...)`, `network=NetworkOptions(...)`,
   `output=OutputOptions(...)`, `distortions=DistortionOptions(...)`.  A flat
   dict of 50 keys with interlocking validity rules (three different
   `spectral_distortions` × `analytic_distortions` × `incomplete_decoupling`
   constraints, `amax` only for large, thermal flags only with vegas, …) is
   the main obstacle to "easy to modify".  Keep the flat-dict constructor as a
   thin compatibility shim.
4. **Make flag validation match the docstrings.**  The comment at
   `config.py:60-62` says analytic distortions "can be used with or without
   incomplete_decoupling", but the constructor *raises* for
   `analytic_distortions=True` + `incomplete_decoupling=True`.  One of the two
   is wrong — decide and align.
5. **Naming hygiene**: `numba_installed` is an input that
   gets mutated into a detection result; rename to `use_numba=auto|True|False`.
   `nTOp_Born_approximation`, `compute_nTOp_thermal`, `include_nTOp_thermal`,
   `save_nTOp_thermal`, `sampling_nTOp_thermal` → a single
   `weak.thermal_corrections="precomputed"|"recompute"|"off"` plus accuracy knobs.

---

## 4. Architecture simplifications

1. **Eliminate module-level mutable state in `plasma.py`.**  The documented
   restriction "multiple `PyPR` instances share and overwrite this state — do
   not run them concurrently" is a real footgun (e.g. comparing
   `QED_corrections=True/False` instances interleaved gives wrong answers
   *silently*; the MC batching code only works because each worker process is
   isolated).  Wrap the state in a `Plasma` class instantiated per `PyPR`
   (the functions already take only `Tg`, so this is mostly mechanical: bind
   the interpolants as attributes).  This unlocks safe in-process parameter
   scans and removes a whole class of test-ordering hazards.
2. **Stop constructing throwaway `PyPRConfig()` objects deep in the call
   stack.**  `reaction_stoichiometry` and `to_filename` (`nuclear.py:403-405`,
   `445-447`) each build a fresh config — which re-reads `nuclides.csv` — just
   to find the data directory, and `_reaction_catalog` re-reads three CSVs on
   every call.  Cache the catalog with `functools.lru_cache` keyed on the data
   dir, or better: load it once in `UpdateNuclearRates` and pass it down.
   This is both a performance and a layering fix (`nuclear` should not import
   `config` mid-function).
3. **Break up `ComputeWeakRates`** (`weak_rates.py:384-913`, ~530 lines, one
   function).  It contains the Born rate, radiative corrections,
   finite-nucleon-mass corrections, thermal corrections, and the integration
   driver.  Split into one function per physical correction (each citing its
   Phys. Rep. equation), assembled by a short driver.  This is the single
   biggest readability win in the package, and it is precisely where new weak
   physics (e.g. sterile neutrinos, non-standard interactions) would be added.
4. **Delete the dynamic `<rxn>_frwrd` method attachment**
   (`nuclear.py:1113-1135`).  Methods stamped onto `UpdateNuclearRates` in a
   module-level loop, used only by the optional rates-TSV output, are
   invisible to readers, linters and IDEs.  Replace with one method
   `forward_rate(name, T)`; the TSV writer loops over `lt_net.names[1:]`.
5. **Prune "legacy" aliases**: `_order_MT`/`_order_LT`/`species_large`/
   `large_NZ` (`nuclear.py:1063-1070`), `PyPRresults()`, the duplicated
   `make_nTOp_pair` MT/LT pairs in `solve()` (`main.py:874-875` creates two
   identical pairs), `_NUC_NAMES_SMALL`/`_NUC_NAMES_FULL` (`main.py:36-38`,
   unused since the by-name embedding).  Each alias doubles the vocabulary a
   reader must hold.
6. **`solve()` is 320 lines with nested function definitions.**  Extract the
   eras into `_solve_HT`, `_solve_MT`, `_solve_LT` methods and `YA` (Saha)
   into a module function — it is pure physics, ideal for direct unit testing
   against Phys. Rep. §V.A, and currently untestable without a full instance.
7. **Stale internal references**: `network_builder.py` docstrings point to
   `:mod:pyprimat.nuclear_net` and `pyprimat.reactions.phase_network` —
   modules that don't exist (the code lives in `pyprimat.nuclear`).
   `config.py:12-13` references `nuclear_data.py` (now in `generate_rates/`).
   CLAUDE.md still describes the tree as `pypr/` with a top-level `rates/`,
   mentions `pypr/large_network.py` (gone), documents flags by their old
   names (`compute_nTOp_flag`), and instructs "run from the repo root so that
   rates/ resolves" (no longer true).  One pass to re-sync all prose with the
   post-rename layout.

---

## 5. Performance

Measured baseline (M-series Mac, small network, `compute_nTOp=True`): full run
5.1 s — roughly: weak rates ~1.8 s, background ODEs ~1 s, network eras ~1 s,
table loading + numba warm-up the rest.

1. **Vectorise the weak-rate integrals.**
   *⚠ Reserved for Fable/Opus, last phase (see §10): replacing adaptive
   quadrature by fixed-order quadrature requires physics-numerics judgment
   (convergence of each correction term at the 10⁻⁴ level on Neff/YP), not
   just code transformation.  Do not delegate to a smaller model.*
   `ComputeWeakRates` integrates with
   scalar `quad` per grid point per correction.  The integrands are smooth;
   fixed-order Gauss–Laguerre/Legendre quadrature evaluated as numpy array
   operations over the whole `sampling_nTOp` grid at once would cut the
   1.8 s to tens of ms and remove the main motivation for the fragile
   pre-tabulation flags (§1.2).  PRIMAT-Main.m itself uses fixed quadrature
   here.
2. **Rate-table loading** — *deferred at the author's request: the ~430
   per-reaction text files stay as they are (they are the readable,
   citable source of truth).*  If large-network init time ever becomes a
   bottleneck, the non-invasive option is a derived cache: keep the text
   files canonical and write the resampled `(fwd_median, expsigma, abg)`
   arrays to one `.npz` per (network, grid), invalidated by the text files'
   mtimes/hashes — the same self-validating pattern as §1.2.  Not needed now.
3. **Background solve**: `np.vectorize(_T_nu_inst)` in the
   instantaneous-decoupling branch (`main.py:302`) is a Python loop in
   disguise, called inside the `a(T)` ODE; tabulate `T_ν(T_γ)` once on the
   `n_temperature_table` grid instead.  Similarly `float(N_NEVO_of_Tg(T))`
   per RHS call could use a precomputed dense table with `np.interp`.
4. **Jacobian sparsity**: for the large network (~59 species) `solve_ivp(BDF)`
   factorises a dense 59×59 each step.  Passing a sparse Jacobian
   (`jac_sparsity` or scipy.sparse from `vr_idx`) is a cheap, likely 2-5×
   LT-era win, and scales to bigger networks (`amax=None`).
5. **numba cache warm-up**: first-ever run pays JIT compilation;
   `cache=True` is already set (good).  Document the first-run cost in the
   README so users don't misattribute it.
6. **Profile before/after**: add a `runfiles/benchmark.py` (or
   `pytest-benchmark` job) printing the era timings already collected under
   `debug=True`, so performance regressions become visible in CI.

---

## 6. Easier to extend with new physics

The clean stoichiometry-driven network (`network_builder.py`) is the model to
follow — extending the *network* is already easy (text file + rate table).
The weak sector and the background are not:

1. **Pluggable energy density.**  EDE is bolted into `PyPR._Hubble` with an
   `if self._rho_EDE is not None` (plus a hard-coded `3.044` in
   `_setup_EDE`).  Generalise to `extra_rho: list[callable(Tg) -> MeV^4]`
   accepted by the constructor; EDE becomes the first plug-in, and users can
   test their own dark components without touching `main.py`.
2. **Pluggable neutrino sector.**
   *⚠ Reserved for Fable/Opus, last phase (see §10): the value of this item
   is in choosing the interface correctly (what belongs inside
   `NeutrinoHistory` vs outside), and a wrong cut here propagates into every
   future extension.  Do not delegate to a smaller model.*
   The trio (`Tnue_of_Tg`, heating `N_NEVO`,
   distortion `dFDneu`) is the natural interface for non-standard neutrino
   physics; today it is woven through 350 lines of `_setup_background_and_cosmo`.
   Define a small `NeutrinoHistory` protocol (temperatures, heating,
   distortion, extra ρ) with two implementations (`NEVOTable`,
   `InstantaneousDecoupling`); `analytic_distortions` becomes a decorator on
   either.  This also makes the §3.4 flag-matrix validation local to one class.
3. **Weak-rate corrections as a list.**  After §4.3 splits `ComputeWeakRates`,
   represent the corrections as an ordered list of named terms that can be
   toggled/inspected individually — mirroring Table 1 of the Phys. Rep. (Born,
   +RC, +FM, +ID, +thermal …).  This gives "one switch per physics effect" and
   the test suite can pin each term's contribution to Neff/YP separately.
4. **Document the one extension recipe** in README: "to add a reaction:
   (1) add `rates/nuclear/tables/<name>.txt`, (2) add the line to a network
   file or pass `network='myfile'`, (3) add the stoichiometry row to
   `reactions_large.csv` (or auto-derive it)".  Step (3) is currently
   undocumented and `reaction_stoichiometry` raises a cryptic `KeyError` if
   forgotten — turn that into an actionable error message.

---

## 7. Test suite: faster *and* more comprehensive

Current state: 181 tests; `-m "not slow"` = 153 selected, **65 s**; full suite
~3 min; 28 `slow` (full solves).  The suite is good (it caught §1.1) but slow
for its coverage class, and `tests/README.md` says `pytest Tests/` (wrong
case) and references `generate_from_primat/` (stale).

1. **Make the session fixtures cheap.**  `conftest.py::solved_small` uses
   `compute_nTOp=True` (~2 s of weak-rate integration) although almost no
   test using it cares about freshly computed rates.  Default the fixtures to
   `compute_nTOp=False` and add *one* dedicated test comparing recomputed vs
   tabulated rates.  Combined with §5.1/§5.2 this should put the fast lane
   under ~20 s.
2. **Add a third speed tier.**  `slow` currently mixes 5-second small-network
   solves with the ~60 s reference runs.  Tier it: default (no solve, <20 s),
   `solve` (any full solve, ~1 min), `reference` (high-precision pins).  CI
   runs default+`solve` on every push, `reference` nightly.
3. **Replace solve-based tests with era-level tests where possible.**  Many
   `slow` tests only need the LT era or even a single RHS evaluation.  Expose
   the era integrations (§4.6) and test e.g. "MT seeding via Saha matches
   PRIMAT values" without integrating 10⁴ steps.
4. **Missing coverage to add** (mostly cheap):
   - a *pip-install smoke test*: build wheel → clean venv → `python -c "from
     pyprimat import PyPR; PyPR({'compute_nTOp': False}).solve()"` (would have
     caught §1.1 and any future data-path break);
   - `mc_uncertainty` import path without joblib (clear error / fallback);
   - output-file resolution (no writes inside the package once §1.5 is fixed);
   - the flag-combination validators (each `ValueError` in `PyPRConfig`);
   - `Y_prime` sign conventions: `sum_s A_s dY_s/dt == 0` for random Y at
     random T (baryon conservation of the *RHS*, not just the final state);
   - detailed-balance round trip: `compute_detailed_balance_coefficients` vs
     the shipped `detailed_balance.csv` for **all** reactions (currently only
     published spot values);
   - extend the 2×2 `incomplete_decoupling`×`QED_corrections` Neff matrix
     with the `spectral_distortions` axis once §1.2 is fixed (this is the
     test that currently fails).
5. **Decide a policy for the 7 skipped tests** — a permanently skipped test is
   dead weight; either make the prerequisite installable in CI or delete it.
6. **Fix `tests/README.md`** paths/names and add the tier table.

---

## 8. Onboarding: runfiles, notebooks, README

1. **README**: fix the quick-start (§1.4), the parameter table (it documents
   `compute_nTOp_flag=True` — neither the name nor the default is right), the
   stated output location of `PyPRIMAT_run.py` (`runfiles/results/` vs actual
   `pyprimat/results/`), and the `He3oH` typo `((He3+T)/H`.  Add: expected
   runtime per network, the numba first-run JIT note, and a 5-line "minimal
   physics modification" example (vary one rate, get ΔD/H).
2. **[DONE in this pass] `PyPRIMAT_run.py` disagreed with the validation
   values.**  It set `nuclear_qed_corrections=True`, so its D/H (2.43449e-5)
   was *outside* the CLAUDE.md validation tolerance (2.43647e-5 ± 3e-9) that
   the same file tells you to check after running this very script.  Now set
   explicitly to `False` (with a comment) in `PyPRIMAT_run.py`,
   `PyPRIMAT_compare.py` and `PyPRIMAT_reference_run.py`.  Remaining:
   `save_nTOp=True` in the default script means a casual run silently
   rewrites tracked data files in `rates/weak/` — harmless once the §1.2
   fingerprinted cache lands (that is the intended behaviour), but until
   then consider defaulting it off.
3. **One canonical "first contact" path**: README quick-start → `pyprimat`
   CLI (§2.2) → `notebooks/StandardPlots.ipynb`.  The notebooks are the best
   onboarding asset in the repo (good READMEs, clear physics framing); link
   them prominently from the main README, and state which ones are *fast*
   (AbundanceEvolution) vs *hours* (MonteCarloRates with num_mc=500).
4. **Runfiles cleanup**: `PyPRIMAT_run.py` prints with mixed Unicode/ASCII and
   stray spaces; `generate_table_CLASS_CAMB.py` (466 lines) deserves a
   README mention since CLASS/CAMB users are a key audience.  Drop the
   `sys.path` boilerplate after pip-install works (§2.6).
5. **Notebook hygiene**: commit them executed-but-stripped (nbstripout) or
   fully executed — mixed states confuse git diffs; add an
   `papermill`-based smoke test that executes the two fast notebooks in CI.

---

## 9. Documentation vs the Physics Reports reference

Spot-checks of cited equations against `doc/Pitrou_etal_PhysReptArxivVersion.pdf`
came out consistent: Eq. (24) is indeed the reduced-entropy relation used for
`s0bar`; Eqs. (A4b/A4c) match the ρ/P Fermi-Dirac integral forms quoted in
`plasma.py`; Eqs. (47)–(49) are the QED pressure corrections cited by
`qed_pressure.py`.  Remaining gaps:

1. `config.py::s0bar` cites "Eq. 24" for `s̄_γ = 4π²/45`; the constant
   actually follows from Eq. (24) plus the photon values just below it —
   cite "Eq. (24) and following" or the photon paragraph explicitly.
2. `weak_rates.py` is the least-cited module relative to its physics density
   (Fermi function, resummed radiative corrections, thermal corrections);
   after the §4.3 split, give each correction its equation/Table number from
   Phys. Rep. §IV (and Froustey–Pitrou–Volpe 2020 / Pitrou–Coc–Uzan–Vangioni
   2021 where appropriate — both PDFs are already in `doc/`).
3. The Saha docstring in `main.py::YA` cites "§V.A" — add the equation number.
4. Keep the existing convention (PRIMAT-Main.m as fallback only, with the
   off-by-one caveat) — it is stated in CLAUDE.md but worth repeating in
   `CONTRIBUTING`/developer docs once one exists.

---

## 10. Suggested execution order

| Phase | Content | Risk | Model |
|-------|---------|------|-------|
| 1 | §1 bug fixes (1.1 done), `.gitignore`, README quick-start, joblib dep | none — all pinned by tests | Sonnet |
| 2 | Output-path resolution (1.5), fingerprinted weak-rate cache (1.2), wheel smoke test | low | Sonnet |
| 3 | Test tiers + cheap fixtures (§7.1–7.3) | low; protects everything after | Sonnet |
| 4 | Config split (§3), plasma de-globalisation (§4.1), catalog caching (§4.2) | medium — do under the tightened suite, one step at a time | Sonnet |
| 5 | `ComputeWeakRates` split into named correction terms (§4.3, §6.3 — *mechanical split only, no change to the quadrature*), pluggable extra energy density (§6.1), CLI (§2.2), notebook CI (§8.5) | low–medium | Sonnet |
| 6 — **final** | §5.1 weak-rate quadrature vectorisation, §6.2 `NeutrinoHistory` interface | medium–high: physics-numerics judgment and interface design | **Fable or Opus only** |

Phase 6 is deliberately last: by then the fingerprinted cache (§1.2), the
tightened test tiers (phase 3) and the per-correction split (phase 5) are in
place, so the two delicate changes land on a fully instrumented codebase
where any 10⁻⁴-level shift in Neff/YP/D-H is caught immediately.  Whatever
the model, the gate is unchanged: `python runfiles/PyPRIMAT_run.py` must
reproduce the CLAUDE.md reference values and `pytest -m reference` must pass
after every phase.

Validation gate for every phase: `python runfiles/PyPRIMAT_run.py` (with
default flags) must reproduce the CLAUDE.md reference values, and
`pytest -m reference` must pass.
