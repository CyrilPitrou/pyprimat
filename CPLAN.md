# CPLAN.md — Plan for CPRIMAT (C port of PyPRIMAT)

This document plans a C99/C11 reimplementation of PyPRIMAT's BBN engine,
named **CPRIMAT**, living in `CPRIMAT/` at the repo root. It does **not**
implement anything yet — implementation starts only after this plan is
approved.

## 0. Scope (confirmed with user)

**In scope** — full numerical parity with the Python CLI's *default physics*
plus these explicitly requested extras:

- Standard background: NEVO non-instantaneous neutrino decoupling (default)
  and instantaneous decoupling fallback, QED plasma-pressure corrections,
  full spectral-distortion correction from the NEVO spectrum file
  (`spectral_distortions=True`, the default).
- n↔p weak rates with all four correction families (CCR, FMCCR/FMNoCCR,
  CCRTh, SD/SD_CCR) and τ_n normalization.
- Nuclear network: `small`, `small_parthenope`, `large`, any `amax`-filtered
  variant, via a **generic stoichiometry-driven engine** (mirrors
  `network_builder.py`), HT/MT/LT eras.
- Early Dark Energy (`fEDE`/`zcEDE`/`wnEDE`).
- `custom_background` (user-supplied T/t/a table, instantaneous-decoupling
  weak rates, Friedmann-based Neff estimate).
- Monte Carlo uncertainty propagation (`mc_uncertainty` equivalent):
  per-reaction `p_*` draws + τ_n draw, parallelised.
- Nuclear rate variation knobs `p_<rxn>` / `NP_delta_<rxn>`, used directly
  and by the MC driver.
- CLI flags + `.ini` config file + `--set KEY=VALUE` escape hatch (mirrors
  `pyprimat/cli.py`).
- Custom NEVO table overrides (`nevo_file`, `nevo_spectral_file`,
  `nevo_grid_file`, `nevo_file_prefix`) — cheap once the table loader is
  generic, included in the same phase as the neutrino-history port.

**Out of scope for v1** (explicitly excluded by the user; revisit later):

- `decay_era` (post-BBN matrix-exponential Decay Time era). Note: the
  *constant-rate beta-decay reactions inside the LT network itself*
  (`tTOHe3Bm`, `Be7TOLi7Bp`, and the `large` network's `decays.txt` entries)
  **are** in scope — they are ordinary T9-independent reactions integrated
  during HT/MT/LT, not the DT-era extension.
- Runtime "custom networks" (add/remove/replace reactions, the GUI's
  Create/Import-custom-network feature).
- The Streamlit GUI itself (not applicable to a CLI/C port anyway).
- Analytic μ/y-type spectral distortions (`analytic_distortions`,
  `delta_xi_nu`, `y_SZ`, `neutrino_history.AnalyticDistortion`). Excluded for
  now — revisit later; the full NEVO-spectrum-based spectral-distortion path
  (`spectral_distortions=True`, the default) stays in scope and is unaffected.

## 1. Why this is hard, and the two risk-reduction tricks that make it tractable

1. **Data-format reuse.** CPRIMAT reads the exact same `pyprimat/rates/`
   tree (text rate tables, NEVO CSVs, QED tables, `nuclides.csv`, network
   list files, `detailed_balance.csv`, `reactions_large.csv`, `decays.txt`).
   No converter, no second source of truth. CPRIMAT locates this directory
   via (in order) `--rates-dir`/`rates_dir` ini key, the `CPRIMAT_RATES_DIR`
   env var, then `../pyprimat/rates` relative to the executable — so the
   default build needs no copying, just the existing checkout layout.

2. **The n↔p weak-rate cache is a free shortcut.** `rates/weak/nTOp_*.txt`
   and `nTOp_thermal_*.txt` are *already* fingerprinted caches keyed by a
   hash of the exact config fields that determine their content
   (`weak_rates._weak_rate_fingerprint`/`_thermal_fingerprint`). CPRIMAT
   reimplements the **same fingerprint hash and file format** (see
   §7.3), so for every config already exercised by the Python test suite or
   `runfiles/`, CPRIMAT gets a cache hit and never needs to evaluate the
   hardest physics integral (the thermal CCRTh double integral, vegas/MC in
   Python) on its own numerical path. This decouples "does the rest of the
   engine reproduce Python" from "does the from-scratch thermal-correction
   quadrature reproduce vegas" — the second, much riskier question, becomes
   an independent, separately-validated sub-task (Phase 3b) instead of a
   blocker for everything downstream.

## 2. Directory layout

```
CPRIMAT/
  Makefile
  README.md
  CPLAN.md                 (this file, copied/symlinked for reference)
  include/cprimat/
    constants.h  config.h  cache.h  table_io.h  ini.h  cli.h
    linalg.h  spline.h  quad.h  rng.h
    ode_rk.h  ode_bdf.h  qed_pressure.h  plasma.h
    neutrino_history.h  weak_rates.h  network_data.h  network_builder.h
    background.h  nuclear_network.h  mc.h  api.h
  src/
    constants.c  config.c  cache.c  table_io.c  ini.c  cli.c  main.c   [Phase 0, done]
    linalg.c  spline.c  quad.c  rng.c
    ode_rk.c  ode_bdf.c
    qed_pressure.c  plasma.c  neutrino_history.c  weak_rates.c
    network_data.c  network_builder.c               [network_data.c's
                                                       file-loading half --
                                                       network lists,
                                                       decays.txt,
                                                       detailed_balance.csv,
                                                       reactions_large.csv
                                                       -- is also done in
                                                       Phase 0; tokenising/
                                                       resampling/
                                                       NetworkDefinition is
                                                       Phase 4]
    background.c  nuclear_network.c
    mc.c  api.c  main.c
  tests/
    unit/   test_linalg.c  test_spline.c  test_quad.c  test_ode_rk.c
            test_ode_bdf.c  test_network_builder.c  test_cache.c  ...
    integration/
            test_reference_small.c
            test_reference_large_amax8.c
            test_custom_background.c
            test_ede_smoke.c
            test_mc_uncertainty.c
    run_tests.sh
  examples/
    run_small.ini  run_large_amax8.ini
```

No vendored third-party C libraries. Only the C standard library, `libm`,
and POSIX threads (`pthread`) for the MC driver. No BLAS/LAPACK, no GSL, no
vegas/MC-integration library — every numerical primitive (linear algebra,
splines, quadrature, ODE integrators, PRNG) is hand-written, per the
"minimal reliance on external libraries" requirement.

## 3. Numerical core (written once, used everywhere)

This is the foundation everything else sits on, and the part the user
specifically called out ("BDF and spline methods should be part of the
code"). Built and unit-tested *before* any physics is ported.

### 3.1 Dense linear algebra (`linalg.c`)
LU decomposition with partial pivoting + forward/back substitution, for
matrices up to ~60×60 (the `large` network's Jacobian). Dense is fine at
this size — no sparse solver needed. Used by the BDF Newton iteration.

### 3.2 Interpolation (`spline.c`)
- **Linear** (`np.interp` / `interp1d(kind='linear')` equivalent): sorted
  x-array + binary search + linear interpolation, with configurable
  extrapolation (constant or linear), and "log-log" extrapolation as used by
  rate-table resampling (`_resample_rate_table` extrapolates outside the
  table's T9 range using a power law fitted in log-log space — must be
  replicated exactly, not approximated).
- **Natural cubic spline** (`CubicSpline` equivalent): standard tridiagonal
  not-a-knot or natural-boundary solve, used for the QED-pressure analytic
  tables and the electron-thermo cache.
- **Quadratic interpolation** (`interp1d(kind='quadratic')`): only needed if
  `rate_interp_order="quadratic"` is exercised; implement via local
  3-point Lagrange (scipy's quadratic spline is more elaborate but PyPRIMAT
  only uses `"linear"` by default — quadratic/cubic rate-table interpolation
  is a Phase-4 stretch item, not a v1 blocker).

### 3.3 Quadrature (`quad.c`)
- **1D adaptive quadrature** (Gauss-Kronrod 21/43-point or adaptive
  Simpson with error estimate + bisection) for the e± thermodynamic
  integrals (`_rho_e_exact` etc., `qed_pressure._dPa`/`_dPe3`) and the Born
  weak-rate phase-space integral (`ComputeFn`).
- **2D adaptive quadrature** for the thermal radiative-correction integral
  (`weak_rates._L_CCRTh_interpolants`, the one Python evaluates with
  `vegas` Monte Carlo or `scipy.dblquad` as fallback). A deterministic
  tensor-product adaptive Gauss-Kronrod (subdividing the 2D domain) is
  preferred over hand-rolled Monte Carlo: deterministic, reproducible, no
  RNG-seed sensitivity, and the fallback path Python itself uses
  (`dblquad`) is already deterministic quadrature — so this is actually the
  *more* faithful port, not a compromise.

### 3.4 PRNG (`rng.c`)
`xoshiro256**` (public-domain, ~10 lines) + Box-Muller normal sampling, for
`mc_uncertainty`'s per-reaction rate draws and τ_n draw. **Not** required
(or expected) to reproduce NumPy's `default_rng` bit-for-bit — `mc_uncertainty`
is a statistical estimate, not a reference value, so CPRIMAT's MC results
are validated statistically (mean/std converge to the same values within
MC noise for the same `num_mc`), not value-for-value against Python.

### 3.5 ODE integrators
- **Nonstiff adaptive RK** (`ode_rk.c`): embedded Dormand-Prince RK45 (or
  DOP853 for extra headroom) with PI step-size control. Used for the HT-era
  n↔p ODE (Python: `LSODA`, non-stiff here) and the two background ODEs
  (`a(T)` entropy-conservation ODE, `t(a)` Hubble-integration ODE — both
  smooth, non-stiff, Python again uses `LSODA`).
- **Stiff BDF** (`ode_bdf.c`): variable-order (1–5) variable-step BDF with
  Newton corrector using the analytic Jacobian (`network_builder`'s
  `_jac_kernel`) and the dense LU solver from §3.1. Algorithm follows the
  classic Krogh/Gear divided-difference formulation used by MATLAB's
  `ode15s` and `scipy.integrate.BDF` (Shampine & Reichelt 1997, "The MATLAB
  ODE Suite", SIAM J. Sci. Comput. 18). This is the single most
  algorithmically demanding piece of the port — see §8 for its dedicated
  validation plan (Robertson problem, Van der Pol, before ever touching the
  real nuclear network).
- **Matrix exponential**: not needed (decay_era is out of scope for v1).

## 4. File-by-file Python → C mapping

| Python | C | Notes |
|---|---|---|
| `constants.py` | `constants.c/.h` | Pure constant definitions; copy values verbatim, no computation differs. |
| `config.py` (`PyPRConfig`, `DEFAULT_PARAMS`) | `config.c/.h` | Tagged-union param table (see §6); validation logic ported 1:1 (the same `ValueError`/warning conditions). |
| `cache_utils.py` | `cache.c/.h` | Fingerprint hash (see §7.3) + read/write of tagged text caches. |
| `qed_pressure.py` | `qed_pressure.c/.h` | `_I01`, `_I2m1` (Bose/Fermi integrals via `quad.c`), `_dPa`, `_dPe3`, table save/load. `_dPb` (O(e⁴)) is unused by `plasma.py` in the default path — port only if a test exercises it. |
| `plasma.py` (`Plasma`) | `plasma.c/.h` | e± thermo via tabulated cubic-spline cache (mirrors `_build_electron_tables`, same on-disk cache file + fingerprint so it is interchangeable with Python's cache). |
| `neutrino_history.py` | `neutrino_history.c/.h` | `NEVOTable` (CSV table load + interpolation), `InstantaneousDecoupling`. `resolve_nevo_path` + `nevo_file_prefix` logic ported. `AnalyticDistortion` is **not** ported (out of scope, see §0). |
| `weak_rates/` package (`integrands.py`, `corrections.py`, `cache.py`, `api.py`) | `weak_rates.c/.h` | The big one — see §7. As of the Python-side split (FUTURE.md P1.1), every Fermi-Dirac kernel and correction term is written **once** as an array-aware function with the scalar case as `T_arr` of length 1 — no more hand-duplicated `FD_nu3`/`_FD_nu3_v` pairs to keep in sync. C gets the same benefit for free (a C function naturally takes a pointer+length and the "scalar" call is just length 1), so there is no `_v`-suffixed shadow API to port either: one `double *` in, one `double *` out, per kernel. The four Python source files do not need four separate C files — `weak_rates.c` (or `weak_rates.c` + `weak_rates_cache.c` for the two on-disk-cache-shaped functions) is enough; collapsing the split back down in C is fine and arguably reads better in a language without Python's package-as-namespace convention. |
| `network_data.py` | `network_data.c/.h` | Tokeniser (`reaction_stoichiometry`), `detailed_balance.csv`/`reactions_large.csv` loaders, `_qed_nuclear_rescale`, `load_network`, master-grid resampling, `NetworkDefinition`/`UpdateNuclearRates` equivalents, `p_*`/`NP_delta_*` application. |
| `network_builder.py` | `network_builder.c/.h` | `compile_network`, `_rhs_kernel`, `_jac_kernel`, `check_conservation` — direct flat-array port, no semantic changes needed (this module was already written for a JIT/array-kernel style, which is exactly what C wants). |
| `background.py` | `background.c/.h` | `StandardBackground` + `CustomBackground`, EDE/ΛCDM setup, `Hubble`, weak-rate normalisation hookup. |
| `nuclear_network.py` | `nuclear_network.c/.h` | HT/MT/LT solve loop, Saha (`YA`) seeding, observable/abundance bookkeeping, TSV output writers. DT-era methods (`_build_decay_matrix`, `_integrate_decay_era`, `_write_decay_evolution`) are **not** ported (out of scope). |
| `main.py` (`PyPR`, `mc_uncertainty`) | `api.c/.h` + `mc.c/.h` | `cprimat_run()` thin wrapper (§9) + threaded MC driver. |
| `cli.py` | `cli.c` + `ini.c` | argv parsing, `.ini` loader, `--set` escape hatch, `main()` printing. |

Deliberately **not ported**: `gui/*`, `plotting.py` (Plotly, GUI-only),
`gui/custom_rates.py`/`gui/params_form.py` (custom-network editor).

## 5. Data formats (read as-is, no conversion)

All formats below are already fixed by the existing `pyprimat/rates/` tree;
CPRIMAT's parsers must match them exactly.

- **`nuclides.csv`**: header `name,N,Z,A,Q,mass_excess_keV,spin`, one row per
  nuclide.
- **Network list files** (`rates/nuclear/networks/<name>.txt`): one reaction
  per line, `<name>` or `<name>, <table_filename>`.
- **Rate tables** (`rates/nuclear/tables/<name>/<name>[_variant].txt`):
  `#`-comment header (often containing the detailed-balance α/β/γ/Q on a
  `# detailed balance: ...` line, redundant with `detailed_balance.csv`),
  then whitespace-separated `T9 rate error` rows.
- **`decays.txt`**: flat file, one row per β±/EC decay, T9-independent rate.
- **`detailed_balance.csv`**: `reaction,Q_keV,alpha,beta,gamma`.
- **`reactions_large.csv`**: `name,reactants,products,source,ref`.
- **NEVO CSVs** (`rates/NEVO/*.csv`): comma-delimited, no header; 6/7-column
  thermo table, 86-column spectral table, `NEVOGrid.csv` 1D y-grid.
- **QED tables** (`rates/plasma/QED_*.txt`): 3 columns (T, O(e²), O(e³)).
- **Weak-rate cache** (`rates/weak/nTOp_<hash>.txt`,
  `nTOp_thermal_<hash>.txt`): fingerprint header + data columns — format
  defined in `weak_rates/cache.py`/`cache_utils.py`; reverse-engineer the exact
  header syntax during Phase 0 so CPRIMAT's reader/writer round-trips
  against existing files (cheap to verify: load a Python-written file, dump
  it back out, diff).

## 6. Configuration: `CPRConfig`

A tagged-union key/value store (`enum { CPR_BOOL, CPR_INT, CPR_DOUBLE,
CPR_STRING }`), pre-populated with the same defaults as `DEFAULT_PARAMS`
(`config.c` ports the dict verbatim, including comments-as-doc-strings kept
as C comments). `p_<rxn>`/`NP_delta_<rxn>` are stored in a small open hash
map keyed by reaction name, exactly mirroring `PyPRConfig.p_rxn`/`NP_delta_rxn`.

Construction order mirrors `PyPRConfig.__init__`:
1. Load defaults.
2. Apply `.ini` file (if given), then CLI flags, then `--set KEY=VALUE`
   entries, in that order (later wins) — same precedence as the Python CLI
   merges `params` dict before constructing `PyPRConfig`.
3. Validate (network file exists, `amax` positive, NEVO override
   shape/column checks, flag-combination checks — same `ValueError`
   conditions as `PyPRConfig.__init__`, returned as error codes/messages
   rather than exceptions).
4. Recompute derived quantities (`eta0b`, `Mpl`, `rhocOverh2`, ...).

**`.ini` format**: one `KEY=VALUE` (or `KEY VALUE`) per line, `#`/`;`
comment lines, blank lines ignored. Values parsed the same way `--set` does
in Python (`ast.literal_eval`-equivalent: try int, then double, then
`true`/`false`/`none`, else literal string). `p_<rxn>` / `NP_delta_<rxn>`
keys recognised by prefix, same as Python.

**CLI flags**: mirror `cli.py`'s exposed flags (`--Omegabh2`, `--DeltaNeff`,
`--network`, `--amax`, `--numerical_precision`, `--json`, `--verbose`,
repeatable `--set`), plus `--ini PATH` and `--rates-dir PATH`.

## 7. Weak rates (`weak_rates.c`) — the hardest physics module

Python-side, this used to be one 1834-line `weak_rates.py` with every
Fermi-Dirac kernel and correction term hand-duplicated as a scalar function
*and* a `_v`-suffixed vectorised twin (`FD_nu3`/`_FD_nu3_v`,
`RadCorrResum`/`_RadCorrResum_v`, ...). It has since been split
(`pyprimat/weak_rates/`) into `integrands.py` (the FD kernels, now each
written **once**, array-aware, with the scalar case handled as a length-1
array — see that module's docstring and `_setup_fd_impls`), `corrections.py`
(`FermiCoulomb`, `RadCorrResum`, `ComputeFn`, the `_L_*` correction terms,
`_RateContext`), `cache.py` (the two fingerprint functions), and `api.py`
(`ComputeWeakRates`/`InterpolateWeakRates`/`RecomputeWeakRates`).

This Python-side de-duplication has a direct, simplifying consequence for
the C port: **there is no scalar/vector API split to replicate**. A C
function over `double *T_arr, size_t n` *is* both the scalar and vectorised
form (call it with `n=1`) — C never had the problem the Python `_v` split
was solving in the first place. So unlike the Python package, the C side
does not need one file per Python submodule; collapsing
`integrands.py`+`corrections.py`+`cache.py`+`api.py` down to a single
`weak_rates.c`/`weak_rates.h` (kernels, correction terms, fingerprinting,
and the three entry points, in that order) is the more readable choice in
C — fewer files to jump between for one tightly-coupled physics module. A
split into `weak_rates.c` + `weak_rates_cache.c` (entry points/corrections
vs. the two on-disk-cache-shaped fingerprint functions) is also fine if
`weak_rates.c` gets unwieldy; either way, prefer fewer, larger,
well-sectioned C files over mirroring the Python file count.

Ported in two independent sub-phases so the riskier half doesn't block
everything else (per the cache-reuse trick in §1):

### 7a. Always-needed pieces
- `FD_nu*` family (`integrands.py`): each kernel as one `double *in, double
  *out, size_t n` function (covers both the scalar and array Python call
  sites with a single C signature — see above).
- `FermiCoulomb` (Fermi function + Coulomb correction), `RadCorrResum` (T=0
  resummed radiative correction `χ`), `ComputeFn` (normalisation
  phase-space integral via `quad.c`) — all from `corrections.py`.
- `_L_BORN`, `_L_CCR`, `_L_FMCCR`, `_L_FMNoCCR`, `_L_SD`, `_L_SD_CCR`: all
  1D-quadrature-based (or closed-form), straightforward ports via `quad.c`.
- `ComputeWeakRates` (`api.py`): assembles the four correction terms per
  `cfg.radiative_corrections`/`finite_mass_corrections`/`thermal_corrections`/
  `spectral_distortions`, builds the forward/backward rate grids, applies
  the noise floor/clamp (`_WEAK_RATE_FLOOR` equivalent).
- `InterpolateWeakRates`: builds the quadratic (or as-implemented)
  interpolant over the cached grid — replicate the exact interpolation
  *kind* used in Python, since the floor-clamping logic in
  `background.py` explicitly compensates for this interpolant's overshoot
  behaviour.

### 7b. Thermal correction (`_L_CCRTh_interpolants`, `corrections.py`)
The double integral evaluated by `vegas`/`dblquad` in Python. Ported as a
from-scratch deterministic adaptive 2D quadrature (§3.3), built and cached
**exactly like Python's** `nTOp_thermal_<hash>.txt` cache — same fingerprint
fields (`cache._thermal_fingerprint`), so a config whose thermal cache
already exists in `rates/weak/` is loaded directly and 7b's own numerics
are never exercised for that run. Validated independently (Phase 3b, §8)
against the *values* in existing `nTOp_thermal_*.txt` files (not just by
re-deriving the same cache key) before being trusted as a from-scratch
fallback.

### 7.3 Fingerprint hash format
`cache_utils.fingerprint_hash` — read its exact algorithm (likely a stable
hash, e.g. SHA-1/MD5 of a canonical `repr()` of the sorted fingerprint
dict, or similar) during Phase 0 and reimplement bit-for-bit, since cache
hits depend on it producing the *same hash string* as Python for the same
fingerprint dict (`weak_rates.cache._weak_rate_fingerprint` /
`_thermal_fingerprint`) — this is a hard compatibility requirement, not
just an implementation detail.

## 8. The BDF solver: dedicated validation plan

Because this is explicitly called out as required ("BDF ... methods should
be part of the code") and is the highest-risk numerical component:

1. Implement against textbook stiff test problems with known reference
   solutions: Robertson's chemical kinetics problem (3-equation classic
   stiff benchmark), the Van der Pol oscillator (stiff parameter regime),
   and a simple linear stiff system with known analytic solution.
2. Compare step counts / accuracy against `scipy.integrate.solve_ivp(method='BDF')`
   run on the *same* test problems in Python, at the *same* `rtol`/`atol`,
   to confirm comparable (not necessarily identical) step-acceptance
   behaviour — variable-order BDF implementations can legitimately take
   different step sequences while both being correct to tolerance, so the
   acceptance criterion is **solution accuracy**, not **identical step
   sequence**.
3. Only once (1)+(2) pass does the BDF integrator get wired into
   `nuclear_network.c`'s MT/LT eras.

## 9. The thin wrapper (`api.c`/`api.h`)

Mirrors `PyPR`: one entry point taking a parameter set, returning a result
set, optionally writing output files as directed by the config (TSV
time-evolution, final-abundance dump, background-evolution TSV — same file
formats as the Python writers, so existing plotting scripts can consume
either).

```c
typedef enum { CPR_NONE, CPR_BOOL, CPR_INT, CPR_DOUBLE, CPR_STRING } CPRType;
typedef struct { const char *key; CPRType type; union { int b; long i; double d; const char *s; } v; } CPRParam;
typedef struct { CPRParam *items; size_t n; } CPRParamSet;

/* Builds the config from `params`, runs the full HT/MT/LT solve, fills
 * `results` (Neff, YPBBN, YPCMB, DoH, He3oH, He3oHe4, Li7oH, per-nuclide
 * final Y, ...), and writes any output files the config requests.
 * Returns 0 on success, nonzero with `*errmsg` set on failure (config
 * validation error, missing data file, integration failure, ...). */
int cprimat_run(const CPRParamSet *params, CPRParamSet *results, char **errmsg);

void cprimat_paramset_free(CPRParamSet *s);
```

`cli.c`'s `main()` is a thin shim: parse argv + `.ini` into a `CPRParamSet`,
call `cprimat_run`, print the same short summary / JSON dump as
`pyprimat/cli.py`'s `main()`.

`mc.c` adds a parallel entry point `cprimat_mc_uncertainty()` that builds
one shared `Background`/`Plasma` per worker thread (read-only after setup,
exactly mirroring `_mc_run_batch`'s "expensive background computed once,
reused across samples" design), spawns `pthread`s over seed chunks, and
reduces to mean/std per requested quantity.

## 10. Build system

Plain `Makefile`, no autotools/CMake (keeps "minimal reliance on external
[build] tooling" in spirit, and the project is small enough not to need
more):

```
make            # builds CPRIMAT/build/cprimat (release, -O3)
make debug      # -O0 -g -fsanitize=address,undefined
make test       # builds + runs tests/unit/* and tests/integration/*
make bench      # runs examples/run_small.ini and run_large_amax8.ini, prints timing
make clean
```

`CFLAGS` baseline: `-std=c11 -Wall -Wextra -O3 -march=native` (with a
non-`-march=native` fallback target for portability/CI). Threading via
`-pthread`; no other link flags beyond `-lm`.

## 11. Test suite

### Unit tests (`tests/unit/`)
One test binary per numerical-core module (§3): LU solve accuracy on
random well-conditioned matrices, linear/cubic spline against known
polynomials, 1D/2D quadrature against closed-form integrals, RK45 against
an analytic IVP, BDF against the stiff benchmarks of §8, fingerprint hash
round-trip against an existing Python-written cache file, network-builder
RHS/Jacobian against a hand-computed 3-reaction toy network (verify both
analytically and via finite-difference check of the Jacobian).

### Integration tests (`tests/integration/`)
Reproduce the exact reference values already documented in `CLAUDE.md`,
**to the same stated tolerances**:

| Network | Observable | Expected | Tolerance |
|---|---|---|---|
| `small` | YP (BBN) | 0.24700028 | ±1e-5 |
| `small` | D/H | 2.43500e-5 | ±3e-9 |
| `large, amax=8` | YP (BBN) | 0.24700363 | ±1e-5 |
| `large, amax=8` | D/H | 2.43571e-5 | ±3e-9 |

plus the per-nuclide final-abundance table (n, p, H2, H3, He4, Li7, Be7) for
both networks, at the same precision CLAUDE.md mandates for reporting
(Neff 8 decimals, YP 8, D/H 7 sig figs, Li7/H 6 sig figs).

Additional smoke/regression tests (looser tolerances, since these features
have no pre-existing documented reference number — first establish the
Python-side number via `runfiles/`, then lock it in):
- `custom_background`: round-trip test exactly like
  `tests/test_custom_background.py` (write a reference background from a
  standard run, re-run through `custom_background`, check observables
  agree to <1e-5 relative).
- EDE smoke test: `fEDE>0` run completes and shifts Neff/YP in the expected
  direction/order-of-magnitude vs. `fEDE=0`.
- `mc_uncertainty`: statistical check — for a fixed `num_mc`, mean ≈
  central value within a few σ/√N, std stable across two independent runs
  with different base seeds (no value-for-value comparison against
  Python's RNG stream, per §3.4).

### Cross-checking against the Python implementation
For every integration test, also dump intermediate Python-side arrays
(QED-pressure tables, electron-thermo tables, NEVO-derived `Tnue_of_Tg`
samples, the weak-rate grids, the compiled-network flat arrays) as plain
text fixtures via a small `runfiles/_export_fixtures.py` helper, and have
the corresponding C unit test load and compare against them directly. This
gives much tighter, localized failure diagnosis than "the final D/H is off
by 1e-6" when something breaks deep in the pipeline.

## 12. Performance

Target: faster than `pyprimat` end-to-end on the same machine for both the
`small` and `large, amax=8` reference runs (the natural apples-to-apples
comparison, since both are in the documented CLAUDE.md table). Sources of
expected speedup: no Python/NumPy call overhead per ODE evaluation, no
interpreter-level dict/dataclass overhead in the hot RHS/Jacobian loop,
dense small-matrix LU instead of going through `scipy.integrate`'s generic
machinery, single static binary with no JIT warm-up (vs. numba's
first-call compilation cost). `make bench` times both reference runs and
reports the ratio against a recorded Python baseline (captured once via
`time python runfiles/PyPRIMAT_run.py` and `..._compare.py`, stored in
`CPRIMAT/examples/baseline_timings.txt`).

## 13. Implementation phases (suggested order)

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0 | **Done.** Skeleton, Makefile, `config.c`/`ini.c`/`cli.c`, all data-file loaders (no physics), fingerprint-hash reimplementation verified against an existing Python cache file. | Loads `pyprimat/rates/`, parses a `.ini`, round-trips a cache file. -- met: `make test` runs `test_cache`/`test_ini`/`test_network_data`/`test_table_io`, all green; `./build/cprimat --ini examples/run_small.ini` round-trips against the real data tree. |
| 1 | **Done.** `linalg.c` (Doolittle LU, partial pivoting), `spline.c` (natural + not-a-knot cubic spline via Thomas-algorithm O(n) solves, `cpr_resample_rate_table`'s log-log/linear two-branch port), `quad.c` (adaptive Simpson), `rng.c` (xoshiro256** + Box-Muller). | Unit tests pass against closed-form references. -- met: `make test` runs `test_linalg`/`test_spline`/`test_quad`/`test_rng`, all green (exact cubic/line reproduction, exact power-law resampling, exact polynomial/transcendental integrals, RNG range/determinism/moment checks). |
| 2 | **Done.** `ode_rk.c` (embedded Dormand-Prince RK5(4)7M, FSAL, mixed rtol/atol step control), `ode_bdf.c` (variable-order 1-5 constant-step-with-restart BDF, Adams-Bashforth predictor of matching order for a sound LTE estimate, Newton corrector via `linalg.c`'s dense LU, forward-difference Jacobian fallback). | §8's stiff benchmarks pass at target tolerance. -- met: `make test` runs `test_ode_rk` (exponential decay, harmonic oscillator, backward integration, all against closed-form solutions) and `test_ode_bdf` (stiff linear system vs. exp(-1000)/exp(-1), Robertson's problem vs. its long-time quasi-steady limit with conservation-law check, Van der Pol mu=100 boundedness), all green. Step-size control: continuous err^(-1/(q+1)) growth, gated to fire only once order has saturated for the current history (order growth and h growth both invalidate/reset the other's uniform-spacing assumption, so interleaving them caused thrashing at low order -- see `ode_bdf.c`'s comments), with a wide growth clamp (1e4x) reserved for genuinely tiny errors and a conservative one (8x) otherwise (an aggressive clamp at merely-comfortable error overshoots in regions of rapidly-changing curvature, e.g. Van der Pol's relaxation transitions, wasting the order-climb that produced it on an immediate rejection). |
| 3a | `qed_pressure.c`, `plasma.c`, `neutrino_history.c` (incl. NEVO overrides), `weak_rates.c` §7a using **existing cache files only**. | Background a(T)/t(a)/weak-rate grids match Python fixtures (§11) for a config with a pre-existing cache hit. |
| 3b | `weak_rates.c` §7b from-scratch thermal integral. | Independently reproduces values in an existing `nTOp_thermal_*.txt`. |
| 4 | `network_data.c`, `network_builder.c`. | RHS/Jacobian unit tests pass; `check_conservation` equivalent passes on `small` and `large`. |
| 5 | `background.c` (Standard + Custom + EDE). | Background fixtures match Python end-to-end. |
| 6 | `nuclear_network.c` (HT/MT/LT, Saha seed, observables, TSV writers). | `small` reference numbers match CLAUDE.md table. |
| 7 | `large`/`amax` support end-to-end. | `large, amax=8` reference numbers match CLAUDE.md table. |
| 8 | `api.c`, `mc.c` threaded MC. | MC statistical sanity tests pass. |
| 9 | Performance pass, `make bench`, docs. | Faster than Python baseline on both reference runs. |

## 14. Open implementation decisions (non-blocking, default chosen, flag if you disagree)

- **Rates directory location**: default search `--rates-dir` →
  `CPRIMAT_RATES_DIR` env → `../pyprimat/rates` relative to the binary.
  Going with this unless you'd rather CPRIMAT have its own copy/symlink
  checked into `CPRIMAT/rates`.
- **Threading**: POSIX `pthread` for MC (ubiquitous, not a "real" external
  dependency). OpenMP was considered but pulls in a compiler-specific
  runtime; plain pthreads keeps the Makefile portable.
- **`rate_interp_order="quadratic"/"cubic"`**: deferred past v1 (linear is
  PyPRIMAT's own default and what all CLAUDE.md reference numbers use).
