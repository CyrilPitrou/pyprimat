# PRIMAT.md — Packaging & Architecture Plan for the `primat` v0.3.0 Release

This document is the detailed plan for turning the current `pyprimat` /
`CPRIMAT` checkout into a single, PyPI-distributable package named
**`primat`**, with a fast C backend by default and a pure-Python fallback,
both reachable from `from primat import PRIMAT` and from a single `primat`
CLI. It supersedes `packaging_and_wrapper_advice.md` (kept around for
historical context; differences from it are called out explicitly below)
and extends `CPLAN.md`'s scope (analytic distortions + custom background in
C, see §8).

**Status: plan only.** Nothing in this document has been applied to the
repository yet — no files moved, no renames performed. Implementation
proceeds in the phases of §9, each as its own reviewable PR.

---

## 1. Decisions already made (do not re-litigate)

1. **Distribution name on PyPI: `primat`.** Public Python import:
   `from primat import PRIMAT`.
2. **C backend is the default**, selected automatically when available;
   Python backend remains fully functional as a fallback and for anything
   the C port hasn't reached parity on yet.
3. **C ↔ Python bridge: a compiled C-extension** (`primat._primat_c`, built
   with the Python C-API), not a subprocess call to a standalone binary.
   The standalone `primat-c` Makefile binary *also* keeps existing for
   non-Python users — same C source, two build paths (§5).
4. **There is exactly one Python package: `primat/`.** It contains the
   physics engine (today's `pyprimat/*.py`), the GUI, the CLI/dispatcher,
   and the rates data — all in one importable tree, one PyPI distribution.
   There is **no** separate `primat-py` Python package; that idea (from an
   earlier draft of this plan) added a split the user never asked for and
   only confused the layout. The only *other* top-level folder is
   `primat-c/`, the standalone C source tree (was `CPRIMAT/`), which is not
   itself a Python package — see §3.
5. **Rates tree**: lives inside `primat/rates/` (i.e. exactly where
   `pyprimat/rates/` already lives today — this part doesn't move). Shipped
   as package-data; a user-overlay mechanism (§4) lets users add custom
   networks/tables without touching the installed package.
6. **Unified time-evolution output**: both backends write the *same*
   documented columnar format, with a single Python loader that works
   regardless of which backend produced the file (§7).
7. **Version bump to 0.3.0** for this whole reorganisation.
8. Renames/moves happen in a **later, separate implementation pass** — this
   document is the plan to review first.

---

## 2. Critique of `packaging_and_wrapper_advice.md`

The earlier advice doc got the big shape right (C-extension + Python
fallback, `cibuildwheel` for wheels) but has problems worth fixing before
building on it:

1. **Three colliding console-script names.** It proposes installing
   `primat`, `primat-python`, *and* `primat-c` as pip console-scripts. But
   `primat-c` is *also* the name of the binary a C-only user builds via
   `make` in the standalone `primat-c/` folder and may put on their `PATH`.
   **Fix:** pip installs only `primat` (with `--backend {auto,c,python}`)
   and `primat-gui`. The standalone Makefile output stays named `primat-c`
   (or `build/primat-c`) and is never installed by pip onto `PATH`
   automatically — only if the user manually copies/symlinks it.
2. **It proposed a separate `pyprimat`-named package plus a C-extension
   bolted onto it (`pyprimat._cprimat`)** — i.e., it never separated "the
   one Python package" question from "the rename" question, which is what
   caused the confusing `primat`/`primat-py` split in an earlier draft of
   *this* document too. **Fix:** one Python package, `primat/`, full stop
   (§3).
3. **`data_dir` resolution via `os.path.join(os.path.dirname(__file__),
   "rates")`.** Works, but should go through `importlib.resources.files()`
   plus the overlay resolver (§4), not a raw `__file__` join, so it behaves
   correctly under the overlay mechanism and any non-standard install
   layout.
4. **No rates-overlay mechanism** — assumed one fixed `data_dir`. Fixed in
   §4 (the user's "small/large could become unavailable" worry).
5. **`-march=native` implied nowhere, but also not excluded** — a wheel
   built with it crashes with "illegal instruction" on an end user's older
   CPU. **Fix:** wheels never use `-march=native` (§5.3).
6. **No sdist story** for platforms `cibuildwheel` doesn't cover. **Fix:**
   §5.4 and §6.
7. **No unified output-format story** beyond final scalars. **Fix:** §7.

---

## 3. Final directory layout

```
PRIMAT_suite/PyPRIMAT/            (repo root, unchanged location)
├── pyproject.toml                 # builds the single "primat" distribution
├── README.md, CLAUDE.md, LICENCE
├── primat/                        # THE Python package (was pyprimat/) — everything Python-side
│   ├── __init__.py                  from .api import PRIMAT, __version__
│   ├── api.py                       PRIMAT facade: builds config, picks backend, runs, returns/loads results
│   ├── backend.py                   HAS_C_BACKEND probe, run_bbn() dispatch, rates_dir resolution
│   ├── cli.py                       `primat` console-script: argparse, --backend flag, .ini loader
│   ├── evolution.py                 shared TSV schema + load_evolution() loader (§7)
│   ├── config.py, constants.py, cache_utils.py
│   ├── background.py, nuclear_network.py, plasma.py, qed_pressure.py
│   ├── network_data.py, network_builder.py, neutrino_history.py
│   ├── weak_rates/                  integrands.py, corrections.py, cache.py, api.py
│   ├── plotting.py
│   ├── rates/                       SHIPPED DEFAULTS — package-data, see §4 (unchanged from pyprimat/rates/)
│   │   ├── plasma/  nuclear/{tables,networks,data}/  weak/  NEVO/
│   ├── _primat_c/                   compiled extension lives here after build (primat._primat_c.<soabi>.so)
│   │   └── _wrapper.c                Python C-API wrapper (was the advice doc's cprimat_wrapper.c)
│   └── gui/                         `primat-gui` Streamlit app (was pyprimat/gui)
│       ├── app.py, launcher.py, panels.py, params_form.py, session_keys.py, custom_rates.py
├── primat-c/                       # standalone C source (was CPRIMAT/); not a Python package
│   ├── Makefile                      `make` -> build/primat-c, standalone CLI binary
│   ├── include/primat_c/...
│   ├── src/*.c                       same sources used by the extension build (minus main.c/cli.c there)
│   ├── tests/, examples/
├── runfiles/                       # updated to `from primat import PRIMAT` (§10)
├── notebooks/                      # updated imports (§10)
├── tests/                          # updated imports; new cross-backend parity tests (§8)
└── .github/workflows/
    ├── wheels.yml                   # cibuildwheel matrix -> PyPI (§6)
    └── tests.yml                    # existing CI, updated for new layout
```

### 3.1 `pyproject.toml` package declaration

Nothing exotic is needed here precisely *because* there's only one Python
package now — no `package_dir` remapping trickery (that machinery was only
needed for the rejected `primat-py` hyphenated-folder idea):

```toml
[project]
name = "primat"
version = "0.3.0"

[tool.setuptools]
packages = ["primat", "primat.gui", "primat.weak_rates", "primat._primat_c"]

[tool.setuptools.package-data]
primat = ["rates/**/*", "gui/.streamlit/*"]
```

`primat-c/` never appears in `packages=` at all — it is C source, not a
Python package; its files reach the build only via the `Extension(sources=...)`
declaration in `setup.py` (§5.2), and reach the sdist automatically because
setuptools includes `Extension` sources in the source distribution.

---

## 4. The rates directory: resolution and overlay design

### 4.1 Where the shipped defaults live

`primat/rates/` — package-data of the `primat` package. This is exactly
where `pyprimat/rates/` already lives today; nothing moves relative to the
Python code, only the enclosing package gets renamed. `pip install primat`
puts this at `site-packages/primat/rates/`; `pip install -e .` leaves it
exactly where it is in the checkout (editable installs map back to the
source tree — editing `primat/rates/` directly works immediately, same as
today's `pyprimat/rates/`).

### 4.2 Why the C side never needs its own copy

The compiled extension and the in-process Python facade always resolve the
rates directory **once, in Python**, and pass the resolved absolute path
string into the C side on every call — both the extension call and (for
the standalone non-Python build) the `--rates-dir` CLI flag. Consequences:

- the pip-installed C extension never needs its own path-search logic for
  "installed via pip" — Python already did the work via
  `importlib.resources.files("primat") / "rates"`;
- the **standalone** `primat-c` (cloned/built with `make`, no Python
  involved at all) keeps its own independent resolution chain exactly as
  already specified in `CPLAN.md` §1/§14: `--rates-dir` flag →
  `PRIMAT_RATES_DIR` env var → `../primat/rates` relative to the executable
  (sibling folders at repo root — `primat-c/build/primat-c` looking one
  level up and into `../primat/rates`).

### 4.3 Overlay resolution (the actual fix for "small/large could become unavailable")

A single relative-path resolver, implemented identically (same precedence,
same merge semantics) in `primat/backend.py` (Python call sites) and
`primat-c/src/rates_resolve.c` (C call sites, including the standalone
binary):

```
resolve(relpath, cfg):
    for base in [cfg.rates_dir_override, cfg.user_rates_dir, <shipped_default>]:
        if base is not None and (base / relpath).exists():
            return base / relpath
    raise FileNotFoundError(relpath)
```

- `<shipped_default>` is always last and always present — `small`/`large`
  networks, AC2024 tables, NEVO tables, QED tables are *never* unavailable,
  regardless of what the user overrides.
- `cfg.user_rates_dir` (new config field, also settable via
  `PRIMAT_USER_RATES_DIR` env var) is the **persistent, mergeable**
  override: a directory the user maintains with *just* their additions —
  e.g. `~/.primat/rates/nuclear/networks/myNetwork.txt` plus one new
  `nuclear/tables/myReaction/myReaction.txt`. Every other relative path
  (`small.txt`, `decays.txt`, `NEVOPRIMAT_col_1_7.csv`, ...) transparently
  falls through to the shipped tree. This is exactly "the GUI zip-import
  pattern, but file-based and persistent."
- `cfg.rates_dir_override` (`--rates-dir` / `rates_dir=` constructor arg) is
  a **full takeover** for power users pointing at an entirely separate,
  self-contained tree — checked first, so it can shadow anything, but still
  followed by the shipped default as a last resort.
- This resolver is what both the GUI's "Import/Create custom network"
  save-as-zip flow and a future "save as a `user_rates_dir` folder" flow
  can target — `primat/gui/custom_rates.py` already builds an in-memory
  equivalent of this overlay; letting it export directly to a
  `user_rates_dir`-shaped folder (not just a zip) is a natural follow-up,
  not required for v0.3.0.

### 4.4 What changes in `PyPRConfig`/`CPRConfig`

Two new fields, both part of the existing fingerprint-field machinery so
cache invalidation stays correct:
- `rates_dir` (today implicit; made an explicit, validated field) — the
  full-takeover override.
- `user_rates_dir` (new) — the persistent overlay-only override.

Both follow the same eager-validation pattern already used for
`nevo_file`/etc. in `config.py`.

---

## 5. The C-extension bridge

### 5.1 Wrapper module: `primat/_primat_c/_wrapper.c`

Same shape as the advice doc's `cprimat_wrapper.c`, relocated and with two
fixes:
- takes the rates directory as an explicit `const char *rates_dir`
  argument from Python (per §4.2), never re-derives it from `__file__`;
- returns the *full* result set needed for the unified evolution format
  (§7), not just the six scalar observables — `cprimat_run()`'s
  `CPRResults` struct (already speced in `CPLAN.md` §9) gains optional
  arrays (`t[]`, `a[]`, `T_gamma[]`, per-nuclide `Y[][]`) populated only
  when `output_time_evolution=True`.

### 5.2 Build target (`setup.py`, alongside `pyproject.toml`)

```python
from setuptools import setup, Extension
import os, platform

c_source_dir = os.path.join("primat-c", "src")
c_sources = [
    os.path.join(c_source_dir, f)
    for f in os.listdir(c_source_dir)
    if f.endswith(".c") and f not in ("main.c", "cli.c")
]
c_sources.append("primat/_primat_c/_wrapper.c")

extra_compile_args = ["-O2"]
if platform.system() != "Windows":
    extra_compile_args.insert(0, "-std=c11")
# NOTE: deliberately no -march=native -- wheels must run on any CPU of the
# target architecture, not just the CI runner's. -march=native is only
# ever added by the user's own `make` invocation in primat-c/.

primat_c_extension = Extension(
    "primat._primat_c",
    sources=c_sources,
    include_dirs=[os.path.join("primat-c", "include")],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[primat_c_extension])
```

### 5.3 Two build paths from one source tree

| Path | Trigger | Output | Flags |
|---|---|---|---|
| Extension build | `pip install primat` (wheel build or sdist compile) | `primat/_primat_c.<soabi>.{so,pyd}` | portable (`-O2`, no `-march=native`) |
| Standalone build | `cd primat-c && make` | `primat-c/build/primat-c` CLI binary | user's own `CFLAGS`, may add `-march=native`, `-O3`, sanitizers (`make debug`) |

Both compile the *same* `primat-c/src/*.c` — there is exactly one C source
of truth, never a forked copy.

### 5.4 Fallback behaviour (`primat/backend.py`)

```python
import logging
logger = logging.getLogger("primat")

HAS_C_BACKEND = False
try:
    from . import _primat_c
    HAS_C_BACKEND = True
except ImportError:
    logger.info(
        "C extension not available (not compiled, or compiler/platform "
        "unsupported at install time) -- using the pure-Python backend. "
        "This still gives identical physics, just slower."
    )

def run_bbn(cfg, force_backend=None):
    backend = force_backend or ("c" if HAS_C_BACKEND else "python")
    if backend == "c":
        if not HAS_C_BACKEND:
            raise RuntimeError("C backend requested but not available.")
        return _primat_c.run(cfg.as_dict(), resolve_rates_dir(cfg))
    from .main import PyPR
    return PyPR(cfg).solve()
```

No exception escapes to the user just because the C extension failed to
build at install time — an `ImportError`-guarded import, same idiom as the
advice doc, just pointed at the corrected module path (`from .main import
PyPR`, since `main.py` now lives inside `primat/` directly, not a separate
`primat_py` package).

---

## 6. Shipping wheels via GitHub Actions (`cibuildwheel`)

### 6.1 `.github/workflows/wheels.yml`

```yaml
name: Build wheels and sdist

on:
  release:
    types: [published]
  workflow_dispatch: {}

jobs:
  build_wheels:
    name: wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-13, macos-14, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: pypa/cibuildwheel@v2.21
        env:
          CIBW_BUILD: "cp310-* cp311-* cp312-* cp313-*"
          CIBW_SKIP: "*-musllinux* pp*"          # skip PyPy + musl initially; add later if needed
          CIBW_ARCHS_MACOS: "x86_64 arm64"
          CIBW_ARCHS_LINUX: "x86_64 aarch64"
          CIBW_ENVIRONMENT: "CFLAGS='-O2'"
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: wheelhouse/*.whl

  build_sdist:
    name: sdist
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pipx run build --sdist
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz

  publish:
    needs: [build_wheels, build_sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write       # PyPI Trusted Publishing (OIDC) -- no token to manage/leak
    steps:
      - uses: actions/download-artifact@v4
        with: { path: dist, merge-multiple: true }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Notes:
- `aarch64` Linux wheels need QEMU emulation in `cibuildwheel` (set up
  automatically when `CIBW_ARCHS_LINUX` includes `aarch64`) — slow but fine
  for a release-triggered job. Drop it if first releases should be faster.
- `manylinux` (the default `cibuildwheel` Linux container) statically links
  against an old-enough glibc baseline automatically — no manual
  configuration needed since `primat-c` has zero non-libc/libm/pthread
  dependencies.
- `windows-latest` uses MSVC; the `extra_compile_args` branch in §5.2
  already drops `-std=c11` there. MSVC's C99/C11 feature support has
  historically lagged GCC/Clang — flag this as a real risk, checked
  empirically in Phase H below. If a feature proves unportable, the Windows
  wheel's extension build simply fails and the pure-Python backend is used
  automatically via §5.4 — `pip install primat` on Windows still works,
  just without the speed boost until the C side is made MSVC-clean.

### 6.2 Registering on PyPI (manual steps)

1. Create an account at pypi.org (and, recommended, `test.pypi.org` for a
   dry run first).
2. Reserve the name early: a manual `twine upload` of a `0.3.0rc0`
   sdist-only build from your laptop is enough to claim `primat` on PyPI
   before automating anything — names are first-come-first-served.
3. Set up **Trusted Publishing** (no API token to store as a secret): on
   PyPI, under the `primat` project → "Publishing" → "Add a new publisher",
   register this GitHub repo + workflow filename (`wheels.yml`) + the
   `pypi` environment name used above. The `id-token: write` permission in
   the workflow then authenticates directly — no long-lived secret in
   GitHub at all.
4. Tag a release on GitHub (`v0.3.0`) → triggers `wheels.yml` via the
   `release: published` event → wheels + sdist built and uploaded
   automatically.
5. Test before the real release: `workflow_dispatch` lets you trigger the
   same job manually against `test.pypi.org` (swap the
   `gh-action-pypi-publish` target via its `repository-url` input) before
   pointing it at the real index.

---

## 7. Unified time-evolution output format

### 7.1 The problem this solves

Today, `pyprimat`'s `output_time_evolution=True` writes a Python-only TSV
(`nuclear_network.py`'s writer) with full `t`, `a`, `T_gamma`,
per-nuclide `Y(t)`, and (for `small`/`small_parthenope`) per-reaction flux
columns. This section pins down a shared schema so a notebook can plot
nuclide evolution from *either* backend's output without caring which one
ran.

### 7.2 Schema (`primat/evolution.py` documents it; both backends implement it)

Plain TSV (tab-separated, `#`-prefixed header line giving column names),
one row per solver output step:

```
# t_s  a  T_gamma_MeV  T_nue_MeV  T_numu_MeV  T_nutau_MeV  Y_n  Y_p  Y_H2  Y_H3  Y_He3  Y_He4  Y_Li7  Y_Be7  [... Y_<nuclide> for every species in the active network, in network order ...]
```

- Column set after the four temperature columns is **network-dependent**
  (small vs. large have different nuclide lists) — the header line is the
  source of truth, read dynamically by the loader, not a fixed column count.
- Units match `CLAUDE.md`'s existing conventions (`t` in seconds, `T` in
  MeV, `Y` as the mass-fraction-style abundance already used throughout).
- Per-reaction flux columns (today small/small_parthenope-only in Python)
  are **deferred from the v0.3.0 unification** — stays a Python-only bonus
  column block until the C network builder's flux bookkeeping is ported.
- File naming: `<run_id>_evolution.tsv`, written next to wherever the
  caller's existing output-location convention already points.

### 7.3 Loader: `primat.evolution.load_evolution(path) -> EvolutionResult`

```python
@dataclass
class EvolutionResult:
    t: np.ndarray            # seconds
    a: np.ndarray
    T_gamma: np.ndarray       # MeV
    T_nu: dict[str, np.ndarray]   # {"e": ..., "mu": ..., "tau": ...}
    Y: dict[str, np.ndarray]      # {"n": ..., "p": ..., "H2": ..., ...}

def load_evolution(path) -> EvolutionResult:
    """Parses the shared TSV schema (§7.2 of PRIMAT.md) written by either
    the Python or C backend, returning the same structure regardless of
    which backend produced the file."""
```

Concrete answer to "speed of C, flexibility of Python for plots": run with
the C backend (`PRIMAT(cfg, backend="c")`, `output_time_evolution=True`),
then `load_evolution(...)` the resulting TSV in the same notebook for
`matplotlib`/`plotly` work — no Python solve needed just to get plottable
arrays.

### 7.4 C-side requirement this adds to `CPLAN.md`

`nuclear_network.c`'s HT/MT/LT solve loop (Phase 6/7 in `CPLAN.md` §13)
must record the same per-step state (`t`, `a`, `T_gamma`, the three `T_nu`,
per-nuclide `Y`) into an in-memory growable array, written out by a
`write_evolution_tsv()` whose output is **column-header-compatible** with
the Python writer (not row-for-row identical, since adaptive solvers don't
take the same steps — but parseable by the same loader, and physically
agreeing to solve tolerance at matching time stamps via interpolation,
exactly like `tests/test_custom_background.py`'s existing comparison
pattern). Add this as an explicit `CPLAN.md` §13 deliverable note on the
existing Phase 6/7 deliverable, not a new phase.

---

## 8. Keeping C and Python in parity (CLAUDE.md + CPLAN.md scope changes)

`CPLAN.md` §0 currently lists `analytic_distortions` (analytic μ/y-type
spectral distortions) as **out of scope for v1**. The user's message asks
for it to be implemented in C. `custom_background` is **already in scope**
per `CPLAN.md` §0 — just re-confirming it isn't lost in this reorganisation.

**Action:** update `CPLAN.md` §0 to move `analytic_distortions` from "out
of scope" to in-scope, ported alongside the NEVO-spectrum-based path it
sits next to in `neutrino_history.py` (`AnalyticDistortion`). This is a
`CPLAN.md` edit, sequenced in §9 below, not a `PRIMAT.md` implementation
detail.

### 8.1 `CLAUDE.md` clause addition

`CLAUDE.md`'s existing "Keeping CPRIMAT and PyPRIMAT in sync" section is
correct and stays, with names updated (`pyprimat`/`CPRIMAT` →
`primat`/`primat-c`) and one new clause: *"Any addition to the unified
time-evolution schema (§7.2 of `PRIMAT.md`) must be implemented by both
backends' writers before being considered complete — a schema column only
one backend populates is a parity bug, not a feature."*

### 8.2 New test category: cross-backend parity

`tests/test_backend_parity.py` (new): for every config already exercised by
existing reference tests (`small`, `large amax=8`), run *both* backends
(skip the C-backend half with a clear skip-reason if the extension isn't
built in the current environment) and assert:
- scalar observables agree to the tolerances already in `CLAUDE.md`'s
  validation table;
- `load_evolution()` on both backends' TSVs (when
  `output_time_evolution=True`) gives back arrays that agree at matching
  time stamps (via 1D interpolation, like `test_custom_background.py`
  already does) to a documented relative tolerance (start at `1e-5`,
  matching the `custom_background` round-trip precedent already in the
  codebase).

---

## 9. Implementation phases

| Phase | Deliverable | Depends on |
|---|---|---|
| A | `pyproject.toml`/`setup.py` rewrite (no file moves yet) targeting the *current* `pyprimat`/`CPRIMAT` paths, proving the `Extension` build + `cibuildwheel` pipeline works on today's layout. De-risks packaging mechanics independently of the rename. | — |
| B | The rename: `git mv pyprimat primat`, `git mv CPRIMAT primat-c`. Update every internal import (`from pyprimat import PyPR` → `from primat import PRIMAT`, etc., per §10). Add `primat/api.py`/`backend.py`/`cli.py`. | A |
| C | Rates overlay resolver (§4.3) implemented in both `primat/backend.py` and `primat-c/src/rates_resolve.c`; `user_rates_dir`/`rates_dir` config fields added with validation + fingerprinting. | B |
| D | Unified evolution schema (§7): Python writer reviewed/adjusted to the pinned schema if it drifts from it; C writer added per §7.4 (folds into `CPLAN.md` Phase 6/7); `primat/evolution.py` loader + tests. | B, and `CPLAN.md` Phase 6/7 |
| E | `CPLAN.md` scope update (§8: analytic distortions in-scope) + cross-backend parity tests (§8.2). | C, D, `CPLAN.md` Phase 3a/5 |
| F | `runfiles/`, `notebooks/`, `gui/` updated to the new import paths and `--backend` option (§10). | B |
| G | `README.md`, `CLAUDE.md` updated to document the new architecture (§11). | B–F |
| H | `.github/workflows/wheels.yml`, PyPI trusted-publisher setup, first `0.3.0rc0` test-PyPI dry run (includes the empirical MSVC-compatibility check flagged in §6.1). | A–G |
| I | `0.3.0` tag + release. | H |

Each phase is its own PR with the existing CLAUDE.md invariant applied
(reference YP/D-H/per-nuclide values within documented tolerance,
before/after diff attached) wherever it touches solver code.

---

## 10. Runfiles, notebooks, GUI: what changes

- **Import path**: every `from pyprimat import PyPR` becomes `from primat
  import PRIMAT` (the facade keeps the same `solve()`/`get_quantity()`
  surface — a rename + thin re-wrap, not an API redesign). No
  backwards-compatibility shim is added (per standing instruction to avoid
  these) — old import paths simply stop working, and every caller in this
  repo is updated in Phase B/F.
- **New `run_basic.py`** (`runfiles/run_basic.py`, replaces/supplements
  `PyPRIMAT_run.py`): a heavily commented template exposing the most common
  options, all commented out with their default value shown, e.g.:

  ```python
  from primat import PRIMAT

  cfg = dict(
      # backend="auto",        # "auto" (default: C if available, else python), "c", or "python"
      # Omegabh2=0.022425,     # baryon density x h^2 (Planck 2018 default)
      # DeltaNeff=0.0,         # extra relativistic species beyond SM neutrinos
      # network="small",       # "small" / "small_parthenope" / "large" / custom network filename
      # amax=None,             # filter any network to reactions with A <= amax
      # numerical_precision=1e-7,   # rtol for all solve_ivp-equivalent calls
      # output_time_evolution=False,  # write the unified <run_id>_evolution.tsv (§7.2 of PRIMAT.md)
      # user_rates_dir=None,   # overlay directory for your own network/table additions (§4.3)
  )
  result = PRIMAT(cfg).solve()
  print(result["YPBBN"], result["DoH"])
  ```
- **Equivalent `.ini`** (`primat-c/examples/run_basic.ini`): same options,
  `KEY=VALUE` syntax, same commented-by-default convention, for direct
  `primat-c` standalone invocation.
- **GUI**: `primat/gui/launcher.py` entry point renamed from `pyprimat-gui`
  to `primat-gui`; internal imports updated; a new sidebar control for
  backend selection (`auto`/`c`/`python`) — defaulting to `auto`, surfaced
  next to the existing network-choice controls, not a new dialog.
- **Notebooks**: update the import cell at the top of each; no structural
  changes otherwise (the facade's `solve()` return dict is unchanged).

---

## 11. `README.md` / `CLAUDE.md` updates required (content, not yet applied)

**`README.md`** needs a new top section explaining:
- `pip install primat` gets you the fast C backend automatically on
  supported platforms (with the pure-Python fallback silently kicking in
  otherwise) — most users need nothing else;
- `pip install -e .` for development, and what that means for editing
  `primat/rates/` directly;
- how to force a backend (`PRIMAT(cfg, backend="c"/"python")`, or `primat
  --backend python ...` on the CLI);
- how to add custom rate tables/networks via `user_rates_dir` without
  touching the installed package (§4.3);
- how to build/extend the standalone `primat-c` C-only binary (`cd
  primat-c && make`), for users who want zero Python involved.

**`CLAUDE.md`** needs:
- every `pyprimat`/`CPRIMAT` reference in the Architecture section updated
  to `primat`/`primat-c`, and the tree diagram updated to match §3;
- the "Keeping CPRIMAT and PyPRIMAT in sync" section's clause addition from
  §8.1;
- a new short subsection documenting the rates-overlay resolver (§4) as
  part of the architecture description — load-bearing behaviour a future
  Claude Code session needs to know before touching rate-loading code;
- a note that `CPLAN.md`'s analytic-distortions exclusion has been lifted
  (§8).

---

## 12. Open items intentionally left for the implementation PRs, not this plan

- Exact `EvolutionResult`/TSV column ordering bikeshedding (this plan fixes
  the four temperature columns + per-nuclide block; nuclide ordering should
  match whatever `nuclear_network.py`'s current writer already does).
- Whether `primat-c`'s MSVC build needs any C11-feature substitutions
  (flagged in §6.1, resolved empirically in Phase H).
- Per-reaction flux columns in the unified schema (explicitly deferred,
  §7.2).
- `musllinux`/PyPy wheels (skipped initially in §6.1's `CIBW_SKIP`) — add
  later only if requested.
