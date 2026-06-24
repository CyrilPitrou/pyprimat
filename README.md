# primat

A precise Big Bang Nucleosynthesis (BBN) solver. It integrates coupled ODEs
for the cosmological background (photon/neutrino temperatures, scale factor)
and a nuclear reaction network to predict primordial abundances of H, D,
He3, He4, Li7, and heavier nuclides.

`primat` ships two interchangeable backends: a pure-Python implementation
(`primat/`, the one this README documents) and a standalone C99 port
(`primat-c/`, built independently via `make`) intended for speed-critical use
(MCMC loops, large-scale parameter scans). The two backends are kept
numerically in sync and are converging on a shared CLI/output-format
contract; wiring the C engine into the `primat` Python package as a
selectable `--backend` is in progress (see `PRIMAT.md`).

## Three ways to use primat

primat can be driven in three equivalent ways — all three build the same
``params`` dict and call ``PRIMAT(params=params).primat_results()``, so they
always agree on results for the same configuration:

1. **As a Python library** — `from primat import PRIMAT` in a script or
   notebook (see [Quick start](#quick-start)). The most flexible option:
   the full `primat.config.DEFAULT_PARAMS` surface is available.
2. **The `primat` command-line tool** — a quick one-liner for the most
   commonly varied parameters, e.g. `primat --Omegabh2 0.02242 --network
   large --amax 8` (see [Command-line interface](#command-line-interface)).
3. **The `primat-gui` graphical interface** — a browser-based app with a
   grouped parameter form, an interactive abundance-evolution plot, and a
   final-abundances/ratios panel (see [Graphical interface](#graphical-interface-gui)).

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd PyPRIMAT
pip install -e .
```

(Once published, a plain `pip install primat` will pull a pre-built wheel
with the C backend already compiled — no local toolchain needed.)

With optional dependencies for best performance:

```bash
pip install -e ".[recommended]"
```

| Package | Role |
|---------|------|
| `numpy`, `scipy`, `joblib`, `plotly` | **Mandatory** (installed by `pip install -e .`) |
| `numba` | Recommended — JIT compilation gives ~5× speedup on rate kernels |
| `vegas` | Recommended — Monte Carlo integration for thermal weak-rate corrections |

For the graphical interface (`primat-gui`), install the `gui` extra:

```bash
pip install -e ".[gui]"
```

| Package | Role |
|---------|------|
| `streamlit` | **Required for `primat-gui`** — the web app framework |
| `pandas` | **Required for `primat-gui`** — final-abundance table |

For the example notebooks under `notebooks/`, install the `notebooks` extra:

```bash
pip install -e ".[notebooks]"
```

| Package | Role |
|---------|------|
| `matplotlib`, `pandas` | Plotting and tabular display in the notebooks |
| `papermill` | Headless notebook execution |

## Quick start

```python
from primat import PRIMAT

result = PRIMAT({"Omegabh2": 0.022425}).solve()

print(f"YP  (BBN) = {result['YPBBN']:.6f}")  # ~0.246915
print(f"D/H = {result['DoH']:.5e}")          # ~2.43647e-05
```

The constructor accepts an optional parameter dict that overrides any default in `primat/config.py`. All keys are optional.

## Running the example scripts

Scripts live in `runfiles/`. Run from the repo root:

```bash
python runfiles/PyPRIMAT_run.py           # Standard SM run (outputs results/output_tables.tsv)
python runfiles/PyPRIMAT_compare.py       # Small vs large network comparison
python runfiles/PyPRIMAT_reference_run.py # High-precision reference run (~2 min)
```

## Command-line interface

The `primat` console script wraps the same "build a `params` dict and call
`PRIMAT`" pattern, exposing the most commonly varied options without writing
any Python:

```bash
primat --Omegabh2 0.02242 --network large --amax 8
```

```
Neff       = 3.04397730
YP (BBN)   = 0.24699808
YP (CMB)   = 0.24567178
D/H        = 2.4365389e-05
He3/H      = 1.0397042e-05
He3/He4    = 1.2677615e-04
Li7/H      = 5.501865e-10
Li6/Li7    = 1.418945e-05
--- running time: 3.67 seconds ---
```

| Flag | Description |
|------|-------------|
| `--Omegabh2 VALUE` | Baryon density Ω_b h² (default: 0.022425) |
| `--DeltaNeff VALUE` | Extra relativistic degrees of freedom (default: 0) |
| `--network {small,small_parthenope,large}` | Nuclear reaction network (default: small) |
| `--amax A` | Drop reactions involving any nuclide with mass number > A (integer >= 1); applies to any `--network` |
| `--numerical_precision RTOL` | `solve_ivp` relative tolerance (default: 1e-7) |
| `--json` | Print the full results dict as JSON instead of the short summary |
| `--verbose` | Enable primat's internal progress messages (timings, cache hits, ...) |

Only flags you pass are forwarded to `PRIMAT`; anything else falls back to
`primat.config.DEFAULT_PARAMS`. For options not exposed as flags, write a
short script that builds a `params` dict and calls `PRIMAT` directly (see
[Quick start](#quick-start)).

## Graphical interface (GUI)

After installing the `gui` extra (`pip install -e ".[gui]"`), launch the
browser-based app with:

```bash
primat-gui
```

From a source checkout you can also run it directly with Streamlit:

```bash
streamlit run primat/gui/app.py
# or
python -m primat.gui.launcher
```

The app mirrors a single CLI/script run:

- **Sidebar** — a parameter form grouped into *Cosmology*, *Nuclear
  reactions* (network/amax + the "Import/Create custom network" popups) and
  *Physics* (weak rates, plasma physics, nuclear QED corrections) sections,
  plus a *Constants* expander for `GN`/`tau_n` and an *Uncertainty* expander
  with a **Quick MC uncertainty (30 samples)** toggle, and a **Run BBN**
  button.
- **Final abundances tab** — the standard ratios (`Neff`, `YP` (BBN/CMB),
  `D/H`, `³He/H`, `³He/⁴He`, `⁷Li/H`) as metric cards, a sortable table of
  every tracked nuclide's final abundance, and a download button for an
  `output_final.dat`-style table. With **Quick MC uncertainty** enabled, an
  extra "± 1σ (quick MC, 30 samples)" column shows a fast Monte Carlo estimate
  (varying every nuclear-rate `p_*` and the neutron lifetime `tau_n`, see
  `mc_uncertainty`) — a quick, noisy estimate, not a publication-quality error
  bar.
- **Abundance evolution tab** — an interactive log-log plot of `A_i·Y_i(t)`
  for any selection of nuclides (with "Light elements" / "All" / "Clear"
  presets), with a toggle between cosmic time and photon temperature on the
  x-axis.

The GUI builds the same `params` dict and calls
`PRIMAT(params=params).primat_results()` as the Python API and the `primat` CLI,
so all three agree on results for the same configuration.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Omegabh2` | 0.022425 | Baryon density |
| `DeltaNeff` | 0.0 | Extra relativistic degrees of freedom |
| `network` | `"small"` | `"small"` (12 reactions) / `"small_parthenope"` (12, Parthenope 3.0 tables) / `"large"` (~429), optionally restricted via `amax`. |
| `numerical_precision` | 1e-7 | ODE solver rtol |
| `sampling_temperature_per_decade` | 400 | Background grid points per decade of T |
| `sampling_nTOp_per_decade` | 80 | n↔p rate grid points per decade of T |
| `weak_rate_cache` | True | If False, never load n↔p rates from `rates/weak/` (always recompute) |
| `save_nTOp` | True | Save recomputed n↔p rates to `rates/weak/` with a fingerprint header |
| `thermal_corrections` | True | Include thermal radiative corrections (CCRTh) to the n↔p rates |
| `save_nTOp_thermal` | True | Save recomputed thermal corrections to `rates/weak/` with a fingerprint header |
| `output_time_evolution` | False | Write time-evolution table to `output_file` |
| `output_file` | `results/output_tables.tsv` | Output file path (relative paths resolve against the current working directory) |
| `output_n_points` | 500 | Number of interpolated rows in output file |

### n↔p weak rate workflow

The n↔p weak rates are the most expensive part of initialisation (~1.8 s). The
non-thermal rate (Born+FM+CCR+SD) is cached in `rates/weak/nTOp_<hash>.txt`
(forward and backward columns together); the finite-temperature radiative
correction (CCRTh) is cached separately in `rates/weak/nTOp_thermal_<hash>.txt`.
Each file is tagged with a *fingerprint* header: a hash of every config field
that affects its numeric content (background thermodynamics,
`sampling_nTOp_per_decade`/`sampling_nTOp_thermal_per_decade`,
`radiative_corrections`, `finite_mass_corrections`, `thermal_corrections`, etc.
— see `primat.weak_rates`). At every run:

- If `weak_rate_cache=True` (default) and a cache file's fingerprint matches the
  current configuration, the corresponding rates are loaded directly —
  initialisation is effectively instantaneous.
- Otherwise (fingerprint mismatch, missing file, or `weak_rate_cache=False`), the
  rates are recomputed from scratch by numerical integration (~1.8 s).
- `save_nTOp` and `save_nTOp_thermal` (both default **`True`**) write the
  (re)computed rates back to `rates/weak/` with a fresh fingerprint header, so
  future runs with the same configuration load the cache. The hash is part of
  the filename, so different configurations coexist without overwriting each
  other — set either flag to `False` only to avoid littering `rates/weak/`
  during throwaway experiments.

Recomputing the thermal correction (`thermal_corrections=True`) requires a
`vegas` Monte Carlo integration that can take minutes to hours; the
fingerprint mechanism above is what makes this avoidable across runs that
share the same configuration.

**Typical workflow for a high-precision study:**
```python
# Step 1 – compute and save high-precision rates once (non-default
# sampling_nTOp_per_decade gives a fingerprint that the shipped cache won't
# match, so this recomputes; save_nTOp=True is the default but spelled out
# here for clarity)
PRIMAT({"save_nTOp": True, "sampling_nTOp_per_decade": 160}).solve()

# Step 2 – all subsequent runs with the same sampling_nTOp_per_decade reuse the saved tables
PRIMAT({"sampling_nTOp_per_decade": 160}).solve()
```

### Custom NEVO tables

The neutrino-decoupling history is read from `rates/NEVO/`. Three optional
parameters point at alternative tables instead (filenames resolved relative
to `rates/NEVO/`, or absolute paths): `nevo_file` (6/7-column thermo table),
`nevo_spectral_file` (spectral-distortion table, used only when
`spectral_distortions=True` and `analytic_distortions=False`), and
`nevo_grid_file` (its y-grid, length must match `nevo_spectral_file`'s
spectral-column count). Each defaults to `None` (the shipped table selected by
`QED_corrections`); a custom file is validated for existence and shape at
construction time, and is included in the n↔p weak-rate cache fingerprint so
a different table correctly triggers a recompute.

Each nuclear reaction rate has a `p_<name>` parameter (e.g. `p_npTOdg`) for uncertainty propagation: setting it to a non-zero float samples the rate at `median × exp(p × σ)`.

### Rates overlay (custom networks/tables without editing the install)

`user_rates_dir` points at a directory with the same `nuclear/networks/`
and/or `nuclear/tables/<name>/` layout as the shipped `rates/` tree; any
network file or per-reaction table found there is used instead of the
shipped one, while everything not overridden still falls back to the
shipped default (an additive overlay, not a takeover). `rates_dir` instead
fully replaces the shipped tree as the first lookup base (still falling
back to the shipped tree as a last resort, so `small`/`large` never
disappear). Both default to `None` and are validated as existing
directories at construction time.

## Output

`solve()` returns a dict:

| Key | Description |
|-----|-------------|
| `YPBBN` | Helium-4 mass fraction (BBN convention) |
| `YPCMB` | Helium-4 mass fraction (CMB convention) |
| `DoH` | D/H |
| `He3oH` | ((He3+T)/H |
| `Li7oH` | (Li7+Be7)/H |
| `Neff` | Effective number of neutrino species |
| `Omeganurel` | Ω_ν h² × 10⁶ (relativistic) |
| `OneOverOmeganunr` | 1 / (Ω_ν h² × 10⁻⁶) (non-relativistic) |

When `output_time_evolution=True`, a TSV file is written with columns:
`a, T, t, H, Tnue, Tnumu, Tnutau, [Nheating], [abundances], n_to_p_weak_rate, p_to_n_weak_rate, [nuclear rates]`

`Nheating` is included only for `incomplete_decoupling=True` (a real NEVO
heating table). `[abundances]` is one `Y<species>` column per nuclide of the
chosen network (8 for small/small_parthenope, ~59 for large, fewer with an
`amax` cutoff). `[nuclear rates]` (`output_rates_time_evolution=True`) is
available for small/small_parthenope only; it is omitted (with a printed
note) for `network="large"`.

## Architecture

```
primat/                    Core package
  config.py              PRIMATConfig: all physical constants + run-time flags
  main.py                PRIMAT: top-level driver
  plasma.py              Plasma thermodynamics (QED corrections, neutrino bath)
  qed_pressure.py        Analytical QED plasma-pressure corrections
  network_data.py        Nuclear network related functions
  network_builder.py     Generic stoichiometry-driven RHS/Jacobian (numba kernels)
  weak_rates.py          n ↔ p weak rate computation

rates/
  plasma/                QED corrections pressure tables
  nuclear/
    tables/              Per-reaction rate tables, one folder per reaction:
                           tables/<name>/<name>.txt (+ sibling alternate tables)
    networks/            Network list files: small_parthenope.txt, large.txt, …
    data/                nuclides.csv, reactions_large.csv, detailed_balance.csv
  weak/                  Pre-tabulated n↔p forward/backward rates
  NEVO/                  Non-instantaneous decoupling table

generate_rates/    Offline one-off generator (run only to refresh the
                         rate/network data from AC2024 + PRIMAT-main.m + NUBASE):
                           python generate_rates/convert_ac2024_rates.py

primat-c/          Standalone C99 port of the same solver (independent `make`
                         build, see primat-c/Makefile), kept numerically in
                         sync with this Python implementation
```

### Networks

Two named networks (plus a Parthenope-rates variant of the small one) are
available via the `network` flag; `amax` (any positive integer) further
restricts *any* of them to reactions whose nuclides all have mass number
A ≤ amax:

| `network` | Reactions | Nuclides | Notes |
|-----------|-----------|----------|-------|
| `"small"`  | 12  | 8  | the key reactions; fastest |
| `"small_parthenope"` | 12 | 8 | same reactions, Parthenope 3.0 rate tables (comparison runs) |
| `"large"`  | ~429 | ~59 | from the AC2024 compilation; LT era only |
| `"large"`, `amax=8` | 68 | 12 | the old "medium" network's exact equivalent |
| `"large"`, `amax=2` | 3 | 3 | the old "deuterium" network's equivalent (n↔p + n_p__d_g + p_p_n__d_p) |

All networks share the HT (n↔p) and MT eras (the MT era always uses a fixed
18-reaction subset, too stiff to run the full network); only the LT reaction
set is filtered by `network`/`amax`. The light-element abundances of the full
large network match the `amax=8` restriction to ≲1e-4; its heavy-nuclide tail
(B, C, N, O, …) is approximate. See `notebooks/AbundanceEvolution.ipynb` for
evolution plots.

### Custom networks (GUI)

The `primat-gui` sidebar's "Nuclear reactions" group offers **"Create
custom network"** (a popup to start from any named network, toggle reactions
in/out by mass-number category, and substitute or upload alternate rate
tables) and **"Import custom network"** (re-load a previously saved
`.zip`).

## Cobaya / MCMC interface

A wrapper for primat is available for use
with [Cobaya](https://cobaya.readthedocs.io), allowing BBN to be embedded directly
in MCMC analyses of CMB or other cosmological data.  The wrapper exposes
`Omegabh2`, `DeltaNeff`, and the nuclear-rate uncertainty parameters as Cobaya
theory/likelihood inputs and returns the standard BBN observables (`YPBBN`, `DoH`,
etc.) for use in a likelihood.

## Citation

If you use primat please cite:

> Pitrou, Coc, Uzan, Vangioni, *Physics Reports* **754** (2018) 1–67.  
> [doi:10.1016/j.physrep.2018.04.005](https://doi.org/10.1016/j.physrep.2018.04.005)

## Authors

Cyril Pitrou (<pitrou@iap.fr>), Julien Froustey
