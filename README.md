# PyPRIMAT

A Python implementation of the [PRIMAT](https://primat.org) package for precise Big Bang Nucleosynthesis (BBN) computations. It integrates coupled ODEs for the cosmological background (photon/neutrino temperatures, scale factor) and a nuclear reaction network to predict primordial abundances of H, D, He3, He4, Li7, and heavier nuclides.

## Three ways to use PyPRIMAT

PyPRIMAT can be driven in three equivalent ways — all three build the same
``params`` dict and call ``PyPR(params=params).PyPRresults()``, so they
always agree on results for the same configuration:

1. **As a Python library** — `from pyprimat import PyPR` in a script or
   notebook (see [Quick start](#quick-start)). The most flexible option:
   the full `pyprimat.config.DEFAULT_PARAMS` surface is available.
2. **The `pyprimat` command-line tool** — a quick one-liner for the most
   commonly varied parameters, e.g. `pyprimat --Omegabh2 0.02242 --network
   medium` (see [Command-line interface](#command-line-interface)).
3. **The `pyprimat-gui` graphical interface** — a browser-based app with a
   grouped parameter form, an interactive abundance-evolution plot, and a
   final-abundances/ratios panel (see [Graphical interface](#graphical-interface-gui)).

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd PyPRIMAT
pip install -e .
```

With optional dependencies for best performance:

```bash
pip install -e ".[recommended]"
```

| Package | Role |
|---------|------|
| `numpy`, `scipy`, `joblib`, `plotly` | **Mandatory** (installed by `pip install -e .`) |
| `numba` | Recommended — JIT compilation gives ~5× speedup on rate kernels |
| `vegas` | Recommended — Monte Carlo integration for thermal weak-rate corrections |

For the graphical interface (`pyprimat-gui`), install the `gui` extra:

```bash
pip install -e ".[gui]"
```

| Package | Role |
|---------|------|
| `streamlit` | **Required for `pyprimat-gui`** — the web app framework |
| `pandas` | **Required for `pyprimat-gui`** — final-abundance table |

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
from pyprimat import PyPR

result = PyPR({"Omegabh2": 0.022425}).solve()

print(f"YP  (BBN) = {result['YPBBN']:.6f}")  # ~0.246915
print(f"D/H = {result['DoH']:.5e}")          # ~2.43647e-05
```

The constructor accepts an optional parameter dict that overrides any default in `pyprimat/config.py`. All keys are optional.

## Running the example scripts

Scripts live in `runfiles/`. Run from the repo root:

```bash
python runfiles/PyPRIMAT_run.py           # Standard SM run (outputs results/output_tables.tsv)
python runfiles/PyPRIMAT_compare.py       # Small vs large network comparison
python runfiles/PyPRIMAT_reference_run.py # High-precision reference run (~2 min)
```

## Command-line interface

The `pyprimat` console script wraps the same "build a `params` dict and call
`PyPR`" pattern, exposing the most commonly varied options without writing
any Python:

```bash
pyprimat --Omegabh2 0.02242 --network medium
```

```
Neff     = 3.04397730
YP (BBN) = 0.24691155
YP (CMB) = 0.24558556
D/H      = 2.4381479e-05
He3/H    = 1.0387101e-05
Li7/H    = 5.489582e-10
--- running time: 1.22 seconds ---
```

| Flag | Description |
|------|-------------|
| `--Omegabh2 VALUE` | Baryon density Ω_b h² (default: 0.022425) |
| `--DeltaNeff VALUE` | Extra relativistic degrees of freedom (default: 0) |
| `--network {small,medium,large}` | Nuclear reaction network (default: small) |
| `--amax A` | With `--network large`, drop reactions involving any nuclide with mass number > A (integer > 7) |
| `--numerical_precision RTOL` | `solve_ivp` relative tolerance (default: 1e-7) |
| `--json` | Print the full results dict as JSON instead of the short summary |
| `--verbose` | Enable PyPRIMAT's internal progress messages (timings, cache hits, ...) |

Only flags you pass are forwarded to `PyPR`; anything else falls back to
`pyprimat.config.DEFAULT_PARAMS`. For options not exposed as flags, write a
short script that builds a `params` dict and calls `PyPR` directly (see
[Quick start](#quick-start)).

## Graphical interface (GUI)

After installing the `gui` extra (`pip install -e ".[gui]"`), launch the
browser-based app with:

```bash
pyprimat-gui
```

From a source checkout you can also run it directly with Streamlit:

```bash
streamlit run pyprimat/gui/app.py
# or
python -m pyprimat.gui.launcher
```

The app mirrors a single CLI/script run:

- **Sidebar** — a parameter form grouped into *Cosmology*, *Network*,
  *Precision*, *Physics* and *Output* sections (plus an *Advanced* expander
  covering the full `pyprimat.config.DEFAULT_PARAMS` surface), an
  *Uncertainty* expander with a **Quick MC uncertainty (30 samples)** toggle,
  and a **Run BBN** button.
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
`PyPR(params=params).PyPRresults()` as the Python API and the `pyprimat` CLI,
so all three agree on results for the same configuration. See `GUI.md` for
the full design.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Omegabh2` | 0.022425 | Baryon density |
| `DeltaNeff` | 0.0 | Extra relativistic degrees of freedom |
| `network` | `"small"` | `"small"` (12 reactions) / `"medium"` (62) / `"large"` (~433). |
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
— see `pyprimat.weak_rates`). At every run:

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
PyPR({"save_nTOp": True, "sampling_nTOp_per_decade": 160}).solve()

# Step 2 – all subsequent runs with the same sampling_nTOp_per_decade reuse the saved tables
PyPR({"sampling_nTOp_per_decade": 160}).solve()
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
chosen network (8 / 12 / ~59 for small/medium/large). `[nuclear rates]`
(`output_rates_time_evolution=True`) is available for small/medium only; it is
omitted (with a printed note) for `network="large"`.

## Architecture

```
pyprimat/                    Core package
  config.py              PyPRConfig: all physical constants + run-time flags
  main.py                PyPR: top-level driver
  plasma.py              Plasma thermodynamics (QED corrections, neutrino bath)
  qed_pressure.py        Analytical QED plasma-pressure corrections
  network_data.py        Nuclear network related functions
  network_builder.py     Generic stoichiometry-driven RHS/Jacobian (numba kernels)
  weak_rates.py          n ↔ p weak rate computation

rates/
  plasma/                QED corrections pressure tables
  nuclear/
    tables/              Per-reaction rate tables (.txt)
    networks/            Network list files: small.txt, medium.txt, large.txt, …
    data/                nuclides.csv, reactions_large.csv, detailed_balance.csv
  weak/                  Pre-tabulated n↔p forward/backward rates
  NEVO/                  Non-instantaneous decoupling table

generate_rates/    Offline one-off generator (run only to refresh the
                         rate/network data from AC2024 + PRIMAT-main.m + NUBASE):
                           python generate_rates/convert_ac2024_rates.py
```

### Networks

Three networks are available via the `network` flag:

| `network` | Reactions | Nuclides | Notes |
|-----------|-----------|----------|-------|
| `"small"`  | 12  | 8  | the key reactions; fastest |
| `"medium"` | 62  | 12 | the standard full network |
| `"large"`  | ~433 | ~59 | from the AC2024 compilation; LT era only |

All three share the HT (n↔p) and MT eras (the MT era always uses a fixed
18-reaction subset, too stiff to run the full network); only the LT reaction set
grows with `network`. The light-element abundances of the large network match the
medium one to ≲1e-4; its heavy-nuclide tail (B, C, N, O, …) is approximate. See
`notebooks/AbundanceEvolution.ipynb` for evolution plots of all three.

## Cobaya / MCMC interface

A wrapper for PyPRIMAT is available for use
with [Cobaya](https://cobaya.readthedocs.io), allowing BBN to be embedded directly
in MCMC analyses of CMB or other cosmological data.  The wrapper exposes
`Omegabh2`, `DeltaNeff`, and the nuclear-rate uncertainty parameters as Cobaya
theory/likelihood inputs and returns the standard BBN observables (`YPBBN`, `DoH`,
etc.) for use in a likelihood.

## Citation

If you use PyPRIMAT please cite:

> Pitrou, Coc, Uzan, Vangioni, *Physics Reports* **754** (2018) 1–67.  
> [doi:10.1016/j.physrep.2018.04.005](https://doi.org/10.1016/j.physrep.2018.04.005)

## Authors

Cyril Pitrou (<pitrou@iap.fr>), Julien Froustey
