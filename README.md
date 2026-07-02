# primat

A precise Big Bang Nucleosynthesis (BBN) solver. It integrates coupled ODEs
for the cosmological background (photon/neutrino temperatures, scale factor)
and a nuclear reaction network to predict primordial abundances of H, D,
He3, He4, Li7, and heavier nuclides.

## Installation

**For most users:**

```
pip install primat
```

That's it. The package includes a fast C backend compiled for your platform,
with a pure-Python fallback if no compiled binary is available — both give
identical results, just different speed. To get started, just type

```
primat --help
```

**For development, examples, and notebooks:**

Clone the repository and install in editable mode:

```bash
git clone https://github.com/CyrilPitrou/primat
cd primat
pip install -e .
```

With optional dependencies for best performance:

```bash
pip install -e ".[recommended]"
```

| Package | Role |
|---------|------|
| `numpy`, `scipy`, `joblib`, `plotly` | **Mandatory** (installed by `pip install primat`) |
| `numba` | Recommended — JIT compilation gives ~5× speedup on rate kernels |
| `vegas` | Recommended — Monte Carlo integration for thermal weak-rate corrections |

For the graphical interface (`primat-gui`), install the `gui` extra:

```bash
pip install "primat[gui]"
```

| Package | Role |
|---------|------|
| `streamlit` | **Required for `primat-gui`** — the web app framework |
| `pandas` | **Required for `primat-gui`** — final-abundance table |

For the example notebooks, install from source:

```bash
pip install -e ".[notebooks]"
```

| Package | Role |
|---------|------|
| `matplotlib`, `pandas` | Plotting and tabular display in the notebooks |

## Quick start

```python
from primat.backend import run_bbn

result = run_bbn({"Omegabh2": 0.022425})

print(f"YP  (BBN) = {result['YPBBN']:.6f}")  # ~0.246915
print(f"D/H = {result['DoH']:.5e}")          # ~2.43647e-05
```

`run_bbn()` is the main entry point and automatically selects the best available
backend (fast C engine by default, pure-Python fallback if needed). Pass an optional
parameter dict to override defaults; all keys are optional and drawn from
`primat/config.py`'s `DEFAULT_PARAMS`.

## Using primat

There are four ways to use primat, all of which produce identical results:

### 1. Python API (recommended)

```python
from primat.backend import run_bbn

# Automatically selects C backend if available, falls back to pure-Python
result = run_bbn({"Omegabh2": 0.022425, "network": "large"})
```

To force a specific backend:

```python
result = run_bbn({"network": "small"}, force_backend="c")       # C only
result = run_bbn({"network": "small"}, force_backend="python")  # Python only
```

### 2. Command-line interface

```bash
primat --Omegabh2 0.02242 --network large --amax 8
```

Output:
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
| `--amax A` | Drop reactions involving any nuclide with mass number > A; applies to any network |
| `--numerical_precision RTOL` | `solve_ivp` relative tolerance (default: 1e-7) |
| `--backend {auto,c,python}` | Force a backend (default: `auto`) |
| `--json` | Print full results dict as JSON instead of summary |
| `--verbose` | Enable progress messages (timings, cache hits, ...) |
| `--set KEY VALUE` | Set any configuration parameter (e.g., `--set tau_n 880.1`); use `primat --help` for the full list |

Run `primat --help` to see all available command-line options. For parameters not exposed as flags, use `--set` or the Python API.

### 3. Graphical interface (GUI)

After installing the `gui` extra:

```bash
primat-gui
```

The browser-based app offers a parameter form, interactive abundance-evolution
plot, and final-abundances panel. It supports custom networks, time-evolution
output, and can use either the C or Python backend (automatically selected
like the CLI, or pinned for the whole session with `primat-gui --backend
{auto,c,python}`, e.g. `primat-gui --backend python` to exercise the
pure-Python backend).

### 4. Example scripts (development/source-only)

Clone the repo and run from the root:

```bash
python runfiles/primat_run.py           # Standard SM run
python runfiles/primat_compare.py       # Network comparison
python runfiles/primat_reference_run.py # High-precision run (~2 min)
```

## Backend selection

`run_bbn()` automatically picks the best available backend:
- **Default (`force_backend=None` or `"auto"`)**: C engine if available
  (pre-compiled in wheels), pure-Python fallback otherwise
- **Force C (`force_backend="c"`)**: Raises if C backend is unavailable
- **Force Python (`force_backend="python"`)**: Useful for development or
  when using Python-only features

Python-only features (that force fallback to pure-Python even with
`force_backend="auto"`):
- `custom_network` (GUI "Create custom network" feature)
- `output_time_evolution=True` (write full time series)
- `extra_rho`, `background=` arguments

### Using primat-c directly

For users who prefer to work directly with the C code, the `primat-c/` directory
contains a standalone C99 implementation that can be compiled independently.
See `primat-c/README.md` for detailed compilation instructions and usage
information for various platforms.


## Key parameters

### Physics parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Omegabh2` | 0.022425 | Baryon density Ω_b h² |
| `DeltaNeff` | 0.0 | Extra relativistic degrees of freedom |
| `tau_n` | 878.4 | Neutron lifetime [s] |
| `network` | `"small"` | `"small"` (12 reactions) / `"small_parthenope"` (12, Parthenope 3.0 tables) / `"large"` (~429), optionally restricted via `amax`. |
| `amax` | None | Maximum mass number A for nuclides in reactions (filters any network) |
| `radiative_corrections` | True | Coulomb + T=0 resummed radiative corrections to n↔p (CCR) |
| `finite_mass_corrections` | True | Fokker-Planck finite-nucleon-mass correction (FM) |
| `thermal_corrections` | True | Finite-temperature radiative corrections to n↔p (CCRTh) |
| `spectral_distortions` | True | Correct n↔p rates for non-FD neutrino distributions (SD) |
| `tau_n_normalization` | True | Normalise weak rates using τ_n (neutron lifetime) |

### Precision parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numerical_precision` | 1e-7 | `solve_ivp` relative tolerance (rtol) for all ODE integration |
| `sampling_temperature_per_decade` | 400 | Background grid points per decade of T |
| `sampling_nTOp_per_decade` | 80 | n↔p rate grid points per decade of T |
| `rate_grid_npts` | 1000 | Points in the master T9 grid used for rate-table resampling |
| `rate_grid_T9_min` | 1e-3 | Minimum T9 [GK] of the master rate grid |
| `rate_grid_T9_max` | 10.0 | Maximum T9 [GK] of the master rate grid |

### Caching parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weak_rate_cache` | True | If False, never load n↔p rates from `data/weak/` (always recompute) |
| `save_nTOp` | True | Save recomputed n↔p rates to `data/weak/` with a fingerprint header |
| `save_nTOp_thermal` | True | Save recomputed thermal corrections to `data/weak/` with a fingerprint header |

### Output parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_time_evolution` | False | Generate time-evolution data (accessible via result["evolution"]) |
| `output_file` | `results/output_tables.tsv` | Output file path for time evolution (relative to current directory) |
| `output_n_points` | 500 | Number of interpolated rows in output file |
| `output_rates_time_evolution` | False | Include per-reaction rate columns in time-evolution output (small/small_parthenope only) |

### n↔p weak rate workflow

The n↔p weak rates are the most expensive part of initialisation (~1.8 s). The
non-thermal rate (Born+FM+CCR+SD) is cached in `data/weak/nTOp_<hash>.txt`
(forward and backward columns together); the finite-temperature radiative
correction (CCRTh) is cached separately in `data/weak/nTOp_thermal_<hash>.txt`.
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
  (re)computed rates back to `data/weak/` with a fresh fingerprint header, so
  future runs with the same configuration load the cache. The hash is part of
  the filename, so different configurations coexist without overwriting each
  other — set either flag to `False` only to avoid littering `data/weak/`
  during throwaway experiments.

Recomputing the thermal correction (`thermal_corrections=True`) requires a
`vegas` Monte Carlo integration that can take a few minutes; the
fingerprint mechanism above is what makes this avoidable across runs that
share the same configuration.

**Typical workflow for a high-precision study:**
```python
from primat.backend import run_bbn

# Step 1 – compute and save high-precision rates once (non-default
# sampling_nTOp_per_decade gives a fingerprint that the shipped cache won't
# match, so this recomputes; save_nTOp=True is the default)
result1 = run_bbn({"save_nTOp": True, "sampling_nTOp_per_decade": 160})

# Step 2 – all subsequent runs with the same sampling_nTOp_per_decade reuse the saved tables
result2 = run_bbn({"sampling_nTOp_per_decade": 160})
```

### Custom NEVO tables

The neutrino-decoupling history is read from `data/NEVO/`. Three optional
parameters point at alternative tables instead (filenames resolved relative
to `data/NEVO/`, or absolute paths): `nevo_file` (6/7-column thermo table),
`nevo_spectral_file` (spectral-distortion table, used only when
`spectral_distortions=True` and `analytic_distortions=False`), and
`nevo_grid_file` (its y-grid, length must match `nevo_spectral_file`'s
spectral-column count). Each defaults to `None` (the shipped table selected by
`QED_corrections`); a custom file is validated for existence and shape at
construction time, and is included in the n↔p weak-rate cache fingerprint so
a different table correctly triggers a recompute. 
For the moment `nevo_grid_file` is assumed to be a Gauss-Laguerre quadrature. 
The format for handling NEVO results to primat will evolve in future releases.


### Data directory override and nuclear overlay

`user_nuclear_dir` points at a directory with the same `networks/`
and/or `tables/<name>/` layout as the shipped `data/nuclear/` folder; any
network file or per-reaction table found there is used instead of the
shipped one, while everything not overridden still falls back to the
shipped default (an additive overlay, not a takeover). `data_dir` instead
fully replaces the entire `primat/data/` tree (NEVO/, weak/, plasma/,
nuclear/, csv/), so all data files are read from the supplied directory.
Both default to `None` and are validated as existing directories at
construction time.


### Nuclear rate variation and sensitivity analysis

primat provides two distinct mechanisms for varying nuclear reaction rates:

#### 1. Log-normal rate variations: `p_<reaction>` parameters

Each nuclear reaction has a corresponding parameter `p_<name>` (e.g., `p_n_p__d_g` for the n + p → d + γ reaction). This varies the rate as:

**Rate = median × exp(p × σ)**

where σ is the rate's log-normal uncertainty width (from the rate table's error column).

- **Primary use case**: Monte Carlo uncertainty propagation. Use `run_mc()` or `mc_uncertainty()` to automatically sample p_* from N(0,1) for each reaction.
- **Manual use**: You can also set `p_<name>` directly to explore fixed variations. For example, `p_n_p__d_g = 1` increases the rate by roughly +1σ, while `p_n_p__d_g = -2` decreases it by roughly -2σ.

For systematic MC runs, use `backend.run_mc()`:

```python
from primat.backend import run_mc

# Run MC with nuclear rate uncertainties
mc_result = run_mc(
    params={"Omegabh2": 0.022425},
    n_samples=100,
    quantities=["DoH", "YPBBN"]  # Quantities to compute statistics for
)

print(f"D/H mean: {mc_result['DoH'].cent:8e} ± {mc_result['DoH'].err:8e}")
```

`p_<reaction>` parameters can be set via the CLI using `--set`:
```bash
primat --set p_n_p__d_g=1  # Fixed variation: increase n+p->d+gamma rate by ~1σ
```

#### 2. Additive rate rescaling: `rescale_nuclear_rates` + `delta_<reaction>`

For deterministic sensitivity studies, enable `rescale_nuclear_rates=True`. This activates additive variation parameters `delta_<name>`. When `p_<name>=0` (the default), the rate becomes:

**Rate = median × (1 + delta_<name>)**

This allows uniform or per-reaction rescaling. When both `rescale_nuclear_rates=True` AND `p_<name>≠0`, the combined formula is:

**Rate = median × (exp(p × σ) + delta_<name>)**

Example:
```python
from primat.backend import run_bbn

# Sensitivity study: vary n+p->d+gamma rate by +10%
result = run_bbn({
    "rescale_nuclear_rates": True,
    "delta_n_p__d_g": 0.1
})
```

**Important**: The `p_<reaction>` mechanism is designed for MC uncertainty propagation (log-normal variations), while `rescale_nuclear_rates` + `delta_<reaction>` is designed for deterministic sensitivity studies (additive variations). They can be used together but interpret the combined effect carefully.

#### 3. Computing the uncertainty: `run_mc()` and `--mc N`

`run_mc()` (or its pure-Python counterpart `primat.main.mc_uncertainty()`)
computes the propagated nuclear-rate/τ_n uncertainty on any observable: it
runs many independent BBN solves, each with randomly-sampled reaction rates
(and neutron lifetime), and reports the spread of results as the
uncertainty.

```python
from primat.backend import run_mc

mc = run_mc(100, ["YPBBN", "DoH"], params={"Omegabh2": 0.022425})

mc["DoH"].central   # nominal (best-estimate) value
mc["DoH"].mean      # mean over the 100 MC samples
mc["DoH"].std       # MC uncertainty (1-sigma) -- this is "the error"
mc["DoH"].values    # full array of per-sample values, length 100
```

Regardless of which `quantities` you ask for, the result also always
includes every standard observable (`Neff`, `YPBBN`, `YPCMB`, `DoH`,
`He3oH`, `He3oHe4`, `Li7oH`, `Li6oLi7`, `YCNO`) and every tracked nuclide's
final abundance, at no extra cost.

From the command line, just add `--mc N`:

```bash
primat --Omegabh2 0.022425 --mc 100
# YP (BBN)   = 0.24700028 +/- 0.00003123
# D/H        = 2.4350000e-05 +/- 1.2000000e-07
```

`--mc-seed` sets the random seed (use the same seed to reproduce a run) and
`--mc-jobs` the number of parallel workers. Pass `--output_mc_samples
--output_mc_file FILE` to additionally dump every raw per-sample value to a
TSV, useful for inspecting the full distribution rather than just its
mean/std.

## Output

`run_bbn()` returns a dict with the following keys:

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

When a Monte Carlo run is requested (`--mc N` on the CLI, or
`run_mc()`/`mc_uncertainty()` via `to_flat_dict()`), every observable above
also gets a matching `sigma_<key>` entry with its 1-sigma MC uncertainty,
e.g. `sigma_DoH` alongside `DoH`, `sigma_YPBBN` alongside `YPBBN`.

When `output_time_evolution=True`, the time evolution data is made available. If `output_file` is set to a path, a TSV file is written with columns:
`a, T, t, H, Tnue, Tnumu, Tnutau, [Nheating], [abundances], n_to_p_weak_rate, p_to_n_weak_rate, [nuclear rates]`.

If `output_file=None` (the default), no file is written to disk, but the time evolution data is still accessible via the `"evolution"` key in the result dictionary returned by `run_bbn()`. The `primat.evolution` and `primat.plotting` modules provide tools for working with and plotting this time evolution data (see the example notebooks for usage).

`Nheating` is included only for `incomplete_decoupling=True` (a real NEVO
heating table). `[abundances]` is one `Y<species>` column per nuclide of the
chosen network (8 for small/small_parthenope, ~59 for large, fewer with an
`amax` cutoff). `[nuclear rates]` (`output_rates_time_evolution=True`) is
available for small/small_parthenope only; it is omitted (with a printed
note) for `network="large"`.

## Architecture

```
primat/                    Core Python package
  backend.py             Main entry point: run_bbn() dispatch (C vs pure-Python)
  config.py              PRIMATConfig: all physical constants + run-time flags
  main.py                PRIMAT class: low-level Python implementation
  background.py          Cosmological background (a<->t<->T, weak rates, Neff)
  nuclear_network.py     Nuclear network ODE integration (HT/MT/LT eras)
  plasma.py              Plasma thermodynamics (QED corrections, neutrino bath)
  qed_pressure.py        Analytical QED plasma-pressure corrections
  network_data.py        Nuclear network definition and loading
  network_builder.py     Generic stoichiometry-driven ODE builders (numba kernels)
  weak_rates/            n <-> p weak rate computation (integrands, corrections, cache)
  neutrino_history.py    NEVO non-instantaneous decoupling table I/O
  evolution.py           Unified time-evolution TSV schema
  cli.py                 `primat` command-line entry point
  gui/                   `primat-gui` Streamlit app (optional, source-only)
  data/                  Shipped default data tree
  _primat_c/             Compiled C extension bridge (wraps primat-c)

primat/data/
  nuclear/            Nuclear reaction data
    tables/          Per-reaction rate tables (one folder per reaction)
    networks/        Network list files (small.txt, large.txt, custom.txt, etc.)
  csv/               Reaction catalog (nuclides.csv, detailed_balance.csv, reactions_large.csv)
  plasma/            Pre-computed QED pressure tables
  weak/              Cached n<->p forward/backward rates
  NEVO/              Neutrino-decoupling history tables

primat-c/                Standalone C99 port (independent build via `make`)
                         Also compiled as extension for the Python backend.
                         See primat-c/README.md for details.

generate_rates/          Offline rate-table generator (one-time use)
                         Converts AC2024 compilation to primat format
```

### Backend dispatch

`run_bbn()` (`primat/backend.py`) is the single entry point:
- **C backend** (default): Precompiled in wheels, ~25× faster, deterministic numerical
  differences (~1e-8 relative) vs. Python that are budgeted separately
- **Python backend** (fallback or explicit): Pure Python, all features, no
  compilation needed, slightly slower, useful for development

All three interfaces (Python API, CLI, GUI) ultimately call `run_bbn()`
or the pure-Python fallback.

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

A wrapper for primat will be available for use
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
