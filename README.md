# PyPRIMAT

A Python implementation of the [PRIMAT](https://primat.org) package for precise Big Bang Nucleosynthesis (BBN) computations. It integrates coupled ODEs for the cosmological background (photon/neutrino temperatures, scale factor) and a nuclear reaction network to predict primordial abundances of H, D, He3, He4, Li7, and heavier nuclides.

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
| `numpy`, `scipy` | **Mandatory** |
| `numba` | Recommended — JIT compilation gives ~5× speedup on rate kernels |
| `numdifftools` | Recommended — numerical entropy derivatives (only if `analytic_entropy_derivative=False`) |
| `vegas` | Recommended — Monte Carlo integration for thermal weak-rate corrections |

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

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Omegabh2` | 0.022425 | Baryon density |
| `DeltaNeff` | 0.0 | Extra relativistic degrees of freedom |
| `network` | `"small"` | `"small"` (12 reactions) / `"medium"` (62) / `"large"` (~433). |
| `numerical_precision` | 1e-7 | ODE solver rtol |
| `n_temperature_table` | 2000 | Background grid density |
| `sampling_nTOp` | 200 | n↔p rate grid size |
| `weak_rate_cache` | True | If False, never load n↔p rates from `rates/weak/` (always recompute) |
| `save_nTOp` | False | Save recomputed n↔p rates to `rates/weak/` with a fingerprint header |
| `include_nTOp_thermal` | True | Include thermal radiative corrections to the n↔p rates |
| `save_nTOp_thermal` | False | Save recomputed thermal corrections to `rates/weak/` with a fingerprint header |
| `output_time_evolution` | False | Write time-evolution table to `output_file` |
| `output_file` | `results/output_tables.tsv` | Output file path (relative paths resolve against the current working directory) |
| `output_n_points` | 500 | Number of interpolated rows in output file |

### n↔p weak rate workflow

The n↔p weak rates are the most expensive part of initialisation (~1.8 s). They are
cached in `rates/weak/nTOp_frwrd.txt` and `rates/weak/nTOp_bkwrd.txt`, each tagged
with a *fingerprint* header: a hash of every config field that affects its numeric
content (background thermodynamics, `sampling_nTOp`, `nTOp_Born_approximation`,
`include_nTOp_thermal`, etc. — see `pyprimat.weak_rates`). At every run:

- If `weak_rate_cache=True` (default) and the cache file's fingerprint matches the
  current configuration, the rates are loaded directly — initialisation is
  effectively instantaneous.
- Otherwise (fingerprint mismatch, missing file, or `weak_rate_cache=False`), the
  rates are recomputed from scratch by numerical integration (~1.8 s).
- Set **`save_nTOp=True`** to write the (re)computed rates back to `rates/weak/`
  with a fresh fingerprint header, so future runs with the same configuration load
  the cache. `save_nTOp` defaults to `False` so that ad-hoc runs with non-default
  settings do not overwrite the shared cache used by the standard configuration.

The thermal radiative corrections follow an analogous pattern via
`include_nTOp_thermal` and `save_nTOp_thermal`, but with a more lenient staleness
policy: recomputing them requires a `vegas` Monte Carlo integration that can take
minutes to hours, so an existing `rates/weak/{nTOp,pTOn}_thermal_corrections.txt`
is always loaded as-is (a fingerprint mismatch only prints a warning in verbose
mode); only a *missing* file triggers recomputation.

**Typical workflow for a high-precision study:**
```python
# Step 1 – compute and save high-precision rates once (non-default sampling_nTOp
# gives a fingerprint that the shipped cache won't match, so this recomputes)
PyPR({"save_nTOp": True, "sampling_nTOp": 400}).solve()

# Step 2 – all subsequent runs with the same sampling_nTOp reuse the saved tables
PyPR({"sampling_nTOp": 400}).solve()
```

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
`a, T, t, H, Tnue, Tnumu, Tnutau, Nheating, [abundances], n_to_p_weak_rate, p_to_n_weak_rate, [nuclear rates]`

## Architecture

```
pyprimat/                    Core package
  config.py              PyPRConfig: all physical constants + run-time flags
  main.py                PyPR: top-level driver
  plasma.py              Plasma thermodynamics (QED corrections, neutrino bath)
  qed_pressure.py        Analytical QED plasma-pressure corrections
  nuclear.py             Nuclear network related functions
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
