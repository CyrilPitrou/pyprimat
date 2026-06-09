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
from PyPR import PyPR

result = PyPR({"Omegabh2": 0.022425}).solve()

print(f"YP  (BBN) = {result['YPBBN']:.6f}")   # ~0.246915
print(f"D/H = {result['DoH']:.5f}") # ~2.43647
```

The constructor accepts an optional parameter dict that overrides any default in `pyprimat/config.py`. All keys are optional.

## Running the example scripts

Scripts live in `runfiles/`. Run from the repo root:

```bash
python runfiles/PyPRIMAT_run.py           # Standard SM run (outputs runfiles/results/output_tables.tsv)
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
| `compute_nTOp_flag` | True | Recompute n↔p weak rates from scratch (vs loading pre-tabulated) |
| `save_nTOp_flag` | False | Save recomputed n↔p rates to `rates/weak/` for future use |
| `compute_nTOp_thermal_flag` | False | Also recompute thermal radiative corrections (very slow, requires `vegas`) |
| `save_nTOp_thermal_flag` | False | Save recomputed thermal corrections to disk |
| `output_time_evolution` | False | Write time-evolution table to `output_file` |
| `output_file` | `runfiles/results/output_tables.tsv` | Output file path |
| `output_n_points` | 500 | Number of interpolated rows in output file |

### n↔p weak rate workflow

The n↔p weak rates are the most expensive part of initialisation (~1.8 s). Two flags control whether they are recomputed or loaded from the pre-tabulated files in `rates/weak/`:

- **`compute_nTOp_flag=True`** (default): rates are computed from scratch by numerical integration. Use this when you change the neutrino temperature history or want higher precision (increase `sampling_nTOp`). Set `save_nTOp_flag=True` at the same time to write the result to `rates/weak/` so future runs can reuse it.
- **`compute_nTOp_flag=False`**: rates are read directly from `rates/weak/`. Initialisation becomes instantaneous. Safe to use as long as the cosmological background has not changed.

The thermal radiative corrections follow the same pattern via `compute_nTOp_thermal_flag` / `save_nTOp_thermal_flag`. They are much slower (require `vegas` Monte Carlo integration) and are disabled by default; the pre-computed corrections shipped in `rates/weak/` are already at high precision.

**Typical workflow for a high-precision study:**
```python
# Step 1 – compute and save high-precision rates once
PyPR({"compute_nTOp_flag": True, "save_nTOp_flag": True,
           "sampling_nTOp": 400}).solve()

# Step 2 – all subsequent runs reuse the saved tables
PyPR({"compute_nTOp_flag": False, ...}).solve()
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
