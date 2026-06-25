# PRIMAT — Python Notebooks

This folder contains demonstration notebooks for PRIMAT.
Each notebook is self-contained and can be run from this directory.
Figures are saved to `plots/`.

---

## Notebooks

### Standard results

| Notebook | Description |
|----------|-------------|
| `StandardPlots.ipynb` | **Schramm diagram**: primordial abundances vs η_b with 1σ nuclear-rate uncertainty bands and observational constraints (YP, D/H, ³He/⁴He, ⁷Li/H). |
| `AbundanceEvolution.ipynb` | **Time evolution** of A_i Y_i(t) for all nuclides from 1 s to 10⁵ s, for both the 12-reaction (small) and 63-reaction (full) networks. |

### Parameter scans

| Notebook | Description |
|----------|-------------|
| `PosteriorBaryons.ipynb` | **Posterior on Ω_b h²** from YP and D/H: scans Ω_b h² ∈ [0.020, 0.024] and computes Gaussian likelihoods from each observable. |
| `AbundancesNrelat.ipynb` | **Abundances vs ΔNeff**: scans ΔNeff ∈ [−2, +2] to show how extra relativistic species shift YP and D/H. |
| `AbundancesXi.ipynb` | **Abundances vs neutrino degeneracy ξ = μ_ν/T_ν**: scans ξ ∈ [−0.05, +0.05] to show the effect of a neutrino chemical potential on BBN. |

### Uncertainty analysis

| Notebook | Description |
|----------|-------------|
| `MonteCarloRates.ipynb` | **Full MC uncertainty budget**: draws nuclear rates, τ_n, and Ω_b h² simultaneously; shows histograms and a corner plot of the joint distribution of all observables. |
| `Sensitivity.ipynb` | **Sensitivity tables**: computes the logarithmic derivative ∂ ln(observable) / ∂ ln(parameter) for each of the 12 nuclear rates, τ_n, G_N, Ω_b h², and ΔNeff. Results are displayed as formatted tables and a heat-map. |

---

## Common conventions

- **Fixed MC seed across grids**: when scanning a parameter, `MC_SEED = 0`
  is used at every grid point so that finite-sample MC bias cancels across the grid.
- **Observational constraints** shown as grey horizontal/vertical bands.
- **Planck baryon density** Ω_b h² = 0.02285 ± 0.00016 shown as a red vertical band.
- Set `num_mc = 500` or more for publication-quality uncertainty bands (default is 50 for speed).

---

## Parallelization and backend choices

### Python vs. C backend parallelization

Both backends support Monte Carlo uncertainty propagation with parallel acceleration, but they use different strategies:

**Python backend** (`primat.main.mc_uncertainty`):
- Parallelizes MC samples across worker processes using **joblib.Parallel**.
- The `n_jobs` parameter controls the number of workers: `-1` uses all available CPU cores, or specify an integer (e.g. `n_jobs=8`).
- Each worker draws independent rate/τ_n samples and runs a full PRIMAT solve; results are aggregated in the main process.
- Uses NumPy's `default_rng` for the per-sample RNG.

**C backend** (`cpr_mc_uncertainty`):
- Parallelizes internally using **POSIX threads (pthread)** within a single process.
- The `n_jobs` parameter is passed to the C extension and controls the thread count.
- Uses xoshiro256** (a fast parallel RNG) for per-thread sample generation.
- Typically faster than joblib for small-to-moderate sample counts (N < a few thousand) due to lower inter-process overhead.

### Equivalence and statistical comparisons

The two backends produce **statistically equivalent but not bit-for-bit identical** MC samples:
- They use different RNG streams (NumPy's `default_rng` vs. xoshiro256**), so individual samples differ.
- Mean and standard deviation converge to the same values as N_MC increases (within numerical precision ~1e-7 to 1e-8).
- When comparing MC results across backends, verify that *statistics* (mean, std, correlations) agree rather than expecting exact sample values.

### Which backend to use

- **Auto** (default, `force_backend=None` or `"auto"`): Uses the C backend if available (faster), else falls back to Python.
- **Force Python** (`force_backend="python"`): Always use joblib parallelization; useful for debugging or if custom-network features are not supported yet by the C backend.
- **Force C** (`force_backend="c"`): Always use pthread; raises if the C extension is not available.

See `primat.backend`'s module docstring (or `primat.cli --help`) for more details.
