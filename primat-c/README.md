# primat-c

A standalone C99 implementation of the BBN solver. This is a complete,
numerically-validated port of the Python `primat` package's core physics
and numerics, with no Python dependencies.

**For Python users:** You almost certainly want to use `pip install primat`
(which includes the compiled C backend automatically). This directory is for:
- Developers working on the C implementation
- Users who need a standalone C binary for integration into C/C++ projects
- Those building custom computational workflows in compiled languages

**For C developers:** This is a ~10k-line, well-commented C99 codebase that can
be built independently and extended. No external libraries required beyond
standard C (math.h, stdio.h, etc.).

## Quick start (standalone C binary)

```bash
cd primat-c
make
./primat_cli examples/run_basic.ini results/output.txt
```

Or with a custom configuration:

```bash
./primat_cli --config examples/run_basic.ini --Omegabh2 0.022425 --network large results/output.txt
```

Output file contains the final abundances and key observables.

## Building

### Build the standalone CLI

```bash
make
```

Produces `primat_cli` (the command-line tool) and optional `primat_bench` (for
performance profiling).

### Clean build

```bash
make clean
make
```

### Compiler options

The Makefile respects standard variables:

```bash
make CC=clang CFLAGS="-O3 -march=native"  # Custom compiler and flags
make DEBUG=1                               # Debug build (-g, no -O3)
```

## Usage

### Command-line tool

```bash
./primat_cli [--help | -h]
./primat_cli <input.ini> <output.txt>
./primat_cli --config <input.ini> [--param VALUE] ... <output.txt>
```

Examples:

```bash
# Standard SM run with default config
./primat_cli examples/run_basic.ini results/sm_run.txt

# Override parameters on the command line
./primat_cli examples/run_basic.ini --Omegabh2 0.022425 --network large results/large.txt

# High-precision run with custom weak-rate grid
./primat_cli examples/run_basic.ini \
  --sampling_nTOp_per_decade 160 \
  --numerical_precision 1e-10 \
  results/highprec.txt
```

#### Output format

The output file contains:

```
# BBN final observables
Neff                   3.04397730
YP_BBN                 0.24700028
YP_CMB                 0.24567395
DoH                    2.43500e-05
He3oH                  1.03970e-05
He3oHe4                1.26776e-04
Li7oH                  5.54721e-10
Li6oLi7                1.41894e-05

# Per-nuclide final abundances
n                      3.99535e-16
p                      7.52941e-01
H2                     1.83341e-05
...
```

Time-evolution output is optional; see `primat-c/examples/run_basic.ini` for
the `output_time_evolution` setting.

### Using as a library

Link against the compiled objects and call the C API:

```c
#include "cprimat/api.h"

int main() {
    CPRConfig cfg = cpr_config_default();
    cfg.Omegabh2 = 0.022425;
    cfg.network = "small";
    
    CPRResults results = cprimat_run(&cfg);
    
    printf("Neff = %.8f\n", results.Neff);
    printf("YPBBN = %.8f\n", results.YPBBN);
    printf("DoH = %.5e\n", results.DoH);
    
    cpr_results_free(&results);
    return 0;
}
```

See `primat-c/include/cprimat/api.h` for the public API and
`primat-c/src/api.c` for implementation details.

## Architecture

```
primat-c/
  include/cprimat/
    api.h              Public C API: main entry points and result structures
    config.h           Configuration structures and defaults (CPRIMAT_VERSION)
    network_data.h     Network definition and reaction catalog
    weak_rates.h       Weak-rate computation (n↔p)
    constants.h        Physical constants
  
  src/
    api.c              Main dispatch and result assembly (cprimat_run, cprimat_mc_uncertainty)
    cli.c              Command-line tool (argument parsing, I/O)
    background.c       Cosmological background (a↔t↔T, weak rates, Neff)
    nuclear_network.c  Nuclear network ODE integration
    network_data.c     Network loading and rate-table parsing
    weak_rates.c       n↔p weak-rate computation (integrands, corrections, caching)
    plasma.c           Plasma thermodynamics (QED, neutrino bath)
    qed_pressure.c     Analytical QED pressure corrections
    neutrino_history.c NEVO table loading/interpolation
    evolution.c        Time-evolution sampling and TSV output
    ode_bdf.c          BDF ODE solver (stiff integration)
    ode_rk45.c         RK45 ODE solver (background evolution)
    utils.c            Utilities (memory, sorting, hashing, caching)
  
  tests/
    unit/              Unit tests for individual modules
    integration/       Full BBN runs vs. reference values
  
  examples/
    run_basic.ini      Template configuration file (all parameters commented)
    small_network.ini  Preset: small network
    large_network.ini  Preset: large network
  
  Makefile             Build system
```

## Numerical guarantees

The C backend reproduces the Python backend's results to:

- **Light-element abundances** (D/H, YP, He3/H, Li7/H): within ~1e-8 relative
  (C and Python agree to better than this on the same platform, but cross-platform
  IEEE 754 rounding differences accumulate at this level)
- **Heavier nuclides and network details**: as above, plus a known ~1.7e-8 relative
  D/H discrepancy for `network="small"` (not yet root-caused; budgeted separately
  in regression testing)
- **Monte Carlo uncertainty estimates**: converge with sample count; quick-MC
  estimates (30 samples) are approximate (~10% noise)

See `tests/test_backend_parity.py` in the Python package for the full parity
test suite.

## Configuration parameters

The full parameter list is documented in `examples/run_basic.ini` with comments
matching `primat/config.py`'s `DEFAULT_PARAMS`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Omegabh2` | 0.022425 | Baryon density Ω_b h² |
| `DeltaNeff` | 0.0 | Extra relativistic d.o.f. |
| `network` | small | "small" / "small_parthenope" / "large" |
| `amax` | (unset) | Max mass number (e.g., `amax=8` for medium) |
| `numerical_precision` | 1e-7 | ODE solver rtol |
| `output_time_evolution` | false | Write per-timestep data |
| `output_file` | (unset) | Output TSV file (optional) |

#### Weak-rate caching

The most expensive part of BBN is computing the n↔p weak rates (~0.5–2 s per
fingerprint). Like the Python side, results are cached:

- Fingerprint-based cache directory: `rates/weak/nTOp_<hash>.txt`
- Cache files are automatically created on first run with a new configuration
- Set `save_nTOp=false` to skip caching (useful for quick experiments)

#### Rate-table overlay

Point at custom rate tables without rebuilding:

```bash
./primat_cli --rates_dir /path/to/custom/rates examples/run_basic.ini output.txt
```

Directory layout must mirror `rates/nuclear/`, with any subset of networks/tables.

## Keeping C and Python in sync

The C port is numerically validated against Python (`tests/test_backend_parity.py`).
Any change to the physics or numerics in either codebase must be mirrored to the
other:

- Formula change → port it
- New correction term → port it  
- Different tolerance or clamp value → port it
- Cache fingerprint field → port it
- Output schema column → port it

Cosmetic changes (comments, variable renames, pure refactoring with no numerical
effect) do not require porting.

`CPRIMAT_VERSION` in `primat-c/include/cprimat/config.h` must match
`version` in `pyproject.toml` (the single source of truth). Update by hand
whenever the version is bumped.

## Development and testing

### Run the unit tests

```bash
make test
```

Runs C unit tests in `tests/unit/`. Each file tests a specific module
(background, network, weak_rates, etc.).

### Benchmark (optional)

```bash
make bench
./primat_bench
```

Profiles BBN solve time across network sizes and parameter sweeps.

### Adding a new test

1. Create `tests/unit/test_<module>.c` with a `test_<module>()` function
2. Register in `tests/test_runner.c`'s switch statement
3. Run `make test`

## Citation

If you use the C port in research, cite the original paper:

> Pitrou, Coc, Uzan, Vangioni, *Physics Reports* **754** (2018) 1–67.  
> [doi:10.1016/j.physrep.2018.04.005](https://doi.org/10.1016/j.physrep.2018.04.005)

## Authors

Cyril Pitrou (original Python implementation and C port), Julien Froustey (Python).
