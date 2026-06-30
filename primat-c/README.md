# primat-c

A standalone C99 implementation of the PRIMAT Big Bang Nucleosynthesis (BBN) solver.

## Overview

`primat-c/` is a complete C99 port of the PRIMAT BBN solver, providing:

- **Identical physics**: Same numerical results as the Python backend (within ~1e-8 relative tolerance)
- **High performance**: Typically ~25× faster than the pure Python implementation
- **Standalone usage**: Can be compiled and run independently of the Python package
- **Python integration**: Compiled as an extension to provide the default fast backend for `primat.backend.run_bbn()`

## Compilation

### Prerequisites

- C99-compatible compiler (GCC, Clang, MSVC)
- GNU Make (for the provided Makefile)
- Python development headers (for the Python extension bridge, optional)

### Building the standalone C executable

From the `primat-c/` directory:

```bash
cd primat-c
make
```

This produces the `primat-c` executable in the `build/` directory.

**Available targets:**
- `make` or `make all` - Build the standalone executable
- `make clean` - Remove build artifacts
- `make debug` - Build with debug symbols
- `make release` - Build with optimizations

### Platform-specific notes

#### Linux (GCC/Clang)

```bash
# Install build essentials on Debian/Ubuntu
sudo apt-get install build-essential

# Then compile as above
make
```

#### macOS (Clang)

```bash
# Install Xcode command line tools if not already present
xcode-select --install

# Then compile
make
```

#### Windows (MSVC)

Use Visual Studio Developer Command Prompt:

```cmd
cd primat-c
nmake /f Makefile.win
```

Or use MinGW/MSYS2:

```bash
pacman -S mingw-w64-x86_64-gcc
make
```

### Building the Python extension

The Python extension is automatically built during `pip install primat` and included in the wheel distribution. To build manually:

```bash
cd primat-c
python setup.py build_ext --inplace
```

## Usage

### Standalone C executable

After compilation, run from the `primat-c/` directory:

```bash
# Basic usage with default parameters
./build/primat-c

# With custom parameters
./build/primat-c --Omegabh2 0.02242 --network large --amax 8

# List all available options
./build/primat-c --help
```

**Common options:**
- `--Omegabh2 VALUE` - Baryon density Ω_b h² (default: 0.022425)
- `--DeltaNeff VALUE` - Extra relativistic degrees of freedom (default: 0)
- `--network NAME` - Nuclear reaction network: small, small_parthenope, large (default: small)
- `--amax N` - Maximum mass number A for reactions (filters any network)
- `--numerical_precision VALUE` - ODE solver relative tolerance (default: 1e-7)
- `--backend c` - Force C backend (default when using the standalone executable)
- `--output_file PATH` - Output file path for results
- `--json` - Output results as JSON

### Configuration file

Instead of command-line arguments, you can use an INI-style configuration file:

```bash
# Run with a configuration file
./build/primat-c --ini examples/run_basic.ini
```

### Via Python API

The C backend is automatically used as the default by `primat.backend.run_bbn()`:

```python
from primat.backend import run_bbn

# Automatically uses C backend if available
result = run_bbn({"Omegabh2": 0.022425, "network": "large"})

# Force C backend explicitly
result = run_bbn({"Omegabh2": 0.022425}, force_backend="c")
```

### Custom networks

Custom nuclear reaction networks can be used with both the standalone executable and Python API:

```bash
# Using a custom network file
./build/primat-c --network my_custom_network
```

The network file should be placed in `data/nuclear/networks/` or made accessible via `--user_nuclear_dir` (additive overlay) or `--data_dir` (full data-tree replacement).

## Output

The C backend produces identical output to the Python backend:

### Return values (via Python API)

`run_bbn()` returns a dictionary with:

- `YPBBN` - Helium-4 mass fraction (BBN convention)
- `YPCMB` - Helium-4 mass fraction (CMB convention)  
- `DoH` - D/H ratio
- `He3oH` - (He3+T)/H ratio
- `Li7oH` - (Li7+Be7)/H ratio
- `Neff` - Effective number of neutrino species
- `Omeganurel` - Ω_ν h² × 10⁶ (relativistic)
- `OneOverOmeganunr` - 1 / (Ω_ν h² × 10⁻⁶) (non-relativistic)
- `evolution` - Time evolution data (when `output_time_evolution=True`)

### Command-line output

Default console output:
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

With `--json` flag, full results are output as JSON.

## Data directory structure

The C backend uses the same data directory structure as the Python package:

```
primat-c/data/
  nuclear/
    tables/              # Per-reaction rate tables (one folder per reaction)
    networks/            # Network list files (small.txt, large.txt, etc.)
  csv/                 # Reaction catalog files (nuclides.csv, detailed_balance.csv, reactions_large.csv)
  plasma/              # Pre-computed QED pressure tables
  weak/                # Cached n<->p forward/backward rates
  NEVO/                # Neutrino-decoupling history tables
```

The standalone executable looks for data files relative to its working directory. When used as a Python extension, paths are resolved relative to the installed package.

## Code structure

```
primat-c/
  include/      # Public headers
    api.h             # Main API functions
    config.h          # Configuration structure and functions
    network_data.h   # Network and reaction data structures
    evolution.h       # Time evolution data structures
    background.h      # Background cosmology structures
    weak_rates.h      # Weak rate computation structures
  
  src/                # Implementation
    api.c             # Main API implementation (cpr_run, cpr_mc_uncertainty)
    config.c          # Configuration parsing and validation
    network_data.c    # Network loading and rate table handling
    nuclear_network.c # Nuclear network ODE integration
    background.c      # Background cosmology computation
    weak_rates.c      # Weak rate computation
    cli.c             # Command-line interface
    neutrino_history.c # NEVO table loading and interpolation
    
  tests/              # Unit tests
    unit/             # C unit test suite
    
  examples/           # Example configurations
    run_basic.ini     # Template configuration file
```

## Custom backgrounds and advanced usage

For advanced users who want to implement custom cosmological backgrounds or test alternative scenarios, the C backend provides hooks for:

- Custom background tables (when `incomplete_decoupling=False`)
- Custom NEVO tables for neutrino decoupling
- Rate variation for sensitivity analysis

See the `include/` headers for the available API functions.

## Error handling and debugging

### Verbose output

Use the `--verbose` flag to see detailed progress messages:

```bash
./build/primat-c --verbose
```

This shows timing information, cache hits, and integration progress.

### Debug builds

To compile with debug symbols and no optimizations:

```bash
make debug
```

### Checking installation

To verify the C backend is working correctly:

```python
from primat.backend import HAS_C_BACKEND
print(f"C backend available: {HAS_C_BACKEND}")
```

## Version compatibility

The `primat-c` version is kept in sync with the main `primat` package version. The version is defined in `include/config.h` as `CPRIMAT_VERSION` and must match the version in `pyproject.toml`.

## Support and contribution

- Report issues on the main PRIMAT repository
- Contributions to the C backend should maintain parity with the Python implementation
- Any changes to physics or numerics must be mirrored in both backends

## License

Same as the main PRIMAT package - see the repository LICENSE file.