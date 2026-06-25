# -*- coding: utf-8 -*-
"""
PyPRIMAT_reference_run.py
=========================
High-precision reference run for updating validation benchmarks in CLAUDE.md.

This script is NOT intended for routine use.  It is designed to produce the
most accurate possible primordial abundances by pushing all precision parameters
to their limits.  Expected running time: several minutes.

Key choices versus the standard run:
  - T_start_cosmo = 100 MeV  (wider background integration range)
  - T_end         = 0.001 MeV (same as standard; already the default)
  - sampling_temperature_per_decade = 2000  (denser background grid)
  - numerical_precision = 1e-10  (tighter ODE tolerances)
  - rate_grid_npts = 4000  (denser nuclear-rate master grid; explicit so this
    reference is decoupled from whatever PRIMATConfig's routine-run default is)
  - sampling_nTOp_per_decade = 125  (denser n<->p rate tables)
  - These settings change the n<->p weak-rate fingerprint relative to the
    shipped rates/weak/*.txt cache (see primat.weak_rates), so the rates
    are automatically recomputed from scratch (not loaded from the cache).
    The thermal corrections are loaded from the existing
    rates/weak/{n__p,pTOn}_thermal_corrections.txt regardless (their
    fingerprint is checked leniently since recomputing them is slow).
  - vegas_n_eval  = 100000, vegas_n_itn = 50  (higher-accuracy MC for
                                               radiative corrections, if used)

Usage::

    python runfiles/PyPRIMAT_reference_run.py
"""

import sys
import os
import time

_pyprimat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pyprimat_path not in sys.path:
    sys.path.insert(0, _pyprimat_path)

from primat import backend

# ---------------------------------------------------------------------------
# Cosmological parameters (standard values)
# ---------------------------------------------------------------------------
omegabh2 = 0.022425

# ---------------------------------------------------------------------------
# High-precision options
# ---------------------------------------------------------------------------
MyOptions = {
    "verbose":               True,
    "debug":                 True,

    # Background thermodynamics
    "T_start_cosmo_MeV":          100.0,   # [MeV]  wider integration range
    "sampling_temperature_per_decade":     2000,    # denser grid for a(T) and t(a)

    # Nuclear rate tables
    "rate_grid_npts":             4000,    # denser master T9 grid (see module docstring)

    # ODE solver
    "numerical_precision":        1e-10,   # rtol for all solve_ivp calls

    # n <-> p weak rates
    "sampling_nTOp_per_decade":           125,   # points per decade for rate tables
    "sampling_nTOp_thermal_per_decade":   25,    # kept as is
    "vegas_n_eval":               100000,  # MC evaluations per iteration
    "vegas_n_itn":                50,      # MC iterations

    # Omegabh2
    "Omegabh2":                   omegabh2,

    # nuclear_qed_corrections: left at its PRIMATConfig default (True) -- the
    # reference values include the radiative-capture QED corrections
    # (Pitrou & Pospelov 2020).

    # Output
    "output_time_evolution":      False,
}

# ---------------------------------------------------------------------------
# Run both networks
# ---------------------------------------------------------------------------

def run_network(label, network, amax=None):
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    extra = {"network": network}
    if amax is not None:
        extra["amax"] = amax
    # force_backend="python": CLAUDE.md's reference table is the source of
    # truth this script regenerates, and there is a documented ~1.7e-8
    # unresolved C-vs-Python D/H gap (tests/test_backend_parity.py) -- the
    # reference values must stay pinned to the Python backend regardless of
    # whether a compiled C extension happens to be available.
    res = backend.run_bbn({**MyOptions, **extra}, force_backend="python")
    elapsed = time.time() - t0
    print(" ")
    print(f" Neff               --> {res['Neff']}")
    print(f" Ωνh2 x 10^6 (rel) --> {res['Omeganurel']}")
    print(f" Σmν/Ωνh2 [eV]     --> {res['OneOverOmeganunr']}")
    print(f" YP (CMB)           --> {res['YPCMB']}")
    print(f" YP (BBN)           --> {res['YPBBN']}")
    print(f" D/H                --> {res['DoH']}")
    print(f" He3/H              --> {res['He3oH']}")
    print(f" Li7/H              --> {res['Li7oH']}")
    print(f" running time: {elapsed:.1f} s")
    return res

total_start = time.time()

res_small  = run_network("small network", network="small")
res_amax8  = run_network("large network, amax=8", network="large", amax=8)
res_large  = run_network("large network", network="large")

print()
print("=" * 76)
print("  Summary")
print("=" * 76)
print(f"  {'':30s}  {'small net':>14}  {'large, amax=8':>14}  {'large net':>14}")
print(f"  {'YP (BBN)':30s}  {res_small['YPBBN']:>14.8f}  {res_amax8['YPBBN']:>14.8f}  {res_large['YPBBN']:>14.8f}")
print(f"  {'D/H':30s}  {res_small['DoH']:>14.5e}  {res_amax8['DoH']:>14.5e}  {res_large['DoH']:>14.5e}")
print(f"  {'He3/H':30s}  {res_small['He3oH']:>14.5e}  {res_amax8['He3oH']:>14.5e}  {res_large['He3oH']:>14.5e}")
print(f"  {'Li7/H':30s}  {res_small['Li7oH']:>14.5e}  {res_amax8['Li7oH']:>14.5e}  {res_large['Li7oH']:>14.5e}")
print()
print(f"--- total running time: {time.time() - total_start:.1f} s ---")
