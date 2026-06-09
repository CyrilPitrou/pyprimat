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
  - n_temperature_table    = 10000     (denser background grid)
  - numerical_precision = 1e-10  (tighter ODE tolerances)
  - sampling_nTOp = 500       (denser n<->p rate tables)
  - compute_nTOp = True  (recompute rates from scratch)
  - compute_nTOp_thermal = False  (thermal rates already pre-computed
                                        with sufficient precision)
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

from pyprimat import PyPR

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
    "numdiff_installed":               True,

    # Background thermodynamics
    "T_start_cosmo_MeV":          100.0,   # [MeV]  wider integration range
    "n_temperature_table":                 10000,   # denser grid for a(T) and t(a)

    # ODE solver
    "numerical_precision":        1e-10,   # rtol for all solve_ivp calls

    # n <-> p weak rates
    "compute_nTOp":          True,
    "compute_nTOp_thermal":  False,   # thermal rates already at high precision
    "sampling_nTOp":              500,     # points per era for rate tables
    "sampling_nTOp_thermal":      100,     # kept as is
    "vegas_n_eval":               100000,  # MC evaluations per iteration
    "vegas_n_itn":                50,      # MC iterations

    # Omegabh2
    "Omegabh2":                   omegabh2,

    # Output
    "output_time_evolution":      False,
}

# ---------------------------------------------------------------------------
# Run both networks
# ---------------------------------------------------------------------------

def run_network(network):
    label = f"{network} network"
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    res = PyPR(params={**MyOptions, "network": network}).PyPRresults()
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

res_small  = run_network(network="small")
res_medium = run_network(network="medium")
res_large  = run_network(network="large")

print()
print("=" * 76)
print("  Summary")
print("=" * 76)
print(f"  {'':30s}  {'small net':>14}  {'medium net':>14}  {'large net':>14}")
print(f"  {'YP (BBN)':30s}  {res_small['YPBBN']:>14.8f}  {res_medium['YPBBN']:>14.8f}  {res_large['YPBBN']:>14.8f}")
print(f"  {'D/H':30s}  {res_small['DoH']:>14.5e}  {res_medium['DoH']:>14.5e}  {res_large['DoH']:>14.5e}")
print(f"  {'He3/H':30s}  {res_small['He3oH']:>14.5e}  {res_medium['He3oH']:>14.5e}  {res_large['He3oH']:>14.5e}")
print(f"  {'Li7/H':30s}  {res_small['Li7oH']:>14.5e}  {res_medium['Li7oH']:>14.5e}  {res_large['Li7oH']:>14.5e}")
print()
print(f"--- total running time: {time.time() - total_start:.1f} s ---")
