# -*- coding: utf-8 -*-
"""
PyPRIMAT_run.py
===============
Standard run script for PyPRIMAT.

All run-time options are passed as a plain dict to ``PyPR``.
No ``PyPR_init.py`` singleton is needed or used.

Usage::

    python PyPRIMAT_run.py
"""

import sys
import os
import time

# Ensure the PyPR package is importable regardless of working directory
_pypr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pypr_path not in sys.path:
    sys.path.insert(0, _pypr_path)

from pypr import PyPR

# ---------------------------------------------------------------------------
# Cosmological parameters
# ---------------------------------------------------------------------------
Nrelat   = 0.
omegabh2 = 0.022425

# ---------------------------------------------------------------------------
# Run-time options
# ---------------------------------------------------------------------------
MyOptions = {
    "verbose":              True,
    "debug":                True,
    "compute_nTOp":         True,
    "compute_nTOp_thermal": False,
    "save_nTOp":            True,
    "save_nTOp_thermal":    False,
    "Omegabh2":                  omegabh2,
    # "eta0b": computed automatically from Omegabh2
    "DeltaNeff":                 Nrelat,  # Note: not exactly the PRIMAT definition
    "network":                   'small',
    "output_time_evolution":     True,
    "numerical_precision":       1e-7,
    "output_final_result":       True,
    "sampling_nTOp":             200,
    "nTOp_Born_approximation":             False,
    "sampling_nTOp_thermal":     100,
    "spectral_distortions":      False,
    "nuclear_qed_corrections":   True
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
start_time = time.time()

PyPRrun = PyPR(params=MyOptions)
res = PyPRrun.PyPRresults()

print(" ")
print(" Neff --> ",             res['Neff'])
print(" Ωνh2 x 10^6 (rel) --> ", res['Omeganurel'])
print(" Σmν/Ωνh2 [eV] --> ",   res['OneOverOmeganunr'])
print(" YP (CMB) --> ",         res['YPCMB'])
print(" YP (BBN) --> ",         res['YPBBN'])
print(" D/H --> ",       res['DoH'])
print(" He3/H --> ",     res['He3oH'])
print(" Li7/H --> ",    res['Li7oH'])
print(" ")

# For the medium and large networks, print the full per-nuclide abundance table.
if MyOptions.get("network", "small") in ("medium", "large"):
    print(" Final nuclide mass-fraction abundances Y_i:")
    print(f"  {'Nuclide':<10}  {'Y_i':>14}")
    print("  " + "-" * 26)
    for name in PyPRrun.abundance_names:
        print(f"  {name:<10}  {PyPRrun._Y_final[name]:14.6e}")
    print(" ")

print("--- running time: %s seconds ---" % (time.time() - start_time))
