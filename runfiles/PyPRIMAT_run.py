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
_pyprimat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pyprimat_path not in sys.path:
    sys.path.insert(0, _pyprimat_path)

from pyprimat import PyPR

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
    # save_nTOp: True is now the default; listed here explicitly for clarity.
    # Each configuration saves to rates/weak/nTOp_<hash>.txt so different
    # configurations coexist without overwriting each other.
    "save_nTOp":            True,
    "Omegabh2":                  omegabh2,
    # "eta0b": computed automatically from Omegabh2
    "DeltaNeff":                 Nrelat,  # Note: not exactly the PRIMAT definition
    "network":                   'large',
    "amax":                      8,
    "output_time_evolution":     True,
    "output_background_evolution":     True,
    "numerical_precision":       1e-7,
    "output_final_result":       True,
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
start_time = time.time()

PyPRrun = PyPR(params=MyOptions)
res = PyPRrun.PyPRresults()

print(" ")
print(" Neff = ",             res['Neff'])
print(" Ωνh2 x 10^6 (rel) = ", res['Omeganurel'])
print(" Σmν/Ωνh2 (in eV, non rel.) = ",   res['OneOverOmeganunr'])
print(" YP (CMB) = ",         res['YPCMB'])
print(" YP (BBN) = ",         res['YPBBN'])
print(" D/H = ",       res['DoH'])
print(" He3/H = ",     res['He3oH'])
print(" He3/He4 = ",     res['He3oHe4'])
print(" Li7/H = ",    res['Li7oH'])
print(" ")

# Print the full per-nuclide abundance table for any network beyond "small".
if MyOptions.get("network", "small") != "small":
    print(" Final nuclide mass-fraction abundances Y_i:")
    print(f"  {'Nuclide':<10}  {'Y_i':>14}")
    print("  " + "-" * 26)
    for name in PyPRrun.abundance_names:
        print(f"  {name:<10}  {PyPRrun.nuclear.Y_final[name]:14.6e}")
    print(" ")

print("--- running time: %s seconds ---" % (time.time() - start_time))
