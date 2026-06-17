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
    # save_nTOp=True: this script's options are exactly the *default*
    # n<->p weak-rate fingerprint (see pyprimat.weak_rates), so running it
    # regenerates rates/weak/nTOp_{frwrd,bkwrd}.txt with a fresh fingerprint
    # header for the default configuration. Other configurations should
    # leave save_nTOp at its default (False) to avoid overwriting this
    # shared cache with a non-default fingerprint.
    "save_nTOp":            True,
    "Omegabh2":                  omegabh2,
    # "eta0b": computed automatically from Omegabh2
    "DeltaNeff":                 Nrelat,  # Note: not exactly the PRIMAT definition
    "network":                   'medium',
    "output_time_evolution":     True,
    "numerical_precision":       1e-7,
    "output_final_result":       True,
    "sampling_nTOp":             200,
    "sampling_nTOp_thermal":     100,
    # radiative_corrections / finite_mass_corrections / thermal_corrections:
    # left at their PyPRConfig defaults (all True) -- full PRIMAT weak rates.
    # spectral_distortions: left at its PyPRConfig default (True) -- the
    # CLAUDE.md reference values include the NEVO spectral-distortion
    # correction to the n<->p weak rates.
    # nuclear_qed_corrections: left at its PyPRConfig default (True) -- the
    # CLAUDE.md reference values include the radiative-capture QED
    # corrections (Pitrou & Pospelov 2020).
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

# For the medium and large networks, print the full per-nuclide abundance table.
if MyOptions.get("network", "small") in ("medium", "large"):
    print(" Final nuclide mass-fraction abundances Y_i:")
    print(f"  {'Nuclide':<10}  {'Y_i':>14}")
    print("  " + "-" * 26)
    for name in PyPRrun.abundance_names:
        print(f"  {name:<10}  {PyPRrun.nuclear.Y_final[name]:14.6e}")
    print(" ")

print("--- running time: %s seconds ---" % (time.time() - start_time))
