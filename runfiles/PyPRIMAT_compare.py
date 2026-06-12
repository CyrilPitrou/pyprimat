# -*- coding: utf-8 -*-
"""
PyPRIMAT_compare.py
===================
Compares primordial abundance predictions between the built-in 12-reaction
small network, the 62-reaction medium file, and the 423-reaction large file.

Usage::

    python runfiles/PyPRIMAT_compare.py
"""

import sys
import os
import time

_pyprimat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pyprimat_path not in sys.path:
    sys.path.insert(0, _pyprimat_path)

from pyprimat import PyPR

# ---------------------------------------------------------------------------
# Cosmological parameters
# ---------------------------------------------------------------------------
Nrelat   = 0.
omegabh2 = 0.022425

_base_opts = {
    "verbose":              True,
    "Omegabh2":                  omegabh2,
    "DeltaNeff":                 Nrelat,
    "numerical_precision":       1e-7,
    # spectral_distortions: left at its PyPRConfig default (True), matching
    # the CLAUDE.md reference values.
    # Standard physics: no radiative-capture QED corrections, so the results
    # are directly comparable to the CLAUDE.md reference values.
    "nuclear_qed_corrections":   False
}

# ---------------------------------------------------------------------------
# Run networks
# ---------------------------------------------------------------------------
networks = ["small","small_parthenope","medium", "large"]
results = {}

for net in networks:
    print("=" * 60)
    print(f"Running {net} network ...")
    print("=" * 60)
    t0 = time.time()
    params = {**_base_opts, "network": net}
    run = PyPR(params=params)
    results[net] = run.PyPRresults()
    print(f"{net.capitalize()} network finished in {time.time()-t0:.1f} s\n")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def _print_res(label, res):
    print(f"  [{label}]")
    print(f"    Neff         = {res['Neff']:.8f}")
    print(f"    YP (BBN)     = {res['YPBBN']:.8f}")
    print(f"    D/H          = {res['DoH']:.7e}")
    print(f"    He3/H        = {res['He3oH']:.7e}")
    print(f"    Li7/H        = {res['Li7oH']:.7e}")

print("\n" + "=" * 60)
print("Results comparison")
print("=" * 60)
for net in networks:
    _print_res(f"{net.capitalize()} net", results[net])
print()
