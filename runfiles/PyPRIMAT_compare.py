# -*- coding: utf-8 -*-
"""
PyPRIMAT_compare.py
===================
Compares primordial abundance predictions between the built-in 12-reaction
small network, the small_parthenope network, the large network restricted to
A <= 8 (68 reactions, equivalent to the old "medium" network), and the full
large network.

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
    # spectral_distortions: left at its PyPRConfig default (True).
    # nuclear_qed_corrections is turned off here (the CLAUDE.md reference
    # table uses the True default), so these results are an internal
    # small-vs-large(amax=8)-vs-large comparison only -- not directly
    # comparable to the CLAUDE.md reference values.
    "nuclear_qed_corrections":   False
}

# ---------------------------------------------------------------------------
# Run networks
# ---------------------------------------------------------------------------
networks = [
    ("small", {}),
    ("small_parthenope", {}),
    ("large_amax8", {"network": "large", "amax": 8}),
    ("large", {"network": "large"}),
]
results = {}

for label, extra in networks:
    print("=" * 60)
    print(f"Running {label} network ...")
    print("=" * 60)
    t0 = time.time()
    params = {**_base_opts, "network": "small", **extra}
    run = PyPR(params=params)
    results[label] = run.PyPRresults()
    print(f"{label.capitalize()} network finished in {time.time()-t0:.1f} s\n")

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
for label, _ in networks:
    _print_res(f"{label} net", results[label])
print()
