#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_qed_tables.py
======================
Standalone script to recompute the QED plasma-pressure correction tables
and write them to ``Rates/plasma/``.

These tables store δP(T), dδP/dT, and d²δP/dT² — the finite-temperature
QED corrections to the EM plasma pressure that enter the background
evolution of the BBN code.  They were originally computed with
PRIMAT-Main.m (Mathematica); this script provides the equivalent Python
computation so the files can be regenerated without Mathematica.

The computation uses :mod:`pypr.qed_pressure` which implements the
analytic formulas from PRIMAT-Main.m:

    δP(T) = δP_a(T)  [O(α), leading]
           + δP_{e3}(T)  [O(α^{3/2}), ring/plasmon]

(The O(α²) two-loop exchange term δP_b is available via --include-dPb
but is not included in the standard files.)

Usage::

    # From the repository root:
    python generate_from_primat/generate_qed_tables.py

    # Higher-resolution grid:
    python generate_from_primat/generate_qed_tables.py --n-pts 1000

    # Also compute the O(e^4) two-loop exchange term (very slow):
    python generate_from_primat/generate_qed_tables.py --include-dPb

The output files are written to ``Rates/plasma/``:
  - ``QED_P_int.txt``     — δP columns: [T, δP_a, δP_{e3}]
  - ``QED_dP_intdT.txt``  — dδP/dT
  - ``QED_d2P_intdT2.txt`` — d²δP/dT²

Physical background
-------------------
The QED interaction pressure corrects the ideal-gas (photon + e±) EM
plasma equation of state.  It is decomposed into an O(e²) leading term
and an O(e³) ring/plasmon term following Phys. Rep. §II.E (PRIMAT
variables ``dPa``, ``dPe3``).  At T = 10 MeV:
  δP_a   ≈ −17  MeV⁴  (negative: reduces pressure)
  δP_{e3} ≈ +0.3 MeV⁴ (positive: ring contribution)
  total  ≈ −16.7 MeV⁴

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095), §II.E
PRIMAT-Main.m: ``dPa``, ``dPe3``, ``dPb`` definitions
"""

import sys
import os
import argparse
import time

# Ensure the repo root is on sys.path so that pypr is importable.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pypr.qed_pressure import compute_qed_pressure_tables, save_qed_tables


def main():
    parser = argparse.ArgumentParser(
        description="Recompute QED plasma-pressure correction tables.")
    parser.add_argument("--n-pts", type=int, default=500,
                        help="Number of log-spaced temperature grid points "
                             "(default: 500, matching the PRIMAT file).")
    parser.add_argument("--T-min", type=float, default=1e-3,
                        help="Minimum temperature [MeV] (default: 1e-3).")
    parser.add_argument("--T-max", type=float, default=100.,
                        help="Maximum temperature [MeV] (default: 100).")
    parser.add_argument("--include-dPb", action="store_true",
                        help="Also compute the O(e^4) two-loop exchange "
                             "correction δP_b (very slow: ~10 s per point).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for QED_*.txt files.  "
                             "Default: <repo_root>/Rates/plasma/")
    args = parser.parse_args()

    plasma_dir = args.output_dir or os.path.join(_repo_root, "Rates", "plasma")
    os.makedirs(plasma_dir, exist_ok=True)

    print(f"Computing QED plasma-pressure tables:")
    print(f"  T grid: {args.T_min:.2e}–{args.T_max:.2e} MeV, {args.n_pts} points")
    print(f"  include δP_b (O(e^4)): {args.include_dPb}")
    print(f"  output: {plasma_dir}/")
    print()

    t0 = time.time()
    tables = compute_qed_pressure_tables(
        T_min=args.T_min,
        T_max=args.T_max,
        n_pts=args.n_pts,
        include_dPb=args.include_dPb,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"\nComputation finished in {dt:.1f} s")

    save_qed_tables(tables, plasma_dir, verbose=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
