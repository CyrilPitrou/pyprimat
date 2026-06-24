# -*- coding: utf-8 -*-
"""
run_basic.py
============
Minimal, heavily-commented template for a standalone BBN run. Copy this file
and uncomment/edit whichever options you need; every option shown below is at
its default value, so running this file unmodified reproduces the standard
run (see CLAUDE.md's "Validation before committing" table for the expected
YPBBN/D-H reference values).

Run from the repo root so that the shipped ``rates/`` data resolve correctly:

    python runfiles/run_basic.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primat.backend import run_bbn

cfg = dict(
    # Omegabh2=0.022425,     # baryon density x h^2 (Planck 2018 default)
    # DeltaNeff=0.0,         # extra relativistic species beyond SM neutrinos
    # network="small",       # "small" / "small_parthenope" / "large" / custom network filename
    # amax=None,             # filter any network to reactions with A <= amax
    # numerical_precision=1e-7,    # rtol for all solve_ivp-equivalent calls
    # output_time_evolution=False,  # write the unified <run_id>_evolution.tsv (PRIMAT.md S7.2);
    #                                forces the python backend (see force_backend below)
    # user_rates_dir=None,   # overlay directory for your own network/table additions (PRIMAT.md S4.3)
)
# force_backend: None/"auto" (default: C extension if built, else pure Python),
# "c", or "python" -- see primat/backend.py's module docstring for exactly
# which features (extra_rho/custom_network/background=, output_time_evolution)
# always fall back to "python" regardless of this setting.
result = run_bbn(cfg, force_backend="auto")
print("Neff  =", result.get("Neff"))
print("YPBBN =", result["YPBBN"])
print("D/H   =", result["DoH"])
