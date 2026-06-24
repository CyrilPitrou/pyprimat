"""Centralised reference BBN observables for the default (small-network) run.

Single source of truth for ``tests/test_cli.py``, ``tests/test_gui.py`` and
any other test asserting the default-config small-network result, so that a
routine default-parameter tweak (e.g. ``rate_grid_npts``, commit ``e00f062``)
only needs a tolerance check here, not a hunt for every literal pin scattered
across the suite.

The tolerances mirror CLAUDE.md's "Validation before committing" table
(``YP (BBN)`` and ``D/H``, small network): a result outside these bounds
indicates a *physics* regression, not test brittleness. ``NEFF_ABS_TOL`` is
not separately documented in CLAUDE.md but is given the same ``1e-5``
margin used for ``YP``, since both observables are driven by the same
n<->p weak-rate / background machinery.
"""

# Default small-network run (network="small", spectral_distortions=True,
# nuclear_qed_corrections=True -- the PRIMATConfig defaults), as produced by
# `primat.cli.main([])`, `primat-gui`'s default "Run BBN", and
# `runfiles/PyPRIMAT_run.py`. Snapshotted after commit e00f062 (rate_grid_npts
# 500->1000, sampling_temperature_per_decade 400->600).
NEFF_REFERENCE  = 3.0439772986
YPBBN_REFERENCE = 0.24700028   # CLAUDE.md "Validation before committing" table
DOH_REFERENCE   = 2.43500e-5   # CLAUDE.md "Validation before committing" table

# Tolerances (CLAUDE.md: "A result outside these bounds indicates a regression").
NEFF_ABS_TOL  = 1e-5
YPBBN_ABS_TOL = 1e-5
DOH_ABS_TOL   = 3e-9

# Per-nuclide final mass fractions, small network (CLAUDE.md per-nuclide table).
P_REFERENCE   = 7.529409e-01
HE4_REFERENCE = 6.174973e-02
NUCLIDE_ABS_TOL = 1e-4  # mirrors the table's own documented precision
