"""
Smoke test for the example scripts in ``runfiles/``.

CLAUDE.md documents ``runfiles/primat_run.py`` as the canonical "run this to
validate a change" entry point, and ``primat_compare.py``/
``primat_run_explanatory.py`` as further worked examples -- but nothing in
the test suite actually executes them (``tests/reference_values.py`` only
mirrors their expected numbers, and ``tests/test_docs_consistency.py`` only
string-checks parameter names in ``primat_reference_run.py``). This means an
import-path bug or an API rename in ``primat.backend``/``primat.main`` could
break these scripts silently until a human runs one by hand.

Each script is run as a real subprocess (``python <script>``) rather than
imported, since none of them are wrapped in a ``main()``/``if __name__``
guard -- they are plain top-level scripts meant to be run directly. The
subprocess's *working directory* is a throwaway ``tmp_path``, not the repo
root: all three scripts write ``results/*.tsv``/``*.dat`` to a path relative
to the *current directory* (not ``__file__``), so running from ``tmp_path``
keeps every output out of the tracked (albeit gitignored) ``results/``
directory at the repo root.

Deliberately excluded:

* ``primat_reference_run.py`` -- several minutes by design (high-precision
  reference run for updating CLAUDE.md's benchmarks), out of scope for a
  smoke test.
* ``generate_weak_rate_caches.py`` -- (re)writes the fingerprinted
  ``rates/weak/nTOp_*.txt`` cache files that are force-added to git; not
  safe to run unattended even from a throwaway cwd (it resolves the cache
  directory relative to the installed package, not cwd).
* ``generate_table_CLASS_CAMB.py`` -- needs an external CLASS/CAMB
  installation and a multi-hour Monte Carlo table generation run.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]

RUNFILES_DIR = Path(__file__).resolve().parents[1] / "runfiles"

FAST_RUNFILES = [
    "primat_run.py",
    "primat_run_explanatory.py",
    "primat_compare.py",
]


@pytest.mark.parametrize("name", FAST_RUNFILES)
def test_runfile_executes_cleanly(name, tmp_path):
    """Run an example script as a subprocess; fail on a nonzero exit code
    or a traceback, and sanity-check it printed the headline observables."""
    result = subprocess.run(
        [sys.executable, str(RUNFILES_DIR / name)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"{name} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr, result.stderr

    # Every one of these scripts prints Neff and D/H (spelled "D/H") for at
    # least the small network -- a quick sanity check that it actually ran
    # the solver rather than exiting early/silently.
    assert "Neff" in result.stdout
    assert "D/H" in result.stdout
