# -*- coding: utf-8 -*-
"""
generate_weak_rate_caches.py
=============================
(Re)generates the fingerprinted n<->p weak-rate cache files
(``rates/weak/nTOp_<hash>.txt`` / ``nTOp_thermal_<hash>.txt``) for the
handful of flag combinations that are force-added to git (see
``.gitignore``'s ``rates/weak/nTOp_*.txt`` pattern -- only the files this
script produces are exempted via ``git add -f``).

These are the combinations actually exercised by the bulk of the test suite
and the example runfiles, so shipping them avoids a (potentially multi-minute,
vegas-based) thermal-correction recompute on a fresh checkout:

1. Full physics, all corrections on (the ``PRIMATConfig`` default): radiative,
   finite-mass, thermal and spectral-distortion corrections + QED pressure,
   with non-instantaneous decoupling (``incomplete_decoupling=True``).
2. Same as (1) but ``QED_corrections=False`` -- the other half of the
   QED on/off comparison used throughout ``tests/test_decoupling_qed.py``.
3. ``incomplete_decoupling=False`` (instantaneous-decoupling limit),
   ``QED_corrections=True``. ``spectral_distortions`` must be ``False``
   here: it requires the NEVO spectral table, which only exists in
   non-instantaneous-decoupling mode (``PRIMATConfig.__init__`` raises
   otherwise).
4. Same as (3) but ``QED_corrections=False``.

Run from the repo root::

    python runfiles/generate_weak_rate_caches.py

The cache filenames embed a hash of the weak-rate / thermal fingerprint
(``weak_rates/cache.py``: ``_weak_rate_fingerprint`` / ``_thermal_fingerprint``
and their ``_WEAK_RATE_BG_FIELDS`` / ``_THERMAL_BG_FIELDS`` field lists).
Whenever those field lists change -- e.g. a field is added to or removed from
``_WEAK_RATE_BG_FIELDS`` -- EVERY hash shifts and the previously shipped files
become orphaned (they would never be hit again, so they only bloat the repo).

To keep the shipped set self-consistent, this script computes the exact set
of filenames the combos below SHOULD produce and then prunes any git-tracked
``nTOp_*.txt`` / ``nTOp_thermal_*.txt`` file in ``rates/weak/`` that is no
longer in that set (only git-tracked files, so a developer's local
non-shipped caches are left untouched). Pruning + (re)generation together
leave the working tree holding exactly the canonical shipped set.

After running, stage the result (new files force-added past .gitignore,
deletions recorded)::

    git add -f rates/weak/nTOp_*.txt rates/weak/nTOp_thermal_*.txt
    git add -u rates/weak/                       # record pruned deletions

Cross-backend agreement check (thermal cache only): for each combo, after
the Python (vegas) thermal table is written and shipped, this script also
recomputes the same table via the C backend (force_backend="c") into a
scratch copy, diffs the two, prints a summary, and discards the C copy --
the file actually shipped is always the Python one written above (see
write_cache_with_fingerprint's `provenance` field, which stamps which
backend produced the file on disk). This is purely an informational
regression check: both backends use independent Monte-Carlo (vegas)
estimates of the same integral with their own noise floor, so some
disagreement -- especially near a zero-crossing of L_nTOpCCRTh/L_pTOnCCRTh,
where any noise is amplified in *relative* terms -- is expected and not by
itself a bug; see weak_rates.c's CCRTh section and
tests/unit/test_weak_rates_thermal.c for the tolerances already accepted
elsewhere for this term. Skipped automatically if the C extension isn't
built (``primat.backend.HAS_C_BACKEND`` False).
"""
import sys
import os
import subprocess
import time

import numpy as np

_primat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _primat_path not in sys.path:
    sys.path.insert(0, _primat_path)

from primat import PRIMAT
from primat.backend import run_bbn, HAS_C_BACKEND
from primat.cache_utils import fingerprint_hash, weak_cache_dir
from primat.weak_rates.cache import (_weak_rate_fingerprint,
                                       _thermal_fingerprint)

# Each entry only lists the flags that deviate from the PRIMATConfig defaults
# (radiative_corrections/finite_mass_corrections/thermal_corrections all
# default to True). spectral_distortions is forced False whenever
# incomplete_decoupling is False, since the two are incompatible.
_COMBOS = [
    ("full physics (defaults: incomplete_decoupling, QED, spectral all on)", {}),
    ("QED off (incomplete_decoupling + spectral on)",
     dict(QED_corrections=False)),
    ("instantaneous decoupling, QED on (spectral forced off)",
     dict(incomplete_decoupling=False, spectral_distortions=False)),
    ("instantaneous decoupling, QED off (spectral forced off)",
     dict(incomplete_decoupling=False, QED_corrections=False,
          spectral_distortions=False)),
]

def _expected_filenames(combos):
    """Filenames (no directory) the given combos should leave on disk.

    For each combo we build the same PRIMATConfig the generation loop uses (but
    without writing anything) and read off both fingerprint hashes, so this
    stays in lockstep with whatever the live fingerprint definition is.

    Returns:
        (set_of_filenames, cache_dir): the expected ``nTOp_*`` /
        ``nTOp_thermal_*`` basenames and the absolute cache directory.
    """
    expected = set()
    cache_dir = None
    for _, extra in combos:
        # Pure inspection: never touch the cache here (weak_rate_cache=False so
        # nothing is loaded, save_* False so nothing is written).
        cfg = PRIMAT(params=dict(extra, verbose=False, weak_rate_cache=False,
                               save_nTOp=False, save_nTOp_thermal=False)).cfg
        cache_dir = weak_cache_dir(cfg)
        expected.add("nTOp_" + fingerprint_hash(_weak_rate_fingerprint(cfg)) + ".txt")
        expected.add("nTOp_thermal_" + fingerprint_hash(_thermal_fingerprint(cfg)) + ".txt")
    return expected, cache_dir


def _tracked_cache_files(cache_dir):
    """git-tracked ``nTOp_*.txt`` / ``nTOp_thermal_*.txt`` basenames in cache_dir.

    Returns an empty set (and prints a warning) if git is unavailable or this
    is not a git checkout -- in that case we simply skip pruning rather than
    risk deleting a developer's local caches.
    """
    try:
        out = subprocess.run(
            ["git", "-C", _primat_path, "ls-files", cache_dir],
            check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  [prune skipped: git unavailable: {exc}]")
        return set()
    tracked = set()
    for line in out.splitlines():
        base = os.path.basename(line.strip())
        if base.startswith(("nTOp_", "nTOp_thermal_")) and base.endswith(".txt"):
            tracked.add(base)
    return tracked


def _thermal_cache_path(extra):
    """Path to the thermal-cache file the given combo's `extra` flags map to."""
    cfg = PRIMAT(params=dict(extra, verbose=False, weak_rate_cache=False,
                           save_nTOp=False, save_nTOp_thermal=False)).cfg
    return os.path.join(weak_cache_dir(cfg),
                         "nTOp_thermal_" + fingerprint_hash(_thermal_fingerprint(cfg)) + ".txt")


def _compare_thermal_with_c(extra, th_path):
    """Recomputes `th_path` via the C backend into a scratch copy, diffs it
    against the just-shipped Python (vegas) table, prints a summary, and
    restores the Python file -- see this file's module docstring. No-op
    (prints a note) if the C extension isn't built.
    """
    if not HAS_C_BACKEND:
        print("    [cross-backend check skipped: primat._primat_c not built]")
        return

    with open(th_path, "rb") as f:
        py_bytes = f.read()
    py_table = np.loadtxt(th_path)

    try:
        os.remove(th_path)  # force a cache miss so the C backend recomputes from scratch
        t0 = time.time()
        run_bbn(params=dict(extra, verbose=False, save_nTOp_thermal=True),
                force_backend="c")
        c_dt = time.time() - t0
        c_table = np.loadtxt(th_path)
    finally:
        # Always restore the shipped (Python-generated) file, whether or not
        # the C recompute above succeeded.
        with open(th_path, "wb") as f:
            f.write(py_bytes)

    if py_table.shape != c_table.shape:
        print(f"    [cross-backend check: grid shape mismatch "
              f"py={py_table.shape} c={c_table.shape}, skipping diff]")
        return

    # Absolute floor avoids relative error blowing up near a zero-crossing
    # of L_nTOpCCRTh/L_pTOnCCRTh (see this file's module docstring) -- 1e-8
    # is two decades below the smallest physically-meaningful value checked
    # in test_weak_rates_thermal.c.
    for col, name in ((1, "L_nTOpCCRTh"), (2, "L_pTOnCCRTh")):
        rel = np.abs(py_table[:, col] - c_table[:, col]) / (np.abs(py_table[:, col]) + 1e-8)
        print(f"    {name}: max rel diff (python vs c) = {rel.max():.3f}, "
              f"median = {np.median(rel):.4f}  (C recompute took {c_dt:.1f} s)")


if __name__ == "__main__":
    for label, extra in _COMBOS:
        print(f"--- {label} ---")
        t0 = time.time()
        # PRIMAT's constructor alone is enough: it computes the n<->p weak
        # rates (and, with the defaults below, writes them back to
        # rates/weak/) without needing a full BBN solve.
        PRIMAT(params=dict(extra, verbose=False, save_nTOp=True,
                          save_nTOp_thermal=True))
        print(f"    done in {time.time() - t0:.1f} s")

        if extra.get("thermal_corrections", True):
            _compare_thermal_with_c(extra, _thermal_cache_path(extra))

    # ---- prune git-tracked files that are no longer part of the shipped set --
    # (stale after a fingerprint change: their hash no longer matches any combo).
    expected, cache_dir = _expected_filenames(_COMBOS)
    orphans = sorted(_tracked_cache_files(cache_dir) - expected)
    print("\n--- pruning stale shipped cache files ---")
    if not orphans:
        print("    none (shipped set already consistent)")
    for base in orphans:
        path = os.path.join(cache_dir, base)
        if os.path.exists(path):
            os.remove(path)
            print(f"    removed orphan {base}")
        else:
            # Tracked but already gone from the working tree; nothing to delete.
            print(f"    orphan {base} (already absent from working tree)")

    print("\nShipped weak-rate cache set is now:")
    for base in sorted(expected):
        print(f"    {base}")
