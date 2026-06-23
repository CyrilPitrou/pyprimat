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

1. Full physics, all corrections on (the ``PyPRConfig`` default): radiative,
   finite-mass, thermal and spectral-distortion corrections + QED pressure,
   with non-instantaneous decoupling (``incomplete_decoupling=True``).
2. Same as (1) but ``QED_corrections=False`` -- the other half of the
   QED on/off comparison used throughout ``tests/test_decoupling_qed.py``.
3. ``incomplete_decoupling=False`` (instantaneous-decoupling limit),
   ``QED_corrections=True``. ``spectral_distortions`` must be ``False``
   here: it requires the NEVO spectral table, which only exists in
   non-instantaneous-decoupling mode (``PyPRConfig.__init__`` raises
   otherwise).
4. Same as (3) but ``QED_corrections=False``.

Run from the repo root::

    python runfiles/generate_weak_rate_caches.py

After running, force-add the new/changed files, e.g.::

    git add -f rates/weak/nTOp_<hash>.txt rates/weak/nTOp_thermal_<hash>.txt
"""
import sys
import os
import time

_pyprimat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pyprimat_path not in sys.path:
    sys.path.insert(0, _pyprimat_path)

from pyprimat import PyPR

# Each entry only lists the flags that deviate from the PyPRConfig defaults
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

if __name__ == "__main__":
    for label, extra in _COMBOS:
        print(f"--- {label} ---")
        t0 = time.time()
        # PyPR's constructor alone is enough: it computes the n<->p weak
        # rates (and, with the defaults below, writes them back to
        # rates/weak/) without needing a full BBN solve.
        PyPR(params=dict(extra, verbose=False, save_nTOp=True,
                          save_nTOp_thermal=True))
        print(f"    done in {time.time() - t0:.1f} s")
