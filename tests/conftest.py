"""
Shared pytest fixtures for primat tests.
"""
import sys
import os
import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories where a non-default PRIMATConfig's fingerprinted cache
# machinery (primat.weak_rates.cache / primat.cache_utils) writes a new file
# by design -- weak_rate_cache=True writes primat/data/weak/nTOp_<hash>.txt
# whenever a test's flags don't match an already-cached fingerprint, and
# nevo_file_prefix tests (tests/test_weak_rates.py) drop renamed NEVO table
# copies into primat/data/NEVO/. This is a deliberate performance feature,
# not a bug -- but left unchecked across a full test run it litters the
# working tree with untracked files (only a fixed handful of nTOp_*.txt are
# actually committed, see generate_weak_rate_caches.py's top comment).
_WATCHED_DATA_DIRS = [
    os.path.join(_REPO_ROOT, "primat", "data", "weak"),
    os.path.join(_REPO_ROOT, "primat", "data", "NEVO"),
]


@pytest.fixture(scope="session", autouse=True)
def _delete_generated_data_byproducts():
    """Session-wide safety net: delete any file created under
    :data:`_WATCHED_DATA_DIRS` while the test session ran.

    Snapshots each directory's contents before the session and removes
    anything not in that snapshot at teardown, so cache files/table copies a
    test writes as a side effect never survive the test run -- regardless of
    which test wrote them or whether its own cleanup (e.g. the try/finally in
    ``test_nevo_file_prefix_reproduces_default``) already ran. Files present
    before the session started (the shipped defaults, or leftovers from a
    previous run) are left untouched.
    """
    before = {d: set(os.listdir(d)) for d in _WATCHED_DATA_DIRS if os.path.isdir(d)}
    yield
    for d, pre_existing in before.items():
        for name in set(os.listdir(d)) - pre_existing:
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass  # already gone (e.g. removed by the test's own cleanup)


@pytest.fixture(scope="session")
def solved_small():
    """A solved PRIMAT instance (small network) reused across the session.

    Uses the default config, i.e. ``weak_rate_cache=True``: with the
    fingerprinted cache (see primat.weak_rates), the shipped
    ``rates/weak/nTOp_*.txt`` tables match this configuration's fingerprint,
    so the n<->p rates are loaded rather than recomputed (~1.8 s saved).
    The recompute path itself (``RecomputeWeakRates`` with
    ``weak_rate_cache=False``) is exercised separately and compared against
    this cached path in
    ``tests/test_weak_rates.py::test_recomputed_rates_match_cached``.
    """
    from primat.main import PRIMAT
    r = PRIMAT({"network": "small"})
    r.solve()
    return r


@pytest.fixture(scope="session")
def solved_large():
    """A solved PRIMAT instance (large network) reused across the session."""
    from primat.main import PRIMAT
    # Default config -> matches the shipped rates/weak/*.txt fingerprint,
    # so this loads the cache instead of recomputing.
    r = PRIMAT({"network": "large"})
    r.solve()
    return r
