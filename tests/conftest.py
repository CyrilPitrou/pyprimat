"""
Shared pytest fixtures for PyPRIMAT tests.
"""
import sys
import os
import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
