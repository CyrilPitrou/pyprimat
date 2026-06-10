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
    """A solved PyPR instance (small network) reused across the session."""
    from pyprimat.main import PyPR
    # weak_rate_cache=False: always recompute the n<->p rates from scratch
    # (exercise ComputeWeakRates, not just the cached interpolants).
    # save_nTOp stays False (the default): tests must not overwrite the
    # tracked rates/weak/*.txt tables.
    r = PyPR({"weak_rate_cache": False, "network": "small"})
    r.solve()
    return r


@pytest.fixture(scope="session")
def solved_large():
    """A solved PyPR instance (large network) reused across the session."""
    from pyprimat.main import PyPR
    # Default config -> matches the shipped rates/weak/*.txt fingerprint,
    # so this loads the cache instead of recomputing.
    r = PyPR({"network": "large"})
    r.solve()
    return r
