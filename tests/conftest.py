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
    # save_nTOp stays False: tests must not overwrite the tracked
    # rates/weak/*.txt tables.
    r = PyPR({"compute_nTOp": True, "network": "small"})
    r.solve()
    return r


@pytest.fixture(scope="session")
def solved_large():
    """A solved PyPR instance (large network) reused across the session."""
    from pyprimat.main import PyPR
    r = PyPR({"compute_nTOp": False, "network": "large"})
    r.solve()
    return r
