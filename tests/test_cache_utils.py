"""Direct unit tests for primat.cache_utils' fingerprinted-cache helpers.

These are exercised indirectly by every test that builds a PRIMATConfig and
solves (the n<->p weak-rate cache and the electron-thermo cache both use this
module), but a direct round-trip test pins the contract precisely:
fingerprint_hash is order-independent, read_cache_fingerprint_hash recovers
exactly the hash write_cache_with_fingerprint wrote, and a corrupted or
missing file is treated as "unknown fingerprint" (None) rather than raising.
The last test is a regression check for the atomic-write fix: the cache file
is written via a per-process temp file + os.replace so concurrent writers
racing on a missing cache cannot tear it (see cache_utils.py's module
docstring for the motivating incident).
"""
import os

import numpy as np
import pytest

from primat.cache_utils import (
    fingerprint_hash,
    read_cache_fingerprint_hash,
    write_cache_with_fingerprint,
)


def test_fingerprint_hash_is_order_independent():
    h1 = fingerprint_hash({"a": 1, "b": 2})
    h2 = fingerprint_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 16


def test_fingerprint_hash_distinguishes_values():
    assert fingerprint_hash({"a": 1}) != fingerprint_hash({"a": 2})


def test_write_then_read_round_trip(tmp_path):
    path = str(tmp_path / "cache.txt")
    fp = {"format_version": 1, "sampling_nTOp_per_decade": 80}
    write_cache_with_fingerprint(path, fp, [np.array([1., 2., 3.]),
                                             np.array([4., 5., 6.])],
                                  col_header="T[K] rate[1/s]")
    assert read_cache_fingerprint_hash(path) == fingerprint_hash(fp)
    # Data rows must be intact (the fingerprint header lines are '#' comments
    # that np.loadtxt ignores).
    data = np.loadtxt(path)
    assert data.shape == (3, 2)
    assert data[:, 0].tolist() == [1., 2., 3.]


def test_read_missing_file_returns_none(tmp_path):
    assert read_cache_fingerprint_hash(str(tmp_path / "does_not_exist.txt")) is None


def test_read_truncated_header_returns_none(tmp_path):
    path = tmp_path / "corrupt.txt"
    # No '# fingerprint_hash:' line at all -- a header-less legacy file.
    path.write_text("# just a comment\n1.0 2.0\n")
    assert read_cache_fingerprint_hash(str(path)) is None


def test_read_mismatched_hash_is_detected(tmp_path):
    path = str(tmp_path / "cache.txt")
    write_cache_with_fingerprint(path, {"a": 1}, [np.array([1.])])
    stored = read_cache_fingerprint_hash(path)
    assert stored != fingerprint_hash({"a": 2})  # caller's mismatch check


def test_write_is_atomic_no_leftover_tmp_file(tmp_path):
    """Regression test for the os.replace() atomic-write fix.

    write_cache_with_fingerprint must write to f"{path}.tmp.{pid}" and
    os.replace() it into place, never np.savetxt directly to `path` -- so a
    reader can never observe a half-written file, and no stray .tmp.<pid>
    file is left behind afterwards.
    """
    path = str(tmp_path / "cache.txt")
    write_cache_with_fingerprint(path, {"a": 1}, [np.array([1., 2.])])
    assert os.path.exists(path)
    leftover = [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert leftover == []


def test_write_overwrites_existing_file_atomically(tmp_path):
    path = str(tmp_path / "cache.txt")
    write_cache_with_fingerprint(path, {"a": 1}, [np.array([1.])])
    write_cache_with_fingerprint(path, {"a": 2}, [np.array([99.])])
    assert read_cache_fingerprint_hash(path) == fingerprint_hash({"a": 2})
    assert np.loadtxt(path).item() == pytest.approx(99.)
