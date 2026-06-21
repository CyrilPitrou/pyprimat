# -*- coding: utf-8 -*-
"""
cache_utils.py — fingerprinted self-validating cache files
============================================================================

Several expensive precomputations (n<->p weak rates, their finite-temperature
radiative corrections, the e+- thermodynamic tables) are written to plain-text
``np.savetxt`` files under ``rates/`` and reloaded on the next run instead of
being recomputed.  Historically these caches were trusted unconditionally:
whatever was on disk was used, even if the configuration that produced it
(neutrino-decoupling treatment, spectral distortions, sampling density, ...)
no longer matches the current run.  This silently makes flags such as
``spectral_distortions`` a no-op.

The fix is a *fingerprint*: a dict of every configuration entry that affects
the cached numbers, serialised as canonical (sorted-key, whitespace-free) JSON
and hashed with sha256 (truncated to 16 hex digits -- short enough to read,
long enough that two different configurations colliding by accident is
astronomically unlikely).  The hash and the JSON dict are written as
``#``-comment header lines of the cache file:

    # fingerprint_hash: a3f9c1b2e4d5f607
    # fingerprint: {"format_version":1,"sampling_nTOp_per_decade":80,...}

``np.loadtxt`` ignores ``#`` lines by default, so the data rows are unaffected.
The JSON line is for humans ("with which flags was this produced?"); only the
hash line is compared by the loader.  A cache file with no header (or an
unparsable one) is reported as having an unknown fingerprint -- the caller
decides whether that counts as a cache hit or a miss.
"""

import hashlib
import json
import os

import numpy as np


def fingerprint_hash(fingerprint: dict) -> str:
    """Return the sha256 hash (first 16 hex digits) of a fingerprint dict.

    The dict is serialised to canonical JSON first (``sort_keys=True`` and no
    extra whitespace) so that the hash depends only on the *values*, not on
    the order in which the caller happened to build the dict.

    Args:
        fingerprint: dict of config values that determine a cache file's
            content (e.g. ``{"format_version": 1, "sampling_nTOp_per_decade": 80, ...}``).

    Returns:
        16-hex-character hash string, e.g. ``"a3f9c1b2e4d5f607"``.
    """
    blob = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def read_cache_fingerprint_hash(path: str):
    """Return the fingerprint hash stored in a cache file's header, or None.

    Reads only the leading ``#``-comment lines of `path`, looking for a line
    of the form ``# fingerprint_hash: <hash>``.  Stops at the first
    non-comment line (the data rows are never parsed).

    Args:
        path: path to a file previously written by
            :func:`write_cache_with_fingerprint`, or a legacy file with no
            header.

    Returns:
        The hash string if found, otherwise ``None`` -- which covers a
        missing file, a header-less legacy file, and a corrupt header.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if line.startswith("# fingerprint_hash:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def write_cache_with_fingerprint(path: str, fingerprint: dict, columns, col_header: str = ""):
    """Write a ``np.savetxt`` cache file with a fingerprint header.

    Args:
        path: output file path; parent directory must already exist.
        fingerprint: dict to hash and embed verbatim as JSON (see
            :func:`fingerprint_hash`).
        columns: sequence of equal-length 1-D arrays, written column-wise
            (``np.column_stack(columns)``).
        col_header: optional human-readable column-name line, written before
            the fingerprint lines (e.g. ``"T[K] rate[1/s]"``).

    Example:
        >>> write_cache_with_fingerprint(
        ...     "nTOp_frwrd.txt",
        ...     {"format_version": 1, "sampling_nTOp_per_decade": 80},
        ...     [T_all, frwrd], col_header="T[K] rate[1/s]")
    """
    fp_hash = fingerprint_hash(fingerprint)
    fp_json = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    header_lines = []
    if col_header:
        header_lines.append(col_header)
    header_lines.append("fingerprint_hash: " + fp_hash)
    header_lines.append("fingerprint: " + fp_json)
    # Write to a per-process temp file then atomically rename into place
    # (os.replace), so concurrent MC workers racing to populate a missing
    # cache never observe a partially-written file.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    np.savetxt(tmp_path, np.column_stack(columns), header="\n".join(header_lines))
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Writable weak-rate cache directory: inspection / cleanup (`pyprimat
# --cache-info` / `--cache-clear`). See FUTURE.md P0.2: every new
# PyPRConfig fingerprint run with spectral_distortions/incomplete_decoupling
# etc. drops another nTOp_<hash>.txt / nTOp_thermal_<hash>.txt file under
# pyprimat/rates/weak/; these are regenerable on demand (a fresh run just
# recomputes and re-caches them), so it is always safe to delete them.
# ---------------------------------------------------------------------------

def weak_cache_dir(cfg) -> str:
    """Return the ``pyprimat/rates/weak/`` directory for ``cfg.data_dir``."""
    return os.path.join(cfg.data_dir, "rates", "weak")


def list_weak_cache_files(cfg):
    """Return the sorted list of ``nTOp_*.txt`` cache file paths on disk."""
    d = weak_cache_dir(cfg)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, name) for name in os.listdir(d)
        if name.startswith("nTOp_") and name.endswith(".txt")
    )


def clear_weak_cache(cfg) -> int:
    """Delete every cached ``nTOp_*.txt`` file. Returns the count removed.

    The cache is purely an optimisation (every entry is reproducible from
    ``cfg`` by recomputing), so removing all of it is always safe -- the
    next run simply pays the one-time recompute cost again per
    configuration touched.
    """
    paths = list_weak_cache_files(cfg)
    for path in paths:
        os.remove(path)
    return len(paths)
