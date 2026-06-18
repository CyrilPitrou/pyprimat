#!/usr/bin/env python3
"""One-off migration: flatten ``tables/<name>.txt`` into ``tables/<name>/<name>.txt``.

Already applied (CUSTOMPOPUP.md §2.2) -- kept for reference / reproducibility,
e.g. if a fresh export from ``convert_ac2024_rates.py --keep-source-grid``
ever needs to be folded into the new per-reaction-folder layout again.

For every ``rates/nuclear/tables/<name>.txt`` (excluding ``decays.txt``, which
stays flat -- it is a single multi-row table backing every Bm/Bp decay
reaction, not a per-reaction rate table), this moves the file into its own
folder ``rates/nuclear/tables/<name>/<name>.txt``.  Alternate-source sibling
tables (e.g. the ``small_parthenope`` network's ``*_parthenope3.0.txt``
variants) are recognised by their ``_parthenope3.0`` suffix and routed into
the *bare* reaction's folder alongside its PRIMAT-default table, since they
are candidate tables for the same reaction, not a reaction of their own.

Uses ``git mv`` so the move is tracked as a rename (preserves
``git log --follow`` history on individual rate-table files).
"""
import os
import subprocess
import sys

TABLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "pyprimat", "rates", "nuclear", "tables"
)

# Known alternate-source suffixes that name a *sibling* table for an existing
# reaction rather than a reaction of their own.
_ALT_SUFFIXES = ["_parthenope3.0"]


def bare_name_for(stem: str) -> str:
    for suffix in _ALT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main():
    tables_dir = os.path.abspath(TABLES_DIR)
    moved = 0
    for fname in sorted(os.listdir(tables_dir)):
        if not fname.endswith(".txt") or fname == "decays.txt":
            continue
        stem = fname[: -len(".txt")]
        bare = bare_name_for(stem)
        folder = os.path.join(tables_dir, bare)
        os.makedirs(folder, exist_ok=True)
        src = os.path.join(tables_dir, fname)
        dst = os.path.join(folder, fname)
        subprocess.run(["git", "mv", src, dst], check=True)
        moved += 1
    print(f"Moved {moved} rate-table files into per-reaction folders.")


if __name__ == "__main__":
    sys.exit(main())
