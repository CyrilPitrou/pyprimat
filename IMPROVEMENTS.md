# IMPROVEMENTS.md — rate-generation scripts (`generate_rates/`) backlog

Lower-priority cleanups for the *offline* rate/network generation scripts
(`convert_ac2024_rates.py`, `nuclide_table.py`, `nuclear_data.py`,
`reformat_scientific.py`).  These scripts run **once** to (re)generate
`pyprimat/rates/nuclear/tables/*.txt` and `pyprimat/rates/nuclear/data/*.csv`,
so none of the items below affect the BBN solver or normal users.

The *blocking* correctness bugs that stop the command from producing correct
output are tracked separately in `FIX.md` (#1 broken import, #2 unused
hard-coded analytic table, #3 stale default paths).  Do those first; the items
here are quality/maintainability only and can wait.

Date: 2026-06-11.  Numbering continues the review (#1–#3 are in FIX.md).

---

## #4 — Duplicate, drift-prone NUBASE parsers [DONE]

`nuclide_table.py` defines `load_nubase_all` + `_parse_spin`
(`nuclide_table.py:102,110`); `nuclear_data.py` defines its own `load_nubase` +
`_parse_spin` (`nuclear_data.py:405,412`).  Two fixed-width readers of the same
NUBASE2020 file, each with independently hard-coded column offsets.  If one is
corrected and the other is not, masses/spins silently disagree between the two
code paths.

**Improve:** keep one reader (e.g. `nuclide_table.load_nubase_all`, which is the
general `(Z,A)`-keyed one) and have the other module import it; delete the
duplicate `_parse_spin`.

**Done:** `nuclear_data.load_nubase` is now a thin wrapper around
`nuclide_table.load_nubase_all` (+ `resolve_token` to get each canonical
name's `(Z,A)`); the duplicate `_parse_spin` and the hard-coded
NUBASE-symbol-string table were removed. `convert_ac2024_rates.py
--produce-csv` still reproduces the committed CSVs/tables byte-for-byte.

## #5 — Generator builds a throwaway `PyPRConfig` just for constants [DONE]

`nuclide_table._DBConfig.__init__` (`nuclide_table.py:202`) does
`base = PyPRConfig()` only to copy seven constants
(`keV, kB, MeV, ma, me, clight, hbar`) into a stand-in config for
`nuclear_data.detailed_balance`.  This is the throwaway-config smell of
IDEAS.md §4.2, and it couples the offline generator to the full runtime config.

**Improve:** import those constants directly from `pyprimat/constants.py` (the
§3.1 constants split) instead of instantiating `PyPRConfig`.  Better still, have
`detailed_balance` accept a small constants object/namedtuple rather than a
config-shaped duck.

**Done:** `_DBConfig.__init__` now copies the seven constants from the frozen
`pyprimat.constants.CONST` instance instead of instantiating `PyPRConfig()`.
Output unchanged (byte-identical regenerated tables/CSVs; `pytest -m
reference` passes).

## #6 — Mathematica→Python translator is unguarded against *silent* mis-translation

`analytic_rate_function` (`convert_ac2024_rates.py:372`) `eval`s a string built
by hand-rolled implicit-multiplication / `With` / `If` / `E^` rewriting.
`build_analytic_blocks` catches *translation exceptions*, but a translation that
is **wrong yet still valid Python** (e.g. an operator-precedence slip between
`^`→`**` and an inserted `*`) evaluates silently to wrong numbers.  The `eval`
itself is fine (trusted source, `__builtins__` disabled).

**Improve:** add a regression test (see "Tests" below) that pins each
`_ANALYTIC_REACTIONS` entry's evaluated rate against the committed rate file, so
a translator change that shifts any number is caught.

## #7 — `reformat_scientific.py` overwrites its input by default and crashes on non-numbers

`reformat_scientific.py` defaults `output_file = input_file` (destructive
in-place rewrite with no backup), and `nums = [float(x) for x in parts]` raises
on any non-`#` line that is not purely numeric.

**Improve:** require an explicit `--in-place` (or a distinct output path) and
skip/guard non-numeric lines.  Add a one-line module docstring.

## #8 — `GRID_NPTS` / T9 bounds duplicated between generator and config [DONE]

`convert_ac2024_rates.py:62-64` hard-codes `GRID_NPTS=500`,
`GRID_T9_MIN=1e-3`, `GRID_T9_MAX=1e1`, duplicating PyPRIMAT's
`rate_grid_npts` / `rate_grid_T9_min` / `rate_grid_T9_max` defaults.  If they
drift, the "standard 500-point grid" comments become false.

**Improve:** single-source them (import from `pyprimat.config`/`constants`), or
at least assert equality at start-up.

**Done:** `GRID_NPTS`/`GRID_T9_MIN`/`GRID_T9_MAX` are now read from
`pyprimat.config.DEFAULT_PARAMS["rate_grid_*"]` (a plain dict, no `PyPRConfig`
instantiation). Output unchanged.

## #9 — Two naming systems + collisions only *warned* [DONE]

Rate **filenames** use the short tokens `a`/`d`/`t` (`_CANON_TOKEN`,
`convert_ac2024_rates.py:82`), while the **CSVs** use `He4`/`H2`/`H3`
(`resolve_token`).  Both work, but the `<reactants>TO<products>` name-collision
check in `main()` only prints `WARNING: ... (last write wins)`
(`convert_ac2024_rates.py:875`).  A collision is almost always a real bug (two
distinct reactions mapping to one file).

**Improve:** make a collision an error (or at minimum list which reactions
collided), and document the dual naming convention in one place.

**Done:** added a documentation block above `_CANON_TOKEN` explaining the two
naming systems and pointing to `nuclide_table.resolve_token`/`canonical_name`
as the CSV-side source of truth. Replaced the warning with
`check_name_collisions()`: a name shared by the *same* reaction's tabulated and
analytic forms is reported as an intentional override (analytic wins, as
before); a name shared by two *different* reactions now raises `ValueError`
listing the offending names. Currently 0 collisions either way, so output is
unchanged.

## #10 — Obscure scalar-broadcast guard [DONE]

`write_analytic_file` (`convert_ac2024_rates.py:466`) uses
`block["rate"](grid) * np.ones_like(grid)` to broadcast T9-independent analytic
rates (constants, decays) to the grid shape.

**Improve:** `np.broadcast_to(block["rate"](grid), grid.shape)` (or
`np.full_like`) reads clearer and states the intent.

**Done:** now `np.array(np.broadcast_to(block["rate"](grid), grid.shape), dtype=float)`.

## #11 — `main()` is one long function [DONE]

`convert_ac2024_rates.py::main` runs four distinct stages inline (tabulated
tables / analytic tables / collision report / network CSVs).

**Improve:** extract one helper per stage for readability and unit-testing.

**Done:** extracted `_parse_args`, `_generate_tabulated`, `_generate_analytic`;
`main()` now calls these plus `check_name_collisions` and
`unified_reactions`/`write_network_files`.

---

## Tests worth adding (currently zero coverage for `generate_rates/`)

- **Translator regression (covers #6, highest value):** evaluate every
  `_ANALYTIC_REACTIONS` entry on the standard grid → assert finite and positive,
  and compare against the committed `tables/<name>.txt` rate column.
- **Embedded-vs-source sync:** `extract_analytic_from_primat(PRIMAT-Main.m)`
  reproduces `_ANALYTIC_REACTIONS` (pins the hard-coded literal to the
  Mathematica source).
- **Committed-CSV invariants:** run `conservation_residual` and the
  detailed-balance cross-check (already implemented inside `write_network_files`)
  against the shipped `nuclides.csv` / `reactions_large.csv` /
  `detailed_balance.csv`.
