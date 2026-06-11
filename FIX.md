# FIX.md — make `convert_ac2024_rates.py` regenerate the rate set correctly

## Hand this task to: **Sonnet**

Rationale: this is wiring/plumbing repair pinned by an exact regenerate-and-diff
acceptance test — no physics, numerics, or interface-design judgment is
involved. It is squarely in the "mechanical, low-risk, test-pinned → Sonnet"
class of IDEAS.md §10. Do **not** change any rate physics, formula, grid, or
number; only fix the broken wiring so the existing logic actually runs and
reproduces the already-committed output files.

---

## Goal

`generate_rates/convert_ac2024_rates.py` is the offline command that
(re)generates, from `BBNRatesAC2024.dat` + the analytic formulas + NUBASE:

- `pyprimat/rates/nuclear/tables/*.txt`   (435 per-reaction rate tables)
- `pyprimat/rates/nuclear/data/nuclides.csv`
- `pyprimat/rates/nuclear/data/reactions_large.csv`
- `pyprimat/rates/nuclear/data/detailed_balance.csv`

Right now the command **cannot reproduce these files**: the `--produce-csv` path
dies on a stale import, the hard-coded analytic table it advertises as the
"single source of truth" is never actually used, and the default paths point at
a folder that no longer exists. Fix the three bugs below so that **one command,
run from the repo root, regenerates all of the above and they match the
committed files.**

The command runs once in a blue moon, so we only care that it produces the
correct output — ignore style/performance. Three other reviewers' notes (#4–#11)
are intentionally **out of scope**; they live in `IMPROVEMENTS.md`.

## Runtime call graph (what runs when the command runs)

```
convert_ac2024_rates.py  (main)
  └─ nuclide_table.py        (build_nuclide_table, conservation_residual,
                              make_detailed_balance, is_decay, resolve_token)
       └─ nuclear_data.py    (detailed_balance)   ← imports pyprimat.config
```

`reformat_scientific.py` is unrelated — leave it alone.

---

## The three bugs to fix

### Bug #1 — broken import kills `--produce-csv`

`generate_rates/nuclide_table.py:221`:

```python
from generate_from_primat.nuclear_data import detailed_balance
```

`generate_from_primat` does not exist (the folder was renamed to
`generate_rates`; `nuclear_data.py` is a sibling). `make_detailed_balance` is
called by `write_network_files`, so every `--produce-csv` run raises
`ModuleNotFoundError` and the three CSVs are never written.

**Fix:** import the sibling module:

```python
from nuclear_data import detailed_balance
```

(See "Import paths" below — you must also make `pyprimat` importable, because
`nuclear_data.py` does `from pyprimat.config import PyPRConfig`.)

### Bug #2 — the hard-coded analytic table is never used

The module docstring and the comment at `convert_ac2024_rates.py:485` advertise
`_ANALYTIC_REACTIONS` (line 496, ~70 reactions: the β-decays, `aaag`, `dng`, …)
as *"the single source of truth at run time"*, with the promise that the rate
set is *"regenerable from `BBNRatesAC2024.dat` alone — `PRIMAT-main.m` not
needed."* But in `main()` (around lines 856–867) the analytic blocks are built
**only** `if args.primat`, via `extract_analytic_from_primat()`.
`_ANALYTIC_REACTIONS` is referenced nowhere except docstrings and
`_dump_analytic_literal`. So without `--primat`:

- no analytic rate files are written, and
- `unified_reactions` receives `ana_blocks == []`, so the CSVs silently drop all
  ~70 analytic reactions.

**Fix:** in `main()`, the analytic blocks must come from the embedded
`_ANALYTIC_REACTIONS` by default, with `--primat` only as an override for
verification. Concretely, replace the `if args.primat: … else: print("Skipping
…")` logic so that:

```python
if args.primat:
    entries = extract_analytic_from_primat(args.primat)   # verification path
else:
    entries = _ANALYTIC_REACTIONS                         # default: embedded table
ana_blocks, skipped = build_analytic_blocks(entries)
for blk in ana_blocks:
    write_analytic_file(blk, grid, args.tabdir, args.suffix)
# keep the existing "built N analytic reactions" / skipped reporting
```

Note `build_analytic_blocks` already accepts exactly the
`(ref, reac, f, forward)` tuple shape that `_ANALYTIC_REACTIONS` uses, so no
other change is needed there.

### Bug #3 — stale / wrong default paths

In `argparse` (and the usage docstring at the top, and the comment near line
493) the defaults point at the non-existent `generate_from_primat/` folder, and
`--datadir` points at the `tables/` dir even though the CSVs live in `data/`:

| Arg | Current default | Should be |
|-----|-----------------|-----------|
| `--input`   | `generate_from_primat/BBNRatesAC2024.dat` | `generate_rates/BBNRatesAC2024.dat` |
| `--nubase`  | `generate_from_primat/nubase_4.mas20.txt` | `generate_rates/nubase_4.mas20.txt` |
| `--tabdir`  | `pyprimat/rates/nuclear/tables` | unchanged (correct) |
| `--datadir` | `pyprimat/rates/nuclear/tables` | `pyprimat/rates/nuclear/data` |

Also fix the `generate_from_primat/` references and the `PRIMAT-main.m` casing
(the real file is `generate_rates/PRIMAT-Main.m`, capital **M**) in the module
docstring usage block and the comment above `_ANALYTIC_REACTIONS`, so the
documented commands are copy-pasteable.

### Import paths (needed for #1 to actually run)

Running `python generate_rates/convert_ac2024_rates.py` puts `generate_rates/`
on `sys.path[0]` (so the sibling imports `nuclide_table` / `nuclear_data` work)
but **not** the repo root, so `from pyprimat.config import PyPRConfig` inside
`nuclear_data.py` / `nuclide_table.py` fails. Make the command self-contained:
near the top of `convert_ac2024_rates.py`, insert both the script directory and
the repo root onto `sys.path` (the repo root is the parent of `generate_rates/`)
**before** `main()` triggers the lazy `from nuclide_table import …` /
`from nuclear_data import …`. A small, well-commented bootstrap is fine, e.g.:

```python
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):     # generate_rates/ and repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

Equivalently you may document `PYTHONPATH=. python …` — but the in-script
bootstrap is preferred so the bare command "just works".

---

## Validation (acceptance test — must pass)

The command is deterministic, so a correct fix reproduces the committed files
exactly. Do **not** overwrite the committed files while testing; write to a temp
dir and diff.

1. From the repo root, regenerate into temp dirs using the **embedded** analytic
   table (no `--primat`), exercising both #1 and #2:

   ```bash
   mkdir -p /tmp/rgen/tables /tmp/rgen/data
   python generate_rates/convert_ac2024_rates.py \
       --tabdir /tmp/rgen/tables \
       --datadir /tmp/rgen/data \
       --produce-csv
   ```

   This must finish without error, print the "formal check OK: all N reactions
   conserve A and Q" and "detailed-balance cross-check" lines, and write 435
   `.txt` files plus the three CSVs.

2. Diff against the committed outputs — **both must be empty**:

   ```bash
   diff -rq /tmp/rgen/tables pyprimat/rates/nuclear/tables
   diff -rq /tmp/rgen/data   pyprimat/rates/nuclear/data
   ```

   (First confirm the grid: a committed tabulated file such as
   `pyprimat/rates/nuclear/tables/npTOdg.txt` has ~500 data rows ⇒ the standard
   grid, so run **without** `--keep-source-grid`. If it has ~60 rows, add
   `--keep-source-grid` and re-test. Pick whichever flag makes the diff empty;
   do not change the grid constants.)

3. Prove the embedded table matches the Mathematica source (i.e. `--primat`
   gives the same result), exercising the extractor path too:

   ```bash
   python generate_rates/convert_ac2024_rates.py \
       --tabdir /tmp/rgen2/tables --datadir /tmp/rgen2/data \
       --primat generate_rates/PRIMAT-Main.m --produce-csv
   diff -rq /tmp/rgen2/tables pyprimat/rates/nuclear/tables
   diff -rq /tmp/rgen2/data   pyprimat/rates/nuclear/data
   ```

4. Confirm the solver still works on the regenerated-and-identical data:

   ```bash
   python runfiles/PyPRIMAT_run.py            # CLAUDE.md refs must hold
   python -m pytest -m reference -q           # must pass
   ```

### If the diffs are NOT empty

Only the **analytic** `.txt` files (and the CSV rows for analytic reactions) can
legitimately differ, and only if the embedded `_ANALYTIC_REACTIONS` has drifted
from the committed `PRIMAT-Main.m`. If so:

1. Regenerate the embedded literal from the committed source and paste it back
   into `convert_ac2024_rates.py`, replacing `_ANALYTIC_REACTIONS`:

   ```bash
   python generate_rates/convert_ac2024_rates.py --dump-analytic generate_rates/PRIMAT-Main.m
   ```

2. Re-run validation. After this, step 1 (no `--primat`) and step 3 (`--primat`)
   must produce identical, committed-matching output.

If tabulated `.txt` files differ, something in the parse/interp wiring was
disturbed — revert and reconsider; tabulated rates come straight from
`BBNRatesAC2024.dat` and must be untouched by these fixes.

---

## Acceptance criteria (all required)

- [ ] `python generate_rates/convert_ac2024_rates.py … --produce-csv` runs from
      the repo root with **no `--primat`** and **no manual `PYTHONPATH`**, and
      exits 0.
- [ ] Both `diff -rq` checks against `pyprimat/rates/nuclear/{tables,data}` are
      empty (regenerated output is byte-identical to committed).
- [ ] The `--primat generate_rates/PRIMAT-Main.m` run produces the same output
      (embedded table == Mathematica source).
- [ ] `python runfiles/PyPRIMAT_run.py` reproduces the CLAUDE.md reference
      values and `pytest -m reference` passes.
- [ ] No change to any rate number, formula, grid, or threshold — diff of
      `convert_ac2024_rates.py` / `nuclide_table.py` touches only imports, the
      `main()` analytic-source selection, default-path strings, and the
      `sys.path` bootstrap.

## Out of scope (do NOT do here)

- Items #4–#11 in `IMPROVEMENTS.md` (duplicate NUBASE readers, throwaway config,
  translator tests, `reformat_scientific.py`, grid-constant single-sourcing,
  collision-as-error, `main()` refactor, etc.).
- Any change to the rate physics or to `BBNRatesAC2024.dat` / `PRIMAT-Main.m`.
- Reformatting / renaming beyond what the four fixes strictly require.

## Suggested commit

One commit, e.g.:

> Fix convert_ac2024_rates.py so it regenerates the rate set (§generate_rates)
>
> - nuclide_table.py: import detailed_balance from the sibling nuclear_data
>   (was the dead generate_from_primat.nuclear_data).
> - main(): use the embedded _ANALYTIC_REACTIONS by default; --primat is now
>   only a verification override.  Restores the "regenerable from
>   BBNRatesAC2024.dat alone" guarantee.
> - Fix stale default paths (generate_from_primat→generate_rates, datadir→data)
>   and add a sys.path bootstrap so the command runs from the repo root.
>
> Regenerated output is byte-identical to the committed tables/ and data/ files;
> PyPRIMAT_run.py and pytest -m reference still pass.
>
> Cyril Pitrou
