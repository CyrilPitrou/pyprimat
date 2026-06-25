# Plan: rename/rework primat-c data-directory & custom-nuclear-dir CLI

## Problem (current state, confirmed by reading the code)

Two different, confusingly-named concepts exist in `primat-c`:

1. `cfg->data_dir` (`primat-c/include/cprimat/config.h:225`) — "directory
   *containing* `rates/`" (the parent of the shipped tree). Set via the CLI
   flag `--rates-dir PATH` (`primat-c/src/cli.c:53,69-70`), or env var
   `CPRIMAT_RATES_DIR` (`cli.c:58`), **defaulting to `".."`**
   (`cli.c:59`). Despite the flag's name, it does *not* set `cfg->rates_dir`
   — it feeds `cpr_config_init_defaults(&cfg, rates_dir, &err)`
   (`cli.c:85`), which copies it into `cfg->data_dir`
   (`primat-c/src/config.c:307`), and the shipped tree is then resolved as
   `<data_dir>/rates/...`.
2. `cfg->rates_dir` / `cfg->user_rates_dir` (`config.h:186-187`) — the
   Python-mirroring overlay knobs (full-takeover vs. additive overlay over
   the shipped `rates/` tree). Both default `NULL`. **No dedicated CLI
   flag exists for either** — reachable today only via the generic
   `--set rates_dir=PATH` / `--set user_rates_dir=PATH`
   (`config.c:280-281`'s `FLD` table feeding `cpr_config_set_by_name`).

The `".."` default for `data_dir` is brittle: it assumes the binary's CWD
(not the binary's own location) is one level below the sibling `primat/`
Python package directory. This works only when invoked with CWD =
`primat-c/` (so `../primat/rates/...` resolves), and silently breaks — falls
back/fails to find `rates/` — whenever invoked from elsewhere (e.g. from
`primat-c/build/`, from the repo root, or installed system-wide), which
matches the user's report that this "fails all the time."

There is currently **no dedicated flag** for "point at a custom nuclear
network directory" (i.e. a directory bundling a `networks/` subfolder and a
`tables/` subfolder together, analogous to the GUI's custom-network zip).
The only existing test, `primat-c/tests/unit/test_network_data_phase4.c:227-283`,
exercises `user_rates_dir` overlay for a *single* `networks/<name>.txt` file
only — no `tables/` override, no full-takeover (`rates_dir`) coverage, no
combined networks+tables directory.

## Goals

1. Rename the "directory containing rates/" concept from `data_dir` to
   something that means "the data folder itself," not its parent — i.e.
   stop making users (and code) pass a *parent* directory and instead point
   directly at the folder that contains `NEVO/`, `weak/`, `plasma/`,
   `nuclear/`.
2. Fix the broken/brittle default so `cprimat` reliably finds the shipped
   data when run from its own build folder, regardless of caller CWD quirks
   — anchor relative to the executable's own location instead of `".."`
   relative to CWD, or fail loudly with a clear message instead of silently
   missing files.
3. Add a real `--custom-nuclear-dir PATH` CLI flag, distinct from
   `--data-dir`, for pointing at one external directory containing both
   `networks/` and `tables/` subfolders (mirrors the GUI's
   "Import custom network" zip shape) — wired as the `user_rates_dir`
   additive-overlay mechanism (or `rates_dir` full-takeover, see open
   question below), but exposed as its own first-class flag instead of
   only via `--set`.
4. Add C unit-test coverage for the combined networks+tables custom
   directory case, which is currently untested.
5. Keep the Python side (`primat/config.py`, `cli.py`) and C side
   (`config.h`/`config.c`/`cli.c`) in sync per CLAUDE.md's "primat-c and
   primat in sync" rule — any rename/flag added on one side must be
   mirrored on the other, including doc updates (`README.md`,
   `primat-c/examples/run_basic.ini`, `runfiles/run_basic.py` if applicable).

## Additional restructuring: move `nuclear/data/` up to `rates/csv/`

Confirmed current layout: `primat/rates/nuclear/` contains three
subfolders — `data/` (`nuclides.csv`, `reactions_large.csv`,
`detailed_balance.csv` — the reaction catalog, *not* overlay-aware per
CLAUDE.md), `networks/` (`<name>.txt` network definitions), and `tables/`
(per-reaction rate tables, plus the flat `decays.txt`).

Because a *custom* nuclear directory (today's GUI export, and the planned
`--custom-nuclear-dir`) only ever supplies `networks/` + `tables/` — never
the reaction catalog — `nuclear/` having a third `data/` subfolder breaks
the mirror: a custom-nuclear-dir is not structurally identical to
`rates/nuclear/`, it's `rates/nuclear/` minus one folder. Moving `data/` out
of `nuclear/` and up to a sibling `rates/csv/` (alongside `nuclear/`,
`networks/` becomes purely `rates/nuclear/{networks,tables}`) makes
`rates/nuclear/` *exactly* the shape a custom-nuclear-dir must have —
no implicit exception to document or special-case.

Concretely:
- `primat/rates/nuclear/data/{nuclides,reactions_large,detailed_balance}.csv`
  → `primat/rates/csv/{nuclides,reactions_large,detailed_balance}.csv`
- Python: every reader of these three files (in `primat/network_data.py`,
  wherever `resolve_rates_path("nuclear", "data", ...)` or equivalent is
  called) updates its path to `resolve_rates_path("csv", ...)`.
- C: `primat-c/src/network_data.c`'s equivalent catalog-loading path updates
  identically (CLAUDE.md already notes these three files are "always read
  from the shipped tree" on both backends, i.e. never overlay-routed — that
  invariant doesn't change, only the literal path components do).
- `decays.txt` stays under `rates/nuclear/tables/` (per CLAUDE.md, "the
  single exception... stays a flat file directly under tables/") — it is
  *not* part of this move, since it's already where a custom-nuclear-dir's
  `tables/` would expect it.
- Update every doc reference to `nuclear/data/` (CLAUDE.md's "Adding a new
  reaction" section, `README.md` if applicable) to `rates/csv/`.
- This is a pure rename/move with no numerical effect — re-validate with
  `python runfiles/PyPRIMAT_run.py` and the C test suite as a no-op check,
  same as the data_dir rename above.

## Naming changes

| Old | New | Meaning |
|-----|-----|---------|
| `cfg->data_dir` (parent of `rates/`) | `cfg->data_dir` (the data folder *itself*, no implicit `/rates` suffix) | Now points directly at the folder containing `NEVO/`, `weak/`, `plasma/`, `nuclear/` — not its parent. |
| `--rates-dir PATH` | `--data-dir PATH` | Renamed flag to match; sets the new `data_dir` directly (no `/rates` appended internally). |
| `CPRIMAT_RATES_DIR` env var | `CPRIMAT_DATA_DIR` | Renamed to match. |
| (none) | `--custom-nuclear-dir PATH` | New flag; wires a directory containing `networks/` + `tables/` subfolders as the nuclear-network overlay (maps to `user_rates_dir`, scoped to nuclear network/table resolution only — same scope `cpr_config_resolve_rates_path` already covers). |
| `cfg->rates_dir` / `cfg->user_rates_dir` | keep names (already accurately named — these are about the nuclear-rates overlay, not the whole data tree) | No change; just expose `user_rates_dir` via the new flag instead of only `--set`. |

This means every internal path-join that currently does
`<data_dir>/rates/...` becomes `<data_dir>/...` (drop the `/rates` segment)
once `data_dir` itself points at the data folder. Same for
`list_or_clear_weak_cache` (`cli.c:25`, currently
`"%s/rates/weak"` → becomes `"%s/weak"`).

## Default value fix

Replace the `".."`-relative-to-CWD default with a path resolved relative to
the executable's own location (so `cprimat` found via `argv[0]`/`/proc/self/exe`-style
resolution, or platform equivalent, looks for a sibling `../primat/rates`
*relative to the binary*, not relative to whatever directory the user
happened to invoke it from). Concretely:

- If `CPRIMAT_DATA_DIR` env var is set, use it verbatim (highest precedence,
  unchanged behavior).
- Else, derive the executable's directory and default to
  `<exe_dir>/../primat/rates` if it exists.
- Else (e.g. installed system-wide with no sibling `primat/`), fail with a
  clear error telling the user to pass `--data-dir` explicitly, rather than
  silently resolving to a nonexistent path.

This directly addresses the user's hypothesis: `".."` was being resolved
against the *caller's* CWD, not the binary's location, which is exactly why
it "fails all the time" whenever invoked from anywhere other than
`primat-c/`.

## Custom nuclear dir semantics

`--custom-nuclear-dir PATH` requires `PATH/networks/` and/or `PATH/tables/`
to exist (at least one); wires `PATH` as `cfg->user_rates_dir` (additive
overlay — falls back to shipped tables/networks for anything not present in
PATH), matching the GUI's existing custom-network directory shape after
unzipping. Do **not** silently allow it to also imply `rates_dir`
(full-takeover) — that stays `--set rates_dir=PATH` only, since takeover for
the *entire* rates tree (including NEVO/weak/plasma) is a different,
broader operation not yet implied by "custom nuclear network."

## Implementation steps

1. **Python side** (`primat/config.py`): confirm whether an analogous
   parent-vs-self ambiguity exists for `PRIMATConfig.data_dir`-equivalent
   logic; if `primat`'s own `rates/` resolution has no such parent-dir
   concept (it's a package-relative path, not user-supplied), this rename
   may be C-only. Verify before touching Python.
2. **`primat-c/include/cprimat/config.h`**: redefine `data_dir` semantics
   (drop "directory containing rates/" framing in the comment at line 225;
   document it as "the data folder itself").
3. **`primat-c/src/config.c`**: update `cpr_config_init_defaults` and
   `cpr_config_resolve_rates_path` to stop appending `/rates` when joining
   `data_dir` with sub-paths.
4. **`primat-c/src/cli.c`**:
   - Rename `--rates-dir` → `--data-dir`, env var `CPRIMAT_RATES_DIR` →
     `CPRIMAT_DATA_DIR`.
   - Implement executable-relative default resolution instead of literal
     `".."`.
   - Add `--custom-nuclear-dir PATH` flag wiring `user_rates_dir`.
   - Update `usage()` (line 48-54) and `list_or_clear_weak_cache`'s path
     join (line 25).
5. **Tests**:
   - Extend `test_network_data_phase4.c`'s overlay block to cover a
     directory with both `networks/` and `tables/` populated (override one
     reaction's rate table + one network file simultaneously), and confirm
     additive fallback still works for non-overlaid files in both
     subfolders.
   - Add a CLI-level test/script exercising `--data-dir` and
     `--custom-nuclear-dir` end-to-end (build dir invocation).
6. **Docs**: update `README.md`, `primat-c/examples/run_basic.ini` if it
   references the old flag/env var names, and CLAUDE.md if needed (it
   already documents `rates_dir`/`user_rates_dir` overlay — add the new
   flag names there).
7. **Validation**: re-run `primat-c/tests` (`make test` or equivalent) and
   the standard Python validation run (`python runfiles/PyPRIMAT_run.py`)
   to confirm no numerical regressions (this is a pure path/CLI refactor,
   should be a no-op numerically).

## Open questions for the user before implementing

- Should `--data-dir`'s default-resolution failure (no sibling `primat/`
  found) be a hard error, or should it still fall back to attempting
  `<cwd>/../primat/rates` as a secondary guess for backward compatibility?
- Should `--custom-nuclear-dir` also accept a `.zip` (matching the GUI's
  export format) and unpack it transparently, or PATH-only (plain directory)
  for now, leaving zip support for later?
