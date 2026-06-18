# CUSTOMPOPUP.md — Network reshuffle + popup-based custom-network workflow

Implementation plan for Sonnet. Do all work on a new git branch `gui`, branched
off current `master` HEAD:

```bash
git checkout master && git pull --ff-only  # if a remote exists
git checkout -b gui
```

This document supersedes `CUSTOM.md` in full — `CUSTOM.md`'s §1/§2/§6.1–§6.5
(network reshuffle, amax generalisation, "Physics" panel merge, doc/test
updates) are carried over essentially unchanged into §1–§3 below, but its §4/§5
(inline sidebar "Customise Reactions" checkbox list) are replaced wholesale by
the popup-dialog workflow in §5–§8, which also requires reorganising
`rates/nuclear/tables/` into one folder per reaction (§4) — a layout change
`CUSTOM.md` explicitly did not need. Once this plan is implemented and merged,
delete `CUSTOM.md` (its content lives on here).

Design decisions already settled (do not re-litigate):
- `small_parthenope` already exists on disk (`rates/nuclear/networks/small_parthenope.txt`) — keep as-is.
- `network="large", amax=8` is accepted as the new baseline replacing `medium`
  even if its exact reaction set/reference numbers differ slightly from the
  old 67-reaction `medium.txt` — re-measure and document the real numbers
  (§3.1), don't force an exact match.
- Rate-table folder reorganisation is a **hard migration**: no flat-file
  backward compatibility. Every loader, `convert_ac2024_rates.py`, and the GUI
  assume the new `rates/nuclear/tables/<reaction>/<file>.txt` layout.
- The popup is implemented with Streamlit's native `st.dialog(..., width="large")`
  (available since Streamlit 1.37; this repo pins `streamlit` unconditionally
  in `pyproject.toml`, and the installed version is 1.58 — no upgrade needed).
- Categories are derived mechanically from `A <= k` cuts (§6.2's
  `reaction_category`), not a hand-picked nuclide grouping — category *k*
  exactly equals "reactions newly unlocked when `amax` goes from `k-1` to `k`."
  This keeps categories and `amax` filtering perfectly consistent by
  construction.

---

## 0. Target behaviour, end to end

1. Only `small`, `small_parthenope`, `large` are named networks. `medium.txt`
   and `deuterium.txt` are deleted; their old behaviour is reproduced via
   `network="large", amax=8` and `network="large", amax=2` respectively.
2. `amax` (`A <= amax` post-filter) works for *any* network, not just `large`.
3. Rate tables move from `rates/nuclear/tables/<name>.txt` to
   `rates/nuclear/tables/<name>/<name>.txt` (the PRIMAT default), with room
   for sibling files in the same folder (e.g.
   `rates/nuclear/tables/n_p__d_g/n_p__d_g_parthenope3.0.txt`) when a reaction
   has more than one candidate table.
4. GUI sidebar: weak rates + plasma physics + nuclear QED corrections all move
   into one **"Physics"** group. "Nuclear reactions" keeps `network`, `amax`,
   and — below them — two buttons: **"Import custom network"** and **"Create
   custom network"**.
5. **"Create custom network"** opens a large `st.dialog` popup:
   - A network-title text input at the top.
   - "Select Network to modify" dropdown (any named network, *and* any
     custom network already imported this session) + its own `amax` setting.
   - The reaction list of the selected (network, amax) pair, banded into
     foldable, non-empty-only mass-number categories. Each category has
     "select all" / "deselect all" buttons and one row per reaction: a
     toggle, the visually rendered equation, a dropdown of available rate
     tables for that reaction (only shown when >1 exists), and an "Add new
     rate table" button.
   - Bottom: "Add new rate" (validated against duplicates and against the
     active `amax`), "Save custom network" (download), and a very large
     "Apply and run BBN" button that closes the dialog and runs the solve.
6. **"Import custom network"** uploads a previously saved zip and restores
   network/amax/per-reaction state exactly, including re-deriving `amax` from
   the heaviest nuclide present (no separate metadata needed) and showing
   `customnetworkname (N reactions)` in the network dropdown.
7. Exported zips always contain the **reinterpolated** (on-grid) version of
   every user-supplied table, with the user's original header lines preserved
   and a bookkeeping `#` line appended.

---

## 1. `network_data.py` / `config.py` / `cli.py`: drop `medium`/`deuterium`, generalise `amax`

This section is carried over verbatim in intent from `CUSTOM.md` §1; repeated
here so this document is self-contained.

### 1.1 Delete network files
- Remove `rates/nuclear/networks/medium.txt` and `rates/nuclear/networks/deuterium.txt`.
- `small` has no on-disk file (`ORDER_SMALL` is hardcoded in `network_data.py`)
  — nothing to delete there. Keep `large.txt` and `small_parthenope.txt`.

### 1.2 `network_data.py`
- `_REACTIONS_MEDIUM` (`network_data.py:294-300`, currently
  `load_reaction_names(_network_dir_from_cwd(), "medium")`): change to read
  `"large"` instead and rename to `_REACTIONS_LARGE` throughout the file
  (it only feeds `ORDER_LT` construction and the dynamic `<rxn>_frwrd`
  extractor-method registration loop at the bottom of the file — grep both
  usages before renaming).
- `load_reaction_names`'s `network=None` fallback (`network_data.py:242-291`):
  change default from `"medium"` to `"large"`.
- `_species_order` (`:1012-1017`): unaffected — `SPECIES_MD` is a Python list,
  not file-derived.
- **Generalise `amax`** in `load_network` (`:1198-1546`): functionally the
  per-reaction filter loop (`:1334-1360`) already applies regardless of
  `cfg.network` — only the comments claim "large only." Two real changes:
  1. **Restructure so amax filtering happens before the MT/LT era branch**,
     not inside the `parsed` loop. Currently `bare_names`/`selected` are
     computed, then the era branch (`era=="MT"`/`"LT"`) decides `selected`
     from `bare_names` *unfiltered* by amax, and only the later `parsed` loop
     (after era selection) drops amax-violating reactions. Bug: for
     `network="small"` (or any network) with `amax` set, the MT-era branch at
     `:1272-1275` intersects `ORDER_SMALL`/`ORDER_MT` against `allowed =
     set(bare_names)` computed *before* amax filtering, so an MT solve can
     still try to run a reaction that the LT era would have dropped for amax.
     Fix: compute the amax-filtered `bare_names`/`bare_to_file` once, right
     after they're built (`:1259-1269`) and before the `era == "MT"` branch
     (`:1270+`), then reuse that filtered set for both the MT intersection and
     the `parsed` loop (which then no longer needs its own per-name amax
     check, since its input is already filtered).
  2. Drop "only meaningful for network='large'" wording everywhere it appears
     (`network_data.py:1331-1333`, `config.py:187-192`).
  - Confirm by test (§3.2) that `network="small", amax=2` now correctly
    collapses both MT and LT to just `n_p__d_g` (i.e. behaves like the old
    `deuterium` network).
- Update docstring `Example`s referencing `medium.txt`/`network="medium"`
  (`:787`, `:1208`, `:1234-1236`) to use `large`/`amax`.
- Add the new `reaction_category` / `group_reactions_by_category` helpers and
  the `AMAX_LARGE` constant here — see §6.2 (they're needed by both the
  amax-vs-medium validation in §3.1 and the GUI popup in §6/§7).

### 1.3 `config.py`
- `DEFAULT_PARAMS["network"]` stays `"small"`.
- `amax` comment block (`:187-193`): rewrite to state it applies to *any*
  network, defaulting to `None` (no filter); document the migration mapping
  explicitly:
  - old `network="medium"` → `network="large", amax=8`
  - old `network="deuterium"` → `network="large", amax=2`
  (Verify the exact reaction *count*/*set* match in §3.1 before asserting
  equivalence in prose — if it doesn't match exactly, say so.)
- `amax` validation (`:491-495`) unaffected (already "any positive integer >= 1").
- Update the `network="medium"` example in the class docstring (`:251`) to
  `network="large"`.

### 1.4 `cli.py`
- Update `--network`/`--amax` help text and docstring examples
  (`cli.py:13`, `:59-69`, `:110`) to drop `medium` and the "large only"
  framing for `amax`.

---

## 2. Rate-table storage reorganisation: one folder per reaction

### 2.1 New layout
- `rates/nuclear/tables/<name>/<name>.txt` is the PRIMAT-default table for
  reaction `<name>` (e.g. `rates/nuclear/tables/n_p__d_g/n_p__d_g.txt`).
- Additional candidate tables for the same reaction live alongside it in the
  same folder, e.g. `rates/nuclear/tables/n_p__d_g/n_p__d_g_parthenope3.0.txt`
  for the reactions `small_parthenope.txt` currently points at via the
  `"name, filename.txt"` syntax (`load_reaction_names`/`bare_to_file` parsing,
  `network_data.py:1259-1269`) — these files keep their existing basenames,
  they just move one level deeper.
- `decays.txt` (the single multi-row file backing every `Bm`/`Bp` decay
  reaction, `_load_decay_table`, `:1157-1195`) is **not** reorganised — it
  stays a single flat file directly under `rates/nuclear/tables/`, since it is
  not a per-reaction rate table (one row per decay, T9-independent). Leave
  `_load_decay_table`'s `os.path.join(tables_dir, "decays.txt")` untouched.

### 2.2 Migration script (one-off, run once on this branch)
Write a small one-off script (e.g. `generate_rates/migrate_tables_to_folders.py`,
delete after running, or keep as a documented one-shot tool — your call, but
note it in the script's docstring as "already applied, kept for reference")
that, for every `rates/nuclear/tables/<name>.txt` (excluding `decays.txt`):
```python
os.makedirs(f"rates/nuclear/tables/{name}", exist_ok=True)
shutil.move(f"rates/nuclear/tables/{name}.txt", f"rates/nuclear/tables/{name}/{name}.txt")
```
Use `git mv` (or `git add` after `shutil.move`) so the move is tracked as a
rename in the commit, not a delete+add — this matters for `git log --follow`
on individual rate-table histories. Run it, verify `git status` shows ~403
renames, then delete the now-empty flat files (the move already removes them).

### 2.3 `generate_rates/convert_ac2024_rates.py`
- `write_reaction_file` (`:230-258`): change
  `path = os.path.join(outdir, block["name"] + suffix + ".txt")`
  to create the per-reaction folder first:
  ```python
  reaction_dir = os.path.join(outdir, block["name"])
  os.makedirs(reaction_dir, exist_ok=True)
  path = os.path.join(reaction_dir, block["name"] + suffix + ".txt")
  ```
  `suffix` (used for alternate-source variants today, if any — check call
  sites) keeps producing sibling files inside the same per-reaction folder,
  which is exactly the new multi-table-per-reaction mechanism.
- `decays.txt`'s writer (`write_decay_file`, find by grep) is untouched —
  still written directly to `outdir` (= `rates/nuclear/tables/`), not into a
  per-reaction folder.
- Update the module's own docstring/header comment (`:46-47`, "Output:
  `pyprimat/rates/nuclear/tables/<reactants>TO<products>.txt`") to describe
  the new `…/tables/<name>/<name>.txt` layout.
- Re-run the script (or at minimum a dry-run / `--help`) to confirm it still
  produces output matching the migrated tree from §2.2 byte-for-byte (modulo
  path).

### 2.4 `network_data.py` loader changes
- `_reaction_catalog` (returns `tables_dir, ...`): `tables_dir` still points at
  `rates/nuclear/tables/` — unchanged. What changes is every *consumer* that
  builds a path from `tables_dir` + a bare/custom filename:
  - `load_network`'s table-loading branch (`:1465-1475`):
    ```python
    table_path = os.path.join(tables_dir, filename)
    ```
    becomes
    ```python
    table_path = os.path.join(tables_dir, name, filename)
    ```
    (`name` is the bare reaction name driving the folder; `filename` is
    whatever `bare_to_file` resolved — default `name + ".txt"`, or an
    explicit `"name, alt_filename.txt"` entry from the network file). Since
    `filename` for the default case is already exactly `name + ".txt"`, the
    folder name and the file's own basename coincide for the common case,
    matching §2.1.
  - `_read_reaction_source(table_path)` (reads the header `ref=` line): no
    change needed, it just receives the new deeper path.
  - Any other `os.path.join(tables_dir, ...)` call building a per-reaction
    rate-table path (grep `tables_dir` across `network_data.py` and
    `pyprimat/gui/`) needs the same `name`-subfolder insertion. In particular
    `pyprimat/gui/panels.py`'s `_render_reaction_downloads` reads
    `run.nucl.files` (already-resolved absolute paths from `NetworkDefinition`,
    built inside `load_network`) — those already carry the new folder-aware
    path once `load_network` is fixed, so no separate change needed there.
- Add a small helper, since the GUI popup (§6) needs to enumerate every
  candidate table for a reaction:
  ```python
  def available_rate_tables(name: str, cfg) -> list[str]:
      """Basenames of every rate-table file inside tables/<name>/, sorted with
      the PRIMAT default (<name>.txt) first."""
  ```
  in `network_data.py`, backed by `os.listdir(os.path.join(tables_dir, name))`
  filtered to `*.txt`, with `f"{name}.txt"` sorted to the front if present.
  Returns `[]` for a reaction with no folder yet (a brand-new GUI-added
  reaction that only has a user-uploaded table, never written to disk).

### 2.5 Tests
- `tests/` likely has direct path assertions against the flat layout — grep
  `rates/nuclear/tables/` and `tables_dir` across `tests/*.py` and fix every
  hit to expect the per-reaction folder.
- Add a test that `available_rate_tables("n_p__d_g", cfg)` returns at least
  `["n_p__d_g.txt"]`, and that `small_parthenope`'s reactions resolve to their
  `*_parthenope3.0.txt` sibling correctly post-migration.

---

## 3. Validate numerically, then update docs

### 3.1 Numeric validation (do this *before* touching docs/tests downstream)
```python
from pyprimat import PyPR
ref8 = PyPR(params={"network": "large", "amax": 8}).PyPRresults()   # vs old "medium"
ref2 = PyPR(params={"network": "large", "amax": 2}).PyPRresults()   # vs old "deuterium"
```
- Record the exact reaction count/set for `amax=8` (expected ≈67; if it
  differs from old `medium.txt`'s 67, say so explicitly rather than asserting
  exact equivalence) — compare via `set(name for name, *_ in
  net.describe_reactions())`, not just final-abundance closeness.
- Compute the true `AMAX_LARGE` = max nuclide mass number actually referenced
  by any `large.txt` reaction (not just `nuclides.csv`'s catalog) — used by
  §6.2/§7.1's "no amax filter" detection heuristic. Likely ~23 (Na23) — do not
  assume the user's illustrative "20" without checking.
- Re-run `runfiles/PyPRIMAT_reference_run.py` (after §3.2's edits) to
  regenerate authoritative reference numbers for CLAUDE.md (§3.4).

### 3.2 Runfiles
- `runfiles/PyPRIMAT_run.py:45`: `"network": 'medium'` → `{"network": "large", "amax": 8}`.
  Update the `network in ("medium", "large")` print-table condition (`:73`) to
  match whatever §3.1 determines (e.g. always print the per-nuclide table now
  that `medium`/`small` are no longer the only "short" cases).
- `runfiles/PyPRIMAT_compare.py`: replace the bare `networks = ["small",
  "small_parthenope", "medium", "large"]` list (`:45`) with label/params
  tuples:
  ```python
  networks = [
      ("small", {}),
      ("small_parthenope", {}),
      ("large_amax8", {"network": "large", "amax": 8}),
      ("large", {}),
  ]
  ```
  and adjust the run loop (`:48-56`) and results-printing loop (`:72-73`) to
  use the label for display/indexing and `{"network": "large", **extra}` (or
  just `extra` when it already sets `network`) for params. Update the module
  docstring.
- `runfiles/PyPRIMAT_reference_run.py:105,112-116`: `run_network(network="medium")`
  → `run_network(network="large", amax=8)`; relabel the printed column.
- Notebooks `notebooks/AbundanceEvolution.ipynb`,
  `notebooks/AnimatedAbundances.ipynb`: grep for `"medium"`/`"deuterium"`,
  replace with `"large"` (+`amax` where the intent was "a smaller/faster
  large-like run"), re-run affected cells, confirm `tests/test_notebooks.py`
  still passes.

### 3.3 GUI Physics panel (`pyprimat/gui/params_form.py`)
- Merge `"Plasma physics"` (`QED_corrections`) and `"Weak rates"`
  (`incomplete_decoupling`, `radiative_corrections`, `finite_mass_corrections`,
  `thermal_corrections`, `spectral_distortions`, `analytic_distortions`,
  `delta_xi_nu`, `y_SZ`) into one new group `"Physics"` in `_FORM_METADATA`.
- Move `nuclear_qed_corrections` (currently `"Nuclear reactions"`, `:93-98`)
  into `"Physics"` too.
- `GROUP_ORDER` (`:156`): `["Cosmology", "Nuclear reactions", "Physics"]`.
- Within `"Physics"`, preserve the insertion order needed for `_CONDITIONAL`
  (`spectral_distortions` before `analytic_distortions`/`delta_xi_nu`/`y_SZ`).
  Add lightweight subheadings (`st.markdown("**Weak rates**")` /
  `**Plasma physics**` / `**Nuclear QED**`) before each cluster via a small
  `_SUBHEADING` map checked in the `render_sidebar_form` loop (`:607-635`),
  printing the heading text whenever it changes from the previous key
  rendered, so the merged group doesn't read as an undifferentiated wall of
  toggles.
- `_EXPANDED_GROUPS` (`:157`): leave `"Physics"` collapsed by default (only
  `"Cosmology"` and `"Nuclear reactions"` stay auto-expanded).
- `_CONDITIONAL["amax"] = ("network", "large")` (`:178`): **remove** —
  `amax` is now offered for every network (§1.2). Replace the special case at
  `:615-628` with an unconditional render of the "Limit max mass number"
  checkbox + number_input, regardless of `network`'s value. Drop "(large
  only)" from `_FORM_METADATA["amax"]`'s label/help (`:87-92`).
- Update `_FORM_METADATA["network"]`'s help text (`:80-86`) to drop the
  "'medium' (62 reactions)" clause and mention `small_parthenope`/`large`±amax
  instead.

### 3.4 README.md / CLAUDE.md
- `README.md`: every `medium`/`deuterium` reference (`:16,95,112,171,259-260,
  279,296,302`) — replace CLI examples, the `{small,medium,large}` table row
  (→ `{small,small_parthenope,large}`), the networks/ directory listing, and
  the per-network reaction-count table, using §3.1's measured numbers.
- `CLAUDE.md` (this is itself a project-instructions file, edit carefully,
  preserve tone/structure):
  - "Key configuration flags" table: `network` row → `"small"` / `"small_
    parthenope"` / `"large"` (custom files still work for any other name);
    `amax` row drops the large-only framing.
  - "The three networks share..." paragraph: reframe around the two-named-
    network-plus-amax model.
  - "Validation before committing" + "Per-nuclide final abundances" tables:
    re-measure via the re-run reference script (§3.1/§3.2) rather than
    hand-editing; rename the "Medium network" section/column to "large
    network, amax=8" with freshly measured numbers (do not assume the old
    numbers carry over unless §3.1 confirms the reaction sets are identical).
  - "`large`-only exception (H3/Li7/Be7)" paragraph: check it doesn't
    implicitly assume `medium` exists as a comparison baseline.
  - Add a short paragraph documenting the new `rates/nuclear/tables/<name>/`
    folder layout (§2) under "Adding a new reaction" (step 1 currently says
    "Add the rate table `rates/nuclear/tables/<name>.txt`" — update to the
    folder form) and a short note on the popup-based custom-network workflow
    (§5–§8) replacing the old inline "Customise Reactions" checkbox panel.

### 3.5 Tests (non-GUI)
Carry over `CUSTOM.md` §6.1–§6.2's full list (re-stated here for
completeness — implement all of these):
- New `tests/test_network_data.py` (or extend `test_network_builder.py`):
  unit tests for `reaction_category()`/`group_reactions_by_category()` (§6.2):
  known reactions → known category (`n_p__d_g`→2, `d_d__t_p`→3, `t_p__a_g`→4),
  and every `large.txt` reaction lands in some category without raising.
- `tests/test_regression.py`: add `network="small", amax=2"` case (should
  collapse to the weak rate + `n_p__d_g`, confirming the §1.2 MT-branch fix).
- `tests/test_deuterium_network.py`: repurpose onto
  `{"network": "large", "amax": 2}` (keep the physical scenario, drop the
  network name).
- `tests/test_large_network.py` (`:62-136`): every `network: "medium"` →
  `{"network": "large", "amax": 8}`; re-verify the "agrees with medium to
  ≲1e-3" tolerances still hold.
- `tests/test_config.py` (`:85-99`): `_REACTIONS_MEDIUM`→`_REACTIONS_LARGE`
  import rename; the "67 medium reactions" assertion becomes whatever §3.1
  measured for `large+amax=8`, derived by calling `load_network` (not by
  duplicating the amax filter logic inline).
- `tests/test_network_builder.py` (`:99-100,154,169`): re-parametrize
  `"medium"` fixtures onto `("large", {"amax": 8})`.
- `tests/test_mc.py` (`:280-281`), `tests/test_regression.py` (`:94,131-148`):
  same substitution.
- `tests/test_gui.py` (`:111,115-118`): `_available_networks()` must now
  return exactly `["large", "small", "small_parthenope"]` (sorted); replace
  the `deuterium (1)` label test with a `_network_label("large")` check whose
  count is read dynamically, not hardcoded.
- `tests/test_network_generation.py` (`:20`): reword the "62 medium-network
  reactions" docstring mention.
- `tests/test_notebooks.py`: update docstring wording ("small/medium/large
  solves" → "small/large(±amax)/large solves").
- `tests/test_cli.py` (`:60`): update comment wording.
- Final repo-wide sanity grep at the end of implementation:
  `grep -rn "medium\|deuterium" pyprimat/ runfiles/ tests/ *.md notebooks/`
  to catch anything these targeted greps missed.

---

## 4. `reaction_category` / `group_reactions_by_category` / `AMAX_LARGE`

Add to `pyprimat/network_data.py` (pure, stateless, reusable by both tests and
the GUI — no Streamlit imports here):

```python
def reaction_category(name: str) -> int:
    """Heaviest nuclide's mass number A (=N+Z) among name's reactants/products.

    Drives the GUI popup's mass-number-banded category view (CUSTOMPOPUP.md
    §5/§6): category k contains exactly the reactions that a filter of
    amax=k keeps but amax=k-1 would drop, so categories and the amax filter
    stay consistent by construction. Photon ("g") and lepton ("Bm"/"Bp")
    tokens are excluded from the max (they don't carry a mass number).

    Uses reaction_stoichiometry(name), so it works for shipped catalog
    reactions, network-file "TO"-derived reactions, and brand-new
    GUI-added reactions alike (anything reaction_stoichiometry can parse).
    """
    react, prod = reaction_stoichiometry(name)
    _, _, _, nuc_NZ, _, _ = _reaction_catalog(_default_data_dir())
    nuclides = set(react) | set(prod)
    return max(sum(nuc_NZ[s]) for s in nuclides if s in nuc_NZ)


def group_reactions_by_category(names) -> dict[int, list[str]]:
    """{category_A: [bare_name, ...]}, sorted by category; names keep input order."""
    groups: dict[int, list[str]] = {}
    for name in names:
        groups.setdefault(reaction_category(name), []).append(name)
    return dict(sorted(groups.items()))


# True maximum nuclide mass number reachable in the large network's catalog
# (measured in §3.1 — replace 23 below with whatever was actually measured).
AMAX_LARGE = 23
```

Add both functions and `AMAX_LARGE` to `network_data.py`'s `__all__`.

---

## 5. GUI: "Nuclear reactions" panel — two buttons replace the inline checkbox

### 5.1 Remove the old inline workflow
- Delete `_render_custom_reactions` and `_render_add_reaction`
  (`params_form.py:272-573`) wholesale — the checkbox-per-reaction list, the
  `customise_reactions` checkbox, and their associated `st.session_state`
  bookkeeping (`keep_*`, `upload_*`, `_customise_replaced`, `_customise_added`,
  `_customise_upload_version`, `_customise_network`, `_customise_was_on`,
  `_customise_imported_id`, `custom_network_dict`) are all superseded by the
  popup's own state (§6).
- In `render_sidebar_form`'s `"Nuclear reactions"` group block (`:634-635`),
  replace the `_render_custom_reactions(params)` call with:
  ```python
  if group == "Nuclear reactions":
      _render_custom_network_buttons(params)
  ```

### 5.2 `_render_custom_network_buttons(params)` (new function)
Placed directly below the existing `network`/`amax` widgets, inside the same
`"Nuclear reactions"` expander:
```python
cols = st.columns(2)
if cols[0].button("Import custom network", use_container_width=True):
    st.session_state["_show_import_dialog"] = True
if cols[1].button("Create custom network", use_container_width=True):
    st.session_state["_show_custom_dialog"] = True

if st.session_state.get("_show_import_dialog"):
    _import_dialog()          # st.dialog, see §6.1
if st.session_state.get("_show_custom_dialog"):
    _custom_network_dialog(params)   # st.dialog, see §6.2-§7
```
(Streamlit dialogs are functions decorated `@st.dialog(...)`; calling the
decorated function opens it. Guard with a session-state flag so the dialog
doesn't reopen on every rerun once dismissed — `st.dialog`-decorated functions
naturally close when they return/finish or the user dismisses them, but the
"set a flag, call the function" pattern is the standard idiom for triggering
one from a button click since the button's own rerun happens before the
dialog can render.)

### 5.3 If a custom network is currently active (post-import or post-"Apply and run")
- Show its synthetic name in the `network` selectbox (§7.2) and keep the
  underlying `params["network"]`/`params["amax"]`/`params["custom_network"]`
  wired exactly as today's `custom_network` JSON mechanism already supports
  (`UpdateNuclearRates.__init__`, unchanged — see §8.4).
- Clicking "Create custom network" again while a custom network is active must
  offer that custom network as a selectable base in the popup's "Select
  Network to modify" dropdown (§6.2's `_dialog_network_options()` includes it
  — see §7.4).

---

## 6. The "Create custom network" popup

### 6.1 Session-state model for the dialog
All state lives under `st.session_state["_dialog_*"]` keys so it persists
across the dialog's internal reruns (Streamlit reruns the whole script on every
widget interaction, dialog included) without colliding with the main sidebar's
own keys:

- `_dialog_title`: str, the network's title (text input at the top).
- `_dialog_base_network`: str, the network selected in "Select Network to
  modify" (one of `_available_networks()` plus any custom network names from
  `_known_custom_networks()`, §7.4).
- `_dialog_amax`: int | None, the dialog's own `amax` (independent of the
  sidebar's `amax`, since "Select Network to modify" + amax together define
  *only the starting point* for customisation — the resulting custom network
  has no `amax` of its own once built, only an explicit reaction list).
- `_dialog_keep[name] -> bool`: per-reaction toggle state, keyed by bare name.
- `_dialog_table_choice[name] -> str`: chosen rate-table basename for `name`
  (from `available_rate_tables`, §2.4), defaulting to the PRIMAT default.
- `_dialog_uploaded_tables[name] -> {basename: raw_text}`: any "Add new rate
  table" uploads this session, merged into the dropdown options for `name`.
- `_dialog_added[name] -> raw_text`: brand-new reactions added via "Add new
  rate" at the bottom, exactly mirroring today's `_customise_added` semantics.
- `_dialog_category_expanded[cat] -> bool`: per-category fold state (optional
  polish; Streamlit's `st.expander(expanded=...)` default is enough if you'd
  rather not persist this).

Initialise all of the above (if absent) from `_dialog_base_network`'s
`base_selection` (§6.2) the first time the dialog opens, or from an imported
zip's restored state (§7) when entering via "Import custom network" → "Create
custom network" follow-up edit.

### 6.2 Dialog skeleton
```python
@st.dialog("Create custom network", width="large")
def _custom_network_dialog(params):
    st.session_state.setdefault("_dialog_title", "custom")
    title = st.text_input("Network title", key="_dialog_title")

    st.markdown("**Select Network to modify**")
    col1, col2 = st.columns([2, 1])
    options = _dialog_network_options()          # §7.4: named + custom networks
    base_network = col1.selectbox(
        "Network", options, key="_dialog_base_network",
        format_func=_network_label,
    )
    amax_enabled = col2.checkbox("Limit A", key="_dialog_amax_enabled")
    dialog_amax = None
    if amax_enabled:
        dialog_amax = col2.number_input(
            "amax", min_value=2, value=st.session_state.get("_dialog_amax_value", 20),
            key="_dialog_amax_value",
        )

    # Reset per-reaction state if the (base_network, amax) pair changed since
    # the dialog last computed it -- mirrors today's _customise_network guard.
    sig = (base_network, dialog_amax)
    if st.session_state.get("_dialog_signature") != sig:
        _reset_dialog_reaction_state(base_network, dialog_amax)
        st.session_state["_dialog_signature"] = sig

    base_selection = _dialog_base_selection(base_network, dialog_amax)  # §6.3
    all_entries = _dialog_superset_entries(dialog_amax)                  # §6.3
    groups = group_reactions_by_category(
        [bare(e) for e in all_entries] + list(st.session_state["_dialog_added"])
    )

    for cat in sorted(groups):
        _render_category(cat, groups[cat], base_selection)               # §6.4

    _render_add_rate_section(dialog_amax, all_entries)                    # §6.5
    _render_dialog_footer(params, title, base_network, dialog_amax)       # §6.6
```

### 6.3 `base_selection` vs. the full superset (mirrors `CUSTOM.md` §4.1)
- `base_selection` = `load_reaction_names(cfg, base_network)` filtered by
  `dialog_amax` (the reactions actually active for that named network at that
  `amax` — what starts **checked**).
- The dialog must let the user toggle reactions *outside* `base_network`'s own
  list too (e.g. starting from `small` but adding a `large`-only reaction), so
  the full set of rows shown is always the **large network's reaction list,
  filtered by `dialog_amax`** (`_dialog_superset_entries`), regardless of
  which named network is the base — reactions in `base_selection` start
  checked, every other entry within the `amax` band starts unchecked. This
  exactly matches `CUSTOM.md §4.1`'s resolved behaviour; carry it over.
- `small_parthenope` special case: its entries use the
  `"name, name_parthenope3.0.txt"` syntax from `load_reaction_names`; when it
  is the base network, pre-select that table in `_dialog_table_choice[name]`
  rather than the PRIMAT default, by reading the filename half of the entry.

### 6.4 `_render_category(cat, names, base_selection)`
```python
def _render_category(cat, names, base_selection):
    nuclide_hint = _category_nuclide_hint(cat)   # e.g. "He3, t" for cat 3
    with st.expander(f"Category {cat} (A <= {cat}{nuclide_hint})", expanded=False):
        c1, c2 = st.columns([1, 1])
        if c1.button("Select all", key=f"_dialog_selall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = True
            st.rerun()
        if c2.button("Deselect all", key=f"_dialog_deselall_{cat}"):
            for n in names:
                st.session_state["_dialog_keep"][n] = False
            st.rerun()
        for name in names:
            _render_reaction_row(name, base_selection)
```
(The user's spec calls these "toggle all"/"untangle all"; implement as
"Select all"/"Deselect all" — same intent, clearer English. If literal label
text matters, swap the strings only.)

`_render_reaction_row(name, base_selection)`:
```python
def _render_reaction_row(name, base_selection):
    keep_map = st.session_state["_dialog_keep"]
    default = keep_map.get(name, name in base_selection)
    equation = _equation_for(name)   # _equation_unicode(reaction_equation), reuse panels.py helper
    tables = available_rate_tables(name, cfg) + list(
        st.session_state["_dialog_uploaded_tables"].get(name, {}))
    cols = st.columns([1, 4, 3, 2])
    keep_map[name] = cols[0].toggle("", value=default, key=f"_dialog_keep_{name}")
    cols[1].markdown(equation)
    if len(tables) > 1:
        choice = cols[2].selectbox(
            "table", tables, key=f"_dialog_table_{name}",
            index=tables.index(st.session_state["_dialog_table_choice"].get(name, tables[0])),
            label_visibility="collapsed",
        )
        st.session_state["_dialog_table_choice"][name] = choice
    else:
        cols[2].caption(tables[0] if tables else "(no table)")
    if cols[3].button("Add new rate table", key=f"_dialog_addtable_{name}"):
        st.session_state[f"_dialog_show_uploader_{name}"] = True
    if st.session_state.get(f"_dialog_show_uploader_{name}"):
        up = st.file_uploader("New table", key=f"_dialog_upload_{name}", label_visibility="collapsed")
        if up is not None:
            raw = up.getvalue().decode()
            custom_rates.parse_rate_upload(io.StringIO(raw))  # validates; raises -> st.error
            basename = up.name
            st.session_state["_dialog_uploaded_tables"].setdefault(name, {})[basename] = raw
            st.session_state["_dialog_table_choice"][name] = basename
            st.session_state[f"_dialog_show_uploader_{name}"] = False
            st.rerun()
```
- `cols[0].toggle` is Streamlit's pill-style toggle (matches the user's "toggle
  button" wording more literally than a checkbox).
- A reaction with zero tables on disk and no upload yet (only possible for a
  brand-new "Add new rate" addition, §6.5) shows "(no table)" and cannot be
  checked until a table exists — enforce this in `_render_dialog_footer`'s
  validation (§6.6), not by disabling the toggle (so the user can still see
  it pending in its category).

### 6.5 `_render_add_rate_section` — "Add new rate" button at the bottom
Mirrors today's `_render_add_reaction` (`params_form.py:482-573`) almost
exactly, with two new checks (carried over from `CUSTOM.md` §4.4):
```python
def _render_add_rate_section(dialog_amax, all_entries):
    st.divider()
    with st.popover("Add new rate", use_container_width=True):
        name = st.text_input("Reaction name", key="_dialog_add_name",
                              placeholder="He3_d__He4_p")
        ... # live-parse feedback exactly as today
        if st.button("Add reaction", key="_dialog_add_submit"):
            bare = name.strip()
            existing = {bare_of(e) for e in all_entries} | set(st.session_state["_dialog_added"])
            if bare in existing:
                st.error(f"'{bare}' already exists in the current selection.")
                return
            try:
                cat = reaction_category(bare)
            except (ValueError, KeyError) as exc:
                st.error(str(exc)); return
            if dialog_amax is not None and cat > dialog_amax:
                st.error(
                    f"reaction {bare!r} involves a nuclide with A={cat}, which "
                    f"exceeds the current amax={dialog_amax}.")
                return
            upload = st.file_uploader("Rate table", key="_dialog_add_table")
            if upload is None:
                st.error("Upload a rate table for the new reaction first.")
                return
            raw = upload.getvalue().decode()
            custom_rates.parse_rate_upload(io.StringIO(raw))
            st.session_state["_dialog_added"][bare] = raw
            st.session_state["_dialog_keep"][bare] = True
            st.session_state["_dialog_table_choice"][bare] = upload.name
            st.rerun()
```
Validation order matters: existing-name check first (cheap, no upload needed
to fail fast), then amax, then require the table upload — matches the user's
two explicit checks ("not a reaction which exists already" and "does not
involve nuclides beyond the currently selected amax").

### 6.6 `_render_dialog_footer(params, title, base_network, dialog_amax)`
```python
def _render_dialog_footer(params, title, base_network, dialog_amax):
    st.divider()
    kept_names = _dialog_kept_names()     # §6.7: every name with keep_map[name]==True and a resolved table
    if st.button("Save custom network", use_container_width=True):
        zip_bytes = _build_export_zip(title, kept_names)   # §8.1
        st.download_button(
            f"Download {title}.zip", data=zip_bytes,
            file_name=f"{_sanitize_filename(title)}.zip", mime="application/zip",
            key="_dialog_download",
        )
    if st.button("Apply and run BBN", type="primary", use_container_width=True):
        custom_network = _dialog_to_custom_network(kept_names)   # §8.3
        params["network"] = "large"
        params.pop("amax", None)
        params["custom_network"] = json.dumps(custom_network, sort_keys=True)
        st.session_state["_active_custom_network"] = {
            "title": title, "kept": kept_names, "custom_network": custom_network,
        }
        st.session_state["_show_custom_dialog"] = False
        st.session_state["_trigger_run"] = True   # read by app.py's main loop, see §8.5
        st.rerun()
```
- "Save custom network" needs a two-step button→download_button because
  Streamlit can't trigger a browser download directly from a single button
  click without a render in between; this mirrors the existing
  pattern already used for `st.download_button` elsewhere in the codebase
  (`panels.py:295-303`).
- "Apply and run BBN" closes the dialog (clearing the show-flag) and sets a
  `_trigger_run` flag that `app.py`'s main render loop checks right after
  `render_sidebar_form()` returns, calling the same solve path as the regular
  "Run BBN" button (`app.py` — grep for the current run-button handling and
  reuse it; do not duplicate the solve invocation).

### 6.7 `_dialog_kept_names()`
```python
def _dialog_kept_names():
    keep_map = st.session_state["_dialog_keep"]
    return [n for n, kept in keep_map.items() if kept]
```
A reaction with `kept=True` but no resolved table (the "(no table)" case from
§6.4) should be filtered out here with an `st.warning` rather than silently
included — surface it instead of failing deep inside `load_network`.

---

## 7. Restoring a previously-customised network

### 7.1 Detecting `amax` purely from the reaction list
- After importing a zip (§8.2) or when re-entering the dialog with a custom
  network as base, compute
  `implied_amax = max(reaction_category(n) for n in kept_names)`.
- If `implied_amax == AMAX_LARGE` (§4), treat it as "no amax filter"
  (`amax=None`); otherwise set `_dialog_amax_value = implied_amax` and check
  `_dialog_amax_enabled`. This is an accepted, documented ambiguity (a real
  `amax=N` export looks identical to "happened to use every reaction up to N
  and none above") — already signed off in `CUSTOM.md` §5.1's reasoning,
  carried over here.

### 7.2 Network dropdown shows "customnetworkname (N reactions)"
- The sidebar's `network` selectbox (`_widget_for`, `:249-253`) and its
  `format_func=_network_label` only know about real `.txt` files. When a
  custom network is active (`st.session_state.get("_active_custom_network")`
  is set, from §6.6's "Apply and run" or §8.2's import), `_available_networks()`
  must append the custom network's title as a synthetic entry, and
  `_network_label` must special-case it: count from
  `len(active_custom_network["kept"])` directly rather than trying to
  `load_reaction_names` a nonexistent file.
- `cfg.network` itself must still resolve to `"large"` under the hood — the
  *reaction selection* always flows through `custom_network` JSON exactly as
  it does today (`UpdateNuclearRates.__init__`, unchanged). The synthetic name
  is a **display label only**; manually changing `network` or amax in the
  sidebar (outside the dialog) clears `_active_custom_network` and reverts the
  dropdown to a real network name. Document this in the help text near the
  selectbox.

### 7.3 Restoring per-category toggle state
- Once `kept_names` and the derived `amax` are set (§7.1), `_dialog_keep`
  population from `kept_names` plus `group_reactions_by_category`'s
  recomputation against the now-correct `amax` is sufficient — no separate
  "category state" needs restoring, it falls out of (a) correct `amax` and
  (b) every reloaded name's `keep` flag being set. (Carried over from
  `CUSTOM.md` §5.3 — explicitly noted so no one goes looking for a
  category-state field that doesn't need to exist.)

### 7.4 `_dialog_network_options()` includes imported custom networks
- Maintain `st.session_state["_known_custom_networks"]: dict[title, {"kept":
  [...], "tables": {...}}]`, appended to whenever a zip is imported (§8.2) or
  a custom network is saved/applied this session (§6.6). `_dialog_network_options()`
  returns `_available_networks() + list(_known_custom_networks)`.
- Selecting one of these in "Select Network to modify" sets `base_selection`
  to its stored `kept` list directly (skip `load_reaction_names`/`amax`
  filtering — it's already a concrete, fully-resolved list) and seeds
  `_dialog_table_choice`/`_dialog_uploaded_tables` from its stored per-reaction
  tables, so editing a previously-saved custom network as the new starting
  point works exactly like editing a named network.

---

## 8. Export/import zip format

### 8.1 `_build_export_zip(title, kept_names)` — extends `custom_rates.export_zip`
- Rename the internal network file from the hardcoded `networks/custom.txt`
  (`custom_rates.py:209`) to `networks/<sanitized_title>.txt`, threading a new
  `network_filename` parameter through `export_zip`. `import_zip` must stop
  hardcoding `"networks/custom.txt"` (`:245`) and instead read whichever
  single file exists under `networks/` in the zip (there is always exactly
  one), recovering `title` from its basename.
- For **every** kept reaction (not just replaced/added ones as today), write
  its **reinterpolated, on-grid** table into `tables/<name>/<name>_custom.txt`
  if the user picked a non-default table or uploaded a new one for it; for a
  reaction left at its shipped default, write nothing (it's resolved from the
  shipped `tables/<name>/<name>.txt` on reload — no change from current
  behaviour, just under the new per-reaction-folder path).
  - This directly answers the user's requirement: *"When downloading the
    currently created network the files should already be placed by their
    interpolated version... A custom network zip file contains essentially a
    network/networkname.txt and the tables/namereaction/file.txt which are
    not provided inside PRIMAT (hence the reinterpolation of the files
    provided by the user when modifying a reaction)."* — i.e. the zip's
    `tables/` mirrors the new on-disk folder-per-reaction layout (§2.1), one
    level per reaction, containing only the non-shipped (user-supplied)
    tables, each already resampled onto the master grid.
- **Header-line preservation**: `effective_table_text` (`custom_rates.py:131-
  165`) currently discards the uploaded file's own header lines and replaces
  them with a single hardcoded `# ref=custom upload (name)` line. Fix:
  1. `parse_rate_upload` (`:86-128`) must also return the leading `#`-prefixed
     lines it currently lets `np.loadtxt` silently skip — change its return
     to `(T9, rate, err, header_lines)` and update every call site
     (`_render_reaction_row`/`_render_add_rate_section` in §6, `export_zip`'s
     loop, `import_single`) to thread `header_lines` through.
  2. `effective_table_text(cfg, T9, rate, err, name="custom", source_header=())`
     gains a `source_header` parameter: preserve those lines verbatim, then
     *append* (not replace) a bookkeeping line:
     ```python
     lines = list(source_header)
     lines.append(f"# custom rate (reinterpolated): {name}")
     ```
- New `export_zip(cfg, custom_network, kept_names, network_filename="custom")`
  signature; update its docstring and every call site (§6.6, `panels.py`'s
  `_render_reaction_downloads` post-run export, which should also gain the
  same user-chosen filename — reuse the title stored in
  `st.session_state["_active_custom_network"]["title"]` if present, else keep
  defaulting to `"custom"`).

### 8.2 `_import_dialog()` — "Import custom network" button's popup
A second, much smaller `st.dialog`:
```python
@st.dialog("Import custom network")
def _import_dialog():
    fh = st.file_uploader("Custom network zip", type=["zip"])
    if fh is not None:
        result = custom_rates.import_zip(fh)   # {"kept", "replaced", "title"}
        title = result["title"]
        kept = result["kept"]
        st.session_state["_known_custom_networks"][title] = {
            "kept": kept, "tables": result["replaced"],
        }
        st.session_state["_active_custom_network"] = {
            "title": title, "kept": kept,
            "custom_network": _kept_to_custom_network(kept, result["replaced"]),
        }
        st.session_state["_show_import_dialog"] = False
        st.rerun()
```
`_kept_to_custom_network(kept, replaced)` builds the same `{"removed": [...],
"replaced": {...}, "added": {...}}` shape `UpdateNuclearRates` already expects
(§8.4): `removed` = `(large reactions filtered by the derived amax, §7.1) -
kept`; reactions in `kept` that are *not* part of that filtered large set are
`added` (carrying their table from `replaced`); everything else in `replaced`
is a true replacement.

### 8.3 `_dialog_to_custom_network(kept_names)`
Builds the same dict shape from the live dialog state instead of a reloaded
zip: `removed` = filtered-large-superset minus `kept_names`; `replaced` =
`{name: dialog's chosen table text for name}` for every kept name whose
`_dialog_table_choice` differs from the shipped default *or* who has an entry
in `_dialog_uploaded_tables`; `added` = `dict(st.session_state["_dialog_added"])`.
When a kept name's chosen table is a non-default file that exists on disk
(e.g. picking `n_p__d_g_parthenope3.0.txt` rather than uploading anything new),
read its raw text straight from disk so `replaced` always carries explicit
text — `load_network`'s `custom_tables` mechanism (`network_data.py:1198-
1546`) only resamples from raw arrays, it does not know about on-disk
alternate filenames by name alone in this code path, so resolve to text here
rather than threading filenames through `custom_network`.

### 8.4 Solve-time wiring — unchanged
`UpdateNuclearRates.__init__(cfg, custom_network=...)` (`network_data.py:1558-
1611`) already accepts exactly the `{"removed", "replaced", "added"}` shape
both §6.6 and §8.2/§8.3 produce — no changes needed to the solver-facing API,
only to how the GUI builds that dict.

### 8.5 `app.py` wiring for "Apply and run BBN"
- Find the existing "Run BBN" button's handler in `pyprimat/gui/app.py` (grep
  for the main button / `_solve` call). Factor the actual solve invocation
  into a small helper (e.g. `_run_bbn(params, quick_mc, mc_samples)`) if not
  already one, so both the regular button and `_trigger_run` (set by §6.6) can
  call it identically without duplicating the solve/cache/results-rendering
  logic.
- At the top of the main render function, right after
  `params, quick_mc, mc_samples = render_sidebar_form()`, check:
  ```python
  if st.session_state.pop("_trigger_run", False):
      _run_bbn(params, quick_mc, mc_samples)
  ```

---

## 9. Implementation order

1. §3.1 numeric validation (medium↔large+amax=8, deuterium↔large+amax=2,
   `AMAX_LARGE`) — everything else depends on these numbers.
2. §1 (`network_data.py`/`config.py`/`cli.py` core changes incl. the MT-branch
   amax-ordering fix) + its direct unit tests (§3.5 first half).
3. §2 (rate-table folder migration) — independent of §1, but do it early since
   §4/§6 (categories, table dropdown) depend on `available_rate_tables`.
4. §4 (`reaction_category`/`group_reactions_by_category`/`AMAX_LARGE`) + unit
   tests.
5. §3.2 (runfiles/notebooks) — quick once §1 lands; re-validates end to end.
6. §3.3 (Physics panel merge) — self-contained.
7. §3.4 (README/CLAUDE.md) — do before the GUI popup work so its help text can
   quote already-finalised wording.
8. §5–§8 (popup-based custom-network workflow) — the bulk of the new logic;
   build inside-out: dialog skeleton (§6.2) → category rendering (§6.4) → add-
   rate section (§6.5) → footer/export (§6.6/§8.1) → import dialog (§8.2) →
   reload restoration (§7) → `app.py` run-trigger wiring (§8.5).
9. §3.5 remaining tests + the final repo-wide `grep -rn "medium\|deuterium"`
   sweep.
10. Manual GUI smoke test (per CLAUDE.md's "test the golden path... in a
    browser" guidance for frontend changes): launch `pyprimat-gui`, exercise
    Import → edit in popup → Save → re-import round-trip, and Create → toggle
    a few categories → Apply and run BBN, end to end.
11. Run `graphify update .` (per CLAUDE.md) — this touches a large fraction of
    `pyprimat/gui/` and `pyprimat/network_data.py`.
12. Delete `CUSTOM.md` (superseded by this document) in the same branch.
