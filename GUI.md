# Plan: Streamlit GUI paralleling the PyPRIMAT CLI

## Context

PyPRIMAT is driven today by editing a hard-coded `params` dict in a `runfiles/`
script (or the `pyprimat` console script) and reading printed output. There is
no interactive way to change parameters, launch a run, and inspect results
visually. This plan adds a **Streamlit** app that mirrors a single CLI run
(`PyPRIMAT_run.py`): a grouped parameter form, a run button, and two result
panels — an interactive nuclide time-evolution plot with per-nuclide
checkboxes, and a final-abundances / standard-ratios panel. The intended
outcome is a nice-looking, browser-based front end that binds to the existing
stable contract: **params dict → `PyPR` → results**, with no changes to the
core physics code.

The plan document itself is delivered as `GUI.md` at the repo root (step 0),
so the steps live alongside the code they describe.

## Decisions (from user)

- **Framework:** Streamlit (pure-Python, polished default look, Plotly charts).
- **Scope:** grouped parameter form + time-evolution panel + final
  abundances/ratios panel. No network-comparison or Monte-Carlo panels for now
  (can be added later — leave seams).
- **Packaging:** shipped **inside the package** as `pyprimat/gui/`, installable
  via `pip install ".[gui]"`, launched with a `pyprimat-gui` console script —
  parallel to the existing `pyprimat` CLI entry point.

## API contract the GUI binds to (already exists — do not modify)

- `PyPR(params: dict, extra_rho=None)` — `pyprimat/main.py:76`. Eager, expensive
  setup in the constructor; network solve is deferred.
- `run.PyPRresults()` / `run.solve()` — returns the 9-key results dict
  (`main.py:751-761`): `Neff`, `Omeganurel`, `OneOverOmeganunr`, `YPCMB`,
  `YPBBN`, `DoH`, `He3oH`, `He3oHe4`, `Li7oH`.
- `run.abundance_names` (property, `main.py:939`) — nuclide list in vector order
  (8 small / 12 medium / ~59 large).
- `run[name](t)` (`__getitem__`, `main.py:913`) — callable `Y(t)` interpolator
  for a nuclide.
- `run.A` / `run.N` / `run.Z` — mass-number / neutron / charge maps (for
  plotting `A_i·Y_i`).
- `run.t_of_T` / `run.T_of_t` (`main.py:903-911`) — background interpolators for
  a temperature x-axis option.
- `run.get_quantity(name)` (`main.py:959`) — scalar by result-key or nuclide
  name (final `Y`).
- `pyprimat.config.DEFAULT_PARAMS` (`config.py:26`) — authoritative dict of every
  flag: name → default; inline `#` comments are the tooltip text.
- `pyprimat.nuclear.load_reaction_names(cfg, network)` — reaction names (only if
  we later expose `p_<rxn>` rate-variation controls; not in MVP).

## Files to create / modify

```
pyprimat/gui/
  __init__.py       # marks the subpackage (kept import-light; no streamlit at top level)
  app.py            # Streamlit script: layout, run orchestration, panels
  launcher.py       # main(): resolves app.py path, execs `streamlit run`; the console-script target
  params_form.py    # Builds the grouped parameter form from DEFAULT_PARAMS
  panels.py         # render_evolution_panel(...) and render_results_panel(...)
  .streamlit/
    config.toml     # theme (colors, font) for the "nice looking" requirement, shipped as package-data
GUI.md              # this plan, delivered at repo root (step 0)
pyproject.toml      # MODIFY: add pyprimat.gui to packages, [gui] extra, pyprimat-gui script
```

The app lives **inside** the `pyprimat` package so `pip install` ships it, but
`pyprimat/__init__.py` and the core modules must stay free of any streamlit
import — only `pyprimat/gui/*` may import streamlit/plotly.

### Packaging changes in `pyproject.toml`
- `[tool.setuptools] packages = ["pyprimat", "pyprimat.gui"]` (explicit list; the
  current config does not use `find_packages`, so the subpackage must be added).
- `[tool.setuptools.package-data]` — add `"pyprimat" = ["rates/**/*", "gui/.streamlit/*"]`
  so the theme file ships in the wheel.
- `[project.optional-dependencies]` — add `gui = ["streamlit", "plotly"]`.
- `[project.scripts]` — add `pyprimat-gui = "pyprimat.gui.launcher:main"` next to
  the existing `pyprimat = "pyprimat.cli:main"`.

### Launcher (`pyprimat/gui/launcher.py`)
Streamlit needs a script *path*, not a module. `main()`:
- resolves the installed app file: `app = importlib.resources.files("pyprimat.gui") / "app.py"`
- execs streamlit: `sys.exit(stcli.main(["run", str(app), *sys.argv[1:]]))`
  (or `subprocess.run(["streamlit", "run", str(app)])`). Works identically from a
  source checkout and from site-packages.

## Implementation steps

### 0. Deliver this plan as `GUI.md`
Write this document to `GUI.md` at the repo root.

### 1. Dependencies & packaging
Apply the `pyproject.toml` changes above (`pyprimat.gui` package, `gui` extra,
package-data, `pyprimat-gui` script). `streamlit`/`plotly` stay optional — the
core package must keep importing without them (`pyprimat/__init__.py` imports
nothing from `pyprimat.gui`).

### 2. Parameter form — `pyprimat/gui/params_form.py`
- Enumerate `DEFAULT_PARAMS.items()` to build the form programmatically (no
  hardcoding); use the inline comments from `config.py` as `help=` tooltips.
  Practically, read the source comments once and bundle a small
  `{name: (group, label, help)}` metadata table in `params_form.py` for the
  ~15–20 user-facing flags, since `DEFAULT_PARAMS` itself has no machine-readable
  grouping. Hide internal/legacy flags (`numba_installed`,
  the various `save_*`/`recompute_*` caches)
  behind an "Advanced" expander.
- Group into Streamlit `st.expander`/`st.tabs` sections:
  - **Cosmology:** `Omegabh2`, `DeltaNeff`, `munuOverTnu`.
  - **Network:** `network` (selectbox: `small`, `small_parthenope`, `medium`,
    `large` — discover from `pyprimat/rates/nuclear/networks/*.txt` + `"small"`);
    `amax` (number input, enabled only when `network=="large"`).
  - **Precision:** `numerical_precision`, `T_start_cosmo_MeV`,
    `n_temperature_table`, `sampling_nTOp`.
  - **Physics toggles:** `incomplete_decoupling`, `QED_corrections`,
    `nuclear_qed_corrections`, `spectral_distortions` (and, gated on it,
    `analytic_distortions`, `delta_xi_nu`, `y_SZ`), `nTOp_Born_approximation`.
  - **Output (optional):** `output_time_evolution`, `output_final_result`.
- Widget type derives from the default's Python type: `bool`→`st.toggle`,
  `int`→`st.number_input(step=1)`, `float`→`st.number_input` (scientific
  format), `str`/enum→`st.selectbox`.
- Return a `params` dict containing **only values the user changed** from the
  default (mirror the CLI's "forward only set flags" behavior, `cli.py:108-114`),
  so defaults stay authoritative in `PyPRConfig`.

### 3. Run orchestration — `pyprimat/gui/app.py`
- A "Run BBN" button (in the sidebar or a top bar) triggers the solve.
- Wrap construction+solve in a cached function keyed on the frozen params dict:
  `@st.cache_resource def run_pypr(params_items): return PyPR(dict(params_items)).…`
  — re-running with identical params returns instantly; changing any param
  invalidates. (Use `cache_resource`, not `cache_data`, since `PyPR` is an
  unpicklable live object.)
- Show a `st.spinner("Solving network…")` during the run (large network /
  high precision can take seconds-to-minutes; reference-precision is minutes).
  Surface `ValueError`s from `PyPRConfig`/constructor (bad `network`/`amax`,
  inconsistent distortion flags) via `st.error` instead of a stack trace.
- Persist the solved `PyPR` instance in `st.session_state` so the two panels and
  the checkbox interactions re-render without re-solving.

### 4. Time-evolution panel — `pyprimat/gui/panels.py::render_evolution_panel(run)`
- Build the nuclide list from `run.abundance_names`.
- Checkbox group for nuclide selection: render as a compact multiselect plus
  quick "presets" buttons (e.g. *Light* = n, p, H2, H3, He3, He4, Li7, Be7;
  *All*; *Clear*). For the large (~59) network a `st.multiselect` scrolls better
  than 59 raw checkboxes; default-tick the light set.
- X-axis toggle: cosmic time `t` [s] vs temperature `T₉`/`T` [MeV] (use
  `run.t_of_T` / `run.T_of_t`). Sample a log-spaced grid over the solved range.
- Y: plot `A_i · Y_i(t)` (mass-fraction weighted, matching
  `AbundanceEvolution.ipynb`) on a **log y-axis**, one Plotly trace per ticked
  nuclide, with a legend, hover readout, and a turbo/qualitative colormap.
- Use `st.plotly_chart(fig, use_container_width=True)` for interactive
  zoom/pan/hover. Ticking/unticking nuclides only redraws the figure (the solve
  is cached), so it feels instant.

### 5. Final-abundances + ratios panel — `pyprimat/gui/panels.py::render_results_panel(run)`
- Top row of `st.metric` cards for the standard observables from the results
  dict, formatted to the decimals mandated by `CLAUDE.md`:
  `YPBBN` (8 dp), `YPCMB` (8 dp), `DoH` (7 sig), `He3oHe4`, `He3oH`,
  `Li7oH` (6 sig), `Neff` (8 dp).
- A sortable `st.dataframe` of every tracked nuclide: columns `nuclide`, `A`,
  `Z`, final `Y` (from `run.get_quantity(name)` / `run.abundance_names`),
  formatted in scientific notation.
- "Download results" button: emit the same content the CLI file outputs produce
  (`output_final.dat`-style table and/or `--json` dict) via `st.download_button`,
  reusing the result dict — no need to enable the on-disk `output_*` flags.

### 6. Look & feel ("nice looking")
- `pyprimat/gui/.streamlit/config.toml` with a custom theme (primary color,
  base, font) so it reads as a polished scientific tool, not stock Streamlit.
  The launcher sets the cwd / `STREAMLIT_*` so this config is picked up.
- `st.set_page_config(page_title="PyPRIMAT", layout="wide", page_icon=…)`.
- Sidebar = parameter form + Run button; main area = two panels as `st.tabs`
  ("Abundance evolution" / "Final abundances") or stacked sections.
- A compact header with the project title and a one-line description; show the
  active `Omegabh2`/`network` and run time as a caption after solving.

### 7. Launch & docs
- Primary launch: `pyprimat-gui` (console script → `launcher.main`), runnable
  from anywhere after `pip install ".[gui]"`. `rates/` resolve via the package's
  own paths, not cwd, so it does not require running from the repo root.
- Source-checkout fallback: `streamlit run pyprimat/gui/app.py` or
  `python -m pyprimat.gui.launcher`.
- Document both in `GUI.md` and a short note in the project `README.md`.

## Verification

1. **Imports stay clean:** `python -c "import pyprimat"` works without
   streamlit/plotly installed (importing `pyprimat` must not pull in
   `pyprimat.gui`).
2. **Install & launch:** `pip install ".[gui]"`, then `pyprimat-gui` opens the
   app (and `streamlit run pyprimat/gui/app.py` works in a source checkout); the
   parameter form renders with all groups and tooltips, and the custom theme is
   applied.
3. **Reference run parity:** with defaults (`Omegabh2=0.022425`,
   `network="small"`), click Run and confirm the ratios panel matches the
   CLAUDE.md reference within tolerance: `YPBBN≈0.2469156`, `DoH≈2.43647e-5`.
   Switch to `network="medium"` and confirm `YPBBN≈0.2469190`,
   `DoH≈2.43718e-5`. This proves the GUI drives `PyPR` identically to the CLI.
4. **Evolution panel:** tick/untick nuclides and confirm traces add/remove
   without re-solving (instant); toggle the time/temperature x-axis; confirm
   `A_i·Y_i` curves match the shapes in `AbundanceEvolution.ipynb`.
5. **Error surfacing:** set `network="large"` with an invalid `amax` (or an
   inconsistent distortion combo) and confirm a clean `st.error`, not a crash.
6. **Download:** download the final-abundance table and diff it against a
   `output_final_result=True` CLI run for the same params.

## Out of scope (future seams)

- Network-comparison panel (parallels `PyPRIMAT_compare.py`).
- Monte-Carlo uncertainty bands via `mc_uncertainty(...)` (`main.py:1068`).
- Per-reaction `p_<rxn>` / `NP_delta_<rxn>` rate-variation controls
  (`load_reaction_names`). Keep the form code structured so these slot in as an
  extra expander/tab later.
