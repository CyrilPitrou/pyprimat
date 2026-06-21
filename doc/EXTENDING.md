# Extending PyPRIMAT with new physics

A short guide to the three supported extension points. For the full physics
formalism (rate evaluation, background equations, weak-rate corrections),
the primary reference is `doc/Pitrou_etal_PhysReptArxivVersion.pdf`
(Pitrou, Coc, Uzan, Vangioni, *Physics Reports* 04 (2018) 005); see also the
other PDFs in `doc/`. Internal planning docs (`FUTURE.md`, `CPLAN.md`) are
deliberately not cited here — they describe in-progress refactors, not
physics, and rot once the work lands.

## (a) Add a nuclear reaction

Full step-by-step instructions already live in `CLAUDE.md` under
"Adding a new reaction" — follow that section. Summary: drop a rate table
under `rates/nuclear/tables/<name>/<name>.txt`, add `<name>` to the relevant
network file under `rates/nuclear/networks/`, and let
`reaction_stoichiometry` (`pyprimat/network_data.py`) auto-derive the
stoichiometry from the reaction name's `TO`-separated tokens (falling back
to a manual `reactions_large.csv`/`detailed_balance.csv` row only if the
name can't be tokenised). `load_network` validates A/Z conservation and
rejects duplicate entries, so a malformed addition fails fast and loudly
rather than silently mis-integrating.

For a one-off sensitivity study rather than a permanent addition, use the
existing `p_<reaction>` / `NP_delta_<reaction>` config knobs to perturb a
rate already in the network — no file changes needed (see CLAUDE.md
"Nuclear rate variation").

## (b) Add a dark-sector / non-standard background component

Two mechanisms exist today, both documented in `pyprimat/main.py`'s `PyPR`
docstring:

**`extra_rho`** — the generic plug-in point. Pass a list of callables
`rho(Tg) -> MeV^4` to `PyPR(params, extra_rho=[...])`; each is summed into
`rho_tot` by `StandardBackground.Hubble` (`pyprimat/background.py`) every
time the Friedmann equation is evaluated. This is the right tool for
"add an extra energy-density component to a standard run" — e.g. a constant
dark-radiation density:

```python
from pyprimat import PyPR
PyPR({"network": "small"}, extra_rho=[lambda Tg: dRho])
```

Early Dark Energy (`cfg.fEDE > 0`) is itself implemented this way
(`StandardBackground._setup_EDE` appends to `self.extra_rho`) — read that
method as a worked example of a temperature-dependent, parametrised
contribution rather than a flat constant.

**`custom_background`** — for replacing the expansion history itself rather
than adding to it. Set `cfg.custom_background` to a path to a delimited file
with columns `T` [MeV], `t` [s], `a` (scale factor normalised to 1 today);
`CustomBackground` (`pyprimat/background.py`) reads `T(t)`/`t`/`a(t)`
directly from the table instead of solving the entropy-conservation ODE, and
falls back to the instantaneous-decoupling approximation for neutrino
temperatures (`incomplete_decoupling`/`spectral_distortions` are forced to
`False` — NEVO tables are not loaded in this mode). `Neff` is estimated from
the Friedmann equation given the supplied `a(t)`. See
`tests/test_custom_background.py` for a round-trip example (write a
reference background from a standard run, re-run through
`custom_background`, check observables agree to <1e-5 relative).

A cleaner `PyPR(..., background=<Background instance>)` injection hook
(subclassing `pyprimat.background.Background` directly, rather than going
through `extra_rho`/a file) is planned but not yet wired in — until then,
`extra_rho` covers additive contributions and `custom_background` covers a
fully prescribed expansion history.

## (c) Add a neutrino-history variant

Neutrino-temperature/spectral-distortion history is dispatched by
`make_neutrino_history(cfg, plasma)` (`pyprimat/neutrino_history.py`):

1. base regime: `NEVOTable` (`cfg.incomplete_decoupling=True`, the default —
   reads the non-instantaneous-decoupling table) or
   `InstantaneousDecoupling` (analytic `T_ν(T_γ)` from EM entropy
   conservation);
2. optionally wrapped in `AnalyticDistortion` when
   `cfg.spectral_distortions and cfg.analytic_distortions`, layering an
   analytic μ/y-type spectral distortion on top of the base regime.

A new variant — e.g. a different non-instantaneous-decoupling table, or a
new analytic distortion shape — is a new class exposing the same interface
(`Tnue_of_Tg`/`Tnumu_of_Tg`/`Tnutau_of_Tg`/`N_NEVO_of_Tg`/`dFDneu_func`/
`rho_nu_SD`) plus a new branch in `make_neutrino_history`'s dispatch. For
swapping the *data* underlying the existing `NEVOTable`/`InstantaneousDecoupling`
classes without writing a new class, see the "Advanced: custom NEVO tables"
section of `CLAUDE.md` (`nevo_file`/`nevo_spectral_file`/`nevo_grid_file`/
`nevo_file_prefix`).
