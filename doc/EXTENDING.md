# Extending primat with new physics

A short guide to the three supported extension points. For the full physics
formalism (rate evaluation, background equations, weak-rate corrections),
the primary reference is `doc/Pitrou_etal_PhysReptArxivVersion.pdf`
(Pitrou, Coc, Uzan, Vangioni, *Physics Reports* 04 (2018) 005); see also the
other PDFs in `doc/`. Internal planning docs (`FUTURE.md`, `CPLAN.md`) are
deliberately not cited here — they describe in-progress refactors, not
physics, and rot once the work lands.

## (a) Add a nuclear reaction

Drop a rate table under `primat/data/nuclear/tables/<name>/<name>.txt`, add
`<name>` to the relevant network file under
`primat/data/nuclear/networks/`, and let `reaction_stoichiometry`
(`primat/network_data.py`) auto-derive the
stoichiometry from the reaction name's `TO`-separated tokens (falling back
to a manual `reactions_large.csv`/`detailed_balance.csv` row only if the
name can't be tokenised). `load_network` validates A/Z conservation and
rejects duplicate entries, so a malformed addition fails fast and loudly
rather than silently mis-integrating.

For a one-off sensitivity study rather than a permanent addition, use the
existing `p_<reaction>` / `delta_<reaction>` config knobs to perturb a
rate already in the network — no file changes needed (see CLAUDE.md
"Nuclear rate variation").

## (b) Add a dark-sector / non-standard background component

Two mechanisms exist today, both documented in `primat/main.py`'s `PRIMAT`
docstring:

**`extra_rho`** — the generic plug-in point. Pass a list of callables
`rho(Tg) -> MeV^4` to `PRIMAT(params, extra_rho=[...])`; each is summed into
`rho_tot` by `StandardBackground.Hubble` (`primat/background.py`) every
time the Friedmann equation is evaluated. This is the right tool for
"add an extra energy-density component to a standard run" — e.g. a constant
dark-radiation density:

```python
from primat import PRIMAT
PRIMAT({"network": "small"}, extra_rho=[lambda Tg: dRho])
```

Early Dark Energy (`cfg.fEDE > 0`) is itself implemented this way
(`StandardBackground._setup_EDE` appends to `self.extra_rho`) — read that
method as a worked example of a temperature-dependent, parametrised
contribution rather than a flat constant.

**`custom_background`** — for replacing the expansion history itself rather
than adding to it. Set `cfg.custom_background` to a path to a delimited file
with columns `T` [MeV], `t` [s], `a` (scale factor normalised to 1 today);
`CustomBackground` (`primat/background.py`) reads `T(t)`/`t`/`a(t)`
directly from the table instead of solving the entropy-conservation ODE, and
falls back to the instantaneous-decoupling approximation for neutrino
temperatures (`incomplete_decoupling`/`spectral_distortions` are forced to
`False` — NEVO tables are not loaded in this mode). `Neff` is estimated from
the Friedmann equation given the supplied `a(t)`. See
`tests/test_custom_background.py` for a round-trip example (write a
reference background from a standard run, re-run through
`custom_background`, check observables agree to <1e-5 relative).

### Using custom_background to test alternative cosmological scenarios

The `custom_background` mechanism is particularly powerful for exploring
non-standard cosmological expansion histories. By providing a complete
`(T, t, a)` table, you can bypass **all** of primat's standard cosmology
calculations — the entropy-conservation ODE, the NEVO neutrino-decoupling
history, and the associated weak-rate corrections — and replace them with
your own expansion history. This is the intended path for investigating:

- Modified expansion histories (e.g., early dark energy models, non-standard
  radiation content, or time-varying dark-energy equations of state)
- Alternative neutrino physics (by providing a table that encodes a
  different `T_ν(T_γ)` relationship implicitly through the supplied `a(t)`)
- Non-standard temperature-time relationships (e.g., from modified gravity
  or other exotic scenarios)

**How it works:**

1. Create a tab- or comma-delimited text file with at least three columns
   named `T` [MeV], `t` [s], and `a` (scale factor, normalised to 1 today).
   The file must span the full BBN temperature range (typically from
   `T ~ 10 MeV` down to `T ~ 0.001 MeV`).

2. Set `cfg.custom_background` to the path of this file.

3. primat automatically forces `incomplete_decoupling=False` and
   `spectral_distortions=False`, then uses your table directly:
   - `T_of_t`/`t_of_T`/`a_of_T`/`T_of_a`/`t_of_a`/`a_of_t` are all read from the
     supplied table via linear interpolation.
   - Neutrino temperatures use the **instantaneous-decoupling** approximation
     (`T_ν = (4/11)^(1/3) * T_γ` for all flavours), computed from your supplied
     photon temperature `T`.
   - The n↔p weak rates are computed using these instantaneous-decoupling
     neutrino temperatures (no NEVO spectral distortions).
   - `Neff` is estimated at the end of BBN from the Friedmann equation:
     `H² = 8πG/3 · ρ_tot`, where `ρ_tot = ρ_plasma + ρ_ν`, and `ρ_ν` is inferred
     as the difference between `ρ_tot` (from `H` via your `a(t)`) and the known
     plasma density `ρ_plasma(T_γ)`.

**Example: generating a custom-background table from an external cosmology code**

Suppose you have a cosmology code that outputs `(T_γ, t, a)` for a non-standard
model. You can format this output as a TSV file:

```
T	t	a
10.0	0.001	1.0e-10
5.0	0.01	2.0e-10
...
0.001	1000000	0.001
```

Then run BBN with this background:

```python
from primat.backend import run_bbn

result = run_bbn({
    "custom_background": "my_cosmology.tsv",
    "network": "small",
    "Omegabh2": 0.022425,
})

print(f"YP (BBN) = {result['YPBBN']:.8f}")
print(f"Neff     = {result['Neff']:.8f}")
```

**Important notes:**

- The scale factor `a` in your table **must** be normalised so that
  `a · T_γ → T_0CMB` as `T_γ → 0`. In practice this means `a ≈ 1/T_γ` in
  radiation domination (the standard convention), so `a = 1` today when
  `T_γ = T_0CMB`.
- Rows may be in any order; primat sorts them internally by cosmic time `t`.
- Extra columns in your file are silently ignored.
- This mode is mutually exclusive with `external_scale_factor=True`.
- The Python backend always supports `custom_background`. The C backend
  (`primat-c`) supports it as of the same feature set; both backends give
  identical results when used with the same table.

A `PRIMAT(background=<Background instance>)` injection hook (subclassing
`primat.background.Background` directly, rather than going through
`extra_rho`/a file) is also available — see `primat/main.py`'s `PRIMAT.__init__`
docstring for the full worked example. `extra_rho` covers additive
contributions, `custom_background` covers a fully prescribed expansion
history read from a file, and `background=` covers a fully custom
`Background` subclass built in code.

## (c) Add a neutrino-history variant

Neutrino-temperature/spectral-distortion history is dispatched by
`make_neutrino_history(cfg, plasma)` (`primat/neutrino_history.py`):

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
