# New-physics constants: design for exposing α_QED, mₑ, Q_np

> **Status:** design record only. **No code has been implemented.** This
> document describes how three fundamental constants would be made
> user-overridable in PyPRIMAT, with the physical caveats that motivated the
> chosen scope.

## Context

PyPRIMAT currently hard-codes its fundamental constants in
`pyprimat/constants.py` and re-exposes them as immutable class attributes on
`PyPRConfig` (`config.py:199-245`). We want the user to be able to vary three
constants that govern the weak n↔p rates and the cosmological background, so
they can study new-physics / varying-constant scenarios and propagate the
effect through BBN:

1. **α_QED** — the fine-structure constant (`cfg.alphaem`, currently
   `1/137.035999084`). GUI label: $\alpha_{\text{QED}}$.
2. **mₑ** — the electron mass (`cfg.me`, currently `0.51099895` MeV). GUI label: $m_e$.
3. **Q_np** — the neutron–proton mass difference. The proton mass `mp` stays
   fixed; the neutron mass is **slaved** to `mn = mp + Q_np`. Default
   `Q_np = CONST.mn - CONST.mp = 1.29333236` MeV. GUI label: $Q_{np}$.

All three feed the n↔p weak rates, so they must enter the weak-rate cache
fingerprint (and the electron-thermo / QED-pressure caches where relevant) so
that changing them correctly invalidates stale caches.

> **Deferred:** a fourth constant, the deuterium binding-energy shift
> $\Delta B_D$, was considered but is **explicitly out of scope**. Beyond the
> detailed-balance reverse rates it would also modify the `npTOdg` *forward*
> rate in a way that is not simply derivable from the binding energy alone, so
> it is left for a future, dedicated treatment.

## Key facts established during exploration

- `alphaem`, `me`, `mn`, `mp` are **class attributes** on `PyPRConfig`
  pulled from the frozen `CONST` singleton (`config.py:210,213,214,215`).
  To make them user-overridable they must move into the `DEFAULT_PARAMS`
  dict (the only values the constructor merges user overrides over).
- `weak_rates.py` already references `cfg.alphaem`, `cfg.me`, `cfg.mn`,
  `cfg.mp` and computes `Q = mn - mp` internally — so no formula changes are
  needed there once the config values become overridable.
- The weak-rate cache fingerprint is built in
  `weak_rates.py::_weak_rate_fingerprint` (≈ lines 186-194) and
  `::_thermal_fingerprint` (≈ lines 139-160). Hashing is via
  `cache_utils.fingerprint_hash`. Neither currently lists `alphaem`, `me`, or
  the masses.
- The **electron-thermo cache** fingerprint (`plasma.py:513`) lists only
  `n_electron_table` and `T_start_cosmo_MeV` — it omits `me`, even though the
  electron thermodynamics integrals and the grid floor (`Tmin = cfg.me/…`)
  depend on it.
- **QED plasma-pressure tables** are loaded *blindly* from
  `rates/plasma/QED_*.txt` in the default "file mode"
  (`plasma.py:345-349`) with **no fingerprint**;
  `compute_qed_pressure_tables` (`plasma.py:329`) is called with **no
  `alpha`/`me` arguments**, so it silently uses the module-level defaults
  `_ALPHA_FS`, `_ME_MEV` in `qed_pressure.py`.
- Local **duplicated hard-codings** that must be made to track the config:
  - `nuclear.py:965,967` — `ALPHA`, `ME_MEV` inside `_qed_nuclear_rescale`
    (the Pitrou & Pospelov 2020 radiative-capture QED factor).
  - `qed_pressure.py` — `_ALPHA_FS`, `_ME_MEV` and the `alpha=…, me=…`
    defaults on `_dPa`, `_dPe3`, `_dPb`, `compute_qed_pressure_tables`.
- The live solver reads detailed-balance α/β/γ from the **static**
  `detailed_balance.csv` (`nuclear.py:1222`);
  `compute_detailed_balance_coefficients` is **test-only**. (Relevant only to
  the deferred $\Delta B_D$ work — recorded here for completeness.)

## Implementation plan (recommended approach only)

### 1. `pyprimat/config.py` — make the three constants overridable

- **Remove** the class attributes `alphaem` (210), `me` (213), `mn` (214).
  **Keep** `mp` (215) as a fixed class attribute.
- **Add to `DEFAULT_PARAMS`** (near the other fundamental constants, ~line 70),
  with defaults referencing `CONST` so a default run is bit-for-bit unchanged:
  ```python
  "alphaem": CONST.alphaem,       # fine-structure constant α_QED [dimensionless]
  "me":      CONST.me,            # electron mass [MeV]
  "Q_np":    CONST.mn - CONST.mp, # neutron-proton mass difference [MeV] (1.29333236)
  ```
- **Add a derived property** so `cfg.mn` keeps working and stays slaved:
  ```python
  @property
  def mn(self) -> float:
      """Neutron mass [MeV], slaved to the proton mass + Q_np.
      mp is held fixed; only Q_np is user-overridable."""
      return self.mp + self.Q_np
  ```
  Keeping the names `alphaem`/`me`/`mn` means **every existing `cfg.alphaem`,
  `cfg.me`, `cfg.mn` reference across the package keeps working unchanged.**

### 2. De-hardcode the local duplicates so they follow the config

- `nuclear.py::_qed_nuclear_rescale` — add a `cfg` parameter, replace
  `ALPHA → cfg.alphaem` and `ME_MEV → cfg.me`; update its caller
  (`nuclear.py:1234`, inside `load_network`) to pass `cfg`.
- `plasma.py:329` — call
  `compute_qed_pressure_tables(..., alpha=cfg.alphaem, me=cfg.me)`.
  (The module defaults `_ALPHA_FS`, `_ME_MEV` can remain as the standalone
  fallbacks for direct callers.)

### 3. Cache-fingerprint invalidation (so changed constants never read stale caches)

- **n↔p weak-rate cache** — add `alphaem`, `me`, `mn` to the dicts in both
  `_weak_rate_fingerprint` and `_thermal_fingerprint` (`weak_rates.py`).
  `mn` captures `Q_np` since `mp` is fixed.
- **Electron-thermo cache** — add `"me": cfg.me` to the `fp` dict at
  `plasma.py:513`. (α_QED does not enter the free-electron thermodynamics.)
- **QED plasma-pressure tables** — these on-disk files are unfingerprinted.
  Add a guard in `plasma.py` (~line 320) so that when `cfg.alphaem` or
  `cfg.me` differ from the shipped `CONST` defaults, file mode is **bypassed**
  and the tables are recomputed analytically (≈ 0.3 s) with the user values,
  rather than loading the stale shipped tables.

### 4. GUI (`pyprimat/gui/params_form.py`)

- **Rename the "Physics" panel to "Weak interactions"**: change the
  `"Physics"` group string in `GROUP_ORDER` (line 131) and in every
  `_FORM_METADATA` entry that currently uses it (lines 83-127).
- **Add the three constants to the "Constants" expander** by extending
  `_CONSTANTS_METADATA` (lines 138-148); widgets render automatically via the
  existing loop (lines 262-267), and the "override only if changed from
  default" guard works because the three are now in `DEFAULT_PARAMS`:
  ```python
  "alphaem": (r"$\alpha_{\text{QED}}$  (fine-structure constant)",
              "Electromagnetic coupling. Affects weak n↔p rates (Coulomb/"
              "radiative corrections), QED plasma pressure and the radiative-"
              "capture QED factor. NOT propagated into tabulated nuclear rates."),
  "me":      (r"$m_e$  (electron mass) [MeV]",
              "Electron rest mass. Affects weak rates, QED plasma pressure and "
              "electron thermodynamics."),
  "Q_np":    (r"$Q_{np} = m_n - m_p$  [MeV]",
              "Neutron-proton mass difference. mp is held fixed; mn = mp + Q_np. "
              "Sets the n↔p weak-rate energetics."),
  ```

## Physical caveats (IMPORTANT)

1. **α_QED does NOT reach the tabulated charged-particle nuclear rates.**
   It is de-hardcoded only where it appears explicitly: the weak-rate
   Fermi–Coulomb factor and radiative corrections (`weak_rates.py`), the QED
   plasma pressure (`qed_pressure.py`), and the radiative-capture QED rescale
   (`nuclear.py::_qed_nuclear_rescale`). The forward thermonuclear rates are
   external fits/tables and depend on α through the **Coulomb/Gamow barrier**
   (penetration factor ∝ exp(−2πη), η ∝ α Z₁Z₂); that dependence is **not**
   captured. For charged-particle reactions this effect can dominate, so a
   varied α_QED here is **incomplete** — it models the weak-sector and QED-
   correction effects only. This is the main reason the analogous ΔB_D work
   was deferred.

2. **mₑ likewise does not re-derive the tabulated nuclear rates.** It is
   honoured in the weak rates, QED plasma pressure and electron
   thermodynamics. Minor numerical note: `constants.py` has
   `me = 0.51099895` while `qed_pressure.py` uses `_ME_MEV = 0.5109989461`;
   routing the QED pressure through `cfg.me` unifies them at a negligible
   (~10⁻⁹ relative) baseline shift in the QED pressure.

3. **Q_np holds mp fixed and slaves mn = mp + Q_np.** Changing Q_np therefore
   also changes the neutron mass everywhere `mn` is used (reduced masses,
   finite-nucleon-mass corrections in the weak rates) — physically correct and
   intended.

4. **Defaults reproduce the current constants**, so a run with no overrides is
   bit-for-bit unchanged (modulo the ~10⁻⁹ mₑ note above) and the
   `CLAUDE.md` validation targets remain valid.

## Files touched

- `pyprimat/config.py` — move 3 constants to `DEFAULT_PARAMS`, add `mn` property.
- `pyprimat/nuclear.py` — thread `cfg` into `_qed_nuclear_rescale`.
- `pyprimat/plasma.py` — pass `alpha`/`me` to QED pressure; add `me` to
  electron-thermo fingerprint; bypass stale QED file mode on non-default α/mₑ.
- `pyprimat/weak_rates.py` — add `alphaem`, `me`, `mn` to both fingerprint dicts.
- `pyprimat/gui/params_form.py` — rename panel; add 3 entries to `_CONSTANTS_METADATA`.

## Verification

1. **Baseline unchanged:** `python runfiles/PyPRIMAT_run.py` with defaults must
   still satisfy the `CLAUDE.md` targets (small: YP 0.2469983, D/H 2.43490e-5;
   medium: YP 0.2470017, D/H 2.43561e-5) within tolerance.
2. **Cache invalidation:** run once at default, then with `alphaem` scaled by
   1.01; confirm (verbose) the weak-rate cache is recomputed (fingerprint
   mismatch, not reused) and Neff/YP shift at the documented precision.
   Repeat for `me` (also exercises the electron-thermo + QED-pressure caches)
   and for `Q_np`.
3. **Existing tests** (`tests/`) still pass — in particular detailed-balance
   and network-generation tests, which use `cfg.me` but not these new overrides.
4. **GUI:** launch `pyprimat-gui`; confirm the panel is now "Weak interactions"
   and the "Constants" expander shows $\alpha_{\text{QED}}$, $m_e$, $Q_{np}$
   with correct LaTeX, and that editing them changes the computed observables.
