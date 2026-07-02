# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for primat.

Physical constants and derived unit conversions are *fixed* and computed once
here.  All run-time flags and cosmological/nuclear parameters are carried in a
``PRIMATConfig`` instance and can be overridden by passing a parameter dictionary
to ``PRIMATConfig(params)``.

No file I/O happens here.  Nuclear rate data are loaded separately in
``nuclear_data.py``.
"""

import os
import re
import warnings
import numpy as np

from .constants import CONST

__all__ = ['DEFAULT_PARAMS', 'PRIMATConfig']

# String-valued config keys that represent filesystem paths.
# These are normalized with os.path.expanduser() so CLI users can pass
# quoted "~/" prefixes through --set and still get the expected home-dir
# expansion.
_PATH_PARAMS = {
    "nevo_file",
    "nevo_spectral_file",
    "nevo_grid_file",
    "custom_background",
    "data_dir",
    "user_nuclear_dir",
    "output_file",
    "output_final_file",
    "output_background_file",
    "output_mc_file",
    "output_decay_file",
}


def _expanduser_path(value):
    """Expand a user-home prefix in a path-like config value.

    Parameters
    ----------
    value : str | os.PathLike | None
        Raw path value supplied by the caller. ``None`` is passed through
        unchanged so optional path parameters keep their sentinel value.

    Returns
    -------
    str | None
        The same path with a leading ``~`` resolved against the current
        user home directory, or ``None`` if that was the input.

    Example
    -------
        >>> _expanduser_path("~/Downloads/custom")
        '/home/user/Downloads/custom'
    """
    if value is None:
        return None
    return os.path.expanduser(os.fspath(value))


def _rates_overlay_notice(field: str, path: str) -> str:
    """Render the startup note for a custom data/nuclear overlay directory.

    Parameters
    ----------
    field : str
        Either ``"data_dir"`` (full-takeover data root) or
        ``"user_nuclear_dir"`` (additive nuclear overlay).
    path : str
        Directory path already accepted by the config validator.

    Returns
    -------
    str
        Human-readable notice explaining the effect of the override.
    """
    if field == "data_dir":
        label = "full-takeover data directory"
        detail = "entire data tree (NEVO/, nuclear/, weak/, plasma/, csv/) replaced"
    else:
        label = "additive nuclear overlay"
        detail = "nuclear networks and rate tables"
    return (
        f"[init]  {field} {label} override: {detail} under "
        f"{os.path.abspath(os.path.expanduser(os.fspath(path)))!r}."
    )


def _overlay_candidates(base: str, relpath: str) -> list[str]:
    """Return overlay lookup candidates for a rates-relative path.

    The shipped tree is rooted at ``primat/data`` and therefore uses paths
    such as ``nuclear/networks/small.txt``.  Overlay directories are treated
    as the equivalent of ``primat/data/nuclear`` instead, so the primary
    lookup drops a leading ``nuclear/`` component when present and then
    falls back to the legacy nested layout for compatibility.
    """
    candidates = []
    if relpath.startswith("nuclear/"):
        candidates.append(os.path.join(base, relpath[len("nuclear/"):]))
    candidates.append(os.path.join(base, relpath))
    return candidates

# ---------------------------------------------------------------------------
# Default parameter values exposed as a plain dict so callers can inspect them
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # ---- general behaviour and numerical settings ------------------------------------------------
    "verbose":               False, #If you want the messages from the code to be printed, set this to True.  This is separate from the debug, which controls the printing of extra messages for debugging purposes.
    "debug":                 False, #If you want the debug messages to be printed, set this to True.  This is separate from the verbose, which controls the printing of general messages from the code.
    "show_progress":         True,  # Set to False to hide the compact stderr progress indicators printed
    # when verbose=False: the "[primat]  HT.  MT.  LT.  done." phase markers from a single solve,
    # and the "[MC] Running N samples..." banner / "[MC] i/N samples (XX%)" counter from an MC run.
    # Has no effect when verbose=True (the verbose prints already convey progress). Mirrors
    # primat-c's CPRConfig.show_progress field (primat-c/include/config.h), already wired there.
    "numerical_precision":        1.e-7, # for finite differences (solve_ivp). 1e-6 should be enough.
    "numba_installed":                 True,  # will be re-checked at runtime. Allows just-in-time compilation for faster execution.

    # ---- physics settings ------------------------------------------------
    # ---- neutrino decoupling ----------------------
    "incomplete_decoupling":      True, # True: non-instantaneous neutrino decoupling, read from the pre-computed NEVO table.
    # False: instantaneous decoupling (Tnu/Tgamma fixed by EM entropy conservation; see neutrino_history.InstantaneousDecoupling).
    # incomplete_decoupling=False with spectral_distortions=True (NEVO-based) is physically inconsistent and rejected; see PRIMATConfig validation.

    # ---- electromagnetic plasma -------------------
    "QED_corrections":            True,  # Whether to include QED interaction corrections to the EM plasma equation of state.
    "n_electron_table":           2000,  # number of log-spaced grid points for the electron-thermo (rho_e/p_e and derivatives) tables
    "recompute_electron_thermo":  False, # If False, load rates/plasma/electron_thermo_cache.txt when its fingerprint matches; otherwise (or if True) recompute and overwrite it. See plasma.Plasma._build_electron_tables.
    "recompute_qed_corrections":  False, # True: always compute analytically and overwrite rates/plasma/QED_*.txt; False: load from files if present, otherwise compute on the fly without saving

    # ---- spectral distortions ---------------------
    "spectral_distortions":       True, # Corrections to n<->p weak rates from deviations of the neutrino phase-space distribution from a perfect Fermi-Dirac.
    # Two sub-modes, selected by analytic_distortions (see neutrino_history.py):
    #   False (default): read the distortion from the full NEVO spectrum file
    #     (86-column, not _col_1_7); requires incomplete_decoupling=True.
    #   True: analytic y-type (SZ) + gray-type distortion controlled by
    #     y_SZ/y_gray, also contributing rho_nuSD to the Friedmann equation.
    # NOTE: there is deliberately no mu-type (chemical-potential) spectral
    # distortion -- a neutrino chemical potential is not a spectral distortion;
    # use munuOverTnu instead (it shifts the weak rates AND the energy density).
    "analytic_distortions":       False,
    "y_SZ":                       0., # Amplitude of the y-type (Sunyaev-Zel'dovich-like, Compton) distortion; see neutrino_history.AnalyticDistortion.
    "y_gray":                     0., # Amplitude of the gray-type (gray-body temperature-rescaling) distortion: delta_f(y) = -fd(y) + fd(y/(1+y_gray))/(1+y_gray)**3.
    # Despite the shared "y_*" naming and despite generate_rates/PRIMAT-Main-gray.m
    # calling its equivalent parameter "YSZ", this is NOT the Compton/SZ shape
    # above: it rescales the neutrino spectrum as if its temperature shifted by
    # a factor (1+y_gray), with the (1+y_gray)**-3 prefactor chosen so the
    # rescaled piece's NUMBER density exactly matches the unperturbed
    # Fermi-Dirac (integral{y^2 delta_f dy} = 0 exactly for any y_gray) while
    # its ENERGY density shifts linearly, integral{y^3 delta_f dy} = y_gray *
    # 7*pi**4/120 exactly -- a distinct, independent third distortion shape.
    # See neutrino_history.AnalyticDistortion.

    # ---- custom NEVO tables ------------------------------------------------
    # Override the shipped rates/NEVO/ tables with custom ones (e.g. a
    # higher-resolution or non-standard neutrino-decoupling history).  Each is
    # a filename resolved relative to rates/NEVO/, or an absolute path; None
    # uses the shipped file selected by QED_corrections (see
    # neutrino_history.NEVOTable / resolve_nevo_path).
    "nevo_file":                  None, # 6/7-column thermo table (replaces NEVOPRIMAT[_NoQED]_col_1_7.csv)
    "nevo_spectral_file":         None, # 86-column spectral-distortion table (replaces NEVOPRIMAT[_NoQED].csv); only read when spectral_distortions=True and analytic_distortions=False
    "nevo_grid_file":             None, # y-grid for nevo_spectral_file (replaces NEVOGrid.csv); its length must match nevo_spectral_file's column count minus 6
    "nevo_file_prefix":           "NEVOPRIMAT", # base name for the *default* NEVO thermo/spectral
    # files: "<prefix>[_NoQED]_col_1_7.csv" (thermo) and "<prefix>[_NoQED].csv" (86-col
    # spectral). NEVOGrid.csv is NOT prefixed (shared y-grid). Ignored for any file
    # selected explicitly via nevo_file/nevo_spectral_file (those still win), and has no
    # effect when incomplete_decoupling=False (no NEVO file is read at all).

    # ---- data directory override and nuclear overlay -----------------------
    # See PRIMATConfig.resolve_rates_path and _resolved_data_dir. Both default
    # to None (shipped primat/data/ tree). When data_dir is set it completely
    # replaces the shipped data tree (NEVO/, weak/, plasma/, nuclear/, csv/
    # must all be present under that directory). When user_nuclear_dir is set
    # it is an additive overlay for nuclear networks and rate tables only
    # (checked before the shipped tree, so "small"/"large" remain available
    # even if only user_nuclear_dir is set and it doesn't contain them).
    # Overlay roots for user_nuclear_dir are treated as the equivalent of
    # primat/data/nuclear, so they should contain `networks/` and `tables/`
    # directly.
    "data_dir":          None, # Full-takeover data directory (must exist if set; replaces primat/data/)
    "user_nuclear_dir":  None, # Additive overlay for nuclear networks & rate tables (must exist if set)

    # ---- background mode ---------------------------------------------------
    "external_scale_factor":      False, # If True, read the scale factor a(T_gamma) directly
    # from the NEVO table's x column (a is proportional to x by the NEVO convention)
    # instead of solving the entropy-conservation ODE from the heating
    # function N_NEVO. t(a) is still obtained by Hubble integration (unchanged). Outside
    # the table's T range, both modes extrapolate assuming radiation domination
    # (a ~ 1/T, t ~ 1/T^2). Requires incomplete_decoupling=True.

    "custom_background":         None, # Path (str) to a user-supplied background file
    # containing at minimum the columns T [MeV], t [s], and a (scale factor, normalised
    # so that a·T_γ → T0CMB_MeV as T → 0, i.e. a = 1 today). The file must be
    # tab- or comma-delimited with a header row. Extra columns are ignored.  When set,
    # incomplete_decoupling and spectral_distortions are forced to False (with warnings
    # if they were True); the n<->p weak rates use the instantaneous-decoupling
    # approximation (T_ν(T_γ) from EM entropy conservation). Neff is estimated via the
    # Friedmann equation from the supplied a(t). Incompatible with external_scale_factor.

    # ---- fundamental constants (overridable for sensitivity studies) --------
    "GN":                         6.674299257609439e-11,   # Newton's constant, SI units [m^3 kg^-1 s^-2]

    # ---- background thermodynamics ----------------------------------------
    "T_start_cosmo_MeV":          40.0,
    "T_end_MeV":                  1.e-3,  # end temperature for nuclear integration [MeV]; default 0.001 MeV ≈ 11.6 MK
    "sampling_temperature_per_decade": 600,  # points per decade of T for the background a(T)/t(T) grid

    # ---- n <--> p weak rates ----------------------------------------------
    # rates/weak/nTOp_*.txt carry a fingerprint header recording the config
    # fields that affect their content; RecomputeWeakRates loads the cache
    # only if its fingerprint matches, and otherwise recomputes from scratch
    # (~2 s).  See weak_rates.RecomputeWeakRates for the full cache logic.
    #
    # Four additive correction terms control which physical effects enter the
    # total n<->p rate (mirroring PRIMAT-Main.m §IV.B):
    #
    #   radiative_corrections   -- True: replace the Born chi function with the
    #                              Coulomb + T=0 resummed radiative correction
    #                              (CCR, Phys. Rep. Eq. 101; Czarnecki et al. 2004).
    #                              False: use the bare Born approximation.
    #   finite_mass_corrections -- True: add the Fokker-Planck finite-nucleon-mass
    #                              correction (FMCCR if radiative_corrections=True,
    #                              FMNoCCR otherwise; Phys. Rep. §III.G).
    #   thermal_corrections     -- True: add the finite-temperature radiative
    #                              correction (CCRTh; Brown & Sawyer 2001,
    #                              Phys. Rep. §III.H, Eqs. 107-113).
    #   spectral_distortions    -- (controlled in the neutrino section above)
    #                              Corrections from non-FD neutrino distributions;
    #                              internally uses SD_CCR or SD_Born depending on
    #                              radiative_corrections.
    #
    # Born (crude) mode = radiative_corrections=False, finite_mass_corrections=False,
    #                     thermal_corrections=False.  All True = full PRIMAT rate.
    "radiative_corrections":      True,  # True: Coulomb + T=0 resummed radiative corrections (CCR); False: Born approximation.
    "finite_mass_corrections":    True,  # True: add Fokker-Planck finite-nucleon-mass correction (FMCCR or FMNoCCR).
    "thermal_corrections":        True,  # True: add finite-temperature radiative corrections (CCRTh; Brown & Sawyer 2001).
    ##################### caching/saving options
    "weak_rate_cache":            True,  # If False, never load the cache (always recompute); save_nTOp still controls whether the result is written back.
    "save_nTOp":                  True,  # If True, the computed n<->p rates are saved to rates/weak/ as nTOp_<hash>.txt (forward and backward columns together).
    "sampling_nTOp_per_decade":   80,    # points per decade of T (T_end -> T_start) in the single n<->p rate grid

    "save_nTOp_thermal":          True,  # If True, the computed thermal n<->p rates are saved to rates/weak/ as nTOp_thermal_<hash>.txt (both directions in one file).
    "sampling_nTOp_thermal_per_decade": 20,   # points per decade of T (T_end -> T_start) for the thermal-correction table
    ##################### Normalization of weak rates
    "tau_n_normalization":        True,  # Use neutron lifetime to normalize weak rates (instead of absolute normalization from GF, Vud, gA, etc.)
    "tau_n":                      878.4,  # neutron lifetime [s]; overrides the class-level constant when tau_n_normalization=True
    "std_tau_n":                  0.5,    # 1σ uncertainty on tau_n [s], used for MC sampling

    # Accuracy knobs for the thermal n<->p radiative correction integral, used
    # only when the thermal-correction cache must be recomputed (see
    # weak_rates._L_CCRTh_interpolants).  Evaluated with the `vegas`
    # Monte-Carlo library when available, else scipy.integrate.dblquad.
    "vegas_n_eval":               20000,   # vegas: evaluations per iteration
    "vegas_n_itn":                20,      # vegas: number of iterations
    "epsrel_thermal":             1.e-2,   # dblquad fallback: relative tolerance
    
    # ---- Output options ------------------------------------------------------
    # Writes a TSV (cfg.output_file) with the time evolution of T, t, and of
    # every nuclide's abundance in the chosen network (8/~59 for small/large,
    # fewer for large with an amax cutoff) plus the n<->p weak rates; see
    # nuclear_network.NuclearNetwork._write_time_evolution.
    "output_time_evolution":      False,
    "output_rates_time_evolution": False, #whether to include or not the nuclear rates evolution in the output time evolution file. This is only useful if you want to inspect the rates evolution, otherwise it is better to set it to False to save disk space and speed up the code. Ignored (with a printed note) for network="large", where per-reaction flux columns are omitted.
    "output_n_points":            500,
    "output_file":                "results/output_tables.tsv",
    # Two-column dump (nuclide name, final mass-fraction abundance Y) at the end of BBN.
    "output_final_result":        False,
    "output_final_file":          "results/output_final.dat",

    # Writes a separate TSV (cfg.output_background_file) with the cosmological
    # background's own time evolution (T, t, and -- if available -- a, H,
    # individual neutrino temperatures, NEVO heating function, and
    # plasma/neutrino/extra/total energy densities); see
    # background.Background.write_time_evolution.
    "output_background_evolution": False,
    "output_background_file":     "results/output_background.tsv",

    # Writes a TSV (cfg.output_mc_file) with every Monte-Carlo sample drawn by
    # backend.run_mc/main.mc_uncertainty: one column per requested quantity
    # (nuclide final-Y names and/or result-dict observables), one row per
    # sample. Has no effect by itself -- only the MC entry points write this
    # file, never a plain solve() -- so it is opt-in plumbing for callers that
    # want every sample on disk (e.g. for an external corner plot) rather than
    # just the summary mean/std (see primat.backend.dump_mc_samples).
    "output_mc_samples":           False,
    "output_mc_file":              "results/output_mc_samples.tsv",


    # ---- nuclear network --------------------------------------------------
    "rate_interp_order":          "linear",   # interpolation of every nuclear rate table:
                                              # "linear" (fast np.interp) or "quadratic"/"cubic" (scipy interp1d)

    # Master grid onto which every nuclear reaction rate table is resampled at
    # load time.  This makes load_network grid-agnostic: tables generated with
    # different grids (e.g. via --keep-source-grid in convert_ac2024_rates.py,
    # or from external sources) are all resampled onto this common grid so that
    # fill_buffer's single searchsorted path remains valid.
    "rate_grid_npts":             1000,       # number of points in the master T9 grid
    "rate_grid_T9_min":          1.0e-3,     # minimum T9 [GK] on the master grid
    "rate_grid_T9_max":          10.0,       # maximum T9 [GK] on the master grid

    # Network selector.  "small" is the built-in ORDER_SMALL network.  Any other
    # value loads data/nuclear/networks/<network>.txt -- shipped options are
    # "small_parthenope" and "large"; any other name loads a custom network
    # file of the same form.
    "network":                    "small",

    # Maximum nuclide mass number A = N + Z to include, for *any* network
    # (not just "large" -- a network whose nuclides are all below the cutoff
    # simply sees no reaction dropped). Reactions involving any nuclide with
    # A > amax are dropped. None = no filter (keep all reactions). Must be a
    # positive integer when set.
    # Migration from the old named networks (removed; reproduce them via):
    #   old network="medium"    -> network="large", amax=8   (68 reactions,
    #                                                          identical set)
    #   old network="deuterium" -> network="large", amax=2   (adds
    #                                                          p_p_n__d_p
    #                                                          alongside
    #                                                          n_p__d_g;
    #                                                          D/H matches to
    #                                                          ~1e-9 relative)
    # Example: {"network": "large", "amax": 20} keeps only A ≤ 20 nuclides.
    "amax":                       None,

    # Absolute solve_ivp tolerance for the large-network LT era.  The heavy
    # nuclides reach very small abundances, so this is tighter than the 1e-15
    # used for the small-network LT era (which keeps its validated tolerances).
    "atol_large_LT":              1.e-26,
    "rescale_nuclear_rates":            False, #Use to vary some rates with a uniform factor to explore their impact.

    # Cap applied to the MC rate rescaling factor during Monte Carlo runs.
    # When a p_* parameter is non-zero, the effective variation factor is
    #   variation = sigma^p + delta
    # which can grow very large for extreme draws of p.  This parameter clamps
    # the variation to [1/cap, cap] before multiplying the median rate.
    # A value of 30 means no more than a factor of 30 up or down. Lowered from
    # the former 1e3 default: reactions carrying a flat "uncertainty factor
    # f=10-100" placeholder (e.g. CF88 rates such as He3_t__a_d/He3_t__a_n_p)
    # can otherwise draw a >=3-sigma p and multiply their rate by up to 1000x,
    # which for non-trace species (He3/t, unlike the many trace heavy-nuclide
    # branches sharing the same placeholder error) dominates the MC variance
    # of D/H with an unphysically large single-sample outlier rather than a
    # smooth uncertainty estimate.
    # Set to None to disable the cap entirely.
    "mc_rate_rescale_cap":         30,

    # QED correction to select radiative-capture nuclear rates (Pitrou & Pospelov 2020).
    # Applies a T9-dependent multiplicative rescaling to the forward rate tables of
    # n_p__d_g, d_p__He3_g, t_p__a_g, t_a__Li7_g, He3_a__Be7_g at load time.  When True the
    # corrected values become the new medians, so p_* and delta_* variations
    # work relative to the QED-corrected central value.
    "nuclear_qed_corrections":    True,

    # ---- cosmological inputs ----------------------------------------------
    "Omegabh2":                   0.022425,
    "Omegach2":                   0.11933,  # cold dark matter density parameter Omega_c h^2 (Planck 2018)
    "h":                          0.6766,   # reduced Hubble constant h = H_0 / (100 km/s/Mpc) (Planck 2018)
    "DeltaNeff":                  0.,
    "munuOverTnu":                0., # Reduced chemical potential xi = mu/T of neutrinos (same for all flavours, nu_e, nu_mu, nu_tau; antineutrinos carry -xi).
    # A genuine chemical potential: it shifts the n<->p weak rates (FD_nu3 in the
    # rate integrands) AND raises the neutrino energy density / Neff by
    # rho(xi) = T^4 (7pi^2/120 + xi^2/4 + xi^4/(8 pi^2)) per flavour
    # (plasma.rho_nu_chempot_excess). It is part of the weak-rate cache fingerprint.
    # munuOverTnu != 0 with incomplete_decoupling=True is physically inconsistent (the NEVO table assumes it vanishes); use incomplete_decoupling=False to explore non-zero values.

    # ---- Decay-era options -------------------------------------------------
    # decay_reverse_rates: when True, compute detailed-balance reverse rates
    # for radioactive-decay reactions, instead of treating them as irreversible
    # (abg = (0, 0, 0)).  During standard BBN the forward decay rate is
    # negligible (e.g. C14 T1/2 = 5700 yr ≫ t_end ≈ 10^6 s), so the reverse
    # rate is likewise negligible; enabling this only matters when T_end_MeV
    # is extended far below the standard 0.001 MeV and thermal equilibrium of
    # long-lived isotopes becomes relevant.
    "decay_reverse_rates":        False,

    # decay_era: if True and network="large", run a fourth "Decay Time" (DT)
    # integration era after the LT era, propagating abundances forward in time
    # (at fixed comoving scale) purely under radioactive decay (no Hubble
    # expansion, no thermal production).  The DT era spans t ∈ [t_end, t_end +
    # t_decay_end], log-spaced on decay_n_points time points.
    "decay_era":                  False,
    "t_decay_end":                3.156e16,  # DT era duration [s] (default: 1 Gyr = 3.156e16 s)
    "decay_n_points":             200,        # log-spaced output points in the DT era
    "output_decay_evolution":     False,      # write TSV of DT-era abundance time evolution
    "output_decay_file":          "results/output_decay_evolution.tsv",

    # ---- Early Dark Energy ------------------------------------------------
    "fEDE":                       0.,     # EDE fraction at peak; 0 = disabled
    "zcEDE":                      1.e8,   # redshift of EDE peak
    "wnEDE":                      1.,     # EDE equation-of-state parameter
}


class PRIMATConfig:
    """
    Immutable physical constants + mutable run-time parameters.

    Usage::

        cfg = PRIMATConfig()                    # all defaults
        cfg = PRIMATConfig({"Omegabh2": 0.022, "network": "large", "amax": 8})

    After construction every key in ``DEFAULT_PARAMS`` is an attribute, plus
    all physical constants listed below.
    """

    @property
    def is_small(self) -> bool:
        """True if using the 'small' network."""
        return self.network == "small"

    @property
    def is_large(self) -> bool:
        """True if using the 'large' network."""
        return self.network == "large"

    # ------------------------------------------------------------------
    # Physical constants and unit-conversion factors
    # ------------------------------------------------------------------
    # All fixed PDG values, CGS<->natural-units conversion factors, and the
    # purely-constant derived quantities (sW2, s0bar, s0CMB, n0CMB, mB,
    # HubbleOverh, the fixed temperature eras T_start/T_weak/T_nucl/T_end,
    # ...) live in primat.constants.Constants (see that module for
    # definitions, formulas and citations). They are re-exposed here as
    # plain class attributes so existing code (cfg.me, cfg.MeV_to_Kelvin,
    # cfg.s0bar, ...) is unaffected; new physics code may instead import
    # CONST directly from primat.constants.
    Kelvin         = CONST.Kelvin
    second         = CONST.second
    cm             = CONST.cm
    gram           = CONST.gram
    erg            = CONST.erg
    kB             = CONST.kB
    clight         = CONST.clight
    hbar           = CONST.hbar
    Mpc            = CONST.Mpc
    MeV            = CONST.MeV
    keV            = CONST.keV
    alphaem        = CONST.alphaem
    GF             = CONST.GF
    mZ             = CONST.mZ
    me             = CONST.me
    mn             = CONST.mn
    mp             = CONST.mp
    T0CMB          = CONST.T0CMB
    MeV_to_Kelvin  = CONST.MeV_to_Kelvin
    MeV_to_secm1   = CONST.MeV_to_secm1
    MeV_to_g       = CONST.MeV_to_g
    MeV_to_cmm1    = CONST.MeV_to_cmm1
    MeV4_to_gcmm3  = CONST.MeV4_to_gcmm3
    T_start        = CONST.T_start
    T_weak         = CONST.T_weak
    T_nucl         = CONST.T_nucl
    sW2            = CONST.sW2
    geL            = CONST.geL
    geR            = CONST.geR
    gmuL           = CONST.gmuL
    gmuR           = CONST.gmuR
    gA             = CONST.gA
    kappa_p        = CONST.kappa_p
    kappa_n        = CONST.kappa_n
    deltakappa     = CONST.deltakappa
    Vud            = CONST.Vud
    radproton      = CONST.radproton
    s0bar          = CONST.s0bar
    s0CMB          = CONST.s0CMB
    n0CMB          = CONST.n0CMB
    ma             = CONST.ma
    He4Overma      = CONST.He4Overma
    HOverma        = CONST.HOverma
    Neff_SM        = CONST.Neff_SM
    mB             = CONST.mB
    maOvermB       = CONST.maOvermB
    HubbleOverh    = CONST.HubbleOverh

    # ------------------------------------------------------------------
    # Quantities depending on overridable parameters (GN, T_start_cosmo_MeV)
    # ------------------------------------------------------------------

    # Temperature era set by the overridable T_start_cosmo_MeV [K].
    @property
    def T_start_cosmo(self) -> float:
        return self.T_start_cosmo_MeV * self.MeV_to_Kelvin

    @property
    def T_end(self) -> float:
        """End temperature for nuclear integration [K].

        Set via ``T_end_MeV`` [MeV] in ``DEFAULT_PARAMS`` (default 0.001 MeV,
        i.e. the standard BBN endpoint at ~11.6 MK / ~1.3×10^6 s).  Making
        it configurable allows extending the integration into the Decay Time
        era (``decay_era=True``) or performing custom post-BBN analysis at
        lower temperatures.

        The default 0.001 MeV (≈ 11.6 MK, cosmic time ≈ 1.3×10⁶ s ≈ 15 days)
        is the standard end point of BBN integration.

        Example::

            # Extend BBN integration to 0.0001 MeV (10× lower than default):
            cfg = PRIMATConfig({"T_end_MeV": 1e-4})
        """
        return self.T_end_MeV * self.MeV_to_Kelvin

    # Gravity: GN is overridable, so it lives in DEFAULT_PARAMS only.
    # tau_n [s] is similarly overridable (DEFAULT_PARAMS), used by weak_rates.
    #
    # cfg.GN is stored in SI units [m^3 kg^-1 s^-2] (so it reads/edits like any
    # textbook value of Newton's constant), but the Friedmann equation below is
    # written in the natural-units (hbar=c=1) convention used throughout the
    # rest of the code, where G has dimension [energy]^-2. Convert once here via
    # CONST.GN_SI_to_MeV2 (see that property's docstring for the derivation).
    @property
    def _GN_MeV2(self) -> float:
        """Newton's constant in natural units [MeV^-2], converted from the
        SI-valued ``self.GN``."""
        return self.GN * CONST.GN_SI_to_MeV2

    @property
    def Mpl(self) -> float:
        return 1. / np.sqrt(self._GN_MeV2)

    @property
    def rhocOverh2(self) -> float:
        return 3. / (8. * np.pi * self._GN_MeV2) * self.HubbleOverh**2  # [MeV^4/h^2]

    # ------------------------------------------------------------------
    # Constructor: merge user params over defaults
    # ------------------------------------------------------------------
    def __init__(self, params: dict | None = None):
        # Initialise every default as an instance attribute
        # We bypass our own __setattr__ for the initial dict setup to avoid
        # interference before the base dicts are even created.
        for key, value in DEFAULT_PARAMS.items():
            # Deep copy dictionaries to avoid shared state between instances
            if isinstance(value, dict):
                object.__setattr__(self, key, value.copy())
            else:
                object.__setattr__(self, key, value)

        # Apply data_dir early (before _load_nuclide_data) so nuclides.csv is
        # read from the user-supplied root when one is provided.
        if params and "data_dir" in params:
            object.__setattr__(self, "data_dir", _expanduser_path(params["data_dir"]))

        # Load nuclide data from CSV
        self._load_nuclide_data()

        # Initialize nuclear rate variation dicts as empty for now.  They are
        # populated with the configured network's reactions *after* user
        # overrides are applied below (self.network may itself be one of
        # those overrides), so that the per-reaction defaults match the
        # network actually requested by the caller.
        object.__setattr__(self, "p_rxn", {})
        object.__setattr__(self, "delta_rxn", {})
        object.__setattr__(self, "_init_messages", [])

        user_keys = set(params.keys()) if params else set()

        # Apply user overrides
        if params:
            known_prefixes = ('p_', 'delta_')
            unknown = set()
            for key, value in params.items():
                if key in DEFAULT_PARAMS or any(key.startswith(p) for p in known_prefixes):
                    setattr(self, key, value)
                else:
                    unknown.add(key)
            
            if unknown:
                warnings.warn(
                    f"PRIMATConfig: unknown parameter keys ignored: {unknown}",
                    stacklevel=2,
                )

        # fEDE is a fraction of the total energy density at its peak, so it
        # must satisfy 0 ≤ fEDE < 1.  The formula in background._setup_ede()
        # has (1 - fEDE) in the denominator, which diverges at fEDE = 1.
        if not (0. <= self.fEDE < 1.):
            raise ValueError(
                f"fEDE={self.fEDE!r} is out of range: must satisfy 0 ≤ fEDE < 1 "
                "(fEDE is the EDE fraction of the total energy density at its peak)."
            )

        # custom_background: force instantaneous decoupling and no spectral
        # distortions (the custom-background driver does not load NEVO tables
        # and uses the analytic T_ν(T_γ) formula instead).  Must be checked
        # before the external_scale_factor / spectral_distortions validations
        # below so those see the corrected flag values.
        if self.custom_background is not None:
            if self.external_scale_factor:
                raise ValueError(
                    "custom_background and external_scale_factor are mutually "
                    "exclusive: external_scale_factor reads a(T_γ) from the "
                    "NEVO table, which is not loaded in custom_background mode."
                )
            forced = []
            if self.incomplete_decoupling:
                forced.append("incomplete_decoupling=False")
                object.__setattr__(self, 'incomplete_decoupling', False)
            if self.spectral_distortions:
                forced.append("spectral_distortions=False")
                object.__setattr__(self, 'spectral_distortions', False)
            if forced:
                warnings.warn(
                    f"custom_background: forcing {', '.join(forced)} "
                    "(custom-background mode uses instantaneous-decoupling "
                    "weak rates; spectral distortions are not supported).",
                    stacklevel=2,
                )

        # data_dir/user_nuclear_dir: eagerly validate (mirrors the nevo_file
        # pattern above) so a typo'd override path fails fast at construction
        # time rather than surfacing as a confusing "network not found" later.
        for _field in ("data_dir", "user_nuclear_dir"):
            _value = getattr(self, _field)
            if _value is not None and not os.path.isdir(_value):
                raise ValueError(f"{_field}={_value!r} is not an existing directory")
            if _value is not None:
                self._init_messages.append(_rates_overlay_notice(_field, _value))

        if self.network != "small":
            path = self.resolve_rates_path("nuclear", "networks", f"{self.network}.txt")
            if not os.path.exists(path):
                searched = []
                if self.user_nuclear_dir is not None:
                    searched.extend(_overlay_candidates(
                        self.user_nuclear_dir,
                        os.path.join("nuclear", "networks", f"{self.network}.txt"),
                    ))
                searched.append(path)
                raise ValueError(
                    f"network must be 'small' or name an existing file in "
                    f"data/nuclear/networks; missing {path!r}"
                    + (f" (searched: {', '.join(repr(p) for p in searched)})" if searched else "")
                )

        # Default every reaction of the *configured* network (self.network,
        # finalised by the overrides above) to p_<rxn>=0 / delta_<rxn>=0,
        # i.e. "no rate variation".  Use setdefault so any p_<rxn>/delta_<rxn>
        # override already applied above is not clobbered.
        from .network_data import load_reaction_names, reaction_category
        reactions_with_tables = load_reaction_names(self, self.network)
        # Each entry is "bare_name" or "bare_name, filename.txt"; only the
        # bare reaction name is used as the p_<rxn>/delta_<rxn> key.
        # amax (now meaningful for any network, not just "large") must be
        # applied here too, so p_rxn/delta_rxn don't carry stale keys for
        # reactions load_network would have dropped.
        valid_rxns = set()
        for entry in reactions_with_tables:
            bare = re.split(r'[, ]+', entry, maxsplit=1)[0]
            if self.amax is not None and reaction_category(bare) > self.amax:
                continue
            valid_rxns.add(bare)
        for rxn in valid_rxns:
            self.p_rxn.setdefault(rxn, 0.0)
            self.delta_rxn.setdefault(rxn, 0.0)

        # Catch p_<rxn>/delta_<rxn> typos in the constructor params. This
        # has to happen here rather than in __setattr__ at the time those
        # overrides were applied (above): the override loop runs before this
        # network's reaction list is known (self.network may itself be one of
        # the overrides), and by the time we reach this point __setattr__'s
        # routing has already inserted the (possibly bogus) key into
        # self.p_rxn/self.delta_rxn -- so we must check against
        # ``valid_rxns`` computed just above, not against those dicts.
        for key in user_keys:
            if key.startswith('p_') and key[2:] not in valid_rxns:
                warnings.warn(
                    f"PRIMATConfig: {key!r} does not match any reaction in "
                    f"network {self.network!r}; it has no effect on the run.",
                    stacklevel=2,
                )
            elif key.startswith('delta_') and key[6:] not in valid_rxns:
                warnings.warn(
                    f"PRIMATConfig: {key!r} does not match any reaction in "
                    f"network {self.network!r}; it has no effect on the run.",
                    stacklevel=2,
                )

        # Detect optional libraries for flags not explicitly set by the caller.
        # Messages are stored for deferred printing (after the banner).

        if self.numba_installed:
            try:
                import numba  # noqa: F401
                self._init_messages.append('[init]  numba detected: using it for JIT compilation.')
            except ImportError:
                self.numba_installed = False
                self._init_messages.append('[init]  numba not detected: running without JIT compilation.')

        # Validate amax: must be None or a positive integer.
        if self.amax is not None:
            if not (isinstance(self.amax, int) and self.amax >= 1):
                raise ValueError(
                    f"amax must be None or a positive integer (got {self.amax!r})."
                )

        # Validate any custom NEVO table overrides: check the file exists and
        # has the column count expected by neutrino_history.NEVOTable, so a
        # typo or malformed file is caught here with a clear message rather
        # than as a confusing shape mismatch deep inside an interpolant.
        from .neutrino_history import resolve_nevo_path
        if self.nevo_file is not None:
            path = resolve_nevo_path(self, self.nevo_file, "")
            if not os.path.exists(path):
                raise ValueError(f"nevo_file={self.nevo_file!r} not found "
                                  f"(resolved to {path!r})")
            ncols = np.loadtxt(path, delimiter=',', max_rows=1).size
            if ncols not in (6, 7):
                raise ValueError(f"nevo_file={self.nevo_file!r} ({path!r}) has "
                                  f"{ncols} columns; expected 6 or 7 (the NEVO "
                                  f"x,z,Tnue,Tnumu,Tnutau,N[,extra] thermo table)")

        n_grid_nodes = 80  # default NEVOGrid.csv length, used if nevo_spectral_file is not overridden
        if self.nevo_spectral_file is not None:
            path = resolve_nevo_path(self, self.nevo_spectral_file, "")
            if not os.path.exists(path):
                raise ValueError(f"nevo_spectral_file={self.nevo_spectral_file!r} "
                                  f"not found (resolved to {path!r})")
            ncols = np.loadtxt(path, delimiter=',', max_rows=1).size
            if ncols <= 6:
                raise ValueError(f"nevo_spectral_file={self.nevo_spectral_file!r} "
                                  f"({path!r}) has {ncols} columns; expected "
                                  f"6 thermo columns plus at least one spectral "
                                  f"column (86 in the shipped tables)")
            n_grid_nodes = ncols - 6

        if self.nevo_grid_file is not None:
            path = resolve_nevo_path(self, self.nevo_grid_file, "")
            if not os.path.exists(path):
                raise ValueError(f"nevo_grid_file={self.nevo_grid_file!r} not "
                                  f"found (resolved to {path!r})")
            n_nodes = np.loadtxt(path, delimiter=',').size
            if n_nodes != n_grid_nodes:
                raise ValueError(f"nevo_grid_file={self.nevo_grid_file!r} "
                                  f"({path!r}) has {n_nodes} nodes; expected "
                                  f"{n_grid_nodes} to match the spectral "
                                  f"table's {n_grid_nodes} y-columns")

        # Validate nevo_file_prefix: when not the shipped default, check that
        # the *derived* default filenames it implies exist and have the right
        # shape -- mirrors the nevo_file/nevo_spectral_file checks above, but
        # only for the files that aren't already overridden individually.
        if self.nevo_file_prefix != "NEVOPRIMAT" and self.incomplete_decoupling:
            prefix = self.nevo_file_prefix
            suffix = "" if self.QED_corrections else "_NoQED"

            if self.nevo_file is None:
                fname = f"{prefix}{suffix}_col_1_7.csv"
                path = resolve_nevo_path(self, None, fname)
                if not os.path.exists(path):
                    raise ValueError(f"nevo_file_prefix={prefix!r}: derived "
                                      f"thermo file {fname!r} not found "
                                      f"(resolved to {path!r})")
                ncols = np.loadtxt(path, delimiter=',', max_rows=1).size
                if ncols not in (6, 7):
                    raise ValueError(f"nevo_file_prefix={prefix!r}: "
                                      f"{fname!r} has {ncols} columns; "
                                      f"expected 6 or 7")

            if (self.spectral_distortions and not self.analytic_distortions
                    and self.nevo_spectral_file is None):
                fname = f"{prefix}{suffix}.csv"
                path = resolve_nevo_path(self, None, fname)
                if not os.path.exists(path):
                    raise ValueError(f"nevo_file_prefix={prefix!r}: derived "
                                      f"spectral file {fname!r} not found "
                                      f"(resolved to {path!r})")
                ncols = np.loadtxt(path, delimiter=',', max_rows=1).size
                if ncols <= 6:
                    raise ValueError(f"nevo_file_prefix={prefix!r}: "
                                      f"{fname!r} has {ncols} columns; "
                                      f"expected > 6")

        # external_scale_factor reads a(T) directly from the NEVO table's x
        # column (NEVOTable.x_of_Tg), so it requires the NEVO table to be
        # loaded in the first place.
        if self.external_scale_factor and not self.incomplete_decoupling:
            raise ValueError(
                "external_scale_factor=True requires incomplete_decoupling=True "
                "(a(T) is read from the NEVO table, which is only loaded by "
                "NEVOTable)."
            )

        # Validate spectral-distortion flag combination.
        if self.spectral_distortions:
            if self.analytic_distortions:
                if self.incomplete_decoupling:
                    raise ValueError(
                        "spectral_distortions=True with analytic_distortions=True "
                        "requires instantaneous decoupling (incomplete_decoupling=False)."
                    )
            else:
                if not self.incomplete_decoupling:
                    raise ValueError(
                        "spectral_distortions=True with analytic_distortions=False "
                        "requires incomplete_decoupling=True (the full NEVO spectrum "
                        "file is only available in the non-instantaneous decoupling mode)."
                    )

        # Derived cosmological quantity (depends on Omegabh2)
        self._update_derived()

    def __getattr__(self, name: str):
        """Dynamic lookup for nuclear rate variations p_* and delta_*."""
        if name.startswith("p_"):
            return object.__getattribute__(self, 'p_rxn').get(name[2:], 0.0)
        if name.startswith("delta_"):
            return object.__getattribute__(self, 'delta_rxn').get(name[6:], 0.0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        """Dynamic routing for nuclear rate variations p_* and delta_*."""
        if name.startswith("p_"):
            object.__getattribute__(self, 'p_rxn')[name[2:]] = float(value)
        elif name.startswith("delta_"):
            object.__getattribute__(self, 'delta_rxn')[name[6:]] = float(value)
        else:
            if name in _PATH_PARAMS:
                # Normalize "~" immediately so both direct assignment and
                # --set KEY=VALUE route through the same resolved path.
                value = _expanduser_path(value)
            object.__setattr__(self, name, value)
            if name == "Omegabh2":
                self._update_derived()

    # Omegabh2 is exposed as a property so that the derived baryon-to-photon
    # ratio eta0b is recomputed automatically whenever it is reassigned (by
    # attribute, by the constructor loop, or via __setitem__).
    @property
    def Omegabh2(self) -> float:
        return self._Omegabh2

    @Omegabh2.setter
    def Omegabh2(self, value: float):
        self._Omegabh2 = value
        self._update_derived()

    def _update_derived(self):
        """Recompute quantities that depend on mutable parameters."""
        self.Omegabh2_to_eta0b = (self.rhocOverh2 / self.n0CMB) / (self.ma / self.maOvermB)
        self.eta0b = self.Omegabh2_to_eta0b * self._Omegabh2

    # Convenience: allow dict-style access for backwards compat if needed
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self._update_derived()

    # Class-level storage to avoid AttributeError if accessed before init
    Nuclides = {}
    NuclExcessMass = {}
    NuclSpin = {}

    def _load_nuclide_data(self):
        """Load mass excesses, spins, and (N, Z) from nuclides.csv."""
        import csv
        path = os.path.join(self._resolved_data_dir, "csv", "nuclides.csv")
        
        self.Nuclides = {}
        self.NuclExcessMass = {}
        self.NuclSpin = {}
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name']
                self.Nuclides[name] = [int(row['N']), int(row['Z'])]
                self.NuclExcessMass[name] = float(row['mass_excess_keV'])
                self.NuclSpin[name] = float(row['spin'])

    @property
    def _pkg_data_dir(self) -> str:
        """Package-shipped data root (``primat/data/``, contains NEVO/, nuclear/, weak/, plasma/, csv/).

        This is the fixed fallback used when ``data_dir`` param is ``None``.
        It always points to the installed package's own data tree regardless
        of any user override.
        """
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    @property
    def _resolved_data_dir(self) -> str:
        """Resolved data root: the ``data_dir`` param when set, otherwise ``primat/data/``.

        Use this everywhere a data-root path is needed instead of the old
        ``cfg.data_dir + "/data"`` idiom.  Output paths are still resolved
        against the current working directory (see PRIMAT._write_time_evolution
        / _write_final_result).
        """
        return self.data_dir if self.data_dir else self._pkg_data_dir

    def resolve_rates_path(self, *parts: str) -> str:
        """Resolve a path inside the nuclear data tree through the overlay chain.

        Used by every caller that needs a nuclear network file or rate-table
        file, so a user's ``user_nuclear_dir`` additive overlay (or a full
        ``data_dir`` takeover — see those fields in ``DEFAULT_PARAMS``) is
        honoured without touching the installed ``primat`` package.

        Lookup order (first existing path wins):
          1. ``self.user_nuclear_dir`` (additive nuclear overlay), if set.
          2. ``self._resolved_data_dir`` (either the user-supplied ``data_dir``
             or the shipped ``primat/data/`` tree — always tried last so
             ``small``/``large`` and the default rate tables are never
             unreachable just because ``user_nuclear_dir`` is also configured).

        If the relative path is not found under any candidate base, the
        resolved-default path is returned anyway (not found), so callers get
        a "missing file" error that points at the expected default location
        rather than at whichever overlay happened to be checked last.

        Args:
            *parts: path components relative to a nuclear data root, e.g.
                ``"nuclear", "networks", "large.txt"``.

        Returns:
            str: an absolute path (existing, if found under any candidate
            base; otherwise the resolved-default path, for use in error
            messages).

        Example:
            >>> cfg.resolve_rates_path("nuclear", "networks", "large.txt")
            '/.../primat/data/nuclear/networks/large.txt'
        """
        relpath = os.path.join(*parts) if parts else ""
        bases = []
        if self.user_nuclear_dir:
            bases.append(self.user_nuclear_dir)
        bases.append(self._resolved_data_dir)  # shipped (or overridden) default, always last
        for base in bases:
            if relpath:
                for candidate in _overlay_candidates(base, relpath):
                    if os.path.exists(candidate):
                        return candidate
            else:
                if os.path.exists(base):
                    return base
        return os.path.join(bases[-1], relpath) if relpath else bases[-1]
