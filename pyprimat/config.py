# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for PyPRIMAT.

Physical constants and derived unit conversions are *fixed* and computed once
here.  All run-time flags and cosmological/nuclear parameters are carried in a
``PyPRConfig`` instance and can be overridden by passing a parameter dictionary
to ``PyPRConfig(params)``.

No file I/O happens here.  Nuclear rate data are loaded separately in
``nuclear_data.py``.
"""

import os
import re
import numpy as np

from .constants import CONST

__all__ = ['DEFAULT_PARAMS', 'PyPRConfig']

# ---------------------------------------------------------------------------
# Default parameter values exposed as a plain dict so callers can inspect them
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # ---- general behaviour and numerical settings ------------------------------------------------
    "verbose":               False, #If you want the messages from the code to be printed, set this to True.  This is separate from the debug, which controls the printing of extra messages for debugging purposes.
    "debug":                 False, #If you want the debug messages to be printed, set this to True.  This is separate from the verbose, which controls the printing of general messages from the code.
    "numerical_precision":        1.e-7, # for finite differences (solve_ivp). 1e-6 should be enough.
    "numba_installed":                 True,  # will be re-checked at runtime. Allows just-in-time compilation for faster execution.

    # ---- physics settings ------------------------------------------------
    # ---- neutrino decoupling ----------------------
    "incomplete_decoupling":      True, # True: non-instantaneous neutrino decoupling, read from the pre-computed NEVO table.
    # False: instantaneous decoupling (Tnu/Tgamma fixed by EM entropy conservation; see neutrino_history.InstantaneousDecoupling).
    # incomplete_decoupling=False with spectral_distortions=True (NEVO-based) is physically inconsistent and rejected; see PyPRConfig validation.

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
    #   True: analytic mu-type + y-type (SZ) distortion controlled by
    #     delta_xi_nu/y_SZ, also contributing rho_nuSD to the Friedmann equation.
    "analytic_distortions":       False,
    "delta_xi_nu":                0., # Amplitude of the mu-type (chemical-potential shift) distortion, same for all three flavours; see neutrino_history.AnalyticDistortion.
    "y_SZ":                       0., # Amplitude of the y-type (Sunyaev-Zel'dovich-like) distortion; see neutrino_history.AnalyticDistortion.

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

    # ---- background mode ---------------------------------------------------
    "external_background":       False, # If True, read the scale factor a(T_gamma) directly
    # from the NEVO table's x column (a is proportional to x by the NEVO convention; see
    # NEUTRINOS.md) instead of solving the entropy-conservation ODE from the heating
    # function N_NEVO. t(a) is still obtained by Hubble integration (unchanged). Outside
    # the table's T range, both modes extrapolate assuming radiation domination
    # (a ~ 1/T, t ~ 1/T^2). Requires incomplete_decoupling=True.

    # ---- fundamental constants (overridable for sensitivity studies) --------
    "GN":                         6.70883e-45,   # Newton's constant [MeV^-2]

    # ---- background thermodynamics ----------------------------------------
    "T_start_cosmo_MeV":          40.0,
    "n_temperature_table":        2000,

    # ---- n <--> p weak rates ----------------------------------------------
    # rates/weak/nTOp_*.txt carry a fingerprint header recording the config
    # fields that affect their content; RecomputeWeakRates loads the cache
    # only if its fingerprint matches, and otherwise recomputes from scratch
    # (~2 s).  See weak_rates.RecomputeWeakRates for the full cache logic.
    "weak_rate_cache":            True,  # If False, never load the cache (always recompute); save_nTOp still controls whether the result is written back.
    "save_nTOp":                  False, # If True, the computed n<->p rates are saved to rates/weak/ with a fingerprint header (see weak_rates.RecomputeWeakRates for why this defaults to False).
    "sampling_nTOp":              200,   # total points in the single n<->p rate grid
    "include_nTOp_thermal":       True,  # If True the thermal corrections are used in the rate computation (see weak_rates._L_CCRTh_interpolants for the cache rules).
    "save_nTOp_thermal":          False, #If True, the computed thermal n<->p rates are saved to rates/weak/ with a fingerprint header.
    "sampling_nTOp_thermal":      100,
    "nTOp_Born_approximation":    False, #If True the crude Born rate is used (off by a few percents, hence should be used only for debugging or fair comparison with other codes). 
    "tau_n_flag":                 True, # Use neutron lifetime to normalize weak rates (instead of absolute normalization from GF, Vud, gA, etc.)
    "tau_n":                      878.4,  # neutron lifetime [s]; overrides the class-level constant when tau_n_flag=True
    "std_tau_n":                  0.5,    # 1σ uncertainty on tau_n [s], used for MC sampling
        
    # ---- finite-temperature weak-rate radiative corrections ----------------
    # Accuracy knobs for the thermal n<->p radiative correction integral, used
    # only when the thermal-correction cache must be recomputed (see
    # weak_rates._L_CCRTh_interpolants).  Evaluated with the `vegas`
    # Monte-Carlo library when available, else scipy.integrate.dblquad.
    "vegas_n_eval":               20000,   # vegas: evaluations per iteration
    "vegas_n_itn":                20,      # vegas: number of iterations
    "epsrel_thermal":             1.e-2,   # dblquad fallback: relative tolerance
    
    # ---- Output options ------------------------------------------------------
    # Writes a TSV (cfg.output_file) with the full time evolution of the
    # background and of every nuclide in the chosen network (8/12/~59 for
    # small/medium/large); see main.PyPR._write_time_evolution.
    "output_time_evolution":      False,
    "output_rates_time_evolution": False, #whether to include or not the nuclear rates evolution in the output time evolution file. This is only useful if you want to inspect the rates evolution, otherwise it is better to set it to False to save disk space and speed up the code. Ignored (with a printed note) for network="large", where per-reaction flux columns are omitted.
    "output_n_points":            500,
    "output_file":                "results/output_tables.tsv",
    # Two-column dump (nuclide name, final mass-fraction abundance Y) at the end of BBN. 
    "output_final_result":        False,
    "output_final_file":          "results/output_final.dat",
    

    # ---- nuclear network --------------------------------------------------
    "rate_interp_order":          "linear",   # interpolation of every nuclear rate table:
                                              # "linear" (fast np.interp) or "quadratic"/"cubic" (scipy interp1d)

    # Master grid onto which every nuclear reaction rate table is resampled at
    # load time.  This makes load_network grid-agnostic: tables generated with
    # different grids (e.g. via --keep-source-grid in convert_ac2024_rates.py,
    # or from external sources) are all resampled onto this common grid so that
    # fill_buffer's single searchsorted path remains valid.
    "rate_grid_npts":             500,        # number of points in the master T9 grid
    "rate_grid_T9_min":          1.0e-3,     # minimum T9 [GK] on the master grid
    "rate_grid_T9_max":          10.0,       # maximum T9 [GK] on the master grid

    # Network selector.  "small" is the built-in ORDER_SMALL network.  Any other
    # value loads rates/nuclear/networks/<network>.txt.
    "network":                    "small",

    # Maximum nuclide mass number A = N + Z to include when loading the large
    # network.  Reactions involving any nuclide with A > amax are dropped.
    # None = no filter (keep all reactions).  Must be an integer > 7 when set,
    # because A ≤ 7 is the light-element domain covered by small/medium.
    # Only effective for network="large"; silently ignored otherwise.
    # Example: {"network": "large", "amax": 20} keeps only A ≤ 20 nuclides.
    "amax":                       None,

    # Absolute solve_ivp tolerance for the large-network LT era.  The heavy
    # nuclides reach very small abundances, so this is tighter than the 1e-15
    # used for the small/medium LT era (which keep their validated tolerances).
    "atol_large_LT":              1.e-25,
    "rescale_nuclear_rates":            False, #Use to vary some rates with a uniform factor to explore their impact.

    # QED correction to select radiative-capture nuclear rates (Pitrou & Pospelov 2020).
    # Applies a T9-dependent multiplicative rescaling to the forward rate tables of
    # npTOdg, dpTOHe3g, tpTOag, taTOLi7g, He3aTOBe7g at load time.  When True the
    # corrected values become the new medians, so p_* and NP_delta_* variations
    # work relative to the QED-corrected central value.
    "nuclear_qed_corrections":    True,

    # ---- cosmological inputs ----------------------------------------------
    "Omegabh2":                   0.022425,
    "DeltaNeff":                  0.,
    "munuOverTnu":                0., # Reduced chemical potential xi = mu/T of neutrinos (same for all flavours, nu_e, nu_mu, nu_tau).
    # munuOverTnu != 0 with incomplete_decoupling=True is physically inconsistent (the NEVO table assumes it vanishes); use incomplete_decoupling=False to explore non-zero values.

    # ---- Early Dark Energy ------------------------------------------------
    "fEDE":                       0.,     # EDE fraction at peak; 0 = disabled
    "zcEDE":                      1.e8,   # redshift of EDE peak
    "wnEDE":                      1.,     # EDE equation-of-state parameter
}


class PyPRConfig:
    """
    Immutable physical constants + mutable run-time parameters.

    Usage::

        cfg = PyPRConfig()                    # all defaults
        cfg = PyPRConfig({"Omegabh2": 0.022, "network": "medium"})

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
    # ...) live in pyprimat.constants.Constants (see that module for
    # definitions, formulas and citations). They are re-exposed here as
    # plain class attributes so existing code (cfg.me, cfg.MeV_to_Kelvin,
    # cfg.s0bar, ...) is unaffected; new physics code may instead import
    # CONST directly from pyprimat.constants.
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
    T_end          = CONST.T_end
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

    # Gravity: GN [MeV^-2] is overridable, so it lives in DEFAULT_PARAMS only.
    # tau_n [s] is similarly overridable (DEFAULT_PARAMS), used by weak_rates.
    @property
    def Mpl(self) -> float:
        return 1. / np.sqrt(self.GN)

    @property
    def rhocOverh2(self) -> float:
        return 3. / (8. * np.pi * self.GN) * self.HubbleOverh**2  # [MeV^4/h^2]

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

        # Load nuclide data from CSV
        self._load_nuclide_data()

        # Initialize nuclear rate variation dicts as empty for now.  They are
        # populated with the configured network's reactions *after* user
        # overrides are applied below (self.network may itself be one of
        # those overrides), so that the per-reaction defaults match the
        # network actually requested by the caller.
        object.__setattr__(self, "p_rxn", {})
        object.__setattr__(self, "NP_delta_rxn", {})

        user_keys = set(params.keys()) if params else set()

        # Apply user overrides
        if params:
            known_prefixes = ('p_', 'NP_delta_')
            unknown = set()
            for key, value in params.items():
                if key in DEFAULT_PARAMS or any(key.startswith(p) for p in known_prefixes):
                    setattr(self, key, value)
                else:
                    unknown.add(key)
            
            if unknown:
                import warnings
                warnings.warn(
                    f"PyPRConfig: unknown parameter keys ignored: {unknown}",
                    stacklevel=2,
                )

        if self.network != "small":
            path = os.path.join(self.data_dir, "rates", "nuclear",
                                "networks", f"{self.network}.txt")
            if not os.path.exists(path):
                raise ValueError(
                    f"network must be 'small' or name an existing file in "
                    f"rates/nuclear/networks; missing {path!r}"
                )

        # Default every reaction of the *configured* network (self.network,
        # finalised by the overrides above) to p_<rxn>=0 / NP_delta_<rxn>=0,
        # i.e. "no rate variation".  Use setdefault so any p_<rxn>/NP_delta_<rxn>
        # override already applied above is not clobbered.
        from .nuclear import load_reaction_names
        reactions_with_tables = load_reaction_names(self, self.network)
        for entry in reactions_with_tables:
            # Each entry is "bare_name" or "bare_name, filename.txt"; only the
            # bare reaction name is used as the p_<rxn>/NP_delta_<rxn> key.
            rxn = re.split(r'[, ]+', entry, maxsplit=1)[0]
            self.p_rxn.setdefault(rxn, 0.0)
            self.NP_delta_rxn.setdefault(rxn, 0.0)

        # Detect optional libraries for flags not explicitly set by the caller.
        # Messages are stored for deferred printing (after the banner).
        self._init_messages = []

        if self.numba_installed:
            try:
                import numba  # noqa: F401
                self._init_messages.append('[init]  numba detected: using it for JIT compilation.')
            except ImportError:
                self.numba_installed = False
                self._init_messages.append('[init]  numba not detected: running without JIT compilation.')

        # Validate amax: must be None or an integer > 7.
        if self.amax is not None:
            if not (isinstance(self.amax, int) and self.amax > 7):
                raise ValueError(
                    f"amax must be None or an integer > 7 (got {self.amax!r}); "
                    "values ≤ 7 are the domain of the small/medium networks."
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

        # external_background reads a(T) directly from the NEVO table's x
        # column (NEVOTable.x_of_Tg), so it requires the NEVO table to be
        # loaded in the first place.
        if self.external_background and not self.incomplete_decoupling:
            raise ValueError(
                "external_background=True requires incomplete_decoupling=True "
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
        """Dynamic lookup for nuclear rate variations p_* and NP_delta_*."""
        if name.startswith("p_"):
            return object.__getattribute__(self, 'p_rxn').get(name[2:], 0.0)
        if name.startswith("NP_delta_"):
            return object.__getattribute__(self, 'NP_delta_rxn').get(name[9:], 0.0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        """Dynamic routing for nuclear rate variations p_* and NP_delta_*."""
        if name.startswith("p_"):
            object.__getattribute__(self, 'p_rxn')[name[2:]] = float(value)
        elif name.startswith("NP_delta_"):
            object.__getattribute__(self, 'NP_delta_rxn')[name[9:]] = float(value)
        else:
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
        path = os.path.join(self.data_dir, "rates", "nuclear", "data", "nuclides.csv")
        
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

    # Path helper: the pyprimat/ package directory, where rates/ lives.
    # Used only for *reading* package data; output paths are resolved against
    # the current working directory (see PyPR._write_time_evolution /
    # _write_final_result).
    @property
    def data_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))
