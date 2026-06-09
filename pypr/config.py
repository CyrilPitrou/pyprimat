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
from functools import cached_property
import numpy as np
from scipy.special import zeta

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
    #Legacy should be removed
    "analytic_entropy_derivative": True, # Use analytic derivative of entropy for plasma thermodynamics. If False, the code will compute the derivative numerically at runtime (using numdifftools if available, or finite differences as a fallback).  This is much slower, but allows testing the impact of numerical vs. analytic derivatives on the final BBN results.
    "numdiff_installed":               True,  # will be re-checked at runtime and used only if analytic_entropy_derivative is False.

    # ---- physics settings ------------------------------------------------
    # ---- neutrino decoupling ----------------------
    "incomplete_decoupling":      True, # Whether to use non-instantaneous (incomplete) neutrino decoupling.
    # True (default) = full treatment: neutrino temperatures are read from the pre-computed NEVO table and differ slightly from the instantaneous-decoupling prediction due to partial reheating by e+e- annihilations.  
    # False = instantaneous (complete) decoupling approximation: the three neutrino flavour temperatures are all set equal to the instantaneous-decoupling value derived from EM entropy conservation, Tν/Tγ = (4/11)^(1/3), and the neutrino energy density is fixed to the free-gas value with that temperature ratio.  
    # Note: the NEVO table was itself computed with incomplete decoupling, so the combination incomplete_decoupling=False with QED_corrections=True is physically inconsistent and should be used only for diagnostic purposes.
    
    # ---- electromagnetic plasma -------------------
    "QED_corrections":            True,  # Whether to include QED interaction corrections to the EM plasma equation of state.
    "tabulate_electron_thermo":   True,  # pre-tabulate rho_e/p_e and derivatives once, then interpolate (faster background solve)
    "n_electron_table":           2000,  # number of log-spaced grid points for the electron-thermo tables
    "recompute_electron_thermo":  False, # force recomputation of the electron-thermo table even if a cache file exists
    "recompute_qed_corrections":  False, # True: always compute analytically and overwrite rates/plasma/QED_*.txt; False: load from files if present, otherwise compute on the fly without saving

    # ---- spectral distortions ---------------------
    "spectral_distortions":       False, #Spectral distortions: corrections to n<->p weak rates from deviations of the neutrino phase-space distribution from a perfect Fermi-Dirac.
    # Two sub-modes (selected by analytic_distortions):
    #
    #   analytic_distortions=False (default): uses the full NEVO spectrum file
    #     (86-column version, not _col_1_7).  Requires incomplete_decoupling=True.
    #     The distortion is read directly from the NEVO table columns 6–85.
    #
    #   analytic_distortions=True: parameterises the distortion analytically as
    #     a μ-type (chemical-potential shift) and/or y-type (SZ-like) distortion,
    #     controlled by delta_xi_nu and y_SZ.  Can be used with or without
    #     incomplete_decoupling.  Also adds the extra neutrino energy density
    #     ρ_νSD to the Friedmann equation via closed-form integrals.
    "analytic_distortions":       False,
    # δξ_ν: shift of the reduced chemical potential ξ = μ/T for the μ-type
    # distortion.  The neutrino distribution becomes
    #   f_ν(y) → 1/(e^{y-(ξ+δξ)}+1)  (from 1/(e^{y-ξ}+1))
    # For antineutrinos the chemical potential flips sign. The same chemical potential shift is applied to all three neutrino flavours (ν_e, ν_μ, ν_τ).  
    "delta_xi_nu":                0.,
    # YSZ: amplitude of the y-type (Sunyaev–Zel'dovich-like) distortion,
    #   δf^SZ(y) = (1/y²) d/dy(y⁴ df_FD/dy)
    # This is the leading-order spectral shape produced by heating a Fermi-Dirac.
    "y_SZ":                       0.,

    # ---- fundamental constants (overridable for sensitivity studies) --------
    "GN":                         6.70883e-45,   # Newton's constant [MeV^-2]

    # ---- background thermodynamics ----------------------------------------
    "T_start_cosmo_MeV":          40.0,
    "n_temperature_table":        2000,

    # ---- n <--> p weak rates ----------------------------------------------
    "compute_nTOp":          True, #If False they are read from previously stored files.
    "save_nTOp":             False, # If True, the computed n<->p rates are saved to files for inspection and reuse.
    "sampling_nTOp":              200,   # total points in the single n<->p rate grid
    "compute_nTOp_thermal":  False, # If True the thermal corrections to the weak rates are re-computed. But this is very slow, hence it is better to compute them only once and then read stored results
    "include_nTOp_thermal":  True,  # If True the thermal corrections are used in the rate computation.
    "save_nTOp_thermal":     False, #If True, the computed thermal n<->p rates are saved to files for inspection and reuse.
    "sampling_nTOp_thermal":      100,
    "nTOp_Born_approximation":              False, #If True the crude Born rate is used (off by a few percents, hence should be used only for debugging or fair comparison with other codes). 
    "tau_n_flag":                 True, # Use neutron lifetime to normalize weak rates (instead of absolute normalization from GF, Vud, gA, etc.)
    "tau_n":                      878.4,  # neutron lifetime [s]; overrides the class-level constant when tau_n_flag=True
    "std_tau_n":                  0.5,    # 1σ uncertainty on tau_n [s], used for MC sampling
        
    # ---- finite-temperature weak-rate radiative corrections ----------------
    # Accuracy knobs for the thermal n<->p radiative correction integral, only
    # used when compute_nTOp_thermal=True (see weak_rates.nTOp_rate_).
    # The integral is evaluated with the `vegas` Monte-Carlo library when
    # available; otherwise it falls back to scipy.integrate.dblquad.
    "vegas_n_eval":               20000,   # vegas: evaluations per iteration
    "vegas_n_itn":                20,      # vegas: number of iterations
    "epsrel_thermal":             1.e-2,   # dblquad fallback: relative tolerance
    
    # ---- Output options ------------------------------------------------------
    # Outputs time evolution of all quantities.
    "output_time_evolution":      False,
    "output_rates_time_evolution": False, #whether to include or not the nuclear rates evolution in the output time evolution file. This is only useful if you want to inspect the rates evolution, otherwise it is better to set it to False to save disk space and speed up the code.
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
    "nuclear_qed_corrections":    False,

    # ---- cosmological inputs ----------------------------------------------
    "Omegabh2":                   0.022425,
    "DeltaNeff":                  0.,
    "munuOverTnu":                0., #Reduced chemical potential of neutrinos (same for all flavours, ν_e, ν_μ, ν_τ).  The neutrino distribution becomes f_ν(y) → 1/(e^{y-(ξ+δξ)}+1)  (from 1/(e^{y-ξ}+1)). 
    # Note: the combination munuOverTnu != 0 with incomplete_decoupling=False is physically inconsistent since NEVO tables were obatined assuming it vanishes.
    # To explore such physics it is preferable to work with full decoupling of neutrinos (incomplete_decoupling=False).
     
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
        cfg = PyPRConfig({"Omegabh2": 0.022, "is_small": False})

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
    # Class-level physical constants (identical for every instance)
    # ------------------------------------------------------------------

    # CGS base units (dimensionless by convention)
    Kelvin: float = 1.
    second: float = 1.
    cm:     float = 1.
    gram:   float = 1.

    @cached_property
    def erg(self) -> float:
        return self.gram * self.cm**2 / self.second

    # Fundamental constants (PDG)
    kB:     float = 1.380649e-16          # Boltzmann [erg/K]
    clight: float = 2.99792458e+10        # speed of light [cm/s]
    hbar:   float = 6.62607015 / (2 * np.pi) * 1e-27  # Planck [erg·s]
    Mpc:    float = 3.08567758149e+24     # [cm]
    MeV:    float = 1.602176634e-6        # [erg]
    keV:    float = 1.602176634e-9        # [erg]
    # Electroweak sector (PDG)
    alphaem: float = 1. / 137.035999084
    GF:      float = 1.1663787e-5 * 1.e-6   # [MeV^-2]
    mZ:      float = 91.1876e3               # [MeV]
    # Fermion masses [MeV]
    me: float = 0.51099895
    mn: float = 939.56542052
    mp: float = 938.27208816
    # CMB / cosmological fixed quantities
    T0CMB: float = 2.7255                  # photon temperature today [K]

    # Conversion factors (MeV-based natural units <-> CGS)
    @cached_property
    def MeV_to_Kelvin(self) -> float:
        return self.MeV / self.kB

    @cached_property
    def MeV_to_secm1(self) -> float:
        return self.MeV / self.hbar

    @cached_property
    def MeV_to_g(self) -> float:
        return self.MeV / self.clight**2

    @cached_property
    def MeV_to_cmm1(self) -> float:
        return self.MeV / (self.hbar * self.clight)

    @cached_property
    def MeV4_to_gcmm3(self) -> float:
        return self.MeV_to_g * self.MeV_to_cmm1**3

    # Temperature eras [K]
    @property
    def T_start_cosmo(self) -> float:
        return self.T_start_cosmo_MeV * self.MeV_to_Kelvin

    @cached_property
    def T_start(self) -> float:
        return 10.0 * self.MeV_to_Kelvin

    @cached_property
    def T_weak(self) -> float:
        return 1.0 * self.MeV_to_Kelvin

    @cached_property
    def T_nucl(self) -> float:
        return 0.11 * self.MeV_to_Kelvin

    @cached_property
    def T_end(self) -> float:
        return 1.e-3 * self.MeV_to_Kelvin

    @cached_property
    def sW2(self) -> float:
        return 0.5 * (1. - np.sqrt(1. - 2.*np.sqrt(2.)*np.pi*self.alphaem / (self.GF * self.mZ**2)))

    @cached_property
    def geL(self) -> float:
        return 0.5 + self.sW2

    @cached_property
    def geR(self) -> float:
        return self.sW2

    @cached_property
    def gmuL(self) -> float:
        return -0.5 + self.sW2

    @cached_property
    def gmuR(self) -> float:
        return self.sW2

    # Gravity: GN [MeV^-2] is overridable, so it lives in DEFAULT_PARAMS only.
    @property
    def Mpl(self) -> float:
        return 1. / np.sqrt(self.GN)

    # Weak rate nuclear structure constants (PDG).
    # tau_n [s] is overridable, so it lives in DEFAULT_PARAMS only.
    gA:          float = 1.2756
    kappa_p:     float = 2.79284734463 - 1.
    kappa_n:     float = -1.91304273
    Vud:         float = 0.9738
    radproton:   float = 0.8409e-13        # proton charge radius [cm]

    @cached_property
    def deltakappa(self) -> float:
        return self.kappa_p - self.kappa_n


    @cached_property
    def s0bar(self) -> float:
        """Dimensionless prefactor in the photon entropy density: s_γ = s̄ T³.

        For a relativistic boson gas with g=2 (photon) the entropy density is
            s_γ = (2π²/45) × 2 × T³ = (4π²/45) T³  [Phys. Rep. Eq. 24].
        s̄ ≡ 4π²/45 is factored out so that s_γ(T) = s̄ × T³ wherever T³
        appears, making the scaling explicit.
        """
        return 4. * np.pi**2 / 45.

    @cached_property
    def s0CMB(self) -> float:
        """Present-day CMB photon entropy density  [MeV³].

        s0CMB = s̄ × T0CMB³,  the entropy density of the photon background
        today (T0CMB in Kelvin, converted to MeV by /MeV_to_Kelvin).
        Used to track the comoving entropy and define the scale factor a(T).
        """
        return self.s0bar * (self.T0CMB / self.MeV_to_Kelvin)**3  # [MeV^3]

    @cached_property
    def n0CMB(self) -> float:
        """Present-day CMB photon number density  [MeV³].

        n_γ = (2ζ(3)/π²) T³  for a bosonic gas with g=2 (photon).
        n0CMB = n_γ(T0CMB), the present CMB photon number density in MeV³.
        Used together with eta0b = n_B/n_γ to set the comoving baryon density.
        """
        return (2. * zeta(3)) / np.pi**2 * (self.T0CMB / self.MeV_to_Kelvin)**3  # [MeV^3]

    # Atomic masses
    ma:         float = 931.494061         # 1 u.m.a. [MeV]
    He4Overma:  float = 4.0026032541
    HOverma:    float = 1.00782503223

    @cached_property
    def mB(self) -> float:
        percentHe = 24.7 / 100.
        return ((1. - percentHe) * self.HOverma + percentHe * self.He4Overma / 4.) * self.ma

    @cached_property
    def maOvermB(self) -> float:
        return self.ma / self.mB

    @property
    def HubbleOverh(self) -> float:
        return 100. * (1.e+5 * self.cm * self.MeV_to_cmm1) / (self.second * self.MeV_to_secm1) / (self.Mpc * self.MeV_to_cmm1)

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

        # Initialize nuclear rate variation dicts
        object.__setattr__(self, "p_rxn", {})
        object.__setattr__(self, "NP_delta_rxn", {})
        from .nuclear import _REACTIONS_MEDIUM
        for rxn in _REACTIONS_MEDIUM:
            self.p_rxn[rxn] = 0.0
            self.NP_delta_rxn[rxn] = 0.0

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
            path = os.path.join(self.working_dir, "rates", "nuclear",
                                "networks", f"{self.network}.txt")
            if not os.path.exists(path):
                raise ValueError(
                    f"network must be 'small' or name an existing file in "
                    f"rates/nuclear/networks; missing {path!r}"
                )
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

        if self.numdiff_installed and (not self.analytic_entropy_derivative):
            try:
                import numdifftools  # noqa: F401
                self._init_messages.append('[init]  numdifftools detected: using it for numerical derivative of entropy.')
            except ImportError:
                self.numdiff_installed = False
                self._init_messages.append('[init]  numdiff not detected: using finite differences for numerical derivatives of entropy.')

        # Validate amax: must be None or an integer > 7.
        if self.amax is not None:
            if not (isinstance(self.amax, int) and self.amax > 7):
                raise ValueError(
                    f"amax must be None or an integer > 7 (got {self.amax!r}); "
                    "values ≤ 7 are the domain of the small/medium networks."
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
        path = os.path.join(self.working_dir, "rates", "nuclear", "data", "nuclides.csv")
        
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

    # Path helper: working dir is two levels above this file (package root)
    @property
    def working_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
