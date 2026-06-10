# -*- coding: utf-8 -*-
"""
constants.py
============
Frozen physical constants and unit-conversion factors for PyPRIMAT.

This module is the single source of truth for the *fixed* physical
constants used throughout the code: PDG masses and couplings, the
CGS-vs-natural-units conversion factors, and the handful of purely
numerical quantities (e.g. the high-T entropy coefficient ``s0bar``,
the Weinberg angle ``sW2``) that follow from them.  None of the values
here depend on any run-time configuration choice — quantities that *do*
(``cfg.GN``, ``cfg.Omegabh2``, ``cfg.tau_n``, ``cfg.T_start_cosmo_MeV``, …)
remain user-settable knobs on :class:`pyprimat.config.PyPRConfig`.

``CONST`` is a single frozen, module-level instance.  ``PyPRConfig``
re-exposes every field/property of ``CONST`` as a class attribute (e.g.
``cfg.me``, ``cfg.MeV_to_Kelvin``) so existing physics code is unaffected.
New code may instead import ``CONST`` directly:

    >>> from pyprimat.constants import CONST
    >>> CONST.me
    0.51099895
    >>> CONST.MeV_to_Kelvin   # 1 MeV in Kelvin
    11604518121.5...

Units convention
-----------------
All "CGS" quantities (``Kelvin``, ``second``, ``cm``, ``gram``) are set to
1: lengths/times/temperatures/masses are expressed in natural (MeV-based)
units throughout the code, and the ``MeV_to_*`` factors below convert *to*
CGS only where needed (e.g. for printing or comparison with CGS-valued
inputs such as ``T0CMB`` [K]).
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import zeta

__all__ = ['Constants', 'CONST']


@dataclass(frozen=True)
class Constants:
    """Fixed physical constants and unit-conversion factors (immutable).

    Grouped as:

    - CGS base units (all set to 1; natural units throughout)
    - Fundamental constants (PDG): kB, clight, hbar, Mpc, MeV, keV
    - Electroweak sector (PDG): alphaem, GF, mZ
    - Fermion masses [MeV] (PDG): me, mn, mp
    - CMB: T0CMB
    - Weak-rate nuclear-structure constants (PDG): gA, kappa_p, kappa_n,
      Vud, radproton
    - Atomic masses: ma, He4Overma, HOverma

    plus derived quantities (unit-conversion factors, fixed temperature
    eras, electroweak mixing-angle couplings, and the high-T plasma
    entropy/number-density normalisations) exposed as read-only
    properties computed from the fields above.
    """

    # ---- CGS base units (dimensionless by convention: natural units) ----
    Kelvin: float = 1.
    second: float = 1.
    cm:     float = 1.
    gram:   float = 1.

    # ---- Fundamental constants (PDG) ----
    kB:     float = 1.380649e-16          # Boltzmann constant [erg/K]
    clight: float = 2.99792458e+10        # speed of light [cm/s]
    hbar:   float = 6.62607015 / (2 * np.pi) * 1e-27  # Planck constant [erg s]
    Mpc:    float = 3.08567758149e+24     # megaparsec [cm]
    MeV:    float = 1.602176634e-6        # 1 MeV [erg]
    keV:    float = 1.602176634e-9        # 1 keV [erg]

    # ---- Electroweak sector (PDG) ----
    alphaem: float = 1. / 137.035999084   # fine-structure constant
    GF:      float = 1.1663787e-5 * 1.e-6 # Fermi constant [MeV^-2]
    mZ:      float = 91.1876e3            # Z boson mass [MeV]

    # ---- Fermion masses [MeV] (PDG) ----
    me: float = 0.51099895
    mn: float = 939.56542052
    mp: float = 938.27208816

    # ---- CMB ----
    T0CMB: float = 2.7255                 # photon temperature today [K]

    # ---- Weak-rate nuclear-structure constants (PDG) ----
    gA:        float = 1.2756              # nucleon axial coupling
    kappa_p:   float = 2.79284734463 - 1.  # proton anomalous magnetic moment
    kappa_n:   float = -1.91304273         # neutron anomalous magnetic moment
    Vud:       float = 0.9738              # CKM matrix element |V_ud|
    radproton: float = 0.8409e-13          # proton charge radius [cm]

    # ---- Atomic masses ----
    ma:        float = 931.494061          # 1 unified atomic mass unit [MeV]
    He4Overma: float = 4.0026032541        # M(He4) / u
    HOverma:   float = 1.00782503223       # M(H) / u

    # ------------------------------------------------------------------
    # Derived quantities (pure functions of the constants above)
    # ------------------------------------------------------------------

    @property
    def erg(self) -> float:
        """1 erg in natural units (= gram cm^2 / second^2 with all = 1)."""
        return self.gram * self.cm**2 / self.second

    @property
    def MeV_to_Kelvin(self) -> float:
        """Conversion factor: 1 MeV / kB, in Kelvin."""
        return self.MeV / self.kB

    @property
    def MeV_to_secm1(self) -> float:
        """Conversion factor: 1 MeV / hbar, in s^-1."""
        return self.MeV / self.hbar

    @property
    def MeV_to_g(self) -> float:
        """Conversion factor: 1 MeV / c^2, in g."""
        return self.MeV / self.clight**2

    @property
    def MeV_to_cmm1(self) -> float:
        """Conversion factor: 1 MeV / (hbar c), in cm^-1."""
        return self.MeV / (self.hbar * self.clight)

    @property
    def MeV4_to_gcmm3(self) -> float:
        """Conversion factor for an energy density [MeV^4] to a mass density [g/cm^3]."""
        return self.MeV_to_g * self.MeV_to_cmm1**3

    # ---- Fixed temperature eras [MeV, converted to Kelvin] ----
    @property
    def T_start(self) -> float:
        return 10.0 * self.MeV_to_Kelvin

    @property
    def T_weak(self) -> float:
        return 1.0 * self.MeV_to_Kelvin

    @property
    def T_nucl(self) -> float:
        return 0.11 * self.MeV_to_Kelvin

    @property
    def T_end(self) -> float:
        return 1.e-3 * self.MeV_to_Kelvin

    # ---- Electroweak mixing angle and effective couplings ----
    @property
    def sW2(self) -> float:
        """sin^2(theta_W), from GF, mZ, alphaem (on-shell relation)."""
        return 0.5 * (1. - np.sqrt(1. - 2.*np.sqrt(2.)*np.pi*self.alphaem
                                    / (self.GF * self.mZ**2)))

    @property
    def geL(self) -> float:
        return 0.5 + self.sW2

    @property
    def geR(self) -> float:
        return self.sW2

    @property
    def gmuL(self) -> float:
        return -0.5 + self.sW2

    @property
    def gmuR(self) -> float:
        return self.sW2

    @property
    def deltakappa(self) -> float:
        return self.kappa_p - self.kappa_n

    # ---- High-T plasma entropy/number-density normalisations ----
    @property
    def s0bar(self) -> float:
        """Dimensionless prefactor in the photon entropy density: s_gamma = s0bar T^3.

        For a relativistic boson gas with g=2 (photon) the entropy density is
            s_gamma = (2 pi^2/45) x 2 x T^3 = (4 pi^2/45) T^3  [Phys. Rep. Eq. 24].
        """
        return 4. * np.pi**2 / 45.

    @property
    def s0CMB(self) -> float:
        """Present-day CMB photon entropy density [MeV^3]."""
        return self.s0bar * (self.T0CMB / self.MeV_to_Kelvin)**3

    @property
    def n0CMB(self) -> float:
        """Present-day CMB photon number density [MeV^3].

        n_gamma = (2 zeta(3)/pi^2) T^3 for a bosonic gas with g=2 (photon).
        """
        return (2. * zeta(3)) / np.pi**2 * (self.T0CMB / self.MeV_to_Kelvin)**3

    # ---- Mean baryon mass (H + He4 mixture) ----
    @property
    def mB(self) -> float:
        """Mean baryon mass [MeV], for a 24.7% He4 mass-fraction mixture with H."""
        percentHe = 24.7 / 100.
        return ((1. - percentHe) * self.HOverma
                + percentHe * self.He4Overma / 4.) * self.ma

    @property
    def maOvermB(self) -> float:
        return self.ma / self.mB

    # ---- Hubble constant in natural units, per unit h ----
    @property
    def HubbleOverh(self) -> float:
        """H0 / h, converted to natural (MeV) units, in MeV.

        100 km/s/Mpc converted via the cm/s/Mpc -> MeV chain.
        """
        return (100. * (1.e+5 * self.cm * self.MeV_to_cmm1)
                / (self.second * self.MeV_to_secm1)
                / (self.Mpc * self.MeV_to_cmm1))


# Single shared instance: all fields/properties above are pure constants,
# so one frozen object suffices for the whole process.
CONST = Constants()
