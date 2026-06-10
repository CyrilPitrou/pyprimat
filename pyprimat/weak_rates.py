# -*- coding: utf-8 -*-
"""
weak_rates.py — n ↔ p weak interaction rates for PyPRIMAT
==========================================================

Computes the seven n ↔ p reactions (neutron beta-decay and the six related
processes) that drive the neutron-to-proton ratio during BBN.

Physics
-------
All six weak processes (Phys. Rep. Eq. 68) are combined into a single forward
rate Γ_{n→p} and its reverse Γ_{p→n}.  In the Born (infinite-nucleon-mass)
approximation, the forward rate is (Phys. Rep. Eqs. 77–78):

    Γ_{n→p} = K ∫₀^∞ p² dp [χ₊(E) + χ₊(−E)]

with the chi function (Phys. Rep. Eq. 81):

    χ₊(E) = (E_ν)² g_ν(E_ν, ξ_ν) × g(-E, x)
    E_ν ≡ E − Δ/mₑ,   g(E) ≡ 1/(eˢ+1)

where E = ε/mₑ (dimensionless electron energy), x = mₑ/(kB Tγ),
ξ_ν = μ_ν/T_ν (neutrino degeneracy), and
K = 4G_W²(1+3g_A²) / (2π)³ (Phys. Rep. Eq. 83).

Corrections applied in sequence (sgnq = +1: n→p; sgnq = −1: p→n):

  _L_BORN   — Born approximation (Eqs. 77–78).  Used when cfg.nTOp_Born_approximation.
  _L_CCR    — Born integrand ×  R(b,y,E) [Coulomb × T=0 resummed radiative
              corrections, Phys. Rep. Eq. 101; R from Czarnecki et al. 2004].
  _L_FMCCR  — Finite-nucleon-mass correction × R × Coulomb (Phys. Rep. §III.G,
              Fokker-Planck expansion to first order in T/m_N).
  _L_CCRTh  — Finite-temperature radiative corrections (Phys. Rep. §III.H;
              Brown & Sawyer 2001, Eqs. 5.10–5.15).  Optional; uses vegas or
              scipy.dblquad; results can be cached to rates/weak/*.txt.
  _L_SD     — Spectral-distortion correction: replaces the Fermi-Dirac g_ν
              with the actual neutrino distribution f_ν from NEVO (optional;
              active when dFDneu_func is supplied).

Normalisation: K is obtained from the free neutron decay rate 1/τ_n rather
than from GF/Vud/gA directly (Phys. Rep. Eqs. 89–91), giving better precision.
The factor λ₀ encodes the phase-space integral over the neutron decay spectrum.

Fermi-Dirac helper table
------------------------
The FD_nu_eNpM functions compute Fermi-Dirac-related integrands appearing
in the finite-nucleon-mass Fokker-Planck expansion (Phys. Rep. App. B.2,
PRIMAT-Main.m ~line 1270).  Their arguments are always
(E, phi, x) with E = ε_ν/mₑ (dimensionless), phi = ξ_ν = μ_ν/T_ν,
x = mₑ/(kB T) (inverse temperature ratio):

  FD_nu3(E, phi, x)    — g_ν(xE, phi) = 1/(e^{xE−phi}+1)   [plain neutrino FD]
  FD2(E, x)            — g(xE) = 1/(e^{xE}+1)               [electron/positron FD]
  FD_nu_e2p0(E, phi, x) — E² × g_ν                           [FD × E²]
  FD_nu_e3p0(E, phi, x) — E³ × g_ν                           [FD × E³]
  FD_nu_e2p1(E, phi, x) — (∂/∂x)[x² g_ν] type combination   [1st FP order]
  FD_nu_e3p1(E, phi, x) — E-weighted 1st-order FP term
  FD_nu_e4p1(E, phi, x) — E²-weighted 1st-order FP term
  FD_nu_e2p2(E, phi, x) — 2nd-order FP combination × E⁰
  FD_nu_e3p2(E, phi, x) — 2nd-order FP combination × E
  FD_nu_e4p2(E, phi, x) — 2nd-order FP combination × E²

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095)
— cited below as "Phys. Rep." with equation numbers.
"""

import os
import numpy as np
from scipy.special import gamma as scipy_gamma, spence
from scipy.integrate import quad
from scipy.interpolate import interp1d

from .cache_utils import (fingerprint_hash, read_cache_fingerprint_hash,
                           write_cache_with_fingerprint)

__all__ = ['ComputeWeakRates', 'InterpolateWeakRates', 'RecomputeWeakRates', 'ComputeFn']

exp_cutoff = 3e+2
epsrel_low = 1.e-4
quad_limit = 200

# ---------------------------------------------------------------------------
# Fingerprinted cache for the n<->p weak-rate tables (IDEAS.md §1.2)
# ---------------------------------------------------------------------------
# Bump this whenever a code change alters the *numerical content* of the
# cached files for a fixed configuration (new physics term, changed formula,
# different file layout, ...).  Bumping it invalidates every existing cache
# file regardless of its fingerprint.
WEAK_RATE_FORMAT_VERSION = 1

# Config fields that determine the (Tg, Tnu) background history and the
# neutrino occupation numbers entering every weak-rate-related integral.
# Shared by the n<->p rate cache and the thermal-correction cache below.
#
#   incomplete_decoupling, QED_corrections  -- select the NEVO table, i.e.
#       the Tnu(Tg) relation the rates are integrated over.
#   munuOverTnu, spectral_distortions, analytic_distortions, delta_xi_nu,
#       y_SZ -- shape of the neutrino phase-space distribution (and, for
#       analytic_distortions, an extra contribution to the Friedmann
#       equation that feeds back into Tg(t)).
#   T_start_cosmo_MeV, n_temperature_table -- the (Tg_vec, Tnu_vec) grid
#       passed in as Tvec.
#   DeltaNeff -- extra radiation density alters the background Tg(t)
#       history (PRIMAT-Main.m / Phys. Rep. background ODEs); explicitly
#       called out in IDEAS.md §1.2 even though it is not a "neutrino
#       distribution" parameter per se.
_BACKGROUND_FINGERPRINT_FIELDS = [
    "incomplete_decoupling",
    "QED_corrections",
    "munuOverTnu",
    "spectral_distortions",
    "analytic_distortions",
    "delta_xi_nu",
    "y_SZ",
    "T_start_cosmo_MeV",
    "n_temperature_table",
    "DeltaNeff",
]


def _thermal_fingerprint(cfg):
    """Fingerprint dict for the thermal radiative-correction cache files.

    Identifies the configuration that produced
    ``rates/weak/{nTOp,pTOn}_thermal_corrections.txt``: the background
    fields above, plus the grid density ``sampling_nTOp_thermal``.  Used by
    :func:`ComputeWeakRates` to decide whether the cached thermal
    corrections may be reused, and folded into
    :func:`_weak_rate_fingerprint` so that a stale thermal cache also
    invalidates the n<->p rate cache that was built on top of it.

    Args:
        cfg: PyPRConfig instance.

    Returns:
        dict, JSON-serialisable.
    """
    fp = {"format_version": WEAK_RATE_FORMAT_VERSION,
          "sampling_nTOp_thermal": cfg.sampling_nTOp_thermal}
    for key in _BACKGROUND_FINGERPRINT_FIELDS:
        fp[key] = getattr(cfg, key)
    return fp


def _weak_rate_fingerprint(cfg):
    """Fingerprint dict for the n<->p weak-rate cache files.

    Identifies the configuration that produced
    ``rates/weak/nTOp_{frwrd,bkwrd}.txt``.  Per IDEAS.md §1.2,
    ``tau_n_flag``/``tau_n`` are deliberately *excluded*: they only rescale
    the interpolated rates after the fact (see
    ``PyPR._setup_weak_rates`` / ``_NormWeakRates``), so they never change
    the cached values themselves.

    When ``cfg.include_nTOp_thermal`` is True, the hash of
    :func:`_thermal_fingerprint` is embedded as well, so that changing any
    field relevant to the thermal-correction tables (e.g.
    ``sampling_nTOp_thermal``) also invalidates this cache, even though that
    field does not appear directly in the list below.

    Args:
        cfg: PyPRConfig instance.

    Returns:
        dict, JSON-serialisable; pass to :func:`fingerprint_hash` to get the
        comparable hash string.
    """
    fp = {"format_version":           WEAK_RATE_FORMAT_VERSION,
          "sampling_nTOp":            cfg.sampling_nTOp,
          "nTOp_Born_approximation":  cfg.nTOp_Born_approximation,
          "include_nTOp_thermal":     cfg.include_nTOp_thermal,
          "thermal_fingerprint_hash": (fingerprint_hash(_thermal_fingerprint(cfg))
                                        if cfg.include_nTOp_thermal else None)}
    for key in _BACKGROUND_FINGERPRINT_FIELDS:
        fp[key] = getattr(cfg, key)
    return fp


# ---------------------------------------------------------------------------
# Fermi-Dirac helper functions — JIT-compiled when numba is available.
# These capture nothing from any enclosing scope (only the module-level
# exp_cutoff constant), so they can live at module level and be wrapped
# with @njit.  Call _setup_fd_impls(cfg.numba_installed) before first use.
# ---------------------------------------------------------------------------

def FD_nu3(E, phi, x):
    return 1. / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD2(E, x):
    return 1. / (np.exp(x * E) + 1.) if (x * E) < exp_cutoff else 0.

def FD_nu_e2p0(E, phi, x):
    return E**2 / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD_nu_e3p0(E, phi, x):
    return E**3 / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD_nu_e4p2(E, phi, x):
    if (2. * phi < exp_cutoff) and (E * x + phi < exp_cutoff) and (2. * E * x < exp_cutoff):
        return (E**2 * np.exp(phi) * ((24. - E * x * (E * x + 8.)) * np.exp(E * x + phi)
                + np.exp(2 * E * x) * (E * x - 6.) * (E * x - 2.) + 12 * np.exp(2 * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.

def FD_nu_e2p2(E, phi, x):
    if (3. * phi < exp_cutoff) and (2 * E * x + phi < exp_cutoff) and (E * x < exp_cutoff):
        return (((E * x * (E * x - 4.) + 2.) * np.exp(2 * E * x + phi)
                 + (4. - E * x * (E * x + 4.)) * np.exp(E * x + 2 * phi)
                 + 2 * np.exp(3 * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.

def FD_nu_e4p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E**3 * (4 * np.exp(phi) + np.exp(E * x) * (4. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e2p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E * (2 * np.exp(phi) + np.exp(E * x) * (2. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e3p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E**2 * (3 * np.exp(phi) + np.exp(E * x) * (3. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e3p2(E, phi, x):
    if (2. * phi < exp_cutoff) and (E * x + phi < exp_cutoff) and (2. * E * x < exp_cutoff):
        return (E * np.exp(phi)
                * ((12. - E * x * (E * x + 6.)) * np.exp(E * x + phi)
                   + np.exp(2. * E * x) * (E * x * (E * x - 6.) + 6.)
                   + 6 * np.exp(2. * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.


_fd_impls_initialized = False


def _setup_fd_impls(numba_installed):
    global FD_nu3, FD2, FD_nu_e2p0, FD_nu_e3p0, FD_nu_e4p2, FD_nu_e2p2, \
           FD_nu_e4p1, FD_nu_e2p1, FD_nu_e3p1, FD_nu_e3p2, _fd_impls_initialized
    if _fd_impls_initialized:
        return
    _fd_impls_initialized = True
    if not numba_installed:
        return
    try:
        from numba import njit
        FD_nu3      = njit(FD_nu3)
        FD2         = njit(FD2)
        FD_nu_e2p0  = njit(FD_nu_e2p0)
        FD_nu_e3p0  = njit(FD_nu_e3p0)
        FD_nu_e4p2  = njit(FD_nu_e4p2)
        FD_nu_e2p2  = njit(FD_nu_e2p2)
        FD_nu_e4p1  = njit(FD_nu_e4p1)
        FD_nu_e2p1  = njit(FD_nu_e2p1)
        FD_nu_e3p1  = njit(FD_nu_e3p1)
        FD_nu_e3p2  = njit(FD_nu_e3p2)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Coulomb + radiative correction factors
# ---------------------------------------------------------------------------

def FermiCoulomb(b, cfg):
    """Fermi–Coulomb factor F(b) for the T=0 Coulomb correction to neutron decay.

    Returns the Fermi function F(Z=1, E) = F(b) that accounts for the Coulomb
    interaction between the outgoing electron and the daughter proton (Phys. Rep.
    §III.D).  The exact relativistic Fermi function is (see also Sirlin 1967):

        F(b) = (1 + Γ/2) × 4 × (2 r_p b / λ_C)^{2Γ} / Γ(3+2Γ)²
               × exp(π α / b) / (1−b²)^Γ × |Γ(1+Γ + iα/b)|²

    where b = v/c = p_e / E_e (electron velocity), Γ = √(1−α²) − 1 ≈ −α²/2,
    r_p = cfg.radproton (proton charge radius), λ_C = ℏc/mₑ (electron Compton
    wavelength), and α = cfg.alphaem is the fine-structure constant.

    The function diverges at b=0 (zero electron kinetic energy) but the phase-space
    factor p_e² = b²E² → 0 faster, so the product is integrable.

    Args:
        b   : electron velocity v/c = p_e/E_e  (dimensionless, 0 < b < 1).
        cfg : PyPRConfig instance (provides alphaem, me, radproton, hbar, clight).

    Returns:
        F(b) : float, the Fermi–Coulomb factor (dimensionless).

    Example:
        >>> F = FermiCoulomb(0.5, cfg)  # F at v = c/2
    """
    me      = cfg.me * cfg.MeV
    Gamma   = np.sqrt(1. - cfg.alphaem**2.) - 1.
    gamma1  = 1. + Gamma
    gamma2  = 3. + 2. * Gamma
    Fn_Comp = cfg.hbar * cfg.clight / me  # electron Compton wavelength ℏc/mₑ
    return ((1. + Gamma / 2.)
            * 4. * ((2. * cfg.radproton * b) / Fn_Comp) ** (2. * Gamma)
            / (scipy_gamma(gamma2)**2)
            * np.exp((np.pi * cfg.alphaem) / b)
            / ((1. - b**2) ** Gamma)
            * np.abs(scipy_gamma(gamma1 + (cfg.alphaem / b) * 1j))**2)


def RadCorrResum(b, y, en, cfg):
    """Resummed T=0 radiative correction factor R(b, y, E) for n↔p weak rates.

    Returns the product of Sirlin's outer radiative correction g(b,y,E) and the
    Czarnecki et al. 2004 constants for the resummed QED and short-distance
    corrections (Phys. Rep. Eq. 101–105).  The full corrected integrand for
    _L_CCR is

        Born integrand × FermiCoulomb(b) × RadCorrResum(b, y, en)

    Sirlin's function g(b,y,E) captures the O(α) virtual+real photon corrections
    from the electron and proton legs (Sirlin 1967; Phys. Rep. §III.D):

        g = 3 ln(mₚ/mₑ) − 3/4
            + 4(atanh(b)/b − 1)(y/(3E) − 3/2 + ln(2y))
            + atanh(b)/b × (2(1+b²) + y²/(6E²) − 4b atanh(b)/b)
            − (4/b) Li₂(1 − 2b/(1+b))

    where b = v/c (electron velocity), y = E_ν/mₑ (neutrino energy in mₑ units),
    E = E_e/mₑ (electron energy in mₑ units).

    Outer correction:
        (1 + α/(2π) × (g − 3 ln(mₚ / 2Q)))   [Phys. Rep. Eq. 103]

    Long-distance factor (QED running, Marciano & Sirlin 2006):
        Lndecay = 1.02094                       [L_factor, PRIMAT-Main.m]

    Inner short-distance factor (Czarnecki et al. 2004, Phys. Rev. Lett. 92):
        Sndecay = 1.02248                       [S_factor, PRIMAT-Main.m]
        Cndecay = 0.891                         [C_{nDecay}]
        Agndecay = -0.34                        [A_g, hadronic contribution]
        mA = 1.2 GeV                            [hadronic cutoff matching onto QCD]
        1/(134 × 2π)                            [= α/(2π) × 1/Cndecay for mA log]

    NLL correction:
        NLLndecay = -0.0001                     [next-to-leading log, ~10⁻⁴]

    Args:
        b   : electron velocity v/c = p_e/E_e  (dimensionless).
        y   : neutrino energy in mₑ units = E_ν/mₑ (> 0).
        en  : electron energy in mₑ units = E_e/mₑ (≥ 1).
        cfg : PyPRConfig instance (provides alphaem, me, mn, mp in MeV).

    Returns:
        R(b, y, en) : float, the resummed radiative correction factor
                      (dimensionless, close to 1.04 near the spectrum peak).

    Example:
        >>> b = 0.5; y = 1.0; en = 1.15
        >>> R = RadCorrResum(b, y, en, cfg)   # ≈ 1.040
    """
    # Czarnecki et al. 2004 (Phys. Rev. Lett. 92, 071801) constants.
    mA        = 1.2e+3 * cfg.MeV   # hadronic scale: matching QCD at ~1.2 GeV
    Agndecay  = -0.34              # A_g: hadronic logarithm coefficient
    Cndecay   =  0.891             # C_{nDecay}: inner radiative constant
    deltand   = -0.00043           # δ_{nd}: small correction
    Lndecay   =  1.02094           # long-distance QED running factor
    Sndecay   =  1.02248           # short-distance factor
    NLLndecay = -0.0001            # next-to-leading logarithm

    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp                   # neutron–proton mass difference

    # atanh(b)/b → 1 as b→0 (Taylor: atanh(b)/b = 1 + b²/3 + …)
    Rd = 1. if b == 0 else np.arctanh(b) / b
    # Sirlin's outer radiative function g(b,y,E) [Phys. Rep. Eq. 103]
    Sirlin = (3. * np.log(mp / me) - 3. / 4.
              + 4. * (Rd - 1.) * (y / (3. * en) - 3. / 2. + np.log(2. * y))
              + Rd * (2. * (1. + b**2) + y**2 / (6. * en**2) - 4. * b * Rd)
              - (4. / b) * spence(1. - (2 * b) / (1. + b)))
    return ((1. + cfg.alphaem / (2. * np.pi) * (Sirlin - 3. * np.log(mp / (2 * Q))))
            * (Lndecay + (cfg.alphaem / np.pi) * Cndecay
               + cfg.alphaem / (2 * np.pi) * deltand * 2 * np.pi / cfg.alphaem)
            * (Sndecay + 1. / (134. * 2. * np.pi) * (np.log(mp / mA) + Agndecay)
               + NLLndecay))


# ---------------------------------------------------------------------------
# Neutron-decay phase-space factor
# ---------------------------------------------------------------------------

def ComputeFn(cfg):
    """Compute the neutron-decay phase-space integral Fn used to normalise the weak rates.

    The overall strength K of the n↔p rates is obtained from the measured neutron
    lifetime τ_n rather than from the fundamental constants (G_F, g_A, V_ud) directly,
    because τ_n is known to better relative precision.  The relation is
    (Phys. Rep. Eqs. 89–91):

        1/τ_n = K × Fn(gA, mN, α, …)      ⟹     K = 1/(τ_n × Fn)

    where Fn = ∫₁^{Q/mₑ} integrand(E) dE  is the free-neutron-decay phase-space
    integral over electron energy E = ε_e/mₑ (dimensionless).

    Three contributions are summed (unless cfg.nTOp_Born_approximation=True, Born only):

    1. Fn_Born   — plain phase-space ∫ E (E−Q/mₑ)² √(E²−1) dE  (Born, no corrections).
    2. Fn_rad    — Born integrand × FermiCoulomb(b) × RadCorrResum (Coulomb + T=0
                   radiative corrections, Phys. Rep. Eq. 101).
    3. Fn_FM     — finite-nucleon-mass correction × Coulomb × radiative
                   (Phys. Rep. §III.G; ChiFMnDec encodes the Fokker-Planck expansion
                   coefficients f1n/f2n/f3n from gA and the anomalous magnetic moment
                   δκ = cfg.deltakappa).

    Args:
        cfg : PyPRConfig instance (provides mn, mp, me, gA, deltakappa, alphaem,
              radproton, hbar, clight, MeV, nTOp_Born_approximation).

    Returns:
        Fn : float (MeV⁰ = dimensionless after dividing by mₑ²), the neutron-decay
             phase-space factor.  Divided into K = 1/(τ_n × Fn) in ComputeWeakRates.

    Example:
        >>> Fn = ComputeFn(cfg)   # ≈ 1.636 (Born), ≈ 1.686 (with corrections)
    """
    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    def Fn_Born_int(E):
        if (-1. >= E) or (E >= 1):
            return E * (E - (Q / me))**2 * np.sqrt(E**2 - 1.)
        return 0.

    Fn_Born = quad(Fn_Born_int, 1., Q / me)[0]
    if cfg.nTOp_Born_approximation:
        return Fn_Born

    def Fn_rad_int(e):
        b = np.sqrt(e**2 - 1.) / e
        q = Q / me
        return (e * (e - q)**2 * e * b
                * FermiCoulomb(b, cfg)
                * RadCorrResum(np.sqrt(e**2 - 1.) / e, q - e, e, cfg))

    Fn_rad = quad(Fn_rad_int, 1., Q / me)[0]

    gA         = cfg.gA
    deltakappa = cfg.deltakappa

    def ChiFMnDec(en, pe):
        f1n = ((1. + gA)**2. + 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f2n = ((1. - gA)**2. - 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f3n = (gA**2 - 1.) / (1. + 3. * gA**2)
        mnOme = mn / me
        return (f1n * (en - Q / me)**2 * (pe**2 / (mnOme * en))
                - f2n / mnOme * (en - Q / me)**3
                + (f1n + f2n + f3n) / (2. * mnOme) * (4. * (en - Q / me)**3 + 2 * (en - Q / me) * pe**2)
                + f3n / mnOme * (en - Q / me)**2 * pe**2 / en)

    def Fn_FM_int(pe):
        en = np.sqrt(pe**2 + 1.)
        b  = pe / en
        return (pe**2
                * ChiFMnDec(en, pe)
                * RadCorrResum(b, np.abs(en - Q / me), en, cfg)
                * FermiCoulomb(b, cfg))

    Fn_FM = quad(Fn_FM_int, 0., np.sqrt((Q / me)**2 - 1.))[0]
    return Fn_rad + Fn_FM


# ---------------------------------------------------------------------------
# Main rate computation
# ---------------------------------------------------------------------------

def ComputeWeakRates(Tvec, cfg, dFDneu_func=None):
    """Compute n↔p weak rate tables over the BBN temperature range.

    Evaluates the forward rate Γ_{n→p}(T) and backward rate Γ_{p→n}(T) on the
    photon-temperature grid Tg_vec, including up to five additive corrections
    (depending on cfg flags):

    Γ_{n→p} = K × [_L_BORN + _L_CCR + _L_FMCCR] + _L_CCRTh + _L_SD

    where:
      _L_BORN   — Born rate ∫ p² [χ₊(E)+χ₊(−E)] dp  (Phys. Rep. Eqs. 77–78).
      _L_CCR    — Born integrand × FermiCoulomb × RadCorrResum (T=0 Coulomb
                  + resummed radiative corrections; Phys. Rep. Eq. 101).
      _L_FMCCR  — Finite-nucleon-mass correction × Coulomb × radiative
                  (Fokker-Planck expansion; Phys. Rep. §III.G).
                  Skipped if cfg.nTOp_Born_approximation=True (Born-only mode).
      _L_CCRTh  — Finite-temperature radiative corrections (Brown & Sawyer 2001;
                  Phys. Rep. §III.H, Eqs. 107–113).  Only if
                  cfg.include_nTOp_thermal=True; loaded from the fingerprinted
                  cache in rates/weak/ when valid, otherwise recomputed (slow,
                  uses vegas if available).
      _L_SD     — Spectral-distortion correction: the difference between the
                  actual neutrino distribution f_ν(E) (from NEVO) and the
                  equilibrium Fermi–Dirac, passed in via dFDneu_func.

    The overall rate constant K is normalised via the neutron lifetime:
        K = 1 / (τ_n × Fn)     (Phys. Rep. Eq. 89–91)
    where Fn = ComputeFn(cfg) is the free-decay phase-space integral.

    Parameters
    ----------
    Tvec       : list [Tg_vec, Tnu_vec], both float arrays in Kelvin (as output
                 by PyPR._setup_background_and_cosmo).
    cfg        : PyPRConfig instance.
    dFDneu_func: callable or None.
        If provided, adds the spectral-distortion correction _L_SD.  Signature:
            dFDneu_func(en, x, znu, sgnq) → float
        where en = E/mₑ, x = mₑ/(kB Tγ), znu = mₑ/(kB Tν), sgnq = ±1.
        Must encode the sign convention for blocking factors (en < 0), as
        described in PyPR._setup_background_and_cosmo.

    Returns
    -------
    [T_all, frwrd, bkwrd] : list
        T_all  — 1-D float array, photon temperatures in Kelvin.
        frwrd  — 1-D float array, Γ_{n→p}(T) in s⁻¹.
        bkwrd  — 1-D float array, Γ_{p→n}(T) in s⁻¹.

    Example:
        >>> rates = ComputeWeakRates([Tg_vec, Tnu_vec], cfg)
        >>> T_K, lam_nTOp, lam_pTOn = rates
    """
    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    xi_nu  = cfg.munuOverTnu
    my_dir = cfg.data_dir

    Tg_vec, Tnu_vec = Tvec
    T_nuOverT = interp1d(Tg_vec * cfg.MeV_to_Kelvin, Tnu_vec / Tg_vec,
                         bounds_error=False, fill_value="extrapolate", kind='linear')

    _setup_fd_impls(cfg.numba_installed)

    # ------------------------------------------------------------------
    # Born rate integrands
    # ------------------------------------------------------------------
    def ChiFunc(E, p, x, znu, sgnq):
        return FD_nu3(E - sgnq * (Q / me), sgnq * xi_nu, znu) * FD2(-E, x) * (E - sgnq * (Q / me))**2

    def FermiStat(sgnq, sgnE, b):
        return FermiCoulomb(b, cfg) if (sgnq * sgnE) > 0 else 1.

    def IPENdp(p, x, znu, sgnq):
        E = np.sqrt(p**2 + 1.)
        return p**2 * (ChiFunc(E, p, x, znu, sgnq) + ChiFunc(-E, p, x, znu, sgnq))

    def _L_BORN_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdp(p, x, xnu, sgnq)

    def _L_BORN(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_BORN_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low, limit=quad_limit)[0]

    # ------------------------------------------------------------------
    # Finite-mass corrections
    # ------------------------------------------------------------------
    gA         = cfg.gA
    deltakappa = cfg.deltakappa

    def ChiFunc_FM(en, pe, x, znu, sgnq):
        M_sgnq = (mp + mn - sgnq * Q) / (2 * me)
        f_1 = ((1. + sgnq * gA)**2. + 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
        f_2 = ((1. - sgnq * gA)**2. - 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
        f_3 = (gA**2 - 1.) / (1. + 3. * gA**2)
        enu    = en - sgnq * Q / me
        FD2_en = FD2(-en, x)
        return (f_1 * FD_nu_e2p0(enu, 0, znu) * FD2_en * (pe**2 / (M_sgnq * en))
                + f_2 * FD_nu_e3p0(enu, 0, znu) * FD2_en * (-1. / M_sgnq)
                + (f_1 + f_2 + f_3) / (2. * x * M_sgnq)
                  * (FD_nu_e4p2(enu, 0, znu) * FD2_en + FD_nu_e2p2(enu, 0, znu) * FD2_en * pe**2)
                + (f_1 + f_2 + f_3) / (2. * M_sgnq)
                  * (FD_nu_e4p1(enu, 0, znu) * FD2_en + FD_nu_e2p1(enu, 0, znu) * FD2_en * pe**2)
                - (f_1 + f_2) / (x * M_sgnq)
                  * (FD_nu_e3p1(enu, 0, znu) * FD2_en + FD_nu_e2p1(enu, 0, znu) * FD2_en * pe**2 / (-en))
                - f_3 * 3. / (x * M_sgnq) * FD_nu_e2p0(enu, 0, znu) * FD2_en
                + f_3 / (3 * M_sgnq) * FD_nu_e3p1(enu, 0, znu) * FD2_en * pe**2 / en
                + f_3 * 2. / (2. * x * 3. * M_sgnq) * FD_nu_e3p2(enu, 0, znu) * FD2_en * pe**2 / en
                - (f_1 + f_2 + f_3) * 3. / (2. * x) * (1. - (mn / mp)**sgnq)
                  * (FD_nu_e2p1(enu, 0, znu) * FD2_en))

    def IPENdpFMCCR(p, x, znu, sgnq):
        eOFpe = np.sqrt(p**2 + 1.)
        b     = p / eOFpe
        return p**2 * (ChiFunc_FM(eOFpe,  p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me - eOFpe), eOFpe, cfg)
                       * FermiStat(sgnq,  1, b)
                       + ChiFunc_FM(-eOFpe, p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me + eOFpe), eOFpe, cfg)
                       * FermiStat(sgnq, -1, b))

    def _L_FMCCR_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdpFMCCR(p, x, xnu, sgnq)

    def _L_FMCCR(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_FMCCR_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low, limit=quad_limit)[0]

    # ------------------------------------------------------------------
    # T=0 radiative corrections
    # ------------------------------------------------------------------
    def IPENdpCCR(p, x, znu, sgnq):
        E = np.sqrt(p**2 + 1.)
        b = p / E
        return p**2 * (ChiFunc(E,  p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                       * FermiStat(sgnq,  1, b)
                       + ChiFunc(-E, p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                       * FermiStat(sgnq, -1, b))

    def _L_CCR_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdpCCR(p, x, xnu, sgnq)

    def _L_CCR(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_CCR_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low, limit=quad_limit)[0]

    # ------------------------------------------------------------------
    # Spectral-distortion correction to the Born rate (optional)
    # Ref: PRIMAT-Main.m, δχ / IPENdpSD / ΛnTOpSD.
    #
    # The integrand is built from dFDneu_func (passed in from main.py), which
    # returns the deviation δf of the actual neutrino distribution from the
    # Fermi-Dirac at temperature Tν.  The correction to χ is:
    #
    #   δχ(en, pe, x, znu, sgnq) =
    #       dFDneu(en − sgnq Q/me, x, znu, sgnq) × FD(-en, x) × (en − sgnq Q/me)²
    #
    # which has the same pe-integrand structure as the Born IPENdp:
    #   IPENdpSD = pe² × [δχ(en_pe, ...) + δχ(−en_pe, ...)]
    #
    # This is added on top of the CCR (or Born) contribution, exactly as the
    # finite-nucleon-mass term is added.
    # ------------------------------------------------------------------
    if dFDneu_func is not None:
        def DeltaChiFunc(en, pe, x, znu, sgnq):
            """Spectral-distortion correction to χ.

            en  : E/me (electron energy in units of me)
            pe  : p/me (electron momentum)
            x   : me/(kB Tγ)
            znu : me/(kB Tν)
            sgnq: +1 (n→p) or −1 (p→n)
            """
            # Neutrino energy shifted by the weak endpoint sgnq·Q/me
            en_nu = en - sgnq * (Q / me)
            return dFDneu_func(en_nu, x, znu, sgnq) * FD2(-en, x) * en_nu**2

        def IPENdpSD(p, x, znu, sgnq):
            E = np.sqrt(p**2 + 1.)
            return p**2 * (DeltaChiFunc( E, p, x, znu, sgnq)
                         + DeltaChiFunc(-E, p, x, znu, sgnq))

        def _L_SD_int(p, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return IPENdpSD(p, x, xnu, sgnq)

        def _L_SD(T, sgnq):
            """Born-level spectral-distortion contribution to the n<->p rate."""
            x = me / (cfg.kB * T)
            return quad(_L_SD_int, 0., max(7., 30. / x),
                        args=(T, sgnq), epsrel=epsrel_low, limit=quad_limit)[0]
    else:
        _L_SD = None

    # ------------------------------------------------------------------
    # Finite-temperature radiative corrections (optional, uses vegas)
    #
    # Loaded from the fingerprinted cache in rates/weak/ when
    # cfg.include_nTOp_thermal=True and a cache file is present (see the
    # module docstring and cache_utils).  A fingerprint mismatch (or a
    # header-less legacy file) is reported but used anyway: recomputing this
    # term is a multi-minute Monte-Carlo integration, far too slow to trigger
    # automatically for what is itself only a ~1e-3-level refinement of
    # L_CCR + L_FMCCR.  Only a *missing* cache file triggers a fresh
    # computation.  Set cfg.include_nTOp_thermal=False to skip this term, or
    # delete the cache files and re-run with save_nTOp_thermal=True to force
    # a refresh stamped with the current configuration's fingerprint.
    # ------------------------------------------------------------------
    _td       = my_dir + "/rates/weak/"
    _nTh_path = _td + "nTOp_thermal_corrections.txt"
    _pTh_path = _td + "pTOn_thermal_corrections.txt"
    _have_thermal_cache = (cfg.include_nTOp_thermal
                           and os.path.exists(_nTh_path) and os.path.exists(_pTh_path))

    if cfg.include_nTOp_thermal and not _have_thermal_cache:
        try:
            import vegas
            _have_vegas = True
            n_eval = getattr(cfg, 'vegas_n_eval', 20000)
            n_itn  = getattr(cfg, 'vegas_n_itn',  20)
        except ImportError:
            _have_vegas = False
            from scipy.integrate import dblquad
            _epsrel_th = getattr(cfg, 'epsrel_thermal', 1.e-2)
            import warnings
            warnings.warn(
                "vegas not found: falling back to scipy.integrate.dblquad for thermal "
                "radiative corrections (epsrel={:.0e}).  Install vegas for better "
                "performance.".format(_epsrel_th),
                ImportWarning, stacklevel=2)

        def A(E, k):
            pE = np.sqrt(E**2 - 1.)
            return (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E

        def B(E):
            pE = np.sqrt(E**2 - 1.)
            return 2. * E * np.log((E + pE) / (E - pE)) - 4. * pE

        def IPENCCRT(E, k, x, znu, sgnq):
            pE = np.sqrt(E**2 - 1.)

            def BE(EkBT):
                resvec = np.zeros(len(EkBT))
                my_index = np.where(np.abs(EkBT) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(EkBT[my_index]) - 1.)
                return resvec

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def Chitilde_vec(en, znuval, sgnq):
                q = Q / me
                resvec = np.zeros(len(en))
                argvec = znuval * (en - sgnq * q) - sgnq * xi_nu
                my_index = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(argvec[my_index]) + 1.)
                return resvec * (en - sgnq * q)**2

            return (cfg.alphaem / (2 * np.pi) * (BE(x * k) / k)
                    * (A(E, k) * (FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
                                  * (Chitilde_vec(E - k, znu, sgnq) + Chitilde_vec(E + k, znu, sgnq)
                                     - 2 * Chitilde_vec(E, znu, sgnq))
                                  + FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
                                  * (Chitilde_vec(-E + k, znu, sgnq) + Chitilde_vec(-E - k, znu, sgnq)
                                     - 2 * Chitilde_vec(-E, znu, sgnq)))
                       - k * B(E) * (FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
                                     * (Chitilde_vec(E - k, znu, sgnq) - Chitilde_vec(E + k, znu, sgnq))
                                     + FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
                                     * (Chitilde_vec(-E + k, znu, sgnq) - Chitilde_vec(-E - k, znu, sgnq)))))

        def IPENCCRDiffBremsstrahlung(E, k, x, znu, sgnq):
            q  = Q / me
            pE = np.sqrt(E**2 - 1.)
            Fp = (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E + k * (2. * E * np.log((E + pE) / (E - pE)) - 4. * pE)
            Fm = (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E - k * (2. * E * np.log((E + pE) / (E - pE)) - 4. * pE)

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def Chitilde_vec(en, znuval, sgnq):
                q = Q / me
                resvec = np.zeros(len(en))
                argvec = znuval * (en - sgnq * q) - sgnq * xi_nu
                my_index = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(argvec[my_index]) + 1.)
                return resvec * (en - sgnq * q)**2

            res_fac  = cfg.alphaem / (2. * np.pi * k)
            res1_fac = FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
            res1vec  = Fp * Chitilde_vec(E + k, znu, sgnq)
            argvec   = k
            my_index = np.where(np.abs(argvec) < np.abs(E - sgnq * q))[0]
            res1vec[my_index] -= Fp[my_index] * FD2_vec(E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] - sgnq * q) - k[my_index])**2
            res1vec *= res1_fac
            res2_fac = FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
            res2vec  = Fm * Chitilde_vec(-E + k, znu, sgnq)
            my_index = np.where(np.abs(argvec) < np.abs(E + sgnq * q))[0]
            res2vec[my_index] -= Fp[my_index] * FD2_vec(-E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] + sgnq * q) - k[my_index])**2
            res2vec *= res2_fac
            return res_fac * (res1vec + res2vec)

        def C1dE(E, x, znu, sgnq):
            pE = np.sqrt(E**2 - 1.)
            return (-(cfg.alphaem * E) / (2. * np.pi * pE) * (2. * np.pi**2) / (3. * x**2)
                    * (ChiFunc(E, pE, x, znu, sgnq) + ChiFunc(-E, pE, x, znu, sgnq)))

        def C2dE1dE2(e1v, e2v, x, znu, sgnq):
            resvec       = np.zeros(len(e1v))
            e1pe2        = e1v + e2v
            e1me2        = e1v - e2v
            min_e1pe2    = 2. + np.abs(e1me2)
            max_e1pe2    = 2. + max(10., 15. / x) + np.abs(e1me2)
            index_limits = np.where(((e1pe2 - min_e1pe2) > 0) * ((max_e1pe2 - e1pe2) > 0))[0]

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def D_FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[idx] = -xval * np.exp(argvec[idx]) / (np.exp(argvec[idx]) + 1.)**2
                return resvec

            def FD_nu3_vec(en, phi, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval - phi
                idx = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                return resvec

            def ChiFunc_vec(E, p, x, znu, sgnq):
                return (FD_nu3_vec(E - sgnq * (Q / me), sgnq * xi_nu, znu)
                        * FD2_vec(-E, x) * (E - sgnq * (Q / me))**2)

            e1 = e1v[index_limits]
            e2 = e2v[index_limits]
            p1 = np.sqrt(e1**2 - 1.)
            p2 = np.sqrt(e2**2 - 1.)
            L_fac = np.log((e1 * e2 + p1 * p2 + 1.) / (e1 * e2 - p1 * p2 + 1.))
            resvec_limits = (cfg.alphaem / (2. * np.pi)
                             * (ChiFunc_vec(e1, p1, x, znu, sgnq) + ChiFunc_vec(-e1, p1, x, znu, sgnq))
                             * (-(1. / 4.) * np.log(((p1 + p2) / (p1 - p2))**2)**2
                                * (D_FD2_vec(e2, x) * p2 / p1 * e1**2 / e2 * (e1 + e2)
                                   + FD2_vec(e2, x) * e1**2 / (p1 * p2) * (e2 + e1 / e2**2))
                                + np.log(((p1 + p2) / (p1 - p2))**2)
                                * (D_FD2_vec(e2, x) * (p2**2 * e1 / e2 * (1. / p1**2 + 2.) - e1**2 * p2 / p1 * L_fac)
                                   + FD2_vec(e2, x) * (e1 / (p1**2 * e2**2) * (e2**2 + 2 * p1**2 + 1.)
                                                       - (e1**2 + e2**2) / (e1 + e2)
                                                       - (e1**2 * e2) / (p1 * p2) * L_fac))
                                - FD2_vec(e2, x) * (4. * e1 * p2 / p1 + 2. * e2 * L_fac)))
            resvec[index_limits] = resvec_limits
            return resvec

        def _L_ThermalTruePhoton_int(E, k, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return IPENCCRT(E, k, x, xnu, sgnq)

        def _L_ThermalTruePhoton(T, sgnq):
            x     = me / (cfg.kB * T)
            E_max = max(10., 20. / x)
            k_max = max(10., 20. / x)
            if _have_vegas:
                integ = vegas.Integrator([[1.001, E_max], [0.001, k_max]])
                @vegas.batchintegrand
                def f_batch(xv):
                    E_val, k_val = np.transpose(xv)
                    return {'myres': _L_ThermalTruePhoton_int(E_val, k_val, T, sgnq)}
                integ(f_batch, nitn=n_itn, neval=n_eval)
                result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                return result['myres'].mean
            else:
                return dblquad(
                    lambda k, E: float(_L_ThermalTruePhoton_int(
                        np.atleast_1d(E), np.atleast_1d(k), T, sgnq)[0]),
                    1.001, E_max, 0.001, k_max, epsrel=_epsrel_th)[0]

        def _L_ThermalDiffBremsstrahlung_int(E, k, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return IPENCCRDiffBremsstrahlung(E, k, x, xnu, sgnq)

        def _L_ThermalDiffBremsstrahlung(T, sgnq):
            x     = me / (cfg.kB * T)
            E_max = max(10., 20. / x)
            k_max = max(10., 20. / x)
            if _have_vegas:
                integ = vegas.Integrator([[1.001, E_max], [0.001, k_max]])
                @vegas.batchintegrand
                def f_batch(xv):
                    E_val, k_val = np.transpose(xv)
                    return {'myres': _L_ThermalDiffBremsstrahlung_int(E_val, k_val, T, sgnq)}
                integ(f_batch, nitn=n_itn, neval=n_eval)
                result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                return result['myres'].mean
            else:
                return dblquad(
                    lambda k, E: float(_L_ThermalDiffBremsstrahlung_int(
                        np.atleast_1d(E), np.atleast_1d(k), T, sgnq)[0]),
                    1.001, E_max, 0.001, k_max, epsrel=_epsrel_th)[0]

        def _L_Thermal_1_int(E, T, sgnq):
            return C1dE(E, me / (cfg.kB * T), me / (cfg.kB * T * T_nuOverT(T)), sgnq)

        def _L_Thermal_1(T, sgnq):
            return quad(_L_Thermal_1_int, 1., max(25., 150. * (cfg.kB * T) / me),
                        args=(T, sgnq), epsrel=1.e-2)[0]

        def _L_Thermal_2_3_int(e1pe2, e1me2, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return 0.5 * C2dE1dE2((e1pe2 + e1me2) / 2., (e1pe2 - e1me2) / 2., x, xnu, sgnq)

        def _L_Thermal_2_3(T, sgnq):
            x    = me / (cfg.kB * T)
            half = max(10., 15. / x)
            res_2 = res_3 = 0.
            for min_e1me2, max_e1me2 in [(-half, -0.001), (0.001, half)]:
                lims = [2.001 + min(np.abs(min_e1me2), np.abs(max_e1me2)),
                        2.   + max(np.abs(min_e1me2), np.abs(max_e1me2))]
                if _have_vegas:
                    integ = vegas.Integrator([lims, [min_e1me2, max_e1me2]])
                    @vegas.batchintegrand
                    def f_batch(xv):
                        e1pe2, e1me2 = np.transpose(xv)
                        return {'myres': _L_Thermal_2_3_int(e1pe2, e1me2, T, sgnq)}
                    integ(f_batch, nitn=n_itn, neval=n_eval)
                    result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                    val = result['myres'].mean
                else:
                    val = dblquad(
                        lambda e1me2, e1pe2: float(_L_Thermal_2_3_int(
                            np.atleast_1d(e1pe2), np.atleast_1d(e1me2), T, sgnq)[0]),
                        lims[0], lims[1], min_e1me2, max_e1me2, epsrel=_epsrel_th)[0]
                if min_e1me2 < 0:
                    res_2 = val
                else:
                    res_3 = val
            return res_2 + res_3

        def _L_CCRTh_compute(T, sgnq):
            if sgnq == -1 and T < 10**(8.2):
                return 0.
            return (_L_ThermalTruePhoton(T, sgnq)
                    + _L_ThermalDiffBremsstrahlung(T, sgnq)
                    + _L_Thermal_1(T, sgnq)
                    + _L_Thermal_2_3(T, sgnq))

        if cfg.verbose:
            print("[weak]     Re-evaluating n <--> p thermal corrections. This may take a while ...")

        _T_th      = np.logspace(np.log10(cfg.T_end), np.log10(cfg.T_start), cfg.sampling_nTOp_thermal)
        L_nTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, +1))(_T_th)
        L_pTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, -1))(_T_th)

        if cfg.save_nTOp_thermal:
            os.makedirs(_td, exist_ok=True)
            write_cache_with_fingerprint(_nTh_path, _thermal_fingerprint(cfg),
                                          [_T_th, L_nTh_data], col_header="T[K] L_nTOpCCRTh")
            write_cache_with_fingerprint(_pTh_path, _thermal_fingerprint(cfg),
                                          [_T_th, L_pTh_data], col_header="T[K] L_pTOnCCRTh")

        if cfg.verbose:
            print("n <--> p thermal corrections computed")

        T_th, L_nTh, L_pTh = _T_th, L_nTh_data, L_pTh_data

    elif cfg.include_nTOp_thermal:
        cached_hash  = read_cache_fingerprint_hash(_nTh_path)
        thermal_hash = fingerprint_hash(_thermal_fingerprint(cfg))
        if cfg.verbose and cached_hash is not None and cached_hash != thermal_hash:
            print("[weak]     Warning: nTOp_thermal_corrections.txt fingerprint does not "
                  "match the current configuration; using it anyway (recomputing thermal "
                  "corrections is slow). Delete the cache files and re-run with "
                  "save_nTOp_thermal=True to refresh them.")
        T_th, L_nTh = np.loadtxt(_nTh_path, unpack=True)
        T_th, L_pTh = np.loadtxt(_pTh_path,  unpack=True)

    if cfg.include_nTOp_thermal:
        L_nTOpCCRTh = interp1d(T_th, L_nTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')
        L_pTOnCCRTh = interp1d(T_th, L_pTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')
    else:
        L_nTOpCCRTh = lambda T: 0.0
        L_pTOnCCRTh = lambda T: 0.0

    # ------------------------------------------------------------------
    # Assembled rates  (sgnq = +1: n→p,  sgnq = -1: p→n)
    #
    # nTOp_rate_ dispatches among the correction levels at each temperature T:
    #   Born-only  (cfg.nTOp_Born_approximation=True):  returns _L_BORN(T, sgnq).
    #   Full CCR   (default):                  _L_CCR + _L_FMCCR [+ _L_CCRTh]
    # In both cases the spectral-distortion correction _L_SD is added on top
    # when dFDneu_func was supplied.  T is in Kelvin, return value in s⁻¹.
    # ------------------------------------------------------------------
    def nTOp_rate_(T, sgnq):
        if cfg.nTOp_Born_approximation:
            base = _L_BORN(T, sgnq)
        else:
            L_CCRTh = L_nTOpCCRTh(T) if sgnq == 1 else L_pTOnCCRTh(T)
            base = (_L_CCR(T, sgnq) + _L_FMCCR(T, sgnq))
            if cfg.include_nTOp_thermal:
                base += L_CCRTh
        # Spectral-distortion correction (Born level), added on top of the
        # base rate regardless of whether Born or CCR is used for the base.
        if _L_SD is not None:
            base += _L_SD(T, sgnq)
        return base

    nTOp_frwrd_vec = np.vectorize(lambda T: nTOp_rate_(T, +1))
    nTOp_bkwrd_vec = np.vectorize(lambda T: nTOp_rate_(T, -1))

    # Single grid spanning the whole BBN temperature range (T_end -> T_start).
    # cfg.sampling_nTOp is the *total* number of points (formerly it was the
    # per-era count and the network used three separate HT/MT/LT grids).
    T_all = np.logspace(np.log10(cfg.T_end), np.log10(cfg.T_start), cfg.sampling_nTOp)
    frwrd = nTOp_frwrd_vec(T_all)
    bkwrd = nTOp_bkwrd_vec(T_all)

    # Saving (if requested) is handled by RecomputeWeakRates, which already
    # has the fingerprint dict to stamp into the cache header.
    return [T_all, frwrd, bkwrd]


# ---------------------------------------------------------------------------
# Load / dispatch interface
# ---------------------------------------------------------------------------

def InterpolateWeakRates(cfg):
    """Load n↔p weak rates from the rates/weak/ cache and return interpolants.

    Reads rates/weak/nTOp_frwrd.txt and rates/weak/nTOp_bkwrd.txt (two
    columns each: T in Kelvin, rate in s⁻¹) regardless of whether their
    fingerprint header matches `cfg` -- callers that care about fingerprint
    validity (RecomputeWeakRates) check that *before* calling this function.
    Used directly to inspect "whatever is currently on disk", e.g. in
    tests/test_weak_rates.py.

    Args:
        cfg : PyPRConfig instance (provides data_dir).

    Returns:
        [frwrd, bkwrd] : two scipy interp1d objects (extrapolating), each mapping
                         T in Kelvin → rate in s⁻¹.
    """
    nd = os.path.join(cfg.data_dir, "rates", "weak", "")

    def _load(fname):
        tab = np.loadtxt(nd + fname)
        return interp1d(tab[:, 0], tab[:, 1], bounds_error=False,
                        fill_value="extrapolate", kind='quadratic')

    return [_load("nTOp_frwrd.txt"), _load("nTOp_bkwrd.txt")]


def RecomputeWeakRates(Tvec, cfg, dFDneu_func=None):
    """Load the n<->p weak-rate tables from the fingerprinted cache, or recompute.

    Implements the loader logic of IDEAS.md §1.2:

    1. Compute the fingerprint hash of the current configuration
       (:func:`_weak_rate_fingerprint`).
    2. If `cfg.weak_rate_cache` is True and both
       rates/weak/nTOp_{frwrd,bkwrd}.txt exist with a matching
       `fingerprint_hash` header, load and interpolate them directly
       (cheap: no integration at all).
    3. Otherwise call :func:`ComputeWeakRates` to recompute from scratch
       (~2 s).  If `cfg.save_nTOp` is True, overwrite the cache files with
       the new data and the current fingerprint header.
    4. **Forced recompute**: if `cfg.spectral_distortions and
       cfg.analytic_distortions`, the cache is bypassed entirely (never
       loaded, never written).  Analytic distortions are continuous knobs
       (`delta_xi_nu`, `y_SZ`) typically scanned point-by-point in an MCMC;
       caching them would write one file per parameter point and pollute
       rates/weak/.  The same rule applies to any future user-supplied
       `dFDneu_func` that cannot be fingerprinted.

    `cfg.tau_n_flag`/`cfg.tau_n` do not enter the fingerprint: they only
    rescale the interpolated rates afterwards (see
    `PyPR._setup_weak_rates` / `_NormWeakRates`), so a cache built with one
    `tau_n` remains valid for any other.

    Parameters
    ----------
    Tvec        : [Tg_vec, Tnu_vec]  (arrays in MeV)
    cfg         : PyPRConfig
    dFDneu_func : callable or None — spectral-distortion correction function;
                  forwarded to ComputeWeakRates on a cache miss.

    Returns
    -------
    [frwrd, bkwrd] : two interp1d objects (n->p and p->n) covering the whole
    BBN temperature range.
    """
    forced_recompute = cfg.spectral_distortions and cfg.analytic_distortions

    nd          = os.path.join(cfg.data_dir, "rates", "weak", "")
    frwrd_path  = nd + "nTOp_frwrd.txt"
    bkwrd_path  = nd + "nTOp_bkwrd.txt"
    fp          = _weak_rate_fingerprint(cfg)
    fp_hash     = fingerprint_hash(fp)

    if not forced_recompute and cfg.weak_rate_cache:
        cached_frwrd_hash = read_cache_fingerprint_hash(frwrd_path)
        cached_bkwrd_hash = read_cache_fingerprint_hash(bkwrd_path)
        if cached_frwrd_hash == fp_hash and cached_bkwrd_hash == fp_hash:
            return InterpolateWeakRates(cfg)
        if cfg.verbose:
            reason = "no cache" if cached_frwrd_hash is None else "fingerprint mismatch"
            print(f"[weak]     Recomputing n<->p weak rates ({reason}).")

    T_all, frwrd, bkwrd = ComputeWeakRates(Tvec, cfg, dFDneu_func=dFDneu_func)

    if not forced_recompute and cfg.save_nTOp:
        os.makedirs(nd, exist_ok=True)
        write_cache_with_fingerprint(frwrd_path, fp, [T_all, frwrd], col_header="T[K] Gamma_nTOp[1/s]")
        write_cache_with_fingerprint(bkwrd_path, fp, [T_all, bkwrd], col_header="T[K] Gamma_pTOn[1/s]")

    def _interp(v):
        return interp1d(T_all, v, bounds_error=False,
                        fill_value="extrapolate", kind='quadratic')

    return [_interp(frwrd), _interp(bkwrd)]
