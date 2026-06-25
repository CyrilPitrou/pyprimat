# -*- coding: utf-8 -*-
"""
weak_rates.corrections — n<->p rate correction terms (Born/CCR/FM/SD/CCRTh)
===============================================================================

The physical correction terms applied in sequence to the Born n<->p rate
(Phys. Rep. §III, PRIMAT-Main.m §IV.B), controlled by cfg flags
(sgnq = +1: n->p; sgnq = -1: p->n):

  _L_BORN    — Born approximation (Eqs. 77-78).  Active when
               cfg.radiative_corrections=False.
  _L_CCR     — Born integrand x R(b,y,E) [Coulomb x T=0 resummed radiative
               corrections, Phys. Rep. Eq. 101; R from Czarnecki et al. 2004].
               Active when cfg.radiative_corrections=True (replaces _L_BORN).
  _L_FMCCR   — Finite-nucleon-mass correction x R x Coulomb (Phys. Rep. §III.G,
               Fokker-Planck expansion to first order in T/m_N).  Active when
               cfg.finite_mass_corrections=True and cfg.radiative_corrections=True.
  _L_FMNoCCR — Finite-nucleon-mass correction without Coulomb/radiative factors.
               Active when cfg.finite_mass_corrections=True and
               cfg.radiative_corrections=False.
  _L_CCRTh   — Finite-temperature radiative corrections (Phys. Rep. §III.H;
               Brown & Sawyer 2001, Eqs. 5.10-5.15).  Active when
               cfg.thermal_corrections=True; uses vegas or scipy.dblquad;
               results cached to rates/weak/*.txt.
  _L_SD      — Spectral-distortion correction (Born-level chi): replaces the
               Fermi-Dirac g_nu with the actual distribution f_nu from NEVO.
               Active when dFDneu_func is supplied and
               cfg.radiative_corrections=False.
  _L_SD_CCR  — Spectral-distortion correction with Coulomb x radiative factor.
               Active when dFDneu_func is supplied and
               cfg.radiative_corrections=True.
  _L_SD_FMCCR / _L_SD_FMNoCCR — Finite-nucleon-mass correction to the
               spectral-distortion channel (generate_rates/PRIMAT-Main-gray.m's
               deltaChiFM).  Active when dFDneu_moments is supplied (analytic
               distortion mode) together with cfg.finite_mass_corrections.

:func:`_correction_terms` combines the active subset into one additive list
for :func:`weak_rates.api.ComputeWeakRates`; :func:`_build_rate_context`
builds the shared :class:`_RateContext` threaded through every term above.

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095)
— cited below as "Phys. Rep." with equation numbers.
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as scipy_gamma, spence
from scipy.integrate import quad
from scipy.interpolate import interp1d

from . import integrands
from .integrands import exp_cutoff
from .cache import n_points_per_decade, _thermal_fingerprint
from ..cache_utils import fingerprint_hash, write_cache_with_fingerprint

__all__ = [
    'FermiCoulomb', 'RadCorrResum', 'ComputeFn',
    '_N_GL', '_GL_NODES', '_GL_WEIGHTS',
    '_RateContext', '_chi_func', '_fermi_stat', '_quad_grid',
    '_L_BORN', '_L_CCR', '_chi_func_fm_v', '_L_FMCCR', '_L_FMNoCCR',
    '_L_SD', '_L_SD_CCR', '_chi_func_sd_fm_v',
    '_L_SD_FMCCR', '_L_SD_FMNoCCR', '_L_CCRTh_interpolants',
    '_correction_terms', '_build_rate_context', '_thermal_correction_interpolants',
]

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
        cfg : PRIMATConfig instance (provides alphaem, me, radproton, hbar, clight).

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
        cfg : PRIMATConfig instance (provides alphaem, me, mn, mp in MeV).

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

    # atanh(b)/b → 1 as b→0 (Taylor: atanh(b)/b = 1 + b²/3 + …); np.where
    # (rather than a Python if/else) makes this work elementwise when b is
    # an array too, so this one function serves both the scalar quad/dblquad
    # calls and the Gauss-Legendre array grid -- no separate "_v" twin.
    b_safe = np.where(b == 0., 1., b)
    Rd = np.where(b == 0., 1., np.arctanh(b_safe) / b_safe)
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

    Up to three contributions are summed depending on cfg flags:

    1. Fn_Born   — plain phase-space ∫ E (E−Q/mₑ)² √(E²−1) dE  (Born, no corrections).
                   Used as the base when cfg.radiative_corrections=False.
    2. Fn_rad    — Born integrand × FermiCoulomb(b) × RadCorrResum (Coulomb + T=0
                   radiative corrections, Phys. Rep. Eq. 101).
                   Used as the base when cfg.radiative_corrections=True.
    3. Fn_FM     — finite-nucleon-mass correction; with Coulomb × radiative when
                   cfg.radiative_corrections=True (Fn_FM_CCR), without otherwise
                   (Fn_FM_NoCCR).  Added when cfg.finite_mass_corrections=True.

    Args:
        cfg : PRIMATConfig instance (provides mn, mp, me, gA, deltakappa, alphaem,
              radproton, hbar, clight, MeV, radiative_corrections,
              finite_mass_corrections).

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

    gA         = cfg.gA
    deltakappa = cfg.deltakappa

    def ChiFMnDec(en, pe):
        """T=0 (vacuum) limit of the general finite-mass chi function _chi_func_fm_v.

        Mass convention -- why mp, not mn
        ----------------------------------
        The general (T>0, both directions) finite-mass correction uses, for a
        given sign sgnq (+1: n->p, -1: p->n), the recoil mass
            M_sgnq = (mp + mn - sgnq*Q) / (2*me)
        (see _chi_func_fm_v).  At sgnq=+1 this collapses algebraically to the
        *daughter* nucleon mass:
            M_sgnq(+1) = (mp + mn - (mn-mp)) / (2me) = (2*mp) / (2me) = mp/me,
        i.e. the proton, not the neutron, even though this is "neutron decay".
        ChiFMnDec is, by construction, supposed to BE the en, pe -> vacuum
        (x, znu -> infinity) limit of _chi_func_fm_v(..., sgnq=+1) -- that is
        the entire point of Fn: it normalises K = 1/(tau_n * Fn) so that the
        T -> 0 forward rate reproduces 1/tau_n exactly.  Taking that limit
        term-by-term (using the explicit x->infinity limits of the
        _FD_nu_e{2,3,4}p{1,2} helper functions: every term carrying an
        explicit 1/x prefactor vanishes -- those are the genuinely thermal
        pieces -- while the terms without a 1/x prefactor have finite limits,
        e.g. integrands.FD_nu_e4p1(enu,0,znu) -> 4*enu**3, FD_nu_e2p1 -> 2*enu,
        FD_nu_e3p1 -> 3*enu**2) reproduces exactly the four terms below, with
        M equal to whatever mass _chi_func_fm_v uses at sgnq=+1, i.e. mp, not
        mn.  Using mn/me here (an earlier, inconsistent choice) introduced a
        ~2.85e-6 mismatch between this vacuum normalisation and the T->0
        limit actually reached by ComputeWeakRates, since the finite-mass
        correction is itself only an O(Q/mp) ~ 1.4e-3 effect: an O(1)
        relative error in M (mn vs mp differ by Q/mp ~ 1.4e-3) inside an
        already-suppressed ~1.4e-3 term produces a second-order, ~2e-6
        absolute error in the total normalised rate -- exactly the size
        observed.
        """
        f1n = ((1. + gA)**2. + 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f2n = ((1. - gA)**2. - 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f3n = (gA**2 - 1.) / (1. + 3. * gA**2)
        # M_sgnq(+1) = (mp + mn - Q)/(2me) = mp/me (see docstring above).
        mnOme = mp / me
        return (f1n * (en - Q / me)**2 * (pe**2 / (mnOme * en))
                - f2n / mnOme * (en - Q / me)**3
                + (f1n + f2n + f3n) / (2. * mnOme) * (4. * (en - Q / me)**3 + 2 * (en - Q / me) * pe**2)
                + f3n / mnOme * (en - Q / me)**2 * pe**2 / en)

    if not cfg.radiative_corrections:
        # Born base rate.  Optionally add finite-mass correction without CCR.
        if not cfg.finite_mass_corrections:
            return Fn_Born
        def Fn_FM_NoCCR_int(pe):
            en = np.sqrt(pe**2 + 1.)
            return pe**2 * ChiFMnDec(en, pe)
        Fn_FM_NoCCR = quad(Fn_FM_NoCCR_int, 0., np.sqrt((Q / me)**2 - 1.))[0]
        return Fn_Born + Fn_FM_NoCCR

    # CCR base rate (replaces Born with Coulomb + resummed radiative corrections).
    def Fn_rad_int(e):
        b = np.sqrt(e**2 - 1.) / e
        q = Q / me
        return (e * (e - q)**2 * e * b
                * FermiCoulomb(b, cfg)
                * RadCorrResum(np.sqrt(e**2 - 1.) / e, q - e, e, cfg))

    Fn_rad = quad(Fn_rad_int, 1., Q / me)[0]

    if not cfg.finite_mass_corrections:
        return Fn_rad

    # Add finite-mass correction with Coulomb × radiative (FMCCR).
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
# Fixed-order Gauss-Legendre quadrature for the n<->p rate integrals
# ---------------------------------------------------------------------------
# The Born / CCR / FMCCR / SD integrands are all of the form
#     p^2 * [chi_+(E) + chi_+(-E)]  with  E = sqrt(p^2+1)
# integrated over p in [0, p_max(T)] with p_max = max(7, 30/x), x = m_e/(kB T).
# The integrand is smooth and exponentially damped (the neutrino Fermi-Dirac
# factor cuts it off at p ~ 30/x by construction of p_max), so a *fixed*
# Gauss-Legendre rule reproduces the former adaptive `scipy.integrate.quad`
# results to better than ~1e-6 on the rates while letting us evaluate the whole
# (n_temperature, n_node) grid in a handful of numpy array operations instead of
# one Python `quad` call per grid point per correction (the old ~1.8 s cost).
#
# One subtlety: the radiative-correction factor (RadCorrResum) has a
# log(2y)-type term that is singular-but-integrable exactly at the neutron
# beta-decay kinematic endpoint E = Q/me (p = p_edge, see _quad_grid).  A
# single GL rule spanning that point in the interior of its domain cannot
# resolve it (GL's fast convergence needs the integrand analytic over the
# *whole* panel) -- as T -> 0 this left a ~1-3e-6 floor in Gamma_{n->p}/tau_n^-1
# that did not shrink, and could even grow, with more nodes.  _quad_grid
# therefore places a panel boundary exactly at p_edge and runs two
# independent GL rules on either side, which restores normal fast convergence
# (the Born/CCR vacuum limit then matches the analytic Fn from ComputeFn to
# ~1e-11, see tests/test_weak_rates.py).
#
# _N_GL is pinned by tests/test_weak_rates.py::test_gauss_legendre_converged,
# which checks that doubling the node count moves the rates by <1e-6 over the
# full BBN temperature range.  160 nodes per panel give that margin comfortably
# (the integrand peak sits near p_max/15, where Gauss-Legendre is sparsest, so
# we deliberately oversample rather than tune to the edge).
_N_GL = 160
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_N_GL)


# ---------------------------------------------------------------------------
# Main rate computation
# ---------------------------------------------------------------------------

@dataclass
class _RateContext:
    """Shared per-call quantities for the n<->p weak-rate correction terms.

    Built once per :func:`ComputeWeakRates` call and threaded through every
    correction-term function below (`_L_BORN`, `_L_CCR`, `_L_FMCCR`, `_L_SD`,
    `_L_CCRTh_interpolants`), so that each term is a short, independently
    named module-level function -- mirroring Table 1 of the Phys. Rep. --
    instead of a closure nested 500 lines deep inside one function.

    Attributes
    ----------
    cfg : PRIMATConfig
        Run configuration (kB, alphaem, gA, data_dir, and all weak-rate flags).
    me, mn, mp, Q : float
        Electron/neutron/proton masses and Q = mn - mp, in MeV.
    xi_nu : float
        Reduced neutrino chemical potential mu_nu/T_nu (cfg.munuOverTnu).
    T_nuOverT : callable
        Interpolant T_nu(T_gamma)/T_gamma as a function of T_gamma [K].
    gA, deltakappa : float
        Nucleon axial coupling g_A and kappa_p - kappa_n, used by the
        finite-nucleon-mass Fokker-Planck expansion (_L_FMCCR).
    my_dir : str
        cfg.data_dir, used to locate the thermal-correction cache files.
    """
    cfg: object
    me: float
    mn: float
    mp: float
    Q: float
    xi_nu: float
    T_nuOverT: object
    gA: float
    deltakappa: float
    my_dir: str


# ---------------------------------------------------------------------------
# Shared chi functions (Phys. Rep. Eq. 81 and its corrections)
# ---------------------------------------------------------------------------

def _chi_func(ctx, E, x, znu, sgnq):
    """chi_+/-(E): Born-rate chi function (Phys. Rep. Eq. 81).

    chi_+/-(E) = (E_nu)^2 g_nu(E_nu, xi_nu) g(-E, x), with E_nu = E - sgnq*Q/me.
    Used by _L_BORN and _L_CCR (the latter multiplies it by the Coulomb and
    radiative correction factors).  Works for both scalar E (the
    scipy.quad/dblquad calls in _L_CCRTh_interpolants) and array E (the
    Gauss-Legendre rate-integral grid below) since FD_nu3/FD2 are themselves
    scalar-and-array-capable -- no separate vectorised twin needed.
    """
    Q, me, xi_nu = ctx.Q, ctx.me, ctx.xi_nu
    enu = E - sgnq * (Q / me)
    return integrands.FD_nu3(enu, sgnq * xi_nu, znu) * integrands.FD2(-E, x) * enu**2


def _fermi_stat(ctx, sgnq, sgnE, b):
    """Coulomb-factor switch used by _L_CCR, _L_FMCCR and _L_CCRTh.

    Returns FermiCoulomb(b) when the produced charged lepton is the electron
    (sgnq*sgnE > 0, i.e. it feels the daughter proton's Coulomb field), and 1
    otherwise (positron emission / no Coulomb correction).
    """
    return FermiCoulomb(b, ctx.cfg) if (sgnq * sgnE) > 0 else 1.


# ---------------------------------------------------------------------------
# Quadrature grid
# ---------------------------------------------------------------------------


def _quad_grid(ctx, T_arr):
    """Build the (n_T, 2*_N_GL) Gauss-Legendre momentum grid for ComputeWeakRates.

    Physical picture
    -----------------
    The rate integrals are nominally over p in [0, +inf): the electron/positron
    momentum can in principle be arbitrarily large.  We never integrate to
    actual infinity -- instead we truncate at a finite p_max(T) and rely on
    Gauss-Legendre (GL) on the *finite* interval [0, p_max(T)].  This is valid
    only because the integrand is forced to underflow well before p_max: every
    term carries a Fermi-Dirac factor integrands.FD2(-E,x) / integrands.FD_nu3(...) that decays like
    exp(-x p) at large p for T > 0 (x = m_e/(kB T)).  Choosing
    p_max = max(7, 30/x) keeps exp(-x p_max) <~ exp(-30) ~ 1e-13 at low T,
    i.e. the tail beyond p_max is below double-precision noise, so replacing
    [0, +inf) by [0, p_max] introduces no measurable truncation error -- this
    is the same heuristic the old scalar ``quad`` calls used.  The "max(7, ...)"
    floor matters at LOW T: there 30/x -> 0 (the thermal cutoff would want to
    shrink p_max to 0), but the T=0 vacuum-decay phase space itself extends out
    to p_edge ~ 2.33 (see below) regardless of T, so p_max must never shrink
    below that physical support; 7 is a safe margin above it.

    Why GL needs to be split at p_edge
    -----------------------------------
    A *single* Gauss-Legendre rule on [0, p_max] approximates the integrand by
    one polynomial of degree ~2*_N_GL over the whole interval.  GL's famous
    spectral (exponentially fast) convergence relies on the integrand being
    analytic (smooth, no kinks) in a neighbourhood of the *whole* interval --
    if it has a non-analytic point strictly inside the interval, a single
    global polynomial cannot track it well no matter how many nodes you add
    (you may even get WORSE as N grows, since GL nodes then sample closer to
    the bad point without resolving it, similar in spirit to Runge's
    phenomenon for naive high-order interpolation).

    Such a non-analytic point exists here: the radiative-correction factor
    RadCorrResum (the "Sirlin function", Phys. Rep. Eq. 101) contains a
    log(2*y) term with y = |sgnq*Q/me -+ E|, which is *exactly* zero when the
    electron energy E crosses the neutron-decay kinematic endpoint
    E = Q/me, i.e. p = p_edge = sqrt((Q/me)^2 - 1).  The full integrand stays
    finite there (the chi function itself vanishes like y^2, faster than
    log(y) diverges, so the *product* -> 0 and the singularity is integrable),
    but the function is not analytic across that point: it has a y^2*log(y)
    profile, which is C^0 but not C^infinity (its higher derivatives blow up
    as y -> 0).  A 160-node GL rule spanning [0, 7] sees this point sitting in
    the *middle* of its domain and, empirically, converges to a value that is
    biased low by O(1e-6) relative to tau_n^-1 in the T -> 0 limit, and does
    NOT improve monotonically with more nodes (it can get *worse*, then
    eventually overflow, because nodes start landing pathologically close to
    the singular point and to the unrelated Coulomb/Gamow singularity at
    p -> 0, where FermiCoulomb ~ exp(+pi*alpha/b) blows up for b = p/E -> 0).

    The standard remedy for a known non-analytic interior point is to split
    the integration domain into sub-intervals with a panel boundary placed
    exactly AT that point, and run an independent GL rule on each panel.
    Inside each panel the integrand is then analytic right up to (but not
    across) the shared boundary, so each panel recovers GL's normal fast
    convergence; only the panel that has p_edge as one of its *endpoints*
    still feels the residual log-type endpoint singularity, which is a much
    milder (algebraic, not interior-blind) source of error that the
    reference normalisation Fn (see ComputeFn) already handles correctly,
    since Fn's adaptive `quad` is given the very same edge as an explicit
    integration bound (E from 1 to Q/me).  Splitting here simply makes the
    rate integral structurally match what Fn already does.

    This is conceptually the same fix you would apply to integrate, say,
    |x| over [-1, 1] with a polynomial rule: a single global polynomial rule
    struggles with the kink at x=0, but two rules on [-1,0] and [0,1] each see
    a perfectly smooth (here: linear) integrand and reproduce the exact answer.

    Returns
    -------
    p   : (n_T, 2*_N_GL) momentum nodes [dimensionless, p/m_e], the first
          _N_GL columns covering panel A = [0, p_edge] and the next _N_GL
          covering panel B = [p_edge, p_max(T)].
    w   : (n_T, 2*_N_GL) quadrature weights (each panel's own dp/du Jacobian
          already folded in), so the integral is still simply
          ``np.sum(w * integrand, axis=1)`` over all 2*_N_GL points.
    x   : (n_T, 1) inverse photon-temperature ratio m_e/(kB T).
    xnu : (n_T, 1) inverse neutrino-temperature ratio m_e/(kB T_nu).
    """
    cfg, me, Q = ctx.cfg, ctx.me, ctx.Q
    x   = (me / (cfg.kB * T_arr))[:, None]
    xnu = (me / (cfg.kB * T_arr * ctx.T_nuOverT(T_arr)))[:, None]
    pmax  = np.maximum(7., 30. / x)                  # (n_T, 1)

    # Kinematic endpoint of neutron decay: E = Q/me <=> p_edge = sqrt((Q/me)^2-1).
    # This is T-independent (fixed by particle masses only), unlike p_max(T).
    p_edge = np.sqrt((Q / me)**2 - 1.)

    # Panel A: [0, p_edge], fixed across all T (the vacuum-decay support).
    # Use the node array's own length rather than the module-level _N_GL
    # constant: tests/test_weak_rates.py::test_gauss_legendre_converged swaps
    # _GL_NODES/_GL_WEIGHTS for a higher-order rule without also touching
    # _N_GL, so deriving the count from the array keeps that test's
    # node-doubling check meaningful for the split grid too.
    n_gl = _GL_NODES.shape[0]
    pA = 0.5 * p_edge * (_GL_NODES[None, :] + 1.)        # (1, n_gl) broadcastable
    wA = (0.5 * p_edge) * _GL_WEIGHTS[None, :]
    pA = np.broadcast_to(pA, (T_arr.shape[0], n_gl))
    wA = np.broadcast_to(wA, (T_arr.shape[0], n_gl))

    # Panel B: [p_edge, p_max(T)], width depends on T through p_max only.
    halfwidth_B = 0.5 * (pmax - p_edge)               # (n_T, 1)
    pB = p_edge + halfwidth_B * (_GL_NODES[None, :] + 1.)   # (n_T, n_gl)
    wB = halfwidth_B * _GL_WEIGHTS[None, :]                 # (n_T, n_gl)

    p = np.concatenate([pA, pB], axis=1)              # (n_T, 2*_N_GL)
    w = np.concatenate([wA, wB], axis=1)
    return p, w, x, xnu


# ---------------------------------------------------------------------------
# _L_BORN -- Born approximation (Phys. Rep. Eqs. 77-78)
# ---------------------------------------------------------------------------

def _L_BORN(ctx, T_arr, sgnq):
    """Born-approximation rate over the whole T grid (Phys. Rep. Eqs. 77-78).

    Vectorised: returns ``int p^2 [chi_+(E)+chi_+(-E)] dp`` evaluated at every
    temperature in ``T_arr`` at once via the fixed Gauss-Legendre rule.  Used
    as the base rate when cfg.radiative_corrections=False; otherwise superseded
    by _L_CCR.
    """
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    integ = p**2 * (_chi_func(ctx, E, x, xnu, sgnq)
                    + _chi_func(ctx, -E, x, xnu, sgnq))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_CCR -- T=0 Coulomb + resummed radiative corrections (Phys. Rep. Eq. 101)
# ---------------------------------------------------------------------------

def _L_CCR(ctx, T_arr, sgnq):
    """Born integrand x FermiCoulomb x RadCorrResum (Phys. Rep. Eq. 101).

    Vectorised over the whole T grid.  T=0 Coulomb correction (FermiCoulomb)
    and resummed QED + short-distance radiative corrections (RadCorrResum,
    Czarnecki et al. 2004) applied to the Born chi function.
    """
    cfg, me, Q = ctx.cfg, ctx.me, ctx.Q
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    b = p / E
    integ = p**2 * (_chi_func(ctx, E, x, xnu, sgnq)
                    * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                    * _fermi_stat(ctx, sgnq, 1, b)
                    + _chi_func(ctx, -E, x, xnu, sgnq)
                    * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                    * _fermi_stat(ctx, sgnq, -1, b))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_FMCCR -- finite-nucleon-mass correction (Phys. Rep. §III.G)
# ---------------------------------------------------------------------------

def _chi_func_fm_v(ctx, en, pe, x, znu, sgnq):
    """Vectorised chi_FM: finite-nucleon-mass correction to chi_+/-
    (Phys. Rep. §III.G, Fokker-Planck expansion to first order in T/m_N).

    No separate scalar twin exists (this term was always evaluated on the
    array grid).  f_1, f_2, f_3 are the Fokker-Planck expansion coefficients
    built from g_A and delta_kappa = kappa_p - kappa_n; M_sgnq is the average
    nucleon mass shifted by +/-Q, in units of m_e.
    """
    me, mn, mp, Q = ctx.me, ctx.mn, ctx.mp, ctx.Q
    gA, deltakappa = ctx.gA, ctx.deltakappa
    M_sgnq = (mp + mn - sgnq * Q) / (2 * me)
    f_1 = ((1. + sgnq * gA)**2. + 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
    f_2 = ((1. - sgnq * gA)**2. - 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
    f_3 = (gA**2 - 1.) / (1. + 3. * gA**2)
    enu    = en - sgnq * Q / me
    FD2_en = integrands.FD2(-en, x)
    return (f_1 * integrands.FD_nu_e2p0(enu, 0., znu) * FD2_en * (pe**2 / (M_sgnq * en))
            + f_2 * integrands.FD_nu_e3p0(enu, 0., znu) * FD2_en * (-1. / M_sgnq)
            + (f_1 + f_2 + f_3) / (2. * x * M_sgnq)
              * (integrands.FD_nu_e4p2(enu, 0., znu) * FD2_en + integrands.FD_nu_e2p2(enu, 0., znu) * FD2_en * pe**2)
            + (f_1 + f_2 + f_3) / (2. * M_sgnq)
              * (integrands.FD_nu_e4p1(enu, 0., znu) * FD2_en + integrands.FD_nu_e2p1(enu, 0., znu) * FD2_en * pe**2)
            - (f_1 + f_2) / (x * M_sgnq)
              * (integrands.FD_nu_e3p1(enu, 0., znu) * FD2_en + integrands.FD_nu_e2p1(enu, 0., znu) * FD2_en * pe**2 / (-en))
            - f_3 * 3. / (x * M_sgnq) * integrands.FD_nu_e2p0(enu, 0., znu) * FD2_en
            + f_3 / (3 * M_sgnq) * integrands.FD_nu_e3p1(enu, 0., znu) * FD2_en * pe**2 / en
            + f_3 * 2. / (2. * x * 3. * M_sgnq) * integrands.FD_nu_e3p2(enu, 0., znu) * FD2_en * pe**2 / en
            - (f_1 + f_2 + f_3) * 3. / (2. * x) * (1. - (mn / mp)**sgnq)
              * (integrands.FD_nu_e2p1(enu, 0., znu) * FD2_en))


def _L_FMCCR(ctx, T_arr, sgnq):
    """Finite-nucleon-mass correction x Coulomb x radiative (Phys. Rep. §III.G).

    Vectorised over the whole T grid.  Used when cfg.finite_mass_corrections=True
    and cfg.radiative_corrections=True.
    """
    cfg, me, Q = ctx.cfg, ctx.me, ctx.Q
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    b = p / E
    integ = p**2 * (_chi_func_fm_v(ctx, E, p, x, xnu, sgnq)
                    * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                    * _fermi_stat(ctx, sgnq, 1, b)
                    + _chi_func_fm_v(ctx, -E, p, x, xnu, sgnq)
                    * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                    * _fermi_stat(ctx, sgnq, -1, b))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_FMNoCCR -- finite-nucleon-mass correction WITHOUT Coulomb/radiative
# ---------------------------------------------------------------------------

def _L_FMNoCCR(ctx, T_arr, sgnq):
    """Finite-nucleon-mass correction WITHOUT Coulomb or radiative factors.

    Mirrors PRIMAT-Main.m ``λFMNoCCR``.  Used when cfg.finite_mass_corrections=True and
    cfg.radiative_corrections=False, so that the finite-mass correction is
    self-consistently computed at the same (Born) level as the base rate.

    The Fokker-Planck chi_FM function (_chi_func_fm_v) is identical to the
    one used in _L_FMCCR; the only difference is the absence of the
    FermiCoulomb (_fermi_stat) and RadCorrResum factors.
    """
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    integ = p**2 * (_chi_func_fm_v(ctx,  E, p, x, xnu, sgnq)
                    + _chi_func_fm_v(ctx, -E, p, x, xnu, sgnq))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_SD -- spectral-distortion correction (Born level, optional)
# ---------------------------------------------------------------------------

def _L_SD(ctx, T_arr, sgnq, dFDneu_func):
    """Born-level spectral-distortion contribution to the n<->p rate.

    Mirrors PRIMAT-Main.m ``λSD``.
    Used when cfg.spectral_distortions=True and cfg.radiative_corrections=False.
    See also :func:`_L_SD_CCR` for the version with Coulomb/radiative factors.

    ``dFDneu_func`` is a user-supplied callable (analytic μ/y or NEVO table
    lookup). The analytic-distortion implementation
    (``neutrino_history.AnalyticDistortion``) is itself array-vectorised and
    marked with a ``vectorized`` attribute, so it is called directly; the
    NEVO-table lookup has internal scalar ``if``/interpolator-call branches
    and cannot be expressed in closed numpy form, so it is wrapped in
    ``np.vectorize`` instead (slower, but the table-distortion mode is the
    less commonly used one).
    """
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    dfd = dFDneu_func if getattr(dFDneu_func, "vectorized", False) else np.vectorize(dFDneu_func)

    def delta_chi(en):
        # delta_chi(en) = dFDneu(en - sgnq*Q/me) * g(-en, x) * (en - sgnq*Q/me)^2,
        # the chi function with dFDneu (deviation from FD) in place of g_nu.
        en_nu = en - sgnq * (ctx.Q / ctx.me)
        return dfd(en_nu, x, xnu, sgnq) * integrands.FD2(-en, x) * en_nu**2

    integ = p**2 * (delta_chi(E) + delta_chi(-E))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_SD_CCR -- spectral-distortion correction WITH Coulomb/radiative factors
# ---------------------------------------------------------------------------

def _L_SD_CCR(ctx, T_arr, sgnq, dFDneu_func):
    """Spectral-distortion correction with Coulomb × T=0 resummed radiative factor.

    Mirrors PRIMAT-Main.m ``λSDCCR``.
    Used when cfg.spectral_distortions=True and cfg.radiative_corrections=True.
    Identical algebra to :func:`_L_CCR` but with the SD delta-chi function in
    place of the Born chi: the FermiCoulomb and RadCorrResum factors are applied
    to the SD integrand, making the spectral-distortion correction self-consistent
    with the base CCR rate.

    Same direct-call/``np.vectorize`` choice as :func:`_L_SD`, based on
    ``dFDneu_func``'s ``vectorized`` attribute.
    """
    cfg, me, Q = ctx.cfg, ctx.me, ctx.Q
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    b = p / E
    dfd = dFDneu_func if getattr(dFDneu_func, "vectorized", False) else np.vectorize(dFDneu_func)

    def delta_chi(en):
        # SD chi: replace g_nu in the Born chi function with the deviation δf/f_FD.
        en_nu = en - sgnq * (Q / me)
        return dfd(en_nu, x, xnu, sgnq) * integrands.FD2(-en, x) * en_nu**2

    integ = p**2 * (delta_chi(E)
                    * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                    * _fermi_stat(ctx, sgnq, 1, b)
                    + delta_chi(-E)
                    * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                    * _fermi_stat(ctx, sgnq, -1, b))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_SD_FMCCR / _L_SD_FMNoCCR -- SD-FM: finite-nucleon-mass correction to the
# spectral-distortion channel (analytic-distortion mode only)
# ---------------------------------------------------------------------------

def _chi_func_sd_fm_v(ctx, en, pe, x, znu, sgnq, dFDneu_moments_v):
    """Vectorised delta_chi_FM: finite-nucleon-mass correction to the
    spectral-distortion channel (Phys. Rep. §III.G algebra applied to the
    distortion deviation rather than the plain Fermi-Dirac).

    Mirrors generate_rates/PRIMAT-Main-gray.m's ``deltaChiFM`` (lines
    ~1712-1725), which is :func:`_chi_func_fm_v` with every plain-FD moment
    ``FDνe{n}p{k}[enu,phi,znu]`` (here ``FD_nu_e{n}p{k}(enu, 0., znu)``)
    replaced by the corresponding *distortion* moment
    ``dFDneue{n}p{k}[enu,phi,x,znu,sgnq]`` -- i.e. en^n times the k-th
    en-derivative of the analytic μ+y+gray distortion δf (PyPRIMAT's
    ``dFDneu_func``), instead of en^n times the plain Fermi-Dirac occupation.
    Same f_1, f_2, f_3 Fokker-Planck coefficients and M_sgnq as
    :func:`_chi_func_fm_v`; the substitution is purely mechanical (term by
    term), which is why the two functions are kept structurally identical
    rather than refactored into one with a moment-source parameter -- a
    line-by-line diff against ``_chi_func_fm_v`` is the easiest way to audit
    this for transcription errors.

    ``dFDneu_moments_v`` is the dict of *vectorised* energy-moment functions
    ``primat.neutrino_history.AnalyticDistortion.dFDneu_moments`` (keys
    "e2p0", "e3p0", "e2p1", "e3p1", "e4p1", "e2p2", "e3p2", "e4p2"), each
    called as ``moment(enu, x, znu, sgnq)`` (the "x" argument is unused by the
    analytic distortion -- present only for interface parity with
    dFDneu_func, see neutrino_history._dFDneu_analytic's docstring).
    """
    me, mn, mp, Q = ctx.me, ctx.mn, ctx.mp, ctx.Q
    gA, deltakappa = ctx.gA, ctx.deltakappa
    M_sgnq = (mp + mn - sgnq * Q) / (2 * me)
    f_1 = ((1. + sgnq * gA)**2. + 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
    f_2 = ((1. - sgnq * gA)**2. - 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
    f_3 = (gA**2 - 1.) / (1. + 3. * gA**2)
    enu    = en - sgnq * Q / me
    FD2_en = integrands.FD2(-en, x)
    m = dFDneu_moments_v
    return (f_1 * m["e2p0"](enu, x, znu, sgnq) * FD2_en * (pe**2 / (M_sgnq * en))
            + f_2 * m["e3p0"](enu, x, znu, sgnq) * FD2_en * (-1. / M_sgnq)
            + (f_1 + f_2 + f_3) / (2. * x * M_sgnq)
              * (m["e4p2"](enu, x, znu, sgnq) * FD2_en + m["e2p2"](enu, x, znu, sgnq) * FD2_en * pe**2)
            + (f_1 + f_2 + f_3) / (2. * M_sgnq)
              * (m["e4p1"](enu, x, znu, sgnq) * FD2_en + m["e2p1"](enu, x, znu, sgnq) * FD2_en * pe**2)
            - (f_1 + f_2) / (x * M_sgnq)
              * (m["e3p1"](enu, x, znu, sgnq) * FD2_en + m["e2p1"](enu, x, znu, sgnq) * FD2_en * pe**2 / (-en))
            - f_3 * 3. / (x * M_sgnq) * m["e2p0"](enu, x, znu, sgnq) * FD2_en
            + f_3 / (3 * M_sgnq) * m["e3p1"](enu, x, znu, sgnq) * FD2_en * pe**2 / en
            + f_3 * 2. / (2. * x * 3. * M_sgnq) * m["e3p2"](enu, x, znu, sgnq) * FD2_en * pe**2 / en
            - (f_1 + f_2 + f_3) * 3. / (2. * x) * (1. - (mn / mp)**sgnq)
              * (m["e2p1"](enu, x, znu, sgnq) * FD2_en))


def _L_SD_FMCCR(ctx, T_arr, sgnq, dFDneu_moments):
    """SD-FM correction (finite-mass x spectral-distortion) x Coulomb x radiative.

    Mirrors PRIMAT-Main-gray.m ``λnTOpSDFMCCR``/``λpTOnSDFMCCR``.  Used when
    cfg.spectral_distortions=True, cfg.analytic_distortions=True,
    cfg.finite_mass_corrections=True and cfg.radiative_corrections=True (see
    _correction_terms).  Analytic-distortion mode only (see this module's
    SD-FM section header): ``dFDneu_moments`` is only ever non-None when
    ``cfg.analytic_distortions`` -- the NEVO-table distortion has no closed-
    form en-derivative.
    """
    cfg, me, Q = ctx.cfg, ctx.me, ctx.Q
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    b = p / E
    # dFDneu_moments' callables are natively array-vectorised (np.where-based,
    # see neutrino_history.AnalyticDistortion._make_moment) -- no np.vectorize
    # wrapping needed (a prior version wrapped them here, which was the
    # dominant cost of an analytic_distortions=True run: ~20 s of per-point
    # Python calls for what is now a handful of numpy ops over the whole grid).
    integ = p**2 * (_chi_func_sd_fm_v(ctx, E, p, x, xnu, sgnq, dFDneu_moments)
                    * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                    * _fermi_stat(ctx, sgnq, 1, b)
                    + _chi_func_sd_fm_v(ctx, -E, p, x, xnu, sgnq, dFDneu_moments)
                    * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                    * _fermi_stat(ctx, sgnq, -1, b))
    return np.sum(w * integ, axis=1)


def _L_SD_FMNoCCR(ctx, T_arr, sgnq, dFDneu_moments):
    """SD-FM correction (finite-mass x spectral-distortion) WITHOUT Coulomb/radiative factors.

    Mirrors PRIMAT-Main-gray.m ``λnTOpSDFM``/``λpTOnSDFM``.  Used when
    cfg.spectral_distortions=True, cfg.analytic_distortions=True,
    cfg.finite_mass_corrections=True and cfg.radiative_corrections=False.
    """
    p, w, x, xnu = _quad_grid(ctx, T_arr)
    E = np.sqrt(p**2 + 1.)
    integ = p**2 * (_chi_func_sd_fm_v(ctx,  E, p, x, xnu, sgnq, dFDneu_moments)
                    + _chi_func_sd_fm_v(ctx, -E, p, x, xnu, sgnq, dFDneu_moments))
    return np.sum(w * integ, axis=1)


# ---------------------------------------------------------------------------
# _L_CCRTh -- finite-temperature radiative corrections (Phys. Rep. §III.H,
# Eqs. 107-113; Brown & Sawyer 2001)
# ---------------------------------------------------------------------------

# Below this temperature the correction is clamped to exactly 0 (see
# _L_CCRTh_compute's docstring for the physics/numerics reasons). The
# nTOp_thermal_<hash>.txt cache grid is therefore built down to this fixed
# floor rather than down to cfg.T_end: the integral is never actually
# evaluated below it regardless of how low cfg.T_end_MeV is set, so letting
# the grid (and hence the cache fingerprint, see cache._THERMAL_BG_FIELDS)
# depend on T_end_MeV only caused spurious cache misses -- and the
# multi-minute vegas recompute that goes with them -- for runs that changed
# T_end_MeV but were otherwise identical.
_T_CCRTH_MIN = 10**8.2  # [K]


def _L_CCRTh_interpolants(ctx):
    """Build interpolants for the finite-temperature radiative correction L_CCRTh.

    Returns a pair ``(L_nTOpCCRTh, L_pTOnCCRTh)`` of callables T[K] -> float,
    giving the additive thermal correction for n->p (sgnq=+1) and p->n
    (sgnq=-1) respectively.  When ``ctx.cfg.thermal_corrections`` is False,
    both are the zero function.

    Loaded from the fingerprinted cache in rates/weak/ when
    cfg.thermal_corrections=True and a cache file is present (see the
    module docstring and cache_utils).  A fingerprint mismatch (or a
    header-less legacy file) is reported but used anyway: recomputing this
    term is a multi-minute Monte-Carlo integration, far too slow to trigger
    automatically for what is itself only a ~1e-3-level refinement of
    L_CCR + L_FMCCR.  Only a *missing* cache file triggers a fresh
    computation.  Set cfg.thermal_corrections=False to skip this term, or
    delete the cache files and re-run with save_nTOp_thermal=True to force
    a refresh stamped with the current configuration's fingerprint.
    """
    cfg = ctx.cfg
    me, Q, xi_nu = ctx.me, ctx.Q, ctx.xi_nu
    T_nuOverT = ctx.T_nuOverT
    my_dir = ctx.my_dir

    if not cfg.thermal_corrections:
        return (lambda T: 0.0), (lambda T: 0.0)

    _td        = my_dir + "/data/weak/"
    _th_fp     = _thermal_fingerprint(cfg)
    _th_hash   = fingerprint_hash(_th_fp)
    _th_path   = _td + "nTOp_thermal_" + _th_hash + ".txt"
    _have_thermal_cache = os.path.exists(_th_path)

    if not _have_thermal_cache:
        try:
            import vegas
            _have_vegas = True
            n_eval = cfg.vegas_n_eval
            n_itn  = cfg.vegas_n_itn
        except ImportError:
            _have_vegas = False
            from scipy.integrate import dblquad
            _epsrel_th = cfg.epsrel_thermal
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
                    * (A(E, k) * (FD2_vec(-E, x) * _fermi_stat(ctx, sgnq,  1, pE / E)
                                  * (Chitilde_vec(E - k, znu, sgnq) + Chitilde_vec(E + k, znu, sgnq)
                                     - 2 * Chitilde_vec(E, znu, sgnq))
                                  + FD2_vec(E, x) * _fermi_stat(ctx, sgnq, -1, pE / E)
                                  * (Chitilde_vec(-E + k, znu, sgnq) + Chitilde_vec(-E - k, znu, sgnq)
                                     - 2 * Chitilde_vec(-E, znu, sgnq)))
                       - k * B(E) * (FD2_vec(-E, x) * _fermi_stat(ctx, sgnq,  1, pE / E)
                                     * (Chitilde_vec(E - k, znu, sgnq) - Chitilde_vec(E + k, znu, sgnq))
                                     + FD2_vec(E, x) * _fermi_stat(ctx, sgnq, -1, pE / E)
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
            res1_fac = FD2_vec(-E, x) * _fermi_stat(ctx, sgnq,  1, pE / E)
            res1vec  = Fp * Chitilde_vec(E + k, znu, sgnq)
            argvec   = k
            my_index = np.where(np.abs(argvec) < np.abs(E - sgnq * q))[0]
            res1vec[my_index] -= Fp[my_index] * FD2_vec(E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] - sgnq * q) - k[my_index])**2
            res1vec *= res1_fac
            res2_fac = FD2_vec(E, x) * _fermi_stat(ctx, sgnq, -1, pE / E)
            res2vec  = Fm * Chitilde_vec(-E + k, znu, sgnq)
            my_index = np.where(np.abs(argvec) < np.abs(E + sgnq * q))[0]
            res2vec[my_index] -= Fp[my_index] * FD2_vec(-E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] + sgnq * q) - k[my_index])**2
            res2vec *= res2_fac
            return res_fac * (res1vec + res2vec)

        def C1dE(E, x, znu, sgnq):
            pE = np.sqrt(E**2 - 1.)
            return (-(cfg.alphaem * E) / (2. * np.pi * pE) * (2. * np.pi**2) / (3. * x**2)
                    * (_chi_func(ctx, E, x, znu, sgnq) + _chi_func(ctx, -E, x, znu, sgnq)))

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
            # Clamp the finite-temperature correction to 0 below ~10^8.2 K in
            # BOTH directions.  Two independent reasons coincide there:
            #  (i) Physics: the e+/e-/photon bath is exp(-m_e/kB T)-dilute, so
            #      the thermal radiative correction is negligible (the
            #      well-behaved sub-terms are already <1e-7 by ~7e7 K).
            #  (ii) Numerics: the differential-bremsstrahlung sub-term
            #      (IPENCCRDiffBremsstrahlung) carries an alpha/(2 pi k) infrared
            #      pole that is meant to cancel against its soft subtraction.
            #      That cancellation breaks once the neutrino Fermi-Dirac
            #      threshold width 1/znu (= kB T Tnu/(me T)) shrinks toward the
            #      hard-coded lower photon-momentum cutoff k_min=1e-3, leaving an
            #      uncancelled ln(k_min) residual that the n->p integral
            #      converges to (~ -1.2e-2 at 1.16e7 K, growing logarithmically
            #      as k_min is lowered).  Below ~10^8.2 K this residual would
            #      otherwise spuriously pull the n->p rate ~0.7% below the free
            #      neutron-decay value it must approach as T -> 0.
            # PRIMAT-Main.m already applies this clamp to p->n (sgnq=-1) only
            # (lines 1639/1644/1650); here it is extended to n->p because
            # PyPRIMAT now tabulates the rates down to T_end ~ 1.16e7 K, well
            # into the regime where the unclamped n->p bremsstrahlung misbehaves.
            if T < _T_CCRTH_MIN:
                return 0.
            return (_L_ThermalTruePhoton(T, sgnq)
                    + _L_ThermalDiffBremsstrahlung(T, sgnq)
                    + _L_Thermal_1(T, sgnq)
                    + _L_Thermal_2_3(T, sgnq))

        #if cfg.verbose:
        print(f"[weak-py] Re-evaluating n <--> p thermal corrections "
              f"({'vegas' if _have_vegas else 'scipy.dblquad'}). This may take a while ...")

        # Grid floor is the fixed clamp _T_CCRTH_MIN, not cfg.T_end: every
        # point below it evaluates to exactly 0 anyway (see
        # _L_CCRTh_compute), so anchoring the grid to T_end_MeV only made the
        # cache fingerprint -- and thus a cold, multi-minute recompute --
        # depend on a parameter the integral never actually uses.
        _n_th_pts  = n_points_per_decade(cfg.sampling_nTOp_thermal_per_decade, _T_CCRTH_MIN, cfg.T_start)
        _T_th      = np.logspace(np.log10(_T_CCRTH_MIN), np.log10(cfg.T_start), _n_th_pts)
        L_nTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, +1))(_T_th)
        L_pTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, -1))(_T_th)

        if cfg.save_nTOp_thermal:
            os.makedirs(_td, exist_ok=True)
            _algo = "vegas" if _have_vegas else "scipy.dblquad"
            write_cache_with_fingerprint(
                _th_path, _th_fp, [_T_th, L_nTh_data, L_pTh_data],
                col_header="T[K] L_nTOpCCRTh L_pTOnCCRTh",
                provenance=f"backend=python algorithm={_algo} "
                           f"vegas_n_eval={cfg.vegas_n_eval} vegas_n_itn={cfg.vegas_n_itn}")

        if cfg.verbose:
            print("[weak-py] n <--> p thermal corrections computed")

        T_th, L_nTh, L_pTh = _T_th, L_nTh_data, L_pTh_data

    else:
        if cfg.verbose:
            print("[weak-py] n <--> p thermal corrections loaded from cache.")
        tab   = np.loadtxt(_th_path)
        T_th  = tab[:, 0]
        L_nTh = tab[:, 1]
        L_pTh = tab[:, 2]

    _interp_n = interp1d(T_th, L_nTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')
    _interp_p = interp1d(T_th, L_pTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')

    def _clamp_below_floor(interp):
        # T_th's lowest point is _T_CCRTH_MIN, not cfg.T_end: callers (the HT/MT
        # eras' rate lookup) still query arbitrary T down to cfg.T_end, so
        # anything below the grid floor must be pinned to 0 here rather than
        # left to interp1d's quadratic extrapolation, which is unconstrained
        # there.
        return lambda T: np.where(np.asarray(T) < _T_CCRTH_MIN, 0.,
                                   interp(np.maximum(T, _T_CCRTH_MIN)))

    L_nTOpCCRTh = _clamp_below_floor(_interp_n)
    L_pTOnCCRTh = _clamp_below_floor(_interp_p)
    return L_nTOpCCRTh, L_pTOnCCRTh


# ---------------------------------------------------------------------------
# Ordered list of named correction terms and main driver
# ---------------------------------------------------------------------------

def _correction_terms(ctx, T_arr, sgnq, dFDneu_func, dFDneu_moments=None):
    """Ordered list of (name, value) additive corrections to Gamma_{n<->p}.

    Mirrors PRIMAT-Main.m §IV.B and Table 1 of the Phys. Rep.  These are the
    terms that make up the *non-thermal* n<->p rate stored on disk; the cfg
    flags control which are active:

      radiative_corrections=True  → CCR (replaces Born)
      radiative_corrections=False → Born
      finite_mass_corrections=True + radiative_corrections=True  → FMCCR
      finite_mass_corrections=True + radiative_corrections=False → FMNoCCR
      spectral_distortions=True   → SD_CCR (if radiative_corrections) or SD
      dFDneu_moments is not None (i.e. spectral_distortions=True,
      analytic_distortions=True) + finite_mass_corrections=True →
      additionally SD_FMCCR (if radiative_corrections) or SD_FM
      (generate_rates/PRIMAT-Main-gray.m's δχFM; analytic-distortion mode
      only -- see _L_SD_FMCCR/_L_SD_FMNoCCR's docstrings)

    The finite-temperature radiative correction (CCRTh) is deliberately NOT in
    this list: it is computed and cached separately (``nTOp_thermal_<hash>.txt``
    via :func:`_L_CCRTh_interpolants`) and recombined with the stored
    Born+FM+CCR+SD rate only at point of use, in :func:`RecomputeWeakRates`.
    Keeping it out here is what lets the stored ``nTOp_<hash>.txt`` rate
    correctly approach the free neutron-decay value (1 in units of 1/tau_n) as
    T -> 0, and matches :func:`_weak_rate_fingerprint`, which never depended on
    ``thermal_corrections``.

    ``ComputeWeakRates`` sums these terms; the same list lets the test suite
    (or a notebook) inspect or pin each term's contribution individually.

    Vectorised: every term is evaluated on the whole photon temperature grid
    ``T_arr`` at once and returned as a numpy array.

    Parameters
    ----------
    ctx : _RateContext
    T_arr : np.ndarray
        Photon temperatures [K] (1-D grid).
    sgnq : +1 or -1
        +1 for n->p, -1 for p->n.
    dFDneu_func : callable or None
        Spectral-distortion function, see ComputeWeakRates.
    dFDneu_moments : dict or None
        Energy-moment functions of dFDneu_func, only available in
        analytic-distortion mode (see ComputeWeakRates); enables the SD-FM
        term when finite_mass_corrections is also True.

    Returns
    -------
    list of (str, np.ndarray)
    """
    cfg = ctx.cfg
    terms = []
    if cfg.radiative_corrections:
        terms.append(("CCR", _L_CCR(ctx, T_arr, sgnq)))
        if cfg.finite_mass_corrections:
            terms.append(("FMCCR", _L_FMCCR(ctx, T_arr, sgnq)))
    else:
        terms.append(("Born", _L_BORN(ctx, T_arr, sgnq)))
        if cfg.finite_mass_corrections:
            terms.append(("FMNoCCR", _L_FMNoCCR(ctx, T_arr, sgnq)))
    if dFDneu_func is not None:
        if cfg.radiative_corrections:
            terms.append(("SD", _L_SD_CCR(ctx, T_arr, sgnq, dFDneu_func)))
        else:
            terms.append(("SD", _L_SD(ctx, T_arr, sgnq, dFDneu_func)))
        if dFDneu_moments is not None and cfg.finite_mass_corrections:
            if cfg.radiative_corrections:
                terms.append(("SD_FM", _L_SD_FMCCR(ctx, T_arr, sgnq, dFDneu_moments)))
            else:
                terms.append(("SD_FM", _L_SD_FMNoCCR(ctx, T_arr, sgnq, dFDneu_moments)))
    return terms


def _build_rate_context(Tvec, cfg):
    """Build the :class:`_RateContext` shared by the n<->p rate integrands.

    Factored out of :func:`ComputeWeakRates` so that the thermal-correction
    interpolants (:func:`_thermal_correction_interpolants`) can be rebuilt on a
    cache hit -- when the non-thermal rate is loaded from disk and
    ComputeWeakRates is never called -- without duplicating the masses /
    T_nu(T_gamma) interpolant / numba setup.

    Args:
        Tvec: [Tg_vec, Tnu_vec], both float arrays in Kelvin (photon and
              neutrino temperatures from PRIMAT._setup_background_and_cosmo).
        cfg : PRIMATConfig instance.

    Returns:
        _RateContext instance.
    """
    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    Tg_vec, Tnu_vec = Tvec
    T_nuOverT = interp1d(Tg_vec * cfg.MeV_to_Kelvin, Tnu_vec / Tg_vec,
                         bounds_error=False, fill_value="extrapolate", kind='linear')

    integrands._setup_fd_impls(cfg.numba_installed)

    return _RateContext(cfg=cfg, me=me, mn=mn, mp=mp, Q=Q, xi_nu=cfg.munuOverTnu,
                        T_nuOverT=T_nuOverT, gA=cfg.gA, deltakappa=cfg.deltakappa,
                        my_dir=cfg.data_dir)


def _thermal_correction_interpolants(Tvec, cfg):
    """Finite-temperature radiative correction (CCRTh) as rate interpolants.

    Returns ``(Ln, Lp)``, two callables T[K] -> additive correction to the
    n->p / p->n rate in units of 1/tau_n (i.e. the raw L_CCRTh of
    :func:`_L_CCRTh_interpolants`, divided by the same neutron-decay
    phase-space factor Fn that normalises the stored non-thermal rate, so the
    two are directly addable).  When ``cfg.thermal_corrections`` is False both
    are the zero function.

    This is the "handled separately" half of the n<->p rate: the stored
    ``nTOp_<hash>.txt`` holds Born+FM+CCR+SD, the cached
    ``nTOp_thermal_<hash>.txt`` holds CCRTh, and :func:`RecomputeWeakRates`
    sums the two.  Keeping them apart on disk lets the non-thermal table
    converge cleanly to the free neutron-decay value at low T and lets the
    (slow, vegas-based) thermal table be reused across configurations that
    differ only in non-thermal flags.

    Args:
        Tvec: [Tg_vec, Tnu_vec], both arrays in Kelvin (for the T_nu(T_gamma)
              interpolant inside the thermal integrand).
        cfg : PRIMATConfig instance.

    Returns:
        (Ln, Lp): two callables T[K] -> float (rate correction in 1/tau_n).
    """
    if not cfg.thermal_corrections:
        return (lambda T: 0.0), (lambda T: 0.0)
    ctx = _build_rate_context(Tvec, cfg)
    L_nTOpCCRTh, L_pTOnCCRTh = _L_CCRTh_interpolants(ctx)
    Fn = ComputeFn(cfg)
    return (lambda T: L_nTOpCCRTh(T) / Fn), (lambda T: L_pTOnCCRTh(T) / Fn)

