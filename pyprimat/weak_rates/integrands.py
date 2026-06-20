# -*- coding: utf-8 -*-
"""
weak_rates.integrands — Fermi-Dirac integrand kernels for the n<->p rates
===============================================================================

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

Each kernel is njit-compiled via :func:`_setup_fd_impls` when numba is
available.  Other weak_rates submodules must call through the
``integrands`` module object (e.g. ``integrands.FD_nu3(...)``), never via a
direct ``from .integrands import FD_nu3``: ``_setup_fd_impls`` rebinds these
names in *this* module's namespace, and a plain name-import would freeze a
caller onto whichever variant (jitted or plain Python) was installed first.

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095).
"""

import numpy as np

__all__ = ['exp_cutoff', 'FD_nu3', 'FD2', 'FD_nu_e2p0', 'FD_nu_e3p0',
           'FD_nu_e4p2', 'FD_nu_e2p2', 'FD_nu_e4p1', 'FD_nu_e2p1',
           'FD_nu_e3p1', 'FD_nu_e3p2', '_FD_IMPLS_ORIG', '_fd_impls_numba',
           '_setup_fd_impls']

exp_cutoff = 3e+2

# ---------------------------------------------------------------------------
# Fermi-Dirac helper functions — JIT-compiled when numba is available.
# These capture nothing from any enclosing scope (only the module-level
# exp_cutoff constant), so they can live at module level and be wrapped
# with @njit.  Call _setup_fd_impls(cfg.numba_installed) before first use.
#
# Each kernel is written ONCE, using np.where for the tail cutoff (instead
# of a Python if/else), which makes it simultaneously:
#   - a valid scalar callable (for the scipy.quad/dblquad calls in
#     _L_CCRTh_interpolants -- np.where on scalars returns a 0-d array that
#     behaves like a float in all downstream arithmetic), and
#   - a valid array callable (for the Gauss-Legendre rate-integral grid in
#     _quad_grid and friends) without a separate hand-maintained "_v" twin.
# Before being squashed by np.where, every exp() argument is clamped with
# np.minimum so the discarded (masked-out) branch can never overflow --
# this also makes the functions numba-njit-compatible (numba supports
# np.where/np.minimum on both scalars and arrays, but not the
# np.errstate(...) context manager the old hand-written "_v" twins used to
# silence overflow warnings in the unclamped masked branch).
# ---------------------------------------------------------------------------

def FD_nu3(E, phi, x):
    arg = x * E - phi
    return np.where(arg < exp_cutoff, 1. / (np.exp(np.minimum(arg, exp_cutoff)) + 1.), 0.)

def FD2(E, x):
    arg = x * E
    return np.where(arg < exp_cutoff, 1. / (np.exp(np.minimum(arg, exp_cutoff)) + 1.), 0.)

def FD_nu_e2p0(E, phi, x):
    arg = x * E - phi
    return np.where(arg < exp_cutoff, E**2 / (np.exp(np.minimum(arg, exp_cutoff)) + 1.), 0.)

def FD_nu_e3p0(E, phi, x):
    arg = x * E - phi
    return np.where(arg < exp_cutoff, E**3 / (np.exp(np.minimum(arg, exp_cutoff)) + 1.), 0.)

def FD_nu_e4p2(E, phi, x):
    Ex = E * x
    guard = (2. * phi < exp_cutoff) & (Ex + phi < exp_cutoff) & (2. * Ex < exp_cutoff)
    half = exp_cutoff / 2.
    Exc, phic = np.minimum(Ex, half), np.minimum(phi, half)
    expr = (E**2 * np.exp(phic) * ((24. - Ex * (Ex + 8.)) * np.exp(Exc + phic)
            + np.exp(2. * Exc) * (Ex - 6.) * (Ex - 2.) + 12. * np.exp(2. * phic))
            / (np.exp(Exc) + np.exp(phic))**3)
    return np.where(guard, expr, 0.)

def FD_nu_e2p2(E, phi, x):
    Ex = E * x
    guard = (3. * phi < exp_cutoff) & (2. * Ex + phi < exp_cutoff) & (Ex < exp_cutoff)
    third = exp_cutoff / 3.
    Exc, phic = np.minimum(Ex, third), np.minimum(phi, third)
    expr = (((Ex * (Ex - 4.) + 2.) * np.exp(2. * Exc + phic)
             + (4. - Ex * (Ex + 4.)) * np.exp(Exc + 2. * phic)
             + 2. * np.exp(3. * phic))
            / (np.exp(Exc) + np.exp(phic))**3)
    return np.where(guard, expr, 0.)

def FD_nu_e4p1(E, phi, x):
    Ex = E * x
    guard = (phi < exp_cutoff) & (Ex < exp_cutoff)
    Exc, phic = np.minimum(Ex, exp_cutoff), np.minimum(phi, exp_cutoff)
    expr = (np.exp(phic) * E**3 * (4. * np.exp(phic) + np.exp(Exc) * (4. - Ex))
            / (np.exp(Exc) + np.exp(phic))**2)
    return np.where(guard, expr, 0.)

def FD_nu_e2p1(E, phi, x):
    Ex = E * x
    guard = (phi < exp_cutoff) & (Ex < exp_cutoff)
    Exc, phic = np.minimum(Ex, exp_cutoff), np.minimum(phi, exp_cutoff)
    expr = (np.exp(phic) * E * (2. * np.exp(phic) + np.exp(Exc) * (2. - Ex))
            / (np.exp(Exc) + np.exp(phic))**2)
    return np.where(guard, expr, 0.)

def FD_nu_e3p1(E, phi, x):
    Ex = E * x
    guard = (phi < exp_cutoff) & (Ex < exp_cutoff)
    Exc, phic = np.minimum(Ex, exp_cutoff), np.minimum(phi, exp_cutoff)
    expr = (np.exp(phic) * E**2 * (3. * np.exp(phic) + np.exp(Exc) * (3. - Ex))
            / (np.exp(Exc) + np.exp(phic))**2)
    return np.where(guard, expr, 0.)

def FD_nu_e3p2(E, phi, x):
    Ex = E * x
    guard = (2. * phi < exp_cutoff) & (Ex + phi < exp_cutoff) & (2. * Ex < exp_cutoff)
    half = exp_cutoff / 2.
    Exc, phic = np.minimum(Ex, half), np.minimum(phi, half)
    expr = (E * np.exp(phic)
            * ((12. - Ex * (Ex + 6.)) * np.exp(Exc + phic)
               + np.exp(2. * Exc) * (Ex * (Ex - 6.) + 6.)
               + 6. * np.exp(2. * phic))
            / (np.exp(Exc) + np.exp(phic))**3)
    return np.where(guard, expr, 0.)


# Pristine (pure-Python) implementations, kept aside so _setup_fd_impls can
# re-wrap from scratch if called again with a different numba_installed value
# (otherwise a second PyPRConfig with the opposite setting would silently
# keep reusing whichever variant -- jitted or not -- was installed first).
_FD_IMPLS_ORIG = dict(
    FD_nu3=FD_nu3, FD2=FD2, FD_nu_e2p0=FD_nu_e2p0, FD_nu_e3p0=FD_nu_e3p0,
    FD_nu_e4p2=FD_nu_e4p2, FD_nu_e2p2=FD_nu_e2p2, FD_nu_e4p1=FD_nu_e4p1,
    FD_nu_e2p1=FD_nu_e2p1, FD_nu_e3p1=FD_nu_e3p1, FD_nu_e3p2=FD_nu_e3p2,
)

# Remembers which numba_installed value the module-level FD_* names were last
# wrapped for; None means "not yet set up".
_fd_impls_numba = None


def _setup_fd_impls(numba_installed):
    global FD_nu3, FD2, FD_nu_e2p0, FD_nu_e3p0, FD_nu_e4p2, FD_nu_e2p2, \
           FD_nu_e4p1, FD_nu_e2p1, FD_nu_e3p1, FD_nu_e3p2, _fd_impls_numba
    if _fd_impls_numba == numba_installed:
        return
    _fd_impls_numba = numba_installed
    # Always start from the pristine pure-Python implementations so this is
    # idempotent regardless of which way numba_installed flips.
    FD_nu3      = _FD_IMPLS_ORIG['FD_nu3']
    FD2         = _FD_IMPLS_ORIG['FD2']
    FD_nu_e2p0  = _FD_IMPLS_ORIG['FD_nu_e2p0']
    FD_nu_e3p0  = _FD_IMPLS_ORIG['FD_nu_e3p0']
    FD_nu_e4p2  = _FD_IMPLS_ORIG['FD_nu_e4p2']
    FD_nu_e2p2  = _FD_IMPLS_ORIG['FD_nu_e2p2']
    FD_nu_e4p1  = _FD_IMPLS_ORIG['FD_nu_e4p1']
    FD_nu_e2p1  = _FD_IMPLS_ORIG['FD_nu_e2p1']
    FD_nu_e3p1  = _FD_IMPLS_ORIG['FD_nu_e3p1']
    FD_nu_e3p2  = _FD_IMPLS_ORIG['FD_nu_e3p2']
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

