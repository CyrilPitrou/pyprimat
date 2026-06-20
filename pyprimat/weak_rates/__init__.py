# -*- coding: utf-8 -*-
"""
weak_rates — n ↔ p weak interaction rates for PyPRIMAT
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

Normalisation: K is obtained from the free neutron decay rate 1/τ_n rather
than from GF/Vud/gA directly (Phys. Rep. Eqs. 89–91), giving better precision.

Module layout
-------------
This package mirrors the single-file ``weak_rates.py`` it replaced (see
FUTURE.md P1.1) and re-exports every public AND private name the rest of the
codebase / test suite addresses as ``weak_rates.<name>`` (or ``wr.<name>``),
so the split is invisible to callers:

  integrands.py  — the ~10 Fermi-Dirac kernels (FD_nu3, FD2, FD_nu_e*p*),
                    each written once as a scalar-and-array-capable function
                    (see integrands.py's module docstring), plus the numba
                    JIT-wrapping machinery (_setup_fd_impls).
  corrections.py — the physical correction terms (_L_BORN/_L_CCR/_L_FMCCR/
                    _L_SD/_L_CCRTh/...), ComputeFn, FermiCoulomb, RadCorrResum,
                    and the shared _RateContext.
  cache.py       — fingerprint dicts for the two on-disk caches
                    (nTOp_<hash>.txt, nTOp_thermal_<hash>.txt).
  api.py         — ComputeWeakRates / InterpolateWeakRates / RecomputeWeakRates,
                    the entry points called from pyprimat.background.

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095)
— cited as "Phys. Rep." with equation numbers throughout the submodules.
"""

from ..cache_utils import fingerprint_hash, write_cache_with_fingerprint

from . import integrands
from .integrands import (
    exp_cutoff, FD_nu3, FD2, FD_nu_e2p0, FD_nu_e3p0, FD_nu_e4p2,
    FD_nu_e2p2, FD_nu_e4p1, FD_nu_e2p1, FD_nu_e3p1, FD_nu_e3p2,
)
from .cache import *
from .corrections import *
from .api import *

__all__ = ['ComputeWeakRates', 'InterpolateWeakRates', 'RecomputeWeakRates', 'ComputeFn']


def _setup_fd_impls(numba_installed):
    """Package-level wrapper around :func:`integrands._setup_fd_impls`.

    ``integrands._setup_fd_impls`` rebinds the FD_* names inside the
    ``integrands`` module's own namespace (the only place every other
    submodule looks them up, via ``integrands.FD_nu3(...)`` -- see that
    module's docstring).  The ``weak_rates.FD_nu3`` aliases imported above are
    a one-shot snapshot taken at package-import time, so they go stale the
    moment this is called with a different ``numba_installed`` value unless
    re-synced here -- this wrapper does both, keeping ``wr.FD_nu3`` (used by
    tests/test_weak_rates.py) tracking the real, currently-installed
    implementation.
    """
    integrands._setup_fd_impls(numba_installed)
    globals().update({name: getattr(integrands, name)
                       for name in integrands._FD_IMPLS_ORIG})
