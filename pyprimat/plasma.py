# -*- coding: utf-8 -*-
"""
plasma.py — SM plasma thermodynamics for PyPRIMAT
=================================================

Implements the photon, electron–positron, and neutrino thermodynamics
used in the background evolution and Friedmann equation during BBN.
All thermodynamic quantities follow from the general Fermi-Dirac / Bose-
Einstein integrals defined in App. A of the main reference (Eq. A1–A7):

    ρ = g T⁴/(2π²) × I±^{(2,1)}(x)        (Eq. A4b)
    P = g T⁴/(6π²) × I±^{(0,3)}(x)        (Eq. A4c)
    s = (ρ + P) / T                          (Eq. 21/24)

with x ≡ m/T and g the spin degeneracy (g=2 photons, g=4 e±, g=1 ν).

QED interaction-pressure corrections to the EM plasma thermodynamics
are loaded from pre-saved tables in ``rates/plasma/`` when available,
or computed analytically via :mod:`pyprimat.qed_pressure` otherwise
(see §II.E, Eq. 47–49 of the reference).

Reference
---------
Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095)
— cited below as "Phys. Rep." with equation numbers.

Design
------
:class:`Plasma` bundles all of the state below (QED-pressure
interpolants, e± integrand implementations, optional electron-thermo
tables) as instance attributes bound to one ``PyPRConfig``.  Each
``PyPR`` instance owns its own ``Plasma`` (``self.plasma``), so several
``PyPR`` instances — e.g. comparing ``QED_corrections=True/False``, or
the per-worker instances used by the Monte-Carlo machinery — can coexist
in the same process without overwriting each other's tables.

Aside from a handful of genuinely cfg-independent pure functions
(``rho_g``, ``drho_g_dT``, ``rho_nu``, ``drho_nu_dT``), every quantity here
is config-dependent and lives on a :class:`Plasma` instance — there is no
module-level mutable default. Build one explicitly:

    >>> from pyprimat.config import PyPRConfig
    >>> from pyprimat.plasma import Plasma
    >>> plasma = Plasma(PyPRConfig())
    >>> plasma.rho_e(Tg)
"""

import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, CubicSpline

from .cache_utils import (fingerprint_hash, read_cache_fingerprint_hash,
                           write_cache_with_fingerprint)

# Bump on any change to the electron-thermo cache's numerical content or
# layout (see Plasma._build_electron_tables).
ELECTRON_THERMO_FORMAT_VERSION = 1

__all__ = [
    'Plasma',
    'rho_g', 'drho_g_dT',
    'rho_nu', 'drho_nu_dT',
]

# Below T = me / _ELEC_THERMO_LOWT_RATIO the e± number density is
# Boltzmann-suppressed by exp(−me/T) < exp(−30) ≃ 10⁻¹³ relative to
# photons, so all four e± thermodynamic quantities are set to exactly zero.
# This avoids integrating a numerically negligible and potentially slow
# tail.  (See non-relativistic limit Eq. A8 of Phys. Rep.)
_ELEC_THERMO_LOWT_RATIO = 30.

# High-T limit of spl/Tγ³ (photon + e±) used for T_nu_decoupling.
# From Phys. Rep. Eq. 25d and 26d: s̄_γ = 4π²/45, s̄_e± = 7/8 × s̄_γ each,
# so s̄_pl = s̄_γ + 2 × (7/8) × s̄_γ = (1 + 7/4) s̄_γ = (11/4) × 4π²/45 = 11π²/45.
_sigma_inf = 11. * np.pi**2 / 45.


# ---------------------------------------------------------------------------
# Photons and SM neutrinos: pure functions of temperature, no cfg dependence.
# ---------------------------------------------------------------------------

def rho_g(Tg):
    """Photon energy density [MeV⁴].

    For a Bose-Einstein gas of photons with g=2 spin states and
    vanishing chemical potential (Phys. Rep. Eq. 25b):

        ρ_γ = (π²/15) Tγ⁴ = 2 × (π²/30) Tγ⁴

    Parameters
    ----------
    Tg : float
        Photon temperature [MeV].

    Returns
    -------
    float
        ρ_γ in MeV⁴.

    Example
    -------
    >>> rho_g(1.0)   # at Tγ = 1 MeV
    0.6493...
    """
    return 2. * (np.pi**2 / 30.) * Tg**4


def drho_g_dT(Tg):
    """Temperature derivative of the photon energy density [MeV³].

    From ρ_γ ∝ Tγ⁴ (Phys. Rep. Eq. 25b):

        dρ_γ/dTγ = 4 ρ_γ / Tγ

    Parameters
    ----------
    Tg : float
        Photon temperature [MeV].

    Returns
    -------
    float
        dρ_γ/dTγ in MeV³.
    """
    return 4. * rho_g(Tg) / Tg


def rho_nu(Tnu):
    """Energy density of one SM neutrino flavour (ν + ν̄) [MeV⁴].

    Each species is a relativistic Fermi gas with g=1.  Counting both
    helicities (ν and ν̄) gives a factor 2, and the Fermi factor 7/8
    relative to photons (Phys. Rep. Eq. 26b):

        ρ_ν = 2 × (7/8) × (π²/30) × Tν⁴

    Parameters
    ----------
    Tnu : float
        Neutrino temperature [MeV].

    Returns
    -------
    float
        ρ_ν for one flavour in MeV⁴.

    Example
    -------
    >>> rho_nu(1.0)   # one neutrino flavour at 1 MeV
    0.5674...
    """
    return 2. * (7. / 8.) * (np.pi**2 / 30.) * Tnu**4


def drho_nu_dT(Tnu):
    """Temperature derivative of one-flavour neutrino energy density [MeV³].

    From ρ_ν ∝ Tν⁴ (Phys. Rep. Eq. 26b):

        dρ_ν/dTν = 4 ρ_ν / Tν

    Parameters
    ----------
    Tnu : float
        Neutrino temperature [MeV].

    Returns
    -------
    float
        dρ_ν/dTν in MeV³.
    """
    return 4. * rho_nu(Tnu) / Tnu


# ---------------------------------------------------------------------------
# Plasma: per-instance thermodynamics bound to one PyPRConfig.
# ---------------------------------------------------------------------------

class Plasma:
    """SM plasma (photons + e± + QED corrections + neutrinos) for one config.

    An instance loads/computes everything that depends on ``cfg`` once, in
    :meth:`__init__`, and exposes it through instance methods and attributes:

    - :attr:`PQEDofT`, :attr:`dPQEDdT`, :attr:`d2PQEDdT2`: callables giving
      the QED interaction-pressure correction δP(Tγ) and its first two
      Tγ-derivatives [MeV⁴, MeV³, MeV²] (zero functions if
      ``cfg.QED_corrections`` is False).
    - :meth:`rho_e`, :meth:`drho_e_dT`, :meth:`p_e`, :meth:`dp_e_dT`: e±
      energy density / pressure and their Tγ-derivatives [MeV⁴ / MeV³],
      dispatching to a pre-built cubic interpolant (see
      :meth:`_build_electron_tables`).
    - :meth:`rho_nu_extra`, :meth:`rho_SM`, :meth:`p_SM`, :meth:`spl`,
      :meth:`spl_and_dspl_dT`, :meth:`dspl_dT`, :meth:`T_nu_decoupling`:
      composite quantities built from the above plus the cfg-independent
      :func:`rho_g`/:func:`rho_nu`.

    Two instances built from different configs (e.g. one with
    ``QED_corrections=True`` and one with ``QED_corrections=False``) are
    fully independent — evaluating one never mutates the other, so they may
    be interleaved freely (parameter scans, Monte-Carlo workers, etc.).

    Example
    -------
    >>> from pyprimat import config, plasma
    >>> p = plasma.Plasma(config.PyPRConfig())
    >>> p.rho_g(1.0)   # cfg-independent quantities are also reachable as methods
    0.6493...
    """

    def __init__(self, cfg):
        """Load all tables needed for the thermodynamic functions below.

        Parameters
        ----------
        cfg : PyPRConfig
            Fully initialised configuration object.  A reference is kept
            (``self._cfg``) for the electron mass ``cfg.me``,
            ``cfg.DeltaNeff``, and the various table-loading flags.

        Internally calls :meth:`_load_tables` (QED pressure correction
        files), :meth:`_setup_integrand_impls` (compile/define the e±
        integrands), and :meth:`_build_electron_tables` (optional
        pre-tabulation of e±).
        """
        self._cfg = cfg

        # JIT-compiled (or plain Python) e± integrand implementations.
        # Each integrand is a function (E, Tg) where E = ε/T is dimensionless
        # and Tg is the photon temperature in MeV.  Set by
        # _setup_integrand_impls() and used by the _*_exact() methods below.
        self._rho_e_int_impl  = None   # integrand for ρ_e
        self._drho_e_dT_impl  = None   # integrand for dρ_e/dT
        self._p_e_int_impl    = None   # integrand for p_e
        self._dp_e_dT_impl    = None   # integrand for dp_e/dT

        # QED interaction-pressure correction callables (set by
        # _load_tables).  δP(T) encodes the leading finite-temperature QED
        # correction to the EM plasma pressure; ρ_QED = T dδP/dT − δP,
        # p_QED = δP (Phys. Rep. §II.E).
        self.PQEDofT   = None   # δP(Tγ) [MeV⁴]
        self.dPQEDdT   = None   # d(δP)/dTγ [MeV³]
        self.d2PQEDdT2 = None   # d²(δP)/dTγ² [MeV²]

        # Electron-thermodynamics interpolants, built by
        # _build_electron_tables below (placeholders until that call returns).
        self._rho_e_tab     = None
        self._p_e_tab       = None
        self._drho_e_dT_tab = None
        self._dp_e_dT_tab   = None

        self._load_tables(cfg)
        self._setup_integrand_impls(cfg)
        self._build_electron_tables(cfg)
        if cfg.verbose:
            print("[init]  Tables loaded.")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_tables(self, cfg):
        """Load or compute the QED interaction-pressure correction tables.

        Sets ``self.PQEDofT``, ``self.dPQEDdT``, ``self.d2PQEDdT2``: callables
        representing the finite-temperature QED correction to the EM plasma
        pressure δP and its temperature derivatives.  They enter every
        thermodynamic quantity through (Phys. Rep. §II.E, Eq. 47):

            ρ_QED = Tγ dδP/dTγ − δP
            p_QED = δP

        so that the total EM-plasma energy density and pressure are:

            ρ_pl = ρ_γ + ρ_e + ρ_QED,   p_pl = p_γ + p_e + p_QED

        When ``cfg.QED_corrections`` is False all three are set to the zero
        function, reducing every thermodynamic quantity to the free ideal-gas
        expression; in that limit spl/Tγ³ → 11π²/45 exactly at high T.

        Three modes (controlled by ``cfg.recompute_qed_corrections``):

        **File mode** (default, ``recompute_qed_corrections=False``, files present):
            Loads ``rates/plasma/QED_P_int.txt``, ``QED_dP_intdT.txt``, and
            ``QED_d2P_intdT2.txt``.  Each file has three columns (T, e²-order,
            e³-order); both columns are summed to give the total δP.

        **Analytic fallback** (``recompute_qed_corrections=False``, files absent):
            Calls :func:`pyprimat.qed_pressure.compute_qed_pressure_tables`
            to evaluate the QED corrections analytically (~0.3 s) without
            writing any files.  Useful on a fresh checkout.

        **Recompute mode** (``recompute_qed_corrections=True``):
            Always computes analytically and overwrites the three
            ``rates/plasma/QED_*.txt`` files so they serve as a cached copy
            for subsequent runs.  Use this after changing physical constants or
            to regenerate after deleting the files.

        The QED pressure decomposition follows PRIMAT-Main.m:
          - δP_a [O(e²)]:  leading one-loop correction (dPa in PRIMAT)
          - δP_{e3} [O(e³)]: ring/plasmon contribution (dPe3 in PRIMAT)
          (The O(e⁴) exchange term δP_b is not included; see
          :func:`pyprimat.qed_pressure._dPb` for the optional computation.)
        """
        if not cfg.QED_corrections:
            self.PQEDofT   = lambda T: 0.
            self.dPQEDdT   = lambda T: 0.
            self.d2PQEDdT2 = lambda T: 0.
            return

        plasma_dir = os.path.join(cfg.data_dir, "rates", "plasma")
        p_file   = os.path.join(plasma_dir, "QED_P_int.txt")
        dp_file  = os.path.join(plasma_dir, "QED_dP_intdT.txt")
        d2p_file = os.path.join(plasma_dir, "QED_d2P_intdT2.txt")

        files_present = (os.path.exists(p_file) and os.path.exists(dp_file)
                         and os.path.exists(d2p_file))
        recompute = cfg.recompute_qed_corrections

        if recompute or not files_present:
            # Compute analytically (~0.3 s).  In recompute mode, also save files.
            from .qed_pressure import compute_qed_pressure_tables, save_qed_tables
            if cfg.verbose:
                reason = "recompute requested" if recompute else "files not found"
                print(f"[init]  Computing QED plasma-pressure tables ({reason})…")
            tables = compute_qed_pressure_tables(
                T_min=1e-3, T_max=1e2, n_pts=500, verbose=False)
            if recompute:
                save_qed_tables(tables, plasma_dir, verbose=cfg.verbose)
            # Build interpolants directly from the computed arrays via CubicSpline,
            # which is smoother and more accurate than the linear interp1d used
            # when loading from files.
            T = tables["T"]
            spl_P   = CubicSpline(T, tables["dP_e2"]       + tables["dP_e3"])
            spl_dP  = CubicSpline(T, tables["d_dP_e2_dT"]  + tables["d_dP_e3_dT"])
            spl_d2P = CubicSpline(T, tables["d2_dP_e2_dT2"] + tables["d2_dP_e3_dT2"])
            self.PQEDofT   = lambda T, _s=spl_P:   float(_s(T))
            self.dPQEDdT   = lambda T, _s=spl_dP:  float(_s(T))
            self.d2PQEDdT2 = lambda T, _s=spl_d2P: float(_s(T))
            return

        # Load from the saved files (linear interpolation matches the file precision).
        t = np.loadtxt(p_file)
        self.PQEDofT = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                        fill_value="extrapolate", assume_sorted=False, kind='linear')
        t = np.loadtxt(dp_file)
        self.dPQEDdT = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                        fill_value="extrapolate", assume_sorted=False, kind='linear')
        t = np.loadtxt(d2p_file)
        self.d2PQEDdT2 = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                          fill_value="extrapolate", assume_sorted=False, kind='linear')

    def _setup_integrand_impls(self, cfg):
        """Define the four e± Fermi-Dirac integrands, optionally JIT-compiled.

        Each integrand is a function of dimensionless energy E = ε/Tγ
        (lower integration limit x = me/Tγ) and the photon temperature Tγ.
        The four quantities follow from Eqs. (A4b, A4c) of Phys. Rep. with
        g = 4 (counting e⁺ and e⁻), x = me/Tγ, and ξ = 0 (vanishing
        chemical potential, justified in App. A.2):

            ρ_e = 4/(2π²) Tγ⁴ ∫_{x}^{∞} E² √(E²−x²) / (eᴱ+1) dE    (A4b)
            p_e = 4/(6π²) Tγ⁴ ∫_{x}^{∞} (E²−x²)^{3/2} / (eᴱ+1) dE   (A4c)

        The dρ_e/dTγ and dp_e/dTγ integrands are obtained by differentiating
        the Fermi-Dirac factor w.r.t. Tγ: d/dTγ[1/(eᴱ+1)] = E/(4Tγ cosh²(E/2)).

        When ``cfg.numba_installed`` is True and ``numba`` is importable, all four
        are wrapped with ``@njit(cache=True)`` for JIT compilation; otherwise
        equivalent pure-Python functions are used.  Numerical results are
        identical in both cases — only speed differs.

        Sets ``self._rho_e_int_impl``, ``self._drho_e_dT_impl``,
        ``self._p_e_int_impl``, ``self._dp_e_dT_impl``.
        """
        me_val = cfg.me

        # Pure-Python implementations (single source of truth for the
        # formulae); optionally JIT-wrapped below so the numba and
        # pure-Python code paths cannot drift out of sync with each other.
        def _rho_e_intgd(E, Tg):
            # Integrand of ρ_e: E² √(E²−(me/Tg)²) / (eᴱ+1)
            return E**2 * (E**2 - (me_val / Tg)**2)**0.5 / (np.exp(E) + 1.)

        def _drho_e_dT_intgd(E, Tg):
            # Integrand of dρ_e/dTg: E³ √(E²−(me/Tg)²) / (4 cosh²(E/2))
            # Factor E/(4 cosh²(E/2)) = −Tg d/dTg [1/(eᴱ+1)]
            return E**3 * (E**2 - (me_val / Tg)**2)**0.5 / np.cosh(E / 2.0)**2

        def _p_e_intgd(E, Tg):
            # Integrand of p_e: (E²−(me/Tg)²)^{3/2} / (eᴱ+1)
            return (E**2 - (me_val / Tg)**2)**1.5 / (np.exp(E) + 1.)

        def _dp_e_dT_intgd(E, Tg):
            # Integrand of dp_e/dTg: E (E²−(me/Tg)²)^{3/2} / (4 cosh²(E/2))
            return E*(E**2 - (me_val / Tg)**2)**1.5 / np.cosh(E / 2.0)**2

        if cfg.numba_installed:
            try:
                from numba import njit
                _rho_e_intgd     = njit(_rho_e_intgd)
                _drho_e_dT_intgd = njit(_drho_e_dT_intgd)
                _p_e_intgd       = njit(_p_e_intgd)
                _dp_e_dT_intgd   = njit(_dp_e_dT_intgd)
            except ImportError:
                pass

        self._rho_e_int_impl = _rho_e_intgd
        self._drho_e_dT_impl = _drho_e_dT_intgd
        self._p_e_int_impl   = _p_e_intgd
        self._dp_e_dT_impl   = _dp_e_dT_intgd

    # ------------------------------------------------------------------
    # e± — exact quad-based implementations
    # ------------------------------------------------------------------
    # Each of the four quantities has a ``_*_exact`` implementation using
    # scipy.quad and a public entry point that dispatches to a pre-built
    # interpolant when available (faster; see _build_electron_tables).

    def _rho_e_exact(self, Tg):
        """Exact e± energy density via numerical quadrature (Phys. Rep. Eq. A4b)."""
        me = self._cfg.me
        if Tg < me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        r = quad(self._rho_e_int_impl, me / Tg, 100., args=(Tg,),
                 epsabs=1e-12, epsrel=1e-12)[0]
        # Prefactor: g/(2π²) Tg⁴ with g=4 for e⁺+e⁻.
        return 4. / (2 * np.pi**2) * Tg**4 * r

    def _drho_e_dT_exact(self, Tg):
        """Exact dρ_e/dTγ via numerical quadrature."""
        me = self._cfg.me
        if Tg < me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        r = quad(self._drho_e_dT_impl, me / Tg, 100., args=(Tg,),
                 epsabs=1e-12, epsrel=1e-12)[0]
        # Prefactor: g/(2π²) Tg³ × (1/4) = g/(8π²) Tg³; combined with the
        # extra E in the integrand this gives dρ/dT.
        return 1. / (2 * np.pi**2) * Tg**3 * r

    def _p_e_exact(self, Tg):
        """Exact e± pressure via numerical quadrature (Phys. Rep. Eq. A4c)."""
        me = self._cfg.me
        if Tg < me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        r = quad(self._p_e_int_impl, me / Tg, 100., args=(Tg,),
                 epsabs=1e-12, epsrel=1e-12)[0]
        # Prefactor: g/(6π²) Tg⁴ with g=4.
        return 4. / (6 * np.pi**2) * Tg**4 * r

    def _dp_e_dT_exact(self, Tg):
        """Exact dp_e/dTγ via numerical quadrature."""
        me = self._cfg.me
        if Tg < me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        r = quad(self._dp_e_dT_impl, me / Tg, 100., args=(Tg,),
                 epsabs=1e-12, epsrel=1e-12)[0]
        return 1. / (6 * np.pi**2) * Tg**3 * r

    def _build_electron_tables(self, cfg):
        """Pre-tabulate the four e± thermodynamic quantities on a log-Tγ grid.

        The e± integrands depend only on the (fixed) electron mass and the
        photon temperature, so a single table built here is reused for the
        whole run.  Cubic interpolation on a ``cfg.n_electron_table``-point
        grid reproduces the exact integrals to a few parts in 1e6 over the
        active temperature range, well within BBN tolerances.

        The computed arrays are stored in ``rates/plasma/electron_thermo_cache.txt``
        (one row per temperature, columns: T ρ_e p_e dρ_e/dT dp_e/dT) so that
        subsequent runs load from disk instead of repeating the ~8000 quad
        calls (~0.7 s).  The file carries a fingerprint header (see
        cache_utils) recording ``n_electron_table`` and
        ``T_start_cosmo_MeV`` -- the two config entries that determine the grid
        -- so a stale cache (e.g. left over from a run with a different
        ``n_electron_table``) is detected and rebuilt automatically.  Set
        ``cfg.recompute_electron_thermo = True`` to force a fresh computation
        regardless of the fingerprint.  Whenever the table is recomputed
        (fingerprint mismatch, missing file, or ``recompute_electron_thermo``)
        it is written back to ``rates/plasma/electron_thermo_cache.txt`` with
        the current fingerprint, so the cache is always self-consistent with
        the configuration that last ran.

        Sets ``self._rho_e_tab``, ``self._p_e_tab``, ``self._drho_e_dT_tab``,
        ``self._dp_e_dT_tab``.
        """
        cache_path = os.path.join(cfg.data_dir, "rates", "plasma",
                                  "electron_thermo_cache.txt")

        Tmin = cfg.me / _ELEC_THERMO_LOWT_RATIO
        Tmax = max(cfg.T_start_cosmo_MeV, 100.) * 1.5
        grid = np.logspace(np.log10(Tmin), np.log10(Tmax), cfg.n_electron_table)

        fp = {"format_version":  ELECTRON_THERMO_FORMAT_VERSION,
              "n_electron_table": cfg.n_electron_table,
              "T_start_cosmo_MeV": cfg.T_start_cosmo_MeV}
        fp_hash = fingerprint_hash(fp)

        # Try loading from disk cache first (skips ~0.7 s of quad calls).
        if (not cfg.recompute_electron_thermo
                and read_cache_fingerprint_hash(cache_path) == fp_hash):
            try:
                d = np.loadtxt(cache_path)
                self._rho_e_tab     = interp1d(d[:, 0], d[:, 1], kind='cubic',
                                          bounds_error=False, fill_value="extrapolate")
                self._p_e_tab       = interp1d(d[:, 0], d[:, 2], kind='cubic',
                                          bounds_error=False, fill_value="extrapolate")
                self._drho_e_dT_tab = interp1d(d[:, 0], d[:, 3], kind='cubic',
                                          bounds_error=False, fill_value="extrapolate")
                self._dp_e_dT_tab   = interp1d(d[:, 0], d[:, 4], kind='cubic',
                                          bounds_error=False, fill_value="extrapolate")
                if cfg.verbose:
                    print(f"[init]  Electron-thermo tables loaded from cache ({cfg.n_electron_table} points).")
                return
            except Exception as exc:
                import warnings
                warnings.warn(f"[plasma] Could not read electron-thermo cache "
                               f"({exc}); falling back to recompute.")

        # Compute from scratch.
        rho_e_arr     = np.array([self._rho_e_exact(T)     for T in grid])
        p_e_arr       = np.array([self._p_e_exact(T)       for T in grid])
        drho_e_dT_arr = np.array([self._drho_e_dT_exact(T) for T in grid])
        dp_e_dT_arr   = np.array([self._dp_e_dT_exact(T)   for T in grid])

        self._rho_e_tab     = interp1d(grid, rho_e_arr,     kind='cubic',
                                  bounds_error=False, fill_value="extrapolate")
        self._p_e_tab       = interp1d(grid, p_e_arr,       kind='cubic',
                                  bounds_error=False, fill_value="extrapolate")
        self._drho_e_dT_tab = interp1d(grid, drho_e_dT_arr, kind='cubic',
                                  bounds_error=False, fill_value="extrapolate")
        self._dp_e_dT_tab   = interp1d(grid, dp_e_dT_arr,   kind='cubic',
                                  bounds_error=False, fill_value="extrapolate")

        # Save to disk so future runs with the same fingerprint can load
        # instead of recomputing.
        try:
            write_cache_with_fingerprint(
                cache_path, fp,
                [grid, rho_e_arr, p_e_arr, drho_e_dT_arr, dp_e_dT_arr],
                col_header='grid rho_e p_e drho_e_dT dp_e_dT')
        except Exception as exc:
            import warnings
            warnings.warn(f"[plasma] Could not write electron-thermo cache: {exc}")

        if cfg.verbose:
            print(f"[init]  Electron-thermo tables built ({cfg.n_electron_table} points).")

    # ------------------------------------------------------------------
    # e± — public entry points (dispatch to table or exact)
    # ------------------------------------------------------------------

    def rho_e(self, Tg):
        """e± energy density [MeV⁴] (Phys. Rep. Eq. A4b with g=4, x=me/Tγ).

        For T < me/30 the contribution is exponentially suppressed and
        returned as exactly 0 (see ``_ELEC_THERMO_LOWT_RATIO``).
        Uses the pre-built cubic interpolant from
        :meth:`_build_electron_tables`.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            ρ_e in MeV⁴.

        Example
        -------
        >>> p.rho_e(0.511)   # at Tγ ≃ me, e± contribution is comparable to photons
        """
        if Tg < self._cfg.me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        return float(self._rho_e_tab(Tg))

    def drho_e_dT(self, Tg):
        """Temperature derivative of the e± energy density [MeV³].

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            dρ_e/dTγ in MeV³.
        """
        if Tg < self._cfg.me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        return float(self._drho_e_dT_tab(Tg))

    def p_e(self, Tg):
        """e± pressure [MeV⁴] (Phys. Rep. Eq. A4c with g=4, x=me/Tγ).

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            p_e in MeV⁴.
        """
        if Tg < self._cfg.me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        return float(self._p_e_tab(Tg))

    def dp_e_dT(self, Tg):
        """Temperature derivative of the e± pressure [MeV³].

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            dp_e/dTγ in MeV³.
        """
        if Tg < self._cfg.me / _ELEC_THERMO_LOWT_RATIO:
            return 0.0
        return float(self._dp_e_dT_tab(Tg))

    # ------------------------------------------------------------------
    # Neutrinos (extra species), SM totals, plasma entropy
    # ------------------------------------------------------------------

    def rho_nu_extra(self, Tg):
        """Energy density of ΔNeff extra decoupled relativistic species [MeV⁴].

        Models additional light species (e.g. a sterile neutrino or dark
        radiation) that decoupled instantaneously and whose temperature
        therefore scales as 1/a exactly (same as the instantaneous-decoupling
        SM neutrino temperature :meth:`T_nu_decoupling`).  Returns 0 when
        ``cfg.DeltaNeff == 0``.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            ΔNeff × ρ_ν(Tν_dec) in MeV⁴.
        """
        if self._cfg.DeltaNeff == 0.:
            return 0.
        Tnu_dec = self.T_nu_decoupling(Tg)
        return self._cfg.DeltaNeff * 2. * (7. / 8.) * (np.pi**2 / 30.) * Tnu_dec**4

    def rho_SM(self, Tg, Tnue, Tnumu):
        """Total SM energy density during BBN [MeV⁴] (Phys. Rep. Eq. 43).

        Includes photons, e±, QED interaction-pressure correction, the
        3 SM neutrino flavours (νe with temperature Tnue, νμ and ντ sharing
        Tnumu), and any extra ΔNeff species.

        The QED energy density correction is ρ_QED = Tγ dδP/dTγ − δP
        (Phys. Rep. §II.E, Eq. 47).  Baryons and cold dark matter are
        treated separately via their dilution relation (Eq. 34) and are
        NOT included here.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].
        Tnue : float
            Electron-neutrino temperature [MeV].
        Tnumu : float
            Muon- and tau-neutrino temperature [MeV] (assumed equal).

        Returns
        -------
        float
            ρ_SM in MeV⁴.

        Example
        -------
        >>> p.rho_SM(1.0, 1.0, 1.0)   # all temperatures equal at 1 MeV
        """
        rho_qed = Tg * self.dPQEDdT(Tg) - self.PQEDofT(Tg)   # QED correction: T dδP/dT − δP
        return (rho_g(Tg) + self.rho_e(Tg) + rho_qed
                + rho_nu(Tnue) + 2. * rho_nu(Tnumu)
                + self.rho_nu_extra(Tg))

    def p_SM(self, Tg, Tnue, Tnumu):
        """Total SM pressure during BBN [MeV⁴] (Phys. Rep. Eq. 43).

        The QED pressure correction is p_QED = δP (Phys. Rep. §II.E).
        Massless neutrinos satisfy p_ν = ρ_ν/3.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].
        Tnue : float
            Electron-neutrino temperature [MeV].
        Tnumu : float
            Muon- and tau-neutrino temperature [MeV].

        Returns
        -------
        float
            p_SM in MeV⁴.
        """
        return (rho_g(Tg) / 3. + self.p_e(Tg) + self.PQEDofT(Tg)
                + (rho_nu(Tnue) + 2. * rho_nu(Tnumu)) / 3.
                + self.rho_nu_extra(Tg) / 3.)

    def spl(self, Tg):
        """EM plasma entropy density [MeV³] (Phys. Rep. Eq. 21/24/30).

        The EM plasma (photons + e± + QED corrections) is in local thermal
        equilibrium with vanishing chemical potential.  Its entropy density is
        (Phys. Rep. Eq. 21 with μ=0):

            spl = (ρ_pl + p_pl) / Tγ

        where:

            ρ_pl = ρ_γ + ρ_e + ρ_QED     p_pl = p_γ + p_e + p_QED
            ρ_QED = Tγ dδP/dTγ − δP       p_QED = δP

        The conservation of a³ spl(Tγ) relates Tγ to the scale factor
        (Phys. Rep. Eq. 30–31).  At high temperature (Tγ ≫ me) the e±
        are ultra-relativistic and spl → (11π²/45) Tγ³.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            spl in MeV³.

        Example
        -------
        >>> p.spl(10.0) / (11 * np.pi**2 / 45 * 10.**3)   # should be ≈ 1 at high T
        0.9999...
        """
        rho_pl = rho_g(Tg) + self.rho_e(Tg)
        p_pl   = rho_g(Tg) / 3. + self.p_e(Tg)
        rho_qed = Tg * self.dPQEDdT(Tg) - self.PQEDofT(Tg)   # QED correction to energy density
        p_qed   = self.PQEDofT(Tg)                          # QED correction to pressure
        return (rho_pl + p_pl + rho_qed + p_qed) / Tg

    def spl_and_dspl_dT(self, Tg):
        """Compute spl and dspl_dT together, sharing intermediate quantities.

        Computing both simultaneously is more efficient than two separate calls
        because the e± thermodynamic functions (which involve numerical
        quadrature or table look-ups) are evaluated only once each.

        The derivative is obtained from d/dTγ[(ρ_pl + p_pl)/Tγ]:

            dspl/dTγ = (dρ_pl/dTγ + dp_pl/dTγ + dρ_QED/dTγ + dp_QED/dTγ) / Tγ
                     − spl / Tγ

        where dρ_QED/dTγ = Tγ d²δP/dTγ² and dp_QED/dTγ = dδP/dTγ.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        s : float
            spl(Tg) in MeV³.
        ds_dT : float
            dspl/dTγ at Tg in MeV².

        Example
        -------
        >>> s, dsdT = p.spl_and_dspl_dT(1.0)
        """
        rho_g_val  = rho_g(Tg)
        rho_e_val  = self.rho_e(Tg)
        p_e_val    = self.p_e(Tg)
        PQEDofT_val   = self.PQEDofT(Tg)
        dPQEDdT_val   = self.dPQEDdT(Tg)
        d2PQEDdT2_val = self.d2PQEDdT2(Tg)
        rho_pl  = rho_g_val + rho_e_val
        p_pl    = rho_g_val / 3. + p_e_val
        rho_qed = Tg * dPQEDdT_val - PQEDofT_val
        p_qed   = PQEDofT_val
        s       = (rho_pl + p_pl + rho_qed + p_qed) / Tg
        drho_g_val    = drho_g_dT(Tg)
        drho_pl_dT    = drho_g_val + self.drho_e_dT(Tg)
        dp_pl_dT      = drho_g_val / 3. + self.dp_e_dT(Tg)
        drho_qed_dT   = Tg * d2PQEDdT2_val           # d/dT[T dP/dT - P] = T d²P/dT²
        dp_qed_dT     = dPQEDdT_val                  # d/dT[P] = dP/dT
        ds_dT = (drho_pl_dT + dp_pl_dT + drho_qed_dT + dp_qed_dT) / Tg - s / Tg
        return s, ds_dT

    def dspl_dT(self, Tg):
        """Temperature derivative of the plasma entropy density [MeV²].

        Delegates to :meth:`spl_and_dspl_dT` and discards the entropy value.
        Use :meth:`spl_and_dspl_dT` directly when both s and ds/dT are needed.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            dspl/dTγ in MeV².
        """
        return self.spl_and_dspl_dT(Tg)[1]

    def T_nu_decoupling(self, Tg):
        """Neutrino temperature in the instantaneous-decoupling limit [MeV].

        When neutrinos decouple instantaneously their temperature evolves as
        Tν ∝ 1/a, so aTν = const throughout e⁺e⁻ annihilation.  The EM
        plasma separately conserves its own entropy: a³ spl(Tγ) = const.
        Combining these two conservation laws and normalising at Tγ ≫ me
        where Tν = Tγ and spl → σ_∞ Tγ³ gives (Phys. Rep. Eqs. 30–33):

            Tν(Tγ) = Tγ × (spl(Tγ) / (σ_∞ Tγ³))^{1/3}

        where σ_∞ ≡ 11π²/45 is the free-gas high-T entropy coefficient.

        **QED-correction caveat**: when ``cfg.QED_corrections`` is True,
        spl(T)/T³ ≠ σ_∞ even at high T, so using σ_∞ as the normalisation
        is inconsistent.  This method is self-consistent only when
        ``QED_corrections=False``.  For the incomplete-decoupling mode,
        ``main._setup_background_and_cosmo`` uses a QED-corrected
        normalisation instead.

        This method is used by :meth:`rho_nu_extra` (extra ΔNeff species) and
        as a convenience when ``QED_corrections=False``.

        Parameters
        ----------
        Tg : float
            Photon temperature [MeV].

        Returns
        -------
        float
            Common neutrino temperature Tν [MeV] in the instantaneous-
            decoupling, free-gas approximation.

        Example
        -------
        >>> p.T_nu_decoupling(0.01)   # well after e± annihilation
        0.00714...   # ≃ (4/11)^{1/3} × 0.01
        """
        return Tg * (self.spl(Tg) / (_sigma_inf * Tg**3))**(1. / 3.)

    # ------------------------------------------------------------------
    # Convenience wrappers around the cfg-independent module functions, so
    # callers holding a Plasma instance can reach every quantity through
    # ``self.plasma.<name>(...)`` without also importing the module.
    # ------------------------------------------------------------------

    def rho_g(self, Tg):
        return rho_g(Tg)

    def drho_g_dT(self, Tg):
        return drho_g_dT(Tg)

    def rho_nu(self, Tnu):
        return rho_nu(Tnu)

    def drho_nu_dT(self, Tnu):
        return drho_nu_dT(Tnu)
