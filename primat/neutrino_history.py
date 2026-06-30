# -*- coding: utf-8 -*-
"""
neutrino_history.py — pluggable neutrino-sector background for primat
======================================================================

The neutrino sector entering the cosmological background is the natural
interface for non-standard neutrino physics.  It is fully described by
four ingredients, all functions of the photon temperature T_gamma [MeV]:

    * temperatures  — the three flavour temperatures T_nue, T_numu, T_nutau,
    * heating       — N(T_gamma), the entropy injected into neutrinos during
                      e+e- annihilation, which drives the a(T_gamma) ODE,
    * distortion    — dFDneu(en, x, znu, sgnq), the deviation of the neutrino
                      phase-space distribution from a perfect Fermi-Dirac, used
                      as an additive correction to the n<->p weak-rate integrand
                      (None when there are no spectral distortions),
    * extra rho     — rho_nu_SD(T_nu), the extra neutrino energy density carried
                      by that distortion, fed into the Friedmann equation (None
                      when there is none).

``NeutrinoHistory`` is the protocol exposing these four ingredients as plain
attributes.  Two concrete implementations cover the two decoupling regimes:

    * :class:`NEVOTable`            — incomplete (non-instantaneous) decoupling,
                                      reading the pre-computed NEVO tables;
    * :class:`InstantaneousDecoupling` — complete decoupling, with the
                                      neutrino temperature fixed by EM entropy
                                      conservation and no heating.

The analytic y-type (SZ) + gray-type spectral distortion is a *decorator*,
:class:`AnalyticDistortion`, that wraps either base history and overrides only
``dFDneu_func`` / ``rho_nu_SD`` (leaving the temperatures and heating untouched).
:func:`make_neutrino_history` is the factory that assembles the right object
from a :class:`~primat.config.PRIMATConfig`; the legality of each flag
combination is enforced once, in ``PRIMATConfig`` (spectral_distortions ×
analytic_distortions × incomplete_decoupling), so the factory only needs to
dispatch.

Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095);
PRIMAT-Main.m (MuDistortionNeutrinos / YDistortionNeutrinos / NEVO spectra).
"""

import os

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator


def resolve_nevo_path(cfg, override, default_filename):
    """Resolve a ``rates/NEVO/`` data file, honouring a config override.

    ``override`` is one of ``cfg.nevo_file``, ``cfg.nevo_spectral_file`` or
    ``cfg.nevo_grid_file`` (each ``None`` by default).  When set, it names the
    file to use instead of ``default_filename`` -- either an absolute path, or
    a filename resolved relative to ``rates/NEVO/`` (so a custom table can sit
    alongside the shipped ones without copying the whole directory).  When
    ``None``, ``default_filename`` (itself already chosen based on
    ``cfg.QED_corrections``, see :class:`NEVOTable`) is used unchanged.

    Args:
        cfg: PRIMATConfig instance (used for ``cfg._resolved_data_dir``).
        override: ``cfg.nevo_*`` value, or ``None``.
        default_filename: filename (relative to ``NEVO/``) to use when
            ``override`` is ``None``.

    Returns:
        Absolute path to the resolved file.
    """
    fname = override if override is not None else default_filename
    if os.path.isabs(fname):
        return fname
    return os.path.join(cfg._resolved_data_dir, "NEVO", fname)

__all__ = ["NeutrinoHistory", "NEVOTable", "InstantaneousDecoupling",
           "AnalyticDistortion", "make_neutrino_history"]

# Exponential overflow guard shared by the distortion Fermi-Dirac evaluations
# (matches the weak_rates / PRIMAT exp_cutoff convention).
_EXP_CUT = 3e2

# Number of neutrino flavours sharing the distortion / chemical potential.
# munuOverTnu and the y/gray distortions are applied identically to nu_e,
# nu_mu, nu_tau, so the extra energy density they carry is summed over all three.
_N_NU = 3.0

# Standard single-particle Fermi-Dirac integral ∫₀^∞ y³ f_FD dy (zero μ).
_INTY3_FD = 7. * np.pi**4 / 120.


def _rho_nu_SD_from_int(Tnu, extra_int):
    """Extra neutrino energy density [MeV⁴] from a (genuine) spectral distortion
    of dimensionless, ν+ν̄-summed, per-flavour integral excess ``extra_int`` =
    Δ∫y³f dy, summed over the three flavours.

    Mirrors PRIMAT-Main-gray.m line 832 exactly:

        ρ_νSD = N_ν · (k T_ν)⁴/(2π²ℏ³c⁵) · extra_int = N_ν · (T_ν⁴/2π²) · extra_int

    NOTE: the historical Python port wrote this as ``rho_nu(Tnu)·extra_int/
    Inty3_FD`` which equals ``2·(T_ν⁴/2π²)·extra_int`` -- an effective N_ν = 2,
    because ``rho_nu`` already carries the ν+ν̄ factor of 2 for one flavour and
    that was mistaken for the flavour count 3, making ρ_νSD (and ΔNeff of every
    analytic distortion) a factor 2/3 too small. PRIMAT-Main-gray.m
    (NeutrinosGenerations = 3, explicit single-particle prefactor) was correct.
    """
    return _N_NU * (Tnu**4 / (2. * np.pi**2)) * extra_int


class NeutrinoHistory:
    """Neutrino-sector background interface (see module docstring).

    Subclasses populate, in ``__init__`` order:

    * ``Tnue_of_Tg``, ``Tnumu_of_Tg``, ``Tnutau_of_Tg`` : callables mapping
      T_gamma [MeV] to the flavour neutrino temperature [MeV] (array-safe),
    * ``N_NEVO_of_Tg`` : callable T_gamma [MeV] -> dimensionless heating N,
    * ``dFDneu_func`` : callable ``(en, x, znu, sgnq) -> float`` or ``None``,
    * ``dFDneu_moments`` : dict of energy-moment callables of ``dFDneu_func``
      (keys "e2p0".."e4p2") or ``None`` -- only set by
      :class:`AnalyticDistortion`, consumed by the SD-FM weak-rate term
      (``weak_rates._L_SD_FMCCR``/``_L_SD_FMNoCCR``),
    * ``rho_nu_SD`` : callable ``T_nu -> MeV^4`` or ``None``,
    * ``x_of_Tg`` : callable T_gamma [MeV] -> dimensionless ``x`` (the NEVO
      table's ``x = m_e/(kB T_com)`` column, proportional to the scale
      factor ``a``), array-safe, with radiation-domination extrapolation
      (``x*T_gamma = const``) outside the table; or ``None`` when no NEVO
      table is loaded. Used by ``external_scale_factor`` mode --
      only :class:`NEVOTable` sets this.

    ``cfg`` and ``plasma`` are stored for the subclasses' use.
    """

    def __init__(self, cfg, plasma):
        self.cfg = cfg
        self.plasma = plasma
        # Defaults: no spectral distortion / no extra energy density / no
        # table-based scale factor.  Set here so every implementation has
        # them even if it never overrides them.
        self.dFDneu_func = None
        self.dFDneu_moments = None
        self.rho_nu_SD = None
        self.x_of_Tg = None
        self._build_temperatures()
        self._build_distortion()

    # -- to be provided by concrete implementations -------------------------
    def _build_temperatures(self):
        """Set Tnue_of_Tg / Tnumu_of_Tg / Tnutau_of_Tg / N_NEVO_of_Tg."""
        raise NotImplementedError

    def _build_distortion(self):
        """Optionally set dFDneu_func / rho_nu_SD.

        Default: no distortion.  ``NEVOTable`` overrides this for its
        table-based spectrum; the analytic y/gray distortion is added by the
        :class:`AnalyticDistortion` decorator instead of here.
        """
        pass


class NEVOTable(NeutrinoHistory):
    """Incomplete (non-instantaneous) neutrino decoupling from the NEVO tables.

    Reads the pre-computed NEVO neutrino-decoupling table
    (``rates/NEVO/NEVOPRIMAT_col_1_7.csv`` with QED corrections, or the
    ``_NoQED`` variant without).  The three flavour temperatures are
    interpolated from the table and the NEVO heating function N(T_gamma) — the
    extra entropy injected into neutrinos during e+e- annihilation — drives the
    a(T_gamma) ODE.

    When ``cfg.spectral_distortions`` and not ``cfg.analytic_distortions``, the
    full 86-column NEVO spectrum file is also loaded to build the table-based
    ``dFDneu_func`` (the deviation of the actual neutrino spectrum from a
    Fermi-Dirac at temperature T_nu).
    """

    def _build_temperatures(self):
        cfg = self.cfg
        # Two versions of the table exist:
        #   <prefix>_col_1_7.csv        — computed with QED corrections
        #   <prefix>_NoQED_col_1_7.csv  — computed without QED corrections
        # where <prefix> defaults to "NEVOPRIMAT" (cfg.nevo_file_prefix lets a
        # user point at a complete alternative set of NEVO-format tables).
        # We select the one that is consistent with cfg.QED_corrections so
        # that the neutrino temperatures entering the background are derived
        # from the same plasma equation of state that is used in the rest of
        # the computation.
        prefix = cfg.nevo_file_prefix
        default_file = (f"{prefix}_col_1_7.csv" if cfg.QED_corrections
                        else f"{prefix}_NoQED_col_1_7.csv")
        nevo_path = resolve_nevo_path(cfg, cfg.nevo_file, default_file)
        table = np.loadtxt(nevo_path, delimiter=',', usecols=range(6))
        # Column layout (0-indexed):
        #   0: x = me / (kB T_com)   [dimensionless]
        #   1: z = a T_γ normalised  [dimensionless]
        #   2: T_νe  / T_com         [dimensionless]
        #   3: T_νμ  / T_com         [dimensionless]
        #   4: T_ντ  / T_com         [dimensionless]
        #   5: N_NEVO (heating fn)   [dimensionless, same units as s̄ = s(T)/T³]

        x      = table[:, 0]
        z      = table[:, 1]

        Tg_tab     = cfg.me * z / x              # T_γ [MeV]
        Tnue_tab   = table[:, 2] * cfg.me / x    # T_νe  [MeV]
        Tnumu_tab  = table[:, 3] * cfg.me / x    # T_νμ  [MeV]
        Tnutau_tab = table[:, 4] * cfg.me / x    # T_ντ  [MeV]
        N_NEVO_tab = table[:, 5]                 # dimensionless

        # Ensure the table is ordered by decreasing T_γ (high→low)
        if Tg_tab[0] < Tg_tab[-1]:
            x, Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab = (
                arr[::-1] for arr in (x, Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab))

        # Interpolants for neutrino temperatures as functions of T_γ.
        # We interpolate the dimensionless ratio T_να/T_γ rather than T_να
        # directly so that constant extrapolation at the table boundaries
        # corresponds to T_να ∝ T_γ — the correct scaling both at high T
        # (before any heating) and at low T (after freeze-out).
        _ratio_ue   = Tnue_tab   / Tg_tab
        _ratio_umu  = Tnumu_tab  / Tg_tab
        _ratio_utau = Tnutau_tab / Tg_tab

        _ratiofn_kw = dict(bounds_error=False, kind='linear')
        _ratiofn_ue   = interp1d(Tg_tab, _ratio_ue,   fill_value=(_ratio_ue[-1],   _ratio_ue[0]),   **_ratiofn_kw)
        _ratiofn_umu  = interp1d(Tg_tab, _ratio_umu,  fill_value=(_ratio_umu[-1],  _ratio_umu[0]),  **_ratiofn_kw)
        _ratiofn_utau = interp1d(Tg_tab, _ratio_utau, fill_value=(_ratio_utau[-1], _ratio_utau[0]), **_ratiofn_kw)

        def Tnue_of_Tg(Tg):
            return _ratiofn_ue(Tg) * Tg

        def Tnumu_of_Tg(Tg):
            return _ratiofn_umu(Tg) * Tg

        def Tnutau_of_Tg(Tg):
            return _ratiofn_utau(Tg) * Tg

        self.Tnue_of_Tg   = Tnue_of_Tg
        self.Tnumu_of_Tg  = Tnumu_of_Tg
        self.Tnutau_of_Tg = Tnutau_of_Tg
        self.N_NEVO_of_Tg = interp1d(Tg_tab, N_NEVO_tab, bounds_error=False,
                                     fill_value=(0., 0.), kind='linear')

        # ------------------------------------------------------------------
        # x_of_Tg: table-based scale factor a(T_γ) ∝ x(T_γ), for
        # external_scale_factor mode. By the NEVO
        # Mathematica convention x = m_e/(kB T_com) with T_com ∝ 1/a, so
        # x ∝ a exactly.
        #
        # Outside the table, extrapolate assuming radiation domination,
        # a ∝ 1/T_γ, i.e. x(T)·T = const, equal to its value at the nearest
        # table edge (this matches the existing N_NEVO -> 0 extrapolation of
        # the minimal-mode ODE in this regime).
        # ------------------------------------------------------------------
        _Tg_asc = Tg_tab[::-1]   # ascending T_γ (Tg_tab is stored high→low)
        _x_asc  = x[::-1]
        _Tg_min, _Tg_max = float(_Tg_asc[0]), float(_Tg_asc[-1])
        _x_min,  _x_max  = float(_x_asc[0]),  float(_x_asc[-1])

        _logx_of_logTg = interp1d(np.log(_Tg_asc), np.log(_x_asc),
                                  kind='linear', bounds_error=False)

        def x_of_Tg(Tg):
            Tg = np.asarray(Tg, dtype=float)
            # Clip into the table range for the interior interpolation; the
            # out-of-range branches below override these values with the
            # radiation-domination extrapolation, so the clipped value here
            # is never used out of range -- it only avoids interp1d NaNs.
            log_Tg_clipped = np.clip(np.log(Tg), np.log(_Tg_min), np.log(_Tg_max))
            x_interior = np.exp(_logx_of_logTg(log_Tg_clipped))
            return np.where(Tg < _Tg_min, _x_min * _Tg_min / Tg,
                   np.where(Tg > _Tg_max, _x_max * _Tg_max / Tg, x_interior))

        self.x_of_Tg = x_of_Tg
        self._Tg_table_range = (_Tg_min, _Tg_max)  # diagnostics/tests only

    def _build_distortion(self):
        cfg = self.cfg
        if not (cfg.spectral_distortions and not cfg.analytic_distortions):
            return

        # ---- NEVO-based spectral distortions ----
        # Load the full NEVO file (86 columns: 6 thermo + 80 spectral).
        # The NEVOGrid.csv contains the 80 Gauss-Laguerre y nodes.
        #
        # Column layout (0-indexed):
        #   0: x = me/(kB Tγ)
        #   1–5: same thermo columns as _col_1_7
        #   6–85: fractional spectral perturbation δf at each y node,
        #         defined so that f_actual(y) = (1+δf(y))/(e^y+1).
        prefix = cfg.nevo_file_prefix
        default_full_file = (f"{prefix}.csv" if cfg.QED_corrections
                             else f"{prefix}_NoQED.csv")
        nevo_full_path = resolve_nevo_path(cfg, cfg.nevo_spectral_file, default_full_file)
        grid_path      = resolve_nevo_path(cfg, cfg.nevo_grid_file, "NEVOGrid.csv")

        table_full = np.loadtxt(nevo_full_path, delimiter=',')   # (600, 86)
        y_nodes    = np.loadtxt(grid_path,      delimiter=',')   # (80,)

        x_NEVO_raw = table_full[:, 0]    # me * a
        z_NEVO_raw = table_full[:, 1]    # a * Tγ
        df_raw     = table_full[:, 6:]   # (600, 80) fractional distortion

        # Ensure ascending order in x_NEVO (the 2D interpolant requires it).
        if x_NEVO_raw[0] > x_NEVO_raw[-1]:
            x_NEVO_raw = x_NEVO_raw[::-1]
            z_NEVO_raw = z_NEVO_raw[::-1]
            df_raw     = df_raw[::-1]

        # The weak rates pass x = me / (kB Tγ).
        # In the table, x_table = x_NEVO / z_NEVO = (me*a) / (a*kB*Tγ) = me/(kB Tγ).
        x_table     = x_NEVO_raw / z_NEVO_raw
        x_min_table = np.min(x_table)
        x_max_table = np.max(x_table)

        # We need x_NEVO (the scale factor me*a) as a function of x = me/(kB Tγ)
        # to (1) look up the 2D table and (2) compute y = p*a = (p/me)*(me*a).
        # Since x_table may not be monotonic if a*Tγ varies significantly,
        # we sort it for the 1D interpolant.
        idx_sort = np.argsort(x_table)
        _x_NEVO_of_x = interp1d(x_table[idx_sort], x_NEVO_raw[idx_sort],
                                bounds_error=False, fill_value="extrapolate")

        log_x_NEVO = np.log(x_NEVO_raw)
        y_min = float(y_nodes[0])
        y_max = float(y_nodes[-1])
        _df_2D = RegularGridInterpolator(
            (log_x_NEVO, y_nodes),
            df_raw,
            method='linear',
            bounds_error=False,
            fill_value=0.
        )

        def dFDneu_func(en, x, znu, sgnq):
            """NEVO-based spectral distortion.

            Returns δf = f_NEVO(en,Tγ) − f_FD(en, Tν):
              f_NEVO(en) = (1 + dfNue(x_NEVO, y)) / (exp(y)+1)
              f_FD(en)   = 1 / (exp(en·znu)+1)
            where y = en · x_NEVO  and  x_NEVO = me · a.

            x is the dimensionless photon-temperature variable me/(kB Tγ).
            In the NEVO table, x_NEVO = x * z where z = a · Tγ.

            Four sign cases follow PRIMAT's dFDneu convention:
              en>0, sgnq>0: ν_e in initial state (n→p)
              en>0, sgnq<0: ν̄_e in initial state (p→n)
              en<0, sgnq>0: ν̄_e Pauli-blocking (n→p), extra minus sign
              en<0, sgnq<0: ν_e Pauli-blocking (p→n), extra minus sign
            Outside the NEVO x range the distortion is zero.
            """
            if x < x_min_table or x > x_max_table:
                return 0.

            # Compute the scale factor xNEV = me*a corresponding to this x = me/T.
            xNEV  = float(_x_NEVO_of_x(x))
            en_ph = abs(en)
            y     = en_ph * xNEV
            if y < y_min or y > y_max:
                return 0.

            # Fractional distortion from the 2D table (returns 1-element array)
            df = float(_df_2D([[np.log(xNEV), y]])[0])
            # Actual occupation vs FD at T_ν
            # f_NEVO  = (1 + df) / (exp(y) + 1)
            # f_FD_nu = 1 / (exp(en_ph * znu) + 1)
            arg_y   = y
            arg_nu  = en_ph * znu
            f_nevo  = (0. if arg_y  > _EXP_CUT else (1. + df) / (np.exp(arg_y)  + 1.))
            f_fd_nu = (0. if arg_nu > _EXP_CUT else 1.             / (np.exp(arg_nu) + 1.))
            delta_f = f_nevo - f_fd_nu

            # Apply sign convention from PRIMAT:
            # blocking factors (en < 0) enter with an extra minus sign
            # because they represent (1 - f) rather than f.
            if en < 0.:
                return -delta_f
            return delta_f

        self.dFDneu_func = dFDneu_func


class InstantaneousDecoupling(NeutrinoHistory):
    """Complete (instantaneous) neutrino decoupling.

    All three neutrino flavours share the same temperature, fixed by EM entropy
    conservation:

        T_ν(T_γ) = [spl(T_γ) / sbar_ref]^{1/3}

    where ``sbar_ref`` is the high-T limit of spl(T)/T³ (photons + e+e- at
    T >> m_e), normalised so that T_ν = T_γ before e+e- annihilation.  The NEVO
    heating function is N ≡ 0 (no entropy transfer between neutrinos and the EM
    plasma).  QED corrections to the plasma equation of state are still included
    via the spl/PQEDofT tables.

    The NEVO table is **not** loaded.  Spectral distortions, if any, are the
    analytic mu+y kind added by :class:`AnalyticDistortion`.
    """

    def _build_temperatures(self):
        cfg    = self.cfg
        thermo = self.plasma

        # sbar_ref is the high-T limit of spl(T)/T³ (photons + e+e- at T >> m_e),
        # normalised so that T_ν = T_γ before e+e- annihilation.
        #
        # When QED_corrections=False: sbar_ref = s_∞ = 11π²/45 exactly
        # (free ideal gas).
        # When QED_corrections=True: spl/T³ ≠ s_∞ at high T due to QED
        # interaction corrections at T ~ m_e.  We use the analytical
        # perturbative result (Dodelson & Turner 1992, Heckler 1994):
        #
        #   (T_γ/T_ν)³|_{T→0} = 11/4 − 25α/(8π) + 10α^{3/2} √(π/3) / π²
        #
        # and recover sbar_ref from  (T_γ/T_ν)³ = sbar_ref / (4π²/45),
        # where 4π²/45 is the low-T photon-only entropy coefficient.
        # Using this analytical formula avoids the finite-T artefact that
        # would arise from evaluating spl(T_start)/T_start³ numerically
        # (residual e+e- at T_start, logarithmic high-T QED terms, etc.).
        #
        # np.vectorize is needed because spl (and rho_e, p_e etc.) use
        # scalar comparisons and are not array-safe.
        if cfg.QED_corrections:
            _ratio3 = (11./4.
                       - 25.*cfg.alphaem / (8.*np.pi)
                       + 10.*cfg.alphaem**(3./2.) * np.sqrt(np.pi/3.) / np.pi**2)
            _sbar_ref = _ratio3 * (4.*np.pi**2 / 45.)
        else:
            _sbar_ref = 11.*np.pi**2 / 45.   # = plasma._sigma_inf

        def _T_nu_inst(Tg):
            return (thermo.spl(Tg) / _sbar_ref)**(1. / 3.)

        _T_nu_inst_vec = np.vectorize(_T_nu_inst)

        # All three flavours share the same temperature.
        self.Tnue_of_Tg   = _T_nu_inst_vec
        self.Tnumu_of_Tg  = _T_nu_inst_vec
        self.Tnutau_of_Tg = _T_nu_inst_vec

        def N_NEVO_of_Tg(Tg):
            # No entropy transfer between neutrinos and the EM plasma.
            return np.zeros_like(np.asarray(Tg, dtype=float))

        self.N_NEVO_of_Tg = N_NEVO_of_Tg


class AnalyticDistortion(NeutrinoHistory):
    """Analytic y-type (SZ/Compton) + gray-type spectral distortion.

    Wraps an existing :class:`NeutrinoHistory` (``base``) and adds the analytic
    distortion: it inherits ``base``'s temperatures and heating unchanged and
    overrides only ``dFDneu_func`` (the n<->p weak-rate correction) and
    ``rho_nu_SD`` (the extra neutrino energy density fed into the Friedmann
    equation).  Used when ``cfg.spectral_distortions`` and
    ``cfg.analytic_distortions``; ``PRIMATConfig`` guarantees that pairing implies
    instantaneous decoupling, so ``base`` is an :class:`InstantaneousDecoupling`.

    The distortion amplitudes are continuous knobs: ``cfg.y_SZ`` (y-type /
    SZ/Compton) and ``cfg.y_gray`` (gray-type, a number-density-preserving
    temperature-like rescaling -- see ``cfg.y_gray``'s docstring in config.py
    for why this is a *different* shape from ``y_SZ`` despite
    generate_rates/PRIMAT-Main-gray.m's misleading "YSZ" name for it). There is
    no mu-type distortion: a genuine neutrino chemical potential
    (``cfg.munuOverTnu``) is handled exactly in the weak rates and the energy
    density, not as a (linearised) spectral distortion.
    """

    # Composition rather than calling the base __init__ machinery: we already
    # have a fully built base history and only want to layer the distortion on.
    def __init__(self, base):
        self._base = base
        self.cfg = base.cfg
        self.plasma = base.plasma
        # Inherit the temperatures and heating from the wrapped history.
        self.Tnue_of_Tg   = base.Tnue_of_Tg
        self.Tnumu_of_Tg  = base.Tnumu_of_Tg
        self.Tnutau_of_Tg = base.Tnutau_of_Tg
        self.N_NEVO_of_Tg = base.N_NEVO_of_Tg
        self.dFDneu_func = None
        self.rho_nu_SD = None
        self._build_analytic_distortion()

    def _build_analytic_distortion(self):
        cfg = self.cfg
        xi_nu = cfg.munuOverTnu   # reduced chemical potential ξ = μ/T_ν

        # ---- y-type (SZ/Compton) and gray-type analytic distortions ----
        # There is deliberately NO μ-type distortion: a neutrino chemical
        # potential is not a spectral distortion. A genuine chemical potential
        # (cfg.munuOverTnu) is treated exactly -- in the n<->p weak rates via
        # the FD_nu3(sgnq*xi_nu) integrand, and in the energy density via
        # NeutrinoHistory.rho_nu -- so the old, linearised μ-type "distortion"
        # (delta_xi_nu) has been removed.
        #
        # y-type: the SZ/Compton spectral shape is the energy derivative of
        # the Fermi-Dirac weighted by y²:
        #   δf_y^ν(y)   = (1/y²) d/dy(y⁴ df_FD/dy)
        #               = f_FD(1−f_FD)[4 − y(1−2f_FD)]  (analytic)
        # Ref: PRIMAT-Main.m, YDistortionNeutrinos.
        #
        # gray-type: a third, independent distortion -- not the Compton/SZ
        # shape above despite generate_rates/PRIMAT-Main-gray.m naming its
        # equivalent parameter "YSZ" (a misnomer kept only in that file's
        # comments; see cfg.y_gray's docstring in config.py). It rescales the
        # spectrum as if the neutrino temperature shifted by a factor
        # (1+y_gray), with the (1+y_gray)^-3 prefactor chosen so the rescaled
        # piece's number density exactly cancels against the unperturbed
        # Fermi-Dirac it is subtracted from (the y^2-moment integral of
        # δf_gray vanishes exactly, for any y_gray -- verified in
        # scratch/derive_sd_fm_distortions.py):
        #   δf_gray(y) = -1/(e^y+1) + 1/(e^{y/(1+γ)}+1) / (1+γ)³,  γ=y_gray
        # Same shape for neutrinos and antineutrinos (no ξ dependence), per
        # PRIMAT-Main-gray.m's dFDneuRawy/dFDantineuRawy.
        y_sz   = cfg.y_SZ
        y_gray = cfg.y_gray

        def _fd(arg):
            # Safe Fermi-Dirac, elementwise over scalars or arrays: clamp the
            # exponent before exp() (avoids overflow) and mask the result to
            # 0 above _EXP_CUT. np.where-based (not a Python if/else) so this
            # -- and everything built from it below (dFDneu_func, the M_*p*
            # moment closures) -- stays array-vectorised: the SD-FM
            # correction (_L_SD_FMCCR/_NoCCR in weak_rates/corrections.py)
            # evaluates these over O(1e4-1e5)-point quadrature grids, and a
            # scalar-only _fd previously forced np.vectorize to fall back to
            # a million-plus individual Python calls (~20 s; see git log for
            # the profiling that found this).
            arg = np.asarray(arg)
            return np.where(arg > _EXP_CUT, 0., 1. / (np.exp(np.minimum(arg, _EXP_CUT)) + 1.))

        def _dFDneu_analytic(en, x, znu, sgnq):
            """Analytic y/gray spectral distortion of neutrinos/antineutrinos.

            en   : electron energy E/me (≥ 0 for forward, < 0 for blocking)
            x    : me/(kB Tγ)  — not used in analytic mode, present for
                   interface consistency
            znu  : me/(kB Tν)
            sgnq : +1 (n→p, ν_e initial) or −1 (p→n, ν̄_e initial)

            Returns the distortion δf evaluated at the neutrino energy
            en − sgnq·Q/me (already shifted by the weak endpoint), following
            the δχ convention.  The caller passes the already-shifted energy
            (en_nu = en − sgnq·Q/me), so here en IS the neutrino energy.
            """
            # comoving momentum y = E_ν / T_ν = en * znu
            y  = en * znu
            # signed ξ: positive for neutrinos (sgnq=+1), negative for
            # antineutrinos (sgnq=−1), consistent with chemical-potential
            # convention where μ_{ν̄} = −μ_ν. A genuine chemical potential
            # (cfg.munuOverTnu) is handled directly in the weak-rate integrands
            # (FD_nu3) and in the neutrino energy density (NeutrinoHistory.rho_nu);
            # here it only sets the Fermi-Dirac the y-type distortion sits on.
            xi = sgnq * xi_nu

            # No μ-type spectral distortion any more (it was just the linearised
            # form of a chemical potential, which is now treated genuinely).
            dist = np.zeros_like(y, dtype=float)

            if y_sz != 0.:
                # y-type (SZ/Compton) distortion: (1/y²) d/dy(y⁴ d f_FD/dy)
                # Analytic form: f(1-f)[4 - y(1-2f)]
                f = _fd(y - xi)
                dist += y_sz * f * (1. - f) * (4. - y * (1. - 2. * f))

            if y_gray != 0.:
                # gray-type distortion (no ξ dependence -- same shape for
                # neutrinos and antineutrinos, see the module docstring above).
                dist += -_fd(y) + _fd(y / (1. + y_gray)) / (1. + y_gray) ** 3

            return dist

        def dFDneu_func(en, x, znu, sgnq):
            """Dispatch δf for all sign combinations of en and sgnq.

            PRIMAT's dFDneu handles four cases depending on whether en
            is the initial-state particle (en > 0) or a Pauli-blocking
            factor (en < 0), and whether the reaction is n→p (sgnq=+1)
            or p→n (sgnq=−1).  The blocking cases carry an extra minus
            sign and use the conjugate spectrum.
            """
            # np.where (not a Python if/else) so this stays array-vectorised
            # over `en` -- see _fd's docstring above for why that matters.
            # Both branches are cheap closed-form numpy expressions, so
            # evaluating both unconditionally and selecting is fine.
            en = np.asarray(en)
            # Neutrino (sgnq>0) or antineutrino (sgnq<0) in initial state
            forward = _dFDneu_analytic(en, x, znu, sgnq)
            # Pauli-blocking factor: the conjugate particle is the one that
            # appears in the final state, so flip both en and sgnq.
            blocking = -_dFDneu_analytic(-en, x, znu, -sgnq)
            return np.where(en >= 0., forward, blocking)

        # Marks this callable as already array-vectorised, so
        # weak_rates/corrections.py's _L_SD/_L_SD_CCR can call it directly
        # over a whole quadrature grid instead of wrapping it in
        # np.vectorize (needed for the *other* dFDneu_func implementation,
        # NEVOTable's table-lookup version above, which is genuinely
        # scalar-only).
        dFDneu_func.vectorized = True
        self.dFDneu_func = dFDneu_func

        # ---- en-moment derivatives of δf, for the SD-FM (finite-nucleon- ----
        # ---- mass) weak-rate correction (weak_rates._L_SD_FMCCR/_NoCCR) ----
        # Mirrors PRIMAT-Main-gray.m's delta_chi_FM (lines ~1712-1725), which
        # needs eight energy-moment functions of δf: e2p0/e3p0 (value,
        # weighted by en^2/en^3 -- no new derivation needed, just
        # en^n*dFDneu_func, which already implements the en<0 dispatch above)
        # and e2p1/e3p1/e4p1/e2p2/e3p2/e4p2 (1st/2nd en-derivatives of
        # en^n*δf), exactly mirroring the eight plain-Fermi-Dirac moments
        # already hand-coded in weak_rates.py (_FD_nu_e{2,3,4}p{1,2}_v).
        #
        # The six derivative moments are derived in
        # scratch/derive_sd_fm_distortions.py (closed forms in terms of the
        # logistic function fd and its standard derivative recursion
        # fd'=-fd(1-fd), so no bare exp() ever appears and nothing can
        # overflow; numerically self-checked there against finite
        # differences for all 18 piece x (n,order) combinations) and reused
        # here via the SAME antisymmetric-dispatch identity as dFDneu_func:
        #   en >= 0:  M[n,k](en) = d^k/den^k[en^n*H(en,sgnq)]  ("_raw_M{n}p{k}"
        #             below, sgnq unflipped)
        #   en <  0:  u = -en; M[n,k](en) = sign(n,k) * _raw_M{n}p{k}(u, -sgnq)
        #             with sign(n,1) = (-1)^n, sign(n,2) = -(-1)^n -- derived
        #             from F(en,sgnq) = -H(-en,-sgnq) the same way
        #             dFDneu_func's en<0 branch is (see neutrino_history.py
        #             module-level commit message / PR description for the
        #             chain-rule derivation; the (-1)^n alternation is the
        #             parity of d/d(-en) = -d/du applied k times).
        # The functions below are transcribed VERBATIM (no hand algebra) from
        # the auto-generated combiner output of
        # scratch/derive_sd_fm_distortions.py (each mu/y/gray piece written
        # to its own helper by sympy's cse()+str() printer, then summed by a
        # plain f-string template -- see that script's `to_pycode`/combiner
        # block), after re-running it and confirming "0 mismatch(es) out of
        # 54 self-checks" against finite differences (h=1e-6, threshold
        # 1e-7). This avoids the earlier hand-merge transcription bug (a
        # `x1**3` mistyped as `x1**4` in the y-piece) by construction.
        def _M_2_p1_y(en, znu, xi):
            x0 = en * znu
            x1 = _fd(x0 - xi)
            x2 = en**2 * znu**2
            x3 = x1**2
            return -en * x1 * (-21 * x0 * x1 + 14 * x0 * x3 + 7 * x0 + 6 * x1**3 * x2
                                + 7 * x1 * x2 + 8 * x1 - 12 * x2 * x3 - x2 - 8)

        def _M_2_p1_gray(en, znu):
            x0 = y_gray + 1
            x1 = x0**4
            x2 = en * znu
            x3 = _fd(x2)
            x4 = _fd(x2 / x0)
            return -en * (2 * x0 * (x0**3 * x3 - x4) + x2 * (x1 * x3 * (x3 - 1) - x4 * (x4 - 1))) / x1

        def _raw_M2p1(en, znu, xi):
            return (y_sz * _M_2_p1_y(en, znu, xi)
                    + _M_2_p1_gray(en, znu))

        def _M_3_p1_y(en, znu, xi):
            x0 = en**2
            x1 = en * znu
            x2 = _fd(x1 - xi)
            x3 = x0 * znu**2
            x4 = x2**2
            return -x0 * x2 * (-24 * x1 * x2 + 16 * x1 * x4 + 8 * x1 + 6 * x2**3 * x3
                                + 7 * x2 * x3 + 12 * x2 - 12 * x3 * x4 - x3 - 12)

        def _M_3_p1_gray(en, znu):
            x0 = y_gray + 1
            x1 = x0**4
            x2 = en * znu
            x3 = _fd(x2)
            x4 = _fd(x2 / x0)
            return -en**2 * (3 * x0 * (x0**3 * x3 - x4) + x2 * (x1 * x3 * (x3 - 1) - x4 * (x4 - 1))) / x1

        def _raw_M3p1(en, znu, xi):
            return (y_sz * _M_3_p1_y(en, znu, xi)
                    + _M_3_p1_gray(en, znu))

        def _M_4_p1_y(en, znu, xi):
            x0 = en * znu
            x1 = _fd(x0 - xi)
            x2 = en**2 * znu**2
            x3 = x1**2
            return -en**3 * x1 * (-27 * x0 * x1 + 18 * x0 * x3 + 9 * x0 + 6 * x1**3 * x2
                                   + 7 * x1 * x2 + 16 * x1 - 12 * x2 * x3 - x2 - 16)

        def _M_4_p1_gray(en, znu):
            x0 = y_gray + 1
            x1 = x0**4
            x2 = en * znu
            x3 = _fd(x2)
            x4 = _fd(x2 / x0)
            return -en**3 * (4 * x0 * (x0**3 * x3 - x4) + x2 * (x1 * x3 * (x3 - 1) - x4 * (x4 - 1))) / x1

        def _raw_M4p1(en, znu, xi):
            return (y_sz * _M_4_p1_y(en, znu, xi)
                    + _M_4_p1_gray(en, znu))

        def _M_2_p2_y(en, znu, xi):
            x0 = en * znu
            x1 = _fd(x0 - xi)
            x2 = en**3 * znu**3
            x3 = en**2 * znu**2
            x4 = x1**2
            x5 = 60 * x1**3
            return -x1 * (-66 * x0 * x1 + 44 * x0 * x4 + 22 * x0 + 24 * x1**4 * x2
                           - 15 * x1 * x2 + 70 * x1 * x3 + 8 * x1 + 50 * x2 * x4 - x2 * x5
                           + x2 - 120 * x3 * x4 + x3 * x5 - 10 * x3 - 8)

        def _M_2_p2_gray(en, znu):
            x0 = y_gray + 1
            x1 = x0**5
            x2 = en * znu
            x3 = _fd(x2)
            x4 = 2 * x3
            x5 = _fd(x2 / x0)
            x6 = x3 * (x3 - 1)
            x7 = x5 - 1
            return (en**2 * znu**2 * (-x1 * x6 * (x4 - 1) + x5**2 * x7 + x5 * x7**2)
                    + 2 * x0**2 * x5 - 4 * x0 * x2 * (x0**4 * x6 - x5 * x7) - x1 * x4) / x1

        def _raw_M2p2(en, znu, xi):
            return (y_sz * _M_2_p2_y(en, znu, xi)
                    + _M_2_p2_gray(en, znu))

        def _M_3_p2_y(en, znu, xi):
            x0 = en * znu
            x1 = _fd(x0 - xi)
            x2 = en**3 * znu**3
            x3 = en**2 * znu**2
            x4 = x1**2
            x5 = x1**3
            return -en * x1 * (-108 * x0 * x1 + 72 * x0 * x4 + 36 * x0 + 24 * x1**4 * x2
                                - 15 * x1 * x2 + 84 * x1 * x3 + 24 * x1 + 50 * x2 * x4
                                - 60 * x2 * x5 + x2 - 144 * x3 * x4 + 72 * x3 * x5 - 12 * x3 - 24)

        def _M_3_p2_gray(en, znu):
            x0 = y_gray + 1
            x1 = x0**5
            x2 = en * znu
            x3 = _fd(x2)
            x4 = _fd(x2 / x0)
            x5 = x3 * (x3 - 1)
            x6 = x4 - 1
            return -en * (-en**2 * znu**2 * (-x1 * x5 * (2 * x3 - 1) + x4**2 * x6 + x4 * x6**2)
                           + 6 * x0**2 * (x0**3 * x3 - x4) + 6 * x0 * x2 * (x0**4 * x5 - x4 * x6)) / x1

        def _raw_M3p2(en, znu, xi):
            return (y_sz * _M_3_p2_y(en, znu, xi)
                    + _M_3_p2_gray(en, znu))

        def _M_4_p2_y(en, znu, xi):
            x0 = en**2
            x1 = en * znu
            x2 = _fd(x1 - xi)
            x3 = en**3 * znu**3
            x4 = x0 * znu**2
            x5 = x2**2
            x6 = x2**3
            return -x0 * x2 * (-156 * x1 * x2 + 104 * x1 * x5 + 52 * x1 + 24 * x2**4 * x3
                                - 15 * x2 * x3 + 98 * x2 * x4 + 48 * x2 + 50 * x3 * x5
                                - 60 * x3 * x6 + x3 - 168 * x4 * x5 + 84 * x4 * x6 - 14 * x4 - 48)

        def _M_4_p2_gray(en, znu):
            x0 = en**2
            x1 = y_gray + 1
            x2 = x1**5
            x3 = en * znu
            x4 = _fd(x3)
            x5 = _fd(x3 / x1)
            x6 = x4 * (x4 - 1)
            x7 = x5 - 1
            return -x0 * (-x0 * znu**2 * (-x2 * x6 * (2 * x4 - 1) + x5**2 * x7 + x5 * x7**2)
                           + 12 * x1**2 * (x1**3 * x4 - x5) + 8 * x1 * x3 * (x1**4 * x6 - x5 * x7)) / x2

        def _raw_M4p2(en, znu, xi):
            return (y_sz * _M_4_p2_y(en, znu, xi)
                    + _M_4_p2_gray(en, znu))

        # Antisymmetric dispatch (see derivation above), one closure per
        # (n, order) pair, with the sign(n,order) factor baked in.
        def _make_moment(n, order, raw):
            sign_p1 = (-1.) ** n
            sign_p2 = -(-1.) ** n
            sign = sign_p1 if order == 1 else sign_p2

            def moment(en, x, znu, sgnq):
                # Same np.where-over-both-branches vectorisation as
                # dFDneu_func above (and for the same reason: this is called
                # on whole quadrature-grid arrays from
                # weak_rates/corrections.py's _chi_func_sd_fm_v).
                en = np.asarray(en)
                # The base chemical potential xi_nu shifts the Fermi-Dirac that
                # the y-type distortion sits on: +sgnq for the forward
                # (initial-state) piece, -sgnq for the Pauli-blocking (en<0)
                # piece. The gray-type has no xi dependence.
                forward = raw(en, znu, sgnq * xi_nu)
                blocking = sign * raw(-en, znu, -sgnq * xi_nu)
                return np.where(en >= 0., forward, blocking)

            return moment

        self.dFDneu_moments = {
            "e2p0": lambda en, x, znu, sgnq: en**2 * dFDneu_func(en, x, znu, sgnq),
            "e3p0": lambda en, x, znu, sgnq: en**3 * dFDneu_func(en, x, znu, sgnq),
            "e2p1": _make_moment(2, 1, _raw_M2p1),
            "e3p1": _make_moment(3, 1, _raw_M3p1),
            "e4p1": _make_moment(4, 1, _raw_M4p1),
            "e2p2": _make_moment(2, 2, _raw_M2p2),
            "e3p2": _make_moment(3, 2, _raw_M3p2),
            "e4p2": _make_moment(4, 2, _raw_M4p2),
        }

        # Extra neutrino energy density from the distortion (Friedmann eq.)
        # Analytic integrals ∫₀^∞ y³ δf dy for each distortion type.
        # Ref: PRIMAT-Main.m, Inty3MuDistortion and Inty3SZdistortion.
        #
        # Inty3Mu(ξ, δξ)  = (δξ/4)(δξ+2ξ)(δξ²+2δξξ+2(π²+ξ²))
        # Inty3SZ(ξ)      = 7π⁴/15 + 2π²ξ² + ξ⁴
        # Inty3Gray(γ)    = γ × 7π⁴/120   (exact; see scratch/derive_sd_fm_distortions.py:
        #   substituting w=y/(1+γ) in ∫y³[fd(y/(1+γ))/(1+γ)³ - fd(y)]dy gives
        #   (1+γ)∫w³fd(w)dw - ∫y³fd(y)dy = γ × Inty3_FD exactly, no expansion)
        # ρ_νSD = N_ν (kT_ν)⁴/(2π²ℏ³c⁵) × [Inty3Mu + y_SZ×Inty3SZ + 2×Inty3Gray]
        # (the factor 2 on Inty3Gray sums the neutrino+antineutrino
        # contributions, identical shapes since δf_gray has no ξ dependence;
        # Inty3Mu/Inty3SZ already are the summed neutrino+antineutrino forms.)
        # The overall normalisation (N_ν=3, single-particle prefactor) lives in
        # the module-level helper _rho_nu_SD_from_int (see PRIMAT-Main-gray.m
        # line 832); the historical bug that put N_ν=2 there is documented
        # in that helper's docstring.
        Inty3_FD = _INTY3_FD   # ∫₀^∞ y³ f_FD dy (zero μ)

        # The genuine chemical-potential energy (cfg.munuOverTnu) is carried by
        # the neutrino energy density itself (NeutrinoHistory.rho_nu), NOT here:
        # rho_nu_SD is reserved for the genuine spectral distortions (y/gray).

        def _rho_nu_SD(Tnu):
            """Extra neutrino energy density [MeV⁴] from the y/gray distortions."""
            Inty3_sz = 7.*np.pi**4/15. + 2.*np.pi**2*xi_nu**2 + xi_nu**4
            extra_int = y_sz * Inty3_sz + 2. * y_gray * Inty3_FD
            return _rho_nu_SD_from_int(Tnu, extra_int)

        self.rho_nu_SD = _rho_nu_SD


def make_neutrino_history(cfg, plasma):
    """Build the :class:`NeutrinoHistory` selected by ``cfg``.

    Dispatch only — the legality of every flag combination is validated once in
    :class:`~primat.config.PRIMATConfig` (spectral_distortions ×
    analytic_distortions × incomplete_decoupling), so here we simply:

    1. pick the base regime: :class:`NEVOTable` if ``cfg.incomplete_decoupling``
       else :class:`InstantaneousDecoupling` (which also builds the NEVO-table
       spectral distortion when that mode is active);
    2. if ``cfg.spectral_distortions and cfg.analytic_distortions``, wrap it in
       :class:`AnalyticDistortion` to layer on the analytic y/gray distortion.

    A genuine neutrino chemical potential ``cfg.munuOverTnu`` needs no special
    handling here: it shifts the n<->p weak rates through the FD_nu3 integrand
    and raises the neutrino energy density through ``NeutrinoHistory.rho_nu``
    (both built into the base history), so it is fully accounted for in either
    branch below.

    Args:
        cfg    : PRIMATConfig instance.
        plasma : initialised primat.plasma module (or a Plasma object exposing
                 spl / rho_nu).

    Returns:
        A NeutrinoHistory exposing Tnue_of_Tg / Tnumu_of_Tg / Tnutau_of_Tg /
        N_NEVO_of_Tg / dFDneu_func / rho_nu_SD.
    """
    if cfg.incomplete_decoupling:
        base = NEVOTable(cfg, plasma)
    else:
        base = InstantaneousDecoupling(cfg, plasma)

    if cfg.spectral_distortions and cfg.analytic_distortions:
        return AnalyticDistortion(base)
    return base
