# -*- coding: utf-8 -*-
"""
neutrino_history.py — pluggable neutrino-sector background for PyPRIMAT
======================================================================

IDEAS.md §6.2.  The neutrino sector entering the cosmological background is the
natural interface for non-standard neutrino physics.  It is fully described by
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

The analytic mu-type / y-type (SZ) spectral distortion is a *decorator*,
:class:`AnalyticDistortion`, that wraps either base history and overrides only
``dFDneu_func`` / ``rho_nu_SD`` (leaving the temperatures and heating untouched).
:func:`make_neutrino_history` is the factory that assembles the right object
from a :class:`~pyprimat.config.PyPRConfig`; the legality of each flag
combination is enforced once, in ``PyPRConfig`` (spectral_distortions ×
analytic_distortions × incomplete_decoupling), so the factory only needs to
dispatch.

Reference: Pitrou, Coc, Uzan & Vangioni, Phys. Rep. 2018 (arXiv:1806.11095);
PRIMAT-Main.m (MuDistortionNeutrinos / YDistortionNeutrinos / NEVO spectra).
"""

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

__all__ = ["NeutrinoHistory", "NEVOTable", "InstantaneousDecoupling",
           "AnalyticDistortion", "make_neutrino_history"]

# Exponential overflow guard shared by the distortion Fermi-Dirac evaluations
# (matches the weak_rates / PRIMAT exp_cutoff convention).
_EXP_CUT = 3e2


class NeutrinoHistory:
    """Neutrino-sector background interface (see module docstring).

    Subclasses populate, in ``__init__`` order:

    * ``Tnue_of_Tg``, ``Tnumu_of_Tg``, ``Tnutau_of_Tg`` : callables mapping
      T_gamma [MeV] to the flavour neutrino temperature [MeV] (array-safe),
    * ``N_NEVO_of_Tg`` : callable T_gamma [MeV] -> dimensionless heating N,
    * ``dFDneu_func`` : callable ``(en, x, znu, sgnq) -> float`` or ``None``,
    * ``rho_nu_SD`` : callable ``T_nu -> MeV^4`` or ``None``.

    ``cfg`` and ``plasma`` are stored for the subclasses' use.
    """

    def __init__(self, cfg, plasma):
        self.cfg = cfg
        self.plasma = plasma
        # Defaults: no spectral distortion / no extra energy density.  Set here
        # so every implementation has them even if it never overrides them.
        self.dFDneu_func = None
        self.rho_nu_SD = None
        self._build_temperatures()
        self._build_distortion()

    # -- to be provided by concrete implementations -------------------------
    def _build_temperatures(self):
        """Set Tnue_of_Tg / Tnumu_of_Tg / Tnutau_of_Tg / N_NEVO_of_Tg."""
        raise NotImplementedError

    def _build_distortion(self):
        """Optionally set dFDneu_func / rho_nu_SD.

        Default: no distortion.  ``NEVOTable`` overrides this for its
        table-based spectrum; the analytic mu+y distortion is added by the
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
        #   NEVOPRIMAT_col_1_7.csv        — computed with QED corrections
        #   NEVOPRIMAT_NoQED_col_1_7.csv  — computed without QED corrections
        # We select the one that is consistent with cfg.QED_corrections so
        # that the neutrino temperatures entering the background are derived
        # from the same plasma equation of state that is used in the rest of
        # the computation.
        nevo_file = ("NEVOPRIMAT_col_1_7.csv" if cfg.QED_corrections
                     else "NEVOPRIMAT_NoQED_col_1_7.csv")
        nevo_path = cfg.data_dir + "/rates/NEVO/" + nevo_file
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
            Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab = (
                arr[::-1] for arr in (Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab))

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
        nevo_full_file = ("NEVOPRIMAT.csv" if cfg.QED_corrections
                          else "NEVOPRIMAT_NoQED.csv")
        nevo_full_path = cfg.data_dir + "/rates/NEVO/" + nevo_full_file
        grid_path      = cfg.data_dir + "/rates/NEVO/NEVOGrid.csv"

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
    """Analytic mu-type + y-type (SZ) spectral distortion, decorating a base.

    Wraps an existing :class:`NeutrinoHistory` (``base``) and adds the analytic
    distortion: it inherits ``base``'s temperatures and heating unchanged and
    overrides only ``dFDneu_func`` (the n<->p weak-rate correction) and
    ``rho_nu_SD`` (the extra neutrino energy density fed into the Friedmann
    equation).  Used when ``cfg.spectral_distortions`` and
    ``cfg.analytic_distortions``; ``PyPRConfig`` guarantees that pairing implies
    instantaneous decoupling, so ``base`` is an :class:`InstantaneousDecoupling`.

    The distortion amplitudes are continuous knobs: ``cfg.delta_xi_nu``
    (mu-type, a shift of the reduced chemical potential) and ``cfg.y_SZ``
    (y-type / SZ).
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

        # ---- μ-type and y-type (SZ) analytic distortions ----
        # μ-type: shift of the reduced chemical potential by delta_xi_nu.
        #   δf_μ^ν(y)   = 1/(e^{y-(ξ+δξ)}+1) − 1/(e^{y−ξ}+1)
        #   δf_μ^{ν̄}(y) = 1/(e^{y+(ξ+δξ)}+1) − 1/(e^{y+ξ}+1)
        # y-type: the SZ spectral shape is the energy derivative of
        # the Fermi-Dirac weighted by y²:
        #   δf_y^ν(y)   = (1/y²) d/dy(y⁴ df_FD/dy)
        #               = f_FD(1−f_FD)[4 − y(1−2f_FD)]  (analytic)
        # Ref: PRIMAT-Main.m, MuDistortionNeutrinos / YDistortionNeutrinos.
        dxi  = cfg.delta_xi_nu
        y_sz = cfg.y_SZ

        def _fd(arg):
            # Safe Fermi-Dirac: returns 0 for large positive arguments.
            return 0. if arg > _EXP_CUT else 1. / (np.exp(arg) + 1.)

        def _dFDneu_analytic(en, x, znu, sgnq):
            """Analytic μ+y spectral distortion of neutrinos/antineutrinos.

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
            # convention where μ_{ν̄} = −μ_ν.
            xi = sgnq * xi_nu

            # μ-type distortion (shift chemical potential by dxi)
            mu_dist = _fd(y - (xi + dxi)) - _fd(y - xi)

            if y_sz == 0.:
                return mu_dist

            # y-type (SZ) distortion: (1/y²) d/dy(y⁴ d f_FD/dy)
            # Analytic form: f(1-f)[4 - y(1-2f)]
            f = _fd(y - xi)
            y_dist = f * (1. - f) * (4. - y * (1. - 2. * f))
            return mu_dist + y_sz * y_dist

        def dFDneu_func(en, x, znu, sgnq):
            """Dispatch δf for all sign combinations of en and sgnq.

            PRIMAT's dFDneu handles four cases depending on whether en
            is the initial-state particle (en > 0) or a Pauli-blocking
            factor (en < 0), and whether the reaction is n→p (sgnq=+1)
            or p→n (sgnq=−1).  The blocking cases carry an extra minus
            sign and use the conjugate spectrum.
            """
            if en >= 0.:
                # Neutrino (sgnq>0) or antineutrino (sgnq<0) in initial state
                return _dFDneu_analytic(en, x, znu, sgnq)
            else:
                # Pauli-blocking factor: the conjugate particle is the one
                # that appears in the final state, so flip both en and sgnq.
                return -_dFDneu_analytic(-en, x, znu, -sgnq)

        self.dFDneu_func = dFDneu_func

        # Extra neutrino energy density from the distortion (Friedmann eq.)
        # Analytic integrals ∫₀^∞ y³ δf dy for each distortion type.
        # Ref: PRIMAT-Main.m, Inty3MuDistortion and Inty3SZdistortion.
        #
        # Inty3Mu(ξ, δξ) = (δξ/4)(δξ+2ξ)(δξ²+2δξξ+2(π²+ξ²))
        # Inty3SZ(ξ)     = 7π⁴/15 + 2π²ξ² + ξ⁴
        # ρ_νSD = N_ν (kT_ν)⁴/(2π²ℏ³c⁵) × [Inty3Mu + YSZ × Inty3SZ]
        #
        # In PyPRIMAT units (MeV throughout) the prefactor is
        #   N_ν × (kT_ν)⁴/(2π²) × (MeV_to_secm1/c)³/(c²)
        # which is exactly rho_nu(T_ν) × (Inty3 / Inty3_FD) where the
        # standard FD integral ∫y³ f_FD dy = 7π⁴/120 per degree of freedom.
        # We use the rho_nu function from plasma.py and rescale by the ratio.
        #
        # N_ν = 3 (three flavours, assumed to have the same distortion).
        Inty3_FD = 7. * np.pi**4 / 120.   # ∫₀^∞ y³ f_FD dy (zero μ)

        def _rho_nu_SD(Tnu):
            """Extra neutrino energy density from the analytic distortion."""
            # ξ = 0 assumed for the energy-density integrals (munuOverTnu
            # affects f_FD shape but is typically 0 in standard BBN).
            Inty3_mu = (dxi / 4.) * (dxi + 2.*xi_nu) * (
                dxi**2 + 2.*dxi*xi_nu + 2.*(np.pi**2 + xi_nu**2))
            Inty3_sz = 7.*np.pi**4/15. + 2.*np.pi**2*xi_nu**2 + xi_nu**4
            extra_int = Inty3_mu + y_sz * Inty3_sz
            # rho_nu(Tnu) = Nnu * 7π⁴/120 × (kTnu)⁴/(2π²) × prefactor,
            # so ρ_νSD/rho_nu = Nnu × extra_int / (Nnu × Inty3_FD).
            return self.plasma.rho_nu(Tnu) * extra_int / Inty3_FD

        self.rho_nu_SD = _rho_nu_SD


def make_neutrino_history(cfg, plasma):
    """Build the :class:`NeutrinoHistory` selected by ``cfg``.

    Dispatch only — the legality of every flag combination is validated once in
    :class:`~pyprimat.config.PyPRConfig` (spectral_distortions ×
    analytic_distortions × incomplete_decoupling), so here we simply:

    1. pick the base regime: :class:`NEVOTable` if ``cfg.incomplete_decoupling``
       else :class:`InstantaneousDecoupling` (which also builds the NEVO-table
       spectral distortion when that mode is active);
    2. if ``cfg.spectral_distortions and cfg.analytic_distortions``, wrap it in
       :class:`AnalyticDistortion` to layer on the analytic μ+y distortion.

    Args:
        cfg    : PyPRConfig instance.
        plasma : initialised pyprimat.plasma module (or a Plasma object exposing
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
