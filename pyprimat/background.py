# -*- coding: utf-8 -*-
"""
background.py
==============
Cosmological-background component for PyPRIMAT ("Class 1" of the
``PyPR`` split, see MODULAR.md).

A *background* encapsulates everything the nuclear-network integration
(:class:`pyprimat.nuclear_network.NuclearNetwork`, "Class 2") needs about the
expanding Universe, but nothing about nuclear reactions: the ``a <-> t <-> T``
relations, the Hubble rate, the baryon density ``rho_B(t)``, the n<->p weak
rates (already corrected for the neutrino temperatures/spectral distortions),
and the derived ``N_eff``/``Omega_nu`` quantities.

:class:`Background` is the interface; :class:`StandardBackground` is today's
(and so far only) implementation: NEVO non-instantaneous decoupling or
instantaneous decoupling, selected via ``cfg.incomplete_decoupling``, with the
neutrino sector itself delegated to
:func:`pyprimat.neutrino_history.make_neutrino_history` (already a pluggable
seam).  Following the style of :mod:`pyprimat.neutrino_history`, this is a
plain base class whose interface methods raise ``NotImplementedError`` --
*not* ``abc.ABC``/``@abstractmethod`` -- so a user can subclass
:class:`Background` for a custom cosmology (e.g. non-standard expansion
history) and hand an instance to a future ``PyPR(..., background=...)`` hook.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

from . import weak_rates as PyPRnTOp
from .neutrino_history import make_neutrino_history

__all__ = ["Background", "StandardBackground"]


class Background(object):
    """Interface for the cosmological background consumed by ``NuclearNetwork``.

    ``__init__`` stores the three constructor arguments common to every
    implementation; concrete subclasses are responsible for populating all
    the attributes/methods documented below before returning from their own
    ``__init__``.

    Parameters
    ----------
    cfg : PyPRConfig
        Run-time configuration (physical constants + flags).
    plasma : pyprimat.plasma.Plasma
        Per-instance QED/electron-thermodynamics tables.
    extra_rho : list of callable, optional
        Extra contributions to the total energy density entering the
        Friedmann equation.  Each element is a function
        ``rho(Tg) -> MeV^4`` of the photon temperature ``Tg`` [MeV], summed
        into ``rho_tot`` by :meth:`Hubble`.  Stored (as a list, copied) on
        :attr:`extra_rho`; subclasses may append further components (e.g.
        Early Dark Energy) to that list during their own setup.

    Attributes a subclass must set during ``__init__``
    ----------------------------------------------------
    Time/scale-factor relations (array-safe interpolants, all ``Tg`` in MeV,
    ``t`` in s, ``a`` dimensionless with ``a=1`` "today" up to the small
    entropy-injection correction -- see :attr:`a_of_T`):

    * ``T_of_t``, ``t_of_T``  -- T_gamma(t) <-> t(T_gamma)
    * ``a_of_T``, ``T_of_a``  -- a(T_gamma) <-> T_gamma(a)
    * ``a_of_t``, ``t_of_a``  -- a(t) <-> t(a)
    * ``TnuofT``              -- energy-weighted-average T_nu(T_gamma)

    Sampled background vectors (all evaluated on the same time grid):

    * ``t_vec``      -- cosmic time [s]
    * ``Tg_vec``     -- T_gamma [MeV]
    * ``Tnu_vec``    -- energy-weighted-average T_nu [MeV]
    * ``Tnue_vec``, ``Tnumu_vec``, ``Tnutau_vec`` -- per-flavour T_nu [MeV]

    Neutrino-sector hooks (see :mod:`pyprimat.neutrino_history`), ``None``
    when not applicable:

    * ``dFDneu_func``   -- spectral-distortion correction to the n<->p weak
      rate integrand, or ``None``
    * ``rho_nu_SD``     -- extra neutrino energy density rho_nu_SD(T_nu)
      [MeV^4] from an analytic spectral distortion, or ``None``
    * ``N_NEVO_of_Tg``  -- NEVO heating function N(T_gamma) driving the
      a(T_gamma) ODE (the N=0 stub under instantaneous decoupling)
    * ``has_heating_table`` -- ``True`` iff ``N_NEVO_of_Tg`` is a real NEVO
      heating table (mirrors ``cfg.incomplete_decoupling``)

    Weak-rate raw interpolants (un-normalised, [s^-1]):

    * ``weak_nTOp_frwrd_raw``, ``weak_nTOp_bkwrd_raw``

    and the settable :attr:`NormWeakRates` property.
    """

    def __init__(self, cfg, plasma, extra_rho=None):
        self.cfg = cfg
        self.plasma = plasma
        # Pluggable extra energy-density components (Early Dark Energy etc.);
        # copied so the caller's list is not mutated by subclass setup.
        self.extra_rho = list(extra_rho) if extra_rho is not None else []

    # ======================================================================
    # Friedmann expansion rate
    # ======================================================================

    def Hubble(self, Tg, Tnue, Tnumu, Tnutau):
        """Hubble rate H [s^-1] at photon temperature ``Tg`` [MeV] and the
        three flavour neutrino temperatures ``Tnue``/``Tnumu``/``Tnutau``
        [MeV] (array-safe)."""
        raise NotImplementedError

    # ======================================================================
    # Baryon sector
    # ======================================================================

    def rhoB_of_t(self, t):
        """Baryon mass density rho_B(t) [g cm^-3] -- the prefactor for
        nuclear reaction rates (rate ~ rho_B)."""
        raise NotImplementedError

    def etab_of_T(self, T_K):
        """Baryon-to-photon ratio eta_b(T) = n_B(a(T)) / n_gamma(T) at photon
        temperature ``T_K`` [Kelvin]; used by the Saha (NSE) seed."""
        raise NotImplementedError

    # ======================================================================
    # n <-> p weak rates (normalised)
    # ======================================================================

    def weak_nTOp_frwrd(self, T_K):
        """Normalised n -> p weak rate [s^-1] at photon temperature ``T_K``
        [Kelvin], i.e. ``NormWeakRates * weak_nTOp_frwrd_raw(T_K)``."""
        raise NotImplementedError

    def weak_nTOp_bkwrd(self, T_K):
        """Normalised p -> n weak rate [s^-1] at photon temperature ``T_K``
        [Kelvin], i.e. ``NormWeakRates * weak_nTOp_bkwrd_raw(T_K)``."""
        raise NotImplementedError

    @property
    def NormWeakRates(self):
        """Normalisation factor applied to the raw n<->p weak-rate
        interpolants (settable -- the MC driver rescales it per-sample to
        propagate the neutron-lifetime uncertainty, see
        ``pyprimat.main._mc_run_batch``)."""
        raise NotImplementedError

    @NormWeakRates.setter
    def NormWeakRates(self, value):
        raise NotImplementedError

    # ======================================================================
    # Derived cosmology
    # ======================================================================

    def N_eff(self, Tg, Tnue, Tnumu, Tnutau):
        """Effective number of relativistic neutrino species at photon
        temperature ``Tg`` [MeV] and flavour neutrino temperatures
        ``Tnue``/``Tnumu``/``Tnutau`` [MeV]."""
        raise NotImplementedError

    def Omeganuh2_relnu(self):
        """Omega_nu h^2 x 1e-6 for the relic neutrino background, treated as
        relativistic today (massless-neutrino convention)."""
        raise NotImplementedError

    def Omeganuh2_nrnu(self):
        """Omega_nu h^2 x 1e-6 for the relic neutrino background, treated as
        non-relativistic today (massive-neutrino convention)."""
        raise NotImplementedError


class StandardBackground(Background):
    """The standard PyPRIMAT cosmological background.

    Builds the ``a <-> t <-> T`` relations and n<->p weak rates exactly as
    the pre-split ``PyPR`` did, under either of the two decoupling regimes
    selected by ``cfg.incomplete_decoupling`` (NEVO non-instantaneous
    decoupling, or instantaneous decoupling via EM entropy conservation), with
    the neutrino sector supplied by
    :func:`pyprimat.neutrino_history.make_neutrino_history`.

    Parameters
    ----------
    cfg : PyPRConfig
    plasma : pyprimat.plasma.Plasma
    extra_rho : list of callable, optional
        See :class:`Background`.  Early Dark Energy (``cfg.fEDE > 0``) is
        appended automatically as the first such plug-in (see
        :meth:`_setup_EDE`); callers do not need to include it.
    """

    def __init__(self, cfg, plasma, extra_rho=None):
        super().__init__(cfg, plasma, extra_rho)
        self._setup_EDE()
        self._setup_background_and_cosmo()
        self._setup_derived_cosmo()
        self._setup_weak_rates()

    # ======================================================================
    # Early Dark Energy setup
    # ======================================================================

    def _setup_EDE(self):
        """Build the EDE energy-density function from cfg.fEDE/zcEDE/wnEDE.

        If fEDE > 0, appends a ``rho_EDE(Tg) -> MeV^4`` callable to
        ``self.extra_rho`` (the generic extra-energy-density plug-in list);
        otherwise a no-op.  Must be called after ``self.plasma`` and
        ``self.extra_rho`` are set, since it evaluates rho_g and appends to
        that list.
        """
        cfg = self.cfg
        if cfg.fEDE == 0.:
            return

        thermo  = self.plasma
        acEDE   = 1. / (1. + cfg.zcEDE)
        amaxEDE = acEDE * (4. / (3. * cfg.wnEDE - 1.))**(1. / (3. * cfg.wnEDE + 3.))
        TmaxEDE = cfg.T0CMB / amaxEDE / cfg.MeV_to_Kelvin   # [MeV]
        TcEDE   = cfg.T0CMB / acEDE   / cfg.MeV_to_Kelvin   # [MeV]

        #The final Neff value in the standard case (3.044) is hard coded here.
        rhocEDEac = (cfg.fEDE / (1. - cfg.fEDE)
                     * thermo.rho_g(TmaxEDE)
                     * (1. + 3.044 * 7./8. * (4./11.)**(4./3.))
                     / 2.
                     * (1. + 4. / (3. * cfg.wnEDE - 1.)))

        def rho_EDE(T):
            return 2. * rhocEDEac / (1. + (TcEDE / T)**(3. * cfg.wnEDE + 3.))

        self.extra_rho.append(rho_EDE)

    # ======================================================================
    # Friedmann expansion rate
    # ======================================================================

    def Hubble(self, Tg, Tnue, Tnumu, Tnutau):
        """Friedmann expansion rate H [s^-1] (see :meth:`Background.Hubble`)."""
        cfg     = self.cfg
        thermo  = self.plasma
        rho_pl  = thermo.rho_g(Tg) + thermo.rho_e(Tg) - thermo.PQEDofT(Tg) + Tg * thermo.dPQEDdT(Tg)
        rho_3nu = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau)
        rho_tot = rho_pl + rho_3nu + thermo.rho_nu_extra(Tg)
        for rho_extra in self.extra_rho:
            rho_tot += rho_extra(Tg)
        # For analytic spectral distortions the neutrino phase-space distribution
        # is shifted from a perfect FD, adding extra energy density.  The NEVO
        # case needs no correction: the NEVO temperatures are defined as the
        # energy-equivalent FD temperature, so rho_nu already accounts for the
        # distortion.
        if self.rho_nu_SD is not None:
            # Average T_ν: use the energy-weighted mean of the three flavours.
            Tnu_avg = ((Tnue**4 + Tnumu**4 + Tnutau**4) / 3.)**0.25
            rho_tot += self.rho_nu_SD(Tnu_avg)
        return cfg.MeV_to_secm1 * (rho_tot * 8. * np.pi / (3. * cfg.Mpl**2))**0.5

    # ======================================================================
    # Background thermodynamics + cosmological setup
    # ======================================================================

    def _setup_background_and_cosmo(self):
        """Cosmological background thermodynamics.

        Two operating modes, selected by ``cfg.incomplete_decoupling``:

        *Incomplete decoupling* (``True``, default):
            Reads the pre-computed NEVO neutrino-decoupling table
            (``rates/NEVO/NEVOPRIMAT_col_1_7.csv``).  The three neutrino
            flavour temperatures T_νe, T_νμ, T_ντ are interpolated from the
            table, and the NEVO heating function N(T_γ) — representing the
            extra entropy injected into neutrinos during e+e- annihilations —
            drives the a(T_γ) ODE.

        *Instantaneous (complete) decoupling* (``False``):
            The NEVO table is **not** loaded.  All three neutrino flavours are
            assumed to have decoupled instantaneously from the EM plasma, so
            their common temperature is given by EM entropy conservation:

                T_ν = (spl(T_γ) / s_∞)^{1/3}

            where s_∞ = 11π²/45 is the high-T limit of spl(T)/T³ (photons +
            e+e- pairs at T >> m_e).  The NEVO heating function is set to
            N ≡ 0, reducing the a(T_γ) ODE to standard EM entropy
            conservation.  QED corrections to the plasma equation of state are
            still included via the spl/PQEDofT tables in both modes.

        In both modes the method builds all interpolants used by
        ``_setup_derived_cosmo``, ``_setup_weak_rates``, and the nuclear
        network's ``solve()``.

        Independently of the above, ``cfg.external_background`` selects how
        ``a(T_γ)`` itself is obtained (requires ``incomplete_decoupling=True``):

        *Minimal* (``False``, default):
            ``a(T_γ)`` is reconstructed by solving the entropy-conservation
            ODE ``d(ln a)/d(ln T) = -(3 s̄ + T ds̄/dT)/(N_NEVO + 3 s̄)`` driven
            by the NEVO heating function ``N_NEVO(T_γ)``.

        *External* (``True``):
            ``a(T_γ)`` is read directly from the NEVO table's ``x`` column
            (``x ∝ a`` by the NEVO convention), with radiation-domination
            extrapolation (``a ∝ 1/T_γ``) outside the table. No ODE is
            solved for ``a(T)``. ``t(a)`` is obtained the same way in both
            modes (Hubble integration below), since no NEVO file carries a
            cosmic-time column. See NEUTRINOS.md for the derivation and the
            empirical check that the two modes agree to ~1e-6.
        """
        cfg    = self.cfg
        thermo = self.plasma

        Tstartcosmo  = cfg.T_start_cosmo / cfg.MeV_to_Kelvin
        Tstart = cfg.T_start / cfg.MeV_to_Kelvin   # [MeV]
        Tend   = cfg.T_end   / cfg.MeV_to_Kelvin   # [MeV]

        # ------------------------------------------------------------------
        # Step 1 - Neutrino-sector background (temperatures, heating,
        #          spectral distortion, extra neutrino energy density)
        # ------------------------------------------------------------------
        # The neutrino sector is encapsulated in a NeutrinoHistory object
        # (pyprimat.neutrino_history): NEVOTable for incomplete
        # decoupling, InstantaneousDecoupling otherwise, optionally decorated
        # with the analytic μ+y spectral distortion (AnalyticDistortion).  It
        # exposes the three flavour temperature functions, the NEVO heating
        # N(T_γ) that drives the a(T_γ) ODE, the n<->p weak-rate distortion
        # dFDneu_func, and the extra neutrino energy density rho_nu_SD.
        nh = make_neutrino_history(cfg, thermo)

        Tnue_of_Tg   = nh.Tnue_of_Tg
        Tnumu_of_Tg  = nh.Tnumu_of_Tg
        Tnutau_of_Tg = nh.Tnutau_of_Tg
        N_NEVO_of_Tg = nh.N_NEVO_of_Tg

        # Spectral-distortion hooks consumed by Hubble (extra ρ via
        # self.rho_nu_SD) and _setup_weak_rates (self.dFDneu_func).  Both are
        # None when there are no distortions.
        self.dFDneu_func = nh.dFDneu_func   # None means "no spectral distortions"
        self.rho_nu_SD   = nh.rho_nu_SD     # None means "no extra energy density"

        # ------------------------------------------------------------------
        # Step 2 – Build a(T) / invert to T(a)
        # ------------------------------------------------------------------
        def _sbar(T):
            return thermo.spl(T) / T**3   # dimensionless

        z0   = cfg.T0CMB / cfg.MeV_to_Kelvin   # [MeV]
        # Algebraic entropy-conservation boundary value a(Tend) = zend/Tend,
        # used as the ODE initial condition (minimal mode) and as the
        # table-normalisation anchor (external_background mode) -- see
        # NEUTRINOS.md.  Requires no ODE: _sbar is the analytic
        # electron-thermo spline.
        zend = z0 / (_sbar(Tend) / cfg.s0bar) ** (1. / 3.)
        a_end = zend / Tend

        # Build the log-temperature grid directly with linspace and feed it
        # straight to t_eval.  Do NOT reconstruct it as log(logspace(...)):
        # the log->exp->log roundtrip can push an endpoint 1 ULP outside the
        # integration span [log(Tend), log(Tstartcosmo)], which makes
        # solve_ivp raise "Values in t_eval are not within t_span" for some
        # values of Tend (e.g. T_end = 2e-3 MeV fails while 1e-3 MeV happens
        # to roundtrip exactly).  np.linspace guarantees its endpoints equal
        # the span bounds exactly, so the check always passes.
        lnT_sol = np.linspace(np.log(Tend), np.log(Tstartcosmo), cfg.n_temperature_table)
        T_sol   = np.exp(lnT_sol)

        if cfg.external_background:
            # --------------------------------------------------------------
            # external_background=True: a(T) read directly from the NEVO
            # table's x column (a ∝ x by the NEVO convention; see
            # NEUTRINOS.md), normalised so a(Tend) matches the algebraic
            # a_end above -- the same boundary value the minimal-mode ODE
            # converges to.  No ODE solve.
            # --------------------------------------------------------------
            K = a_end / float(nh.x_of_Tg(Tend))

            def a_of_T(T):
                return K * nh.x_of_Tg(T)
        else:
            # --------------------------------------------------------------
            # minimal mode (default): solve the entropy-conservation ODE
            # d(ln a)/d(ln T) for the EM plasma.  The reduced entropy
            # sbar = s/T^3 and its T-derivative are both obtained
            # analytically from the electron-thermo spline via
            # thermo.spl_and_dspl_dT (a single evaluation returning s and
            # ds/dT): this is exact and fast, so no numerical-derivative
            # fallback (numdifftools / finite differences) is needed.
            # --------------------------------------------------------------
            def _dlnadlnT_NEVO(lnT, y):
                T = np.exp(lnT)
                s, ds_dT = thermo.spl_and_dspl_dT(T)
                sb     = s / T**3
                dsbdT  = ds_dT / T**3 - 3. * s / T**4   # d(s/T^3)/dT, chain rule
                N = float(N_NEVO_of_Tg(T))
                return [-(3. * sb + T * dsbdT) / (N + 3. * sb)]

            lna_end = np.log(a_end)
            _t_nevo_a0 = time.time()
            sol_lna = solve_ivp(_dlnadlnT_NEVO,
                                [np.log(Tend), np.log(Tstartcosmo)],
                                [lna_end],
                                t_eval=lnT_sol,
                                method='LSODA', rtol=0.1*cfg.numerical_precision, atol=1e-10)
            if cfg.debug:
                print((f"[bckg]  Finished a(T) solve in {time.time()- _t_nevo_a0:.2f} s "
                       f"(status={sol_lna.status}, nfev={sol_lna.nfev})"), flush=True)
            _lnalnT = interp1d(sol_lna.t, sol_lna.y[0].flatten(),
                               bounds_error=False, fill_value="extrapolate")

            def a_of_T(T):
                return np.exp(_lnalnT(np.log(T)))

        # ------------------------------------------------------------------
        # Step 3 – Invert a(T) → T(a), then integrate dt/d(ln a) = 1/H(a)
        # ------------------------------------------------------------------
        T_grid = T_sol                          # already sampled low→high
        a_grid = a_of_T(T_grid)                  # low a → high a (a_of_T is array-safe)

        T_of_a = interp1d(a_grid, T_grid, bounds_error=False, fill_value="extrapolate")

        a_ini = a_of_T(Tstartcosmo)
        a_fin = a_of_T(Tend)

        # ------------------------------------------------------------------
        # Step 4 – Integrate dt/d(ln a) = 1/H(a)
        # ------------------------------------------------------------------
        def Hubble_NEVO(Tg):
            return self.Hubble(Tg, Tnue_of_Tg(Tg), Tnumu_of_Tg(Tg), Tnutau_of_Tg(Tg))

        t_ini = 1. / (2. * Hubble_NEVO(Tstartcosmo))

        # Log-scale-factor grid, built directly in log space (see the note on
        # the a(T) solve above): feeding linspace endpoints straight to t_eval
        # avoids the log(logspace(...)) roundtrip that could land the last
        # point 1 ULP outside [log(a_ini), log(a_fin)].
        lna_samp = np.linspace(np.log(a_ini), np.log(a_fin), cfg.n_temperature_table)

        def _dtdlna(lna, t):
            return [1. / Hubble_NEVO(T_of_a(np.exp(lna)))]

        _t_nevo_t0 = time.time()
        sol_t = solve_ivp(_dtdlna,
                          [np.log(a_ini), np.log(a_fin)],
                          [t_ini],
                          t_eval=lna_samp,
                          method='LSODA', rtol=cfg.numerical_precision, atol=1e-12)
        if cfg.debug:
            print((f"[bckg]  Finished t(a) solve in {time.time()-_t_nevo_t0:.2f} s "
                   f"(status={sol_t.status}, nfev={sol_t.nfev})"), flush=True)

        t_of_lna = interp1d(sol_t.t, sol_t.y[0].flatten(),
                            bounds_error=False, fill_value="extrapolate")

        # ------------------------------------------------------------------
        # Step 5 – Sample on the common time grid; set instance attributes
        # ------------------------------------------------------------------
        a_arr  = np.exp(sol_t.t)       # a values at ODE evaluation points
        t_vec  = sol_t.y[0].flatten()  # corresponding t [s]
        Tg_vec = T_of_a(a_arr)         # T_γ [MeV]

        Tnue_vec   = Tnue_of_Tg(Tg_vec)
        Tnumu_vec  = Tnumu_of_Tg(Tg_vec)
        Tnutau_vec = Tnutau_of_Tg(Tg_vec)
        # Energy-weighted average neutrino temperature (for weak rates / Omega_ν)
        Tnu_avg_vec = ((Tnue_vec**4 + Tnumu_vec**4 + Tnutau_vec**4) / 3.)**0.25

        self.t_vec      = t_vec
        self.Tg_vec     = Tg_vec
        self.Tnu_vec    = Tnu_avg_vec   # average, used by _setup_derived_cosmo and _setup_weak_rates
        self.Tnue_vec   = Tnue_vec
        self.Tnumu_vec  = Tnumu_vec
        self.Tnutau_vec = Tnutau_vec

        self.t_of_T = interp1d(Tg_vec, t_vec, bounds_error=False,
                                fill_value="extrapolate", kind='linear')
        self.T_of_t = interp1d(t_vec, Tg_vec, bounds_error=False,
                                fill_value="extrapolate", kind='linear')
        self.TnuofT = interp1d(Tg_vec, Tnu_avg_vec, bounds_error=False,
                                fill_value="extrapolate", kind='linear')
        self.a_of_T = a_of_T   # already vectorised: np.exp(interp1d(log T))
        self.T_of_a = T_of_a
        self.a_of_t = interp1d(t_vec, a_arr, bounds_error=False,
                                fill_value=(a_arr[0], a_arr[-1]))
        self.t_of_a = interp1d(a_arr, t_vec, bounds_error=False,
                                fill_value=(t_vec[0], t_vec[-1]))
        self.N_NEVO_of_Tg = N_NEVO_of_Tg

        # Whether N_NEVO_of_Tg is a *real* heating table (NEVOTable, read from
        # rates/NEVO/) or just the N=0 stub used by InstantaneousDecoupling to
        # close the a(T) ODE under EM entropy conservation.  Consumed by
        # NuclearNetwork._write_time_evolution to decide whether the
        # "Nheating" column carries physical information or would just be a
        # column of zeros.
        self.has_heating_table = cfg.incomplete_decoupling

    def _setup_derived_cosmo(self):
        """Build N_eff and relic-neutrino Omega functions from the stored background.

        Called after _setup_background_and_cosmo.
        Requires self.Tg_vec, self.Tnu_vec to be set.
        """
        cfg    = self.cfg
        thermo = self.plasma

        # N_eff
        def N_eff(Tg, Tnue, Tnumu, Tnutau):
            rho_g   = thermo.rho_g(Tg)
            rho_rad = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau) + rho_g + thermo.rho_nu_extra(Tg)
            # Analytic spectral distortions add an extra neutrino energy
            # density on top of the (possibly shifted) FD temperatures Tnue/
            # Tnumu/Tnutau -- mirror Hubble's treatment exactly: a single
            # aggregate term evaluated at the energy-weighted mean Tnu, added
            # to rho_rad before forming the Neff ratio. As in Hubble, the
            # NEVO (incomplete_decoupling) case needs no correction since its
            # Tnu's are already the energy-equivalent FD temperatures, so
            # self.rho_nu_SD is None there.
            if self.rho_nu_SD is not None:
                Tnu_avg = ((Tnue**4 + Tnumu**4 + Tnutau**4) / 3.)**0.25
                rho_rad += self.rho_nu_SD(Tnu_avg)
            return (rho_rad - rho_g) / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

        self._N_eff = N_eff

        # Relic neutrino abundances
        def Omeganuh2_relnu():
            Tnu0 = self.Tnu_vec[-1] / self.Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (7. * np.pi**2 / 120. * Tnu0**4) / cfg.rhocOverh2

        def Omeganuh2_nrnu():
            Tnu0 = self.Tnu_vec[-1] / self.Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (3. / 2. * zeta(3) / np.pi**2 * Tnu0**3) / cfg.rhocOverh2

        self._Omeganuh2_relnu = Omeganuh2_relnu
        self._Omeganuh2_nrnu  = Omeganuh2_nrnu

    def N_eff(self, Tg, Tnue, Tnumu, Tnutau):
        """Effective number of relativistic neutrino species (see
        :meth:`Background.N_eff`)."""
        return self._N_eff(Tg, Tnue, Tnumu, Tnutau)

    def Omeganuh2_relnu(self):
        """Omega_nu h^2 x 1e-6, relativistic-neutrino convention (see
        :meth:`Background.Omeganuh2_relnu`)."""
        return self._Omeganuh2_relnu()

    def Omeganuh2_nrnu(self):
        """Omega_nu h^2 x 1e-6, non-relativistic-neutrino convention (see
        :meth:`Background.Omeganuh2_nrnu`)."""
        return self._Omeganuh2_nrnu()

    # ======================================================================
    # n <-> p weak rates
    # ======================================================================

    def _setup_weak_rates(self):
        cfg = self.cfg
        _t_weak0 = time.time()
        # Single forward and backward n<->p interpolant over the whole BBN
        # temperature range (the rate is continuous, so one grid suffices).
        self.weak_nTOp_frwrd_raw, self.weak_nTOp_bkwrd_raw = \
            PyPRnTOp.RecomputeWeakRates([self.Tg_vec, self.Tnue_vec], cfg,
                                        dFDneu_func=self.dFDneu_func)
        if cfg.debug:
            # Wording is generic on purpose: RecomputeWeakRates may have either
            # recomputed the rates (~2 s) or loaded them from a fingerprinted
            # cache file (~0 s) -- see pyprimat.weak_rates.RecomputeWeakRates.
            print((f"[weak]  n <--> p weak rates ready in "
                   f"{time.time()-_t_weak0:.2f} s"), flush=True)

        # Normalisation factor
        if cfg.tau_n_flag:
            Fn = PyPRnTOp.ComputeFn(cfg)
            self._norm_weak_rates = 1. / (Fn * cfg.tau_n)   # [s^-1]
        else:
            GFtilde2 = (cfg.GF * cfg.Vud)**2 * (1. + 3. * cfg.gA**2) / (2. * np.pi**3)
            self._norm_weak_rates = cfg.MeV_to_secm1 * (GFtilde2 * cfg.me**5)

    @property
    def NormWeakRates(self):
        """Normalisation factor applied to the raw n<->p weak-rate
        interpolants (see :attr:`Background.NormWeakRates`)."""
        return self._norm_weak_rates

    @NormWeakRates.setter
    def NormWeakRates(self, value):
        self._norm_weak_rates = value

    def weak_nTOp_frwrd(self, T_K):
        """Normalised n -> p weak rate [s^-1] (see
        :meth:`Background.weak_nTOp_frwrd`)."""
        return self._norm_weak_rates * self.weak_nTOp_frwrd_raw(T_K)

    def weak_nTOp_bkwrd(self, T_K):
        """Normalised p -> n weak rate [s^-1] (see
        :meth:`Background.weak_nTOp_bkwrd`)."""
        return self._norm_weak_rates * self.weak_nTOp_bkwrd_raw(T_K)

    # ======================================================================
    # Baryon sector
    # ======================================================================

    def _nB(self, a):
        """Comoving baryon number density n_B(a) = n_B0 / a^3 [MeV^3].

        ``n0CMB`` is the present-day CMB photon number density [MeV^3] and
        ``eta0b = n_B/n_gamma`` the baryon-to-photon ratio (from
        ``cfg.Omegabh2``).  Internal helper for :meth:`etab_of_T`; not part
        of the public :class:`Background` interface (rho_B is exported
        instead, via :meth:`rhoB_of_t`).
        """
        cfg = self.cfg
        return cfg.n0CMB * cfg.eta0b / a**3   # [MeV^3]

    def etab_of_T(self, T_K):
        """Effective baryon-to-photon ratio at photon temperature ``T_K``
        (Kelvin) (see :meth:`Background.etab_of_T`).

        eta_b(T) = n_B(a(T)) / n_gamma(T), n_gamma = (2 zeta(3)/pi^2) T^3.
        """
        cfg = self.cfg
        T_MeV = T_K / cfg.MeV_to_Kelvin
        ngCMB = (2. * zeta(3)) / np.pi**2 * T_MeV**3
        return self._nB(self.a_of_T(T_MeV)) / ngCMB

    def rhoB_BBN(self, a):
        """Baryon mass density at scale factor ``a`` [g cm^-3].

        Used as the prefactor for nuclear reaction rates (rate ~ rho_B).
        """
        cfg = self.cfg
        n0B = cfg.n0CMB * cfg.eta0b
        return cfg.ma * n0B * cfg.MeV4_to_gcmm3 / a**3  # [g cm^-3]

    def rhoB_of_t(self, t):
        """Baryon mass density rho_B(t) [g cm^-3] (see
        :meth:`Background.rhoB_of_t`)."""
        return self.rhoB_BBN(self.a_of_t(t))
