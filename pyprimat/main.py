# -*- coding: utf-8 -*-
"""
main.py
=======
Main class for PyPRIMAT.

Design
------
* ``PyPR.__init__(params)`` accepts an optional dict of parameters,
  builds a ``PyPRConfig``, loads all data files (thermodynamics tables and
  nuclear rate tables), and pre-computes the thermal background.
* ``PyPR.solve()`` runs the full nuclear network ODE integration and
  returns the BBN predictions.
* ``PyPR.PyPRresults()`` calls ``solve()`` and returns the result dict
  (for backwards compatibility).

"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

__all__ = ['PyPR', 'mc_uncertainty']

from .config       import PyPRConfig
from . import plasma      as PyPRthermo
from . import weak_rates   as PyPRnTOp
from .neutrino_history import make_neutrino_history


__version__ = "0.1.0"

# Column order for abundance interpolators; names match PyPRConfig.Nuclides keys
_NUC_NAMES_SMALL = ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"]
_NUC_NAMES_FULL  = ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7",
                    "He6", "Li8", "Li6", "B8"]

_BANNER = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                 ┃
┃   ░█▀█░█░█░█▀█░█▀▄░▀█▀░█▄█░█▀█░▀█▀              ┃
┃   ░█▀▀░░█░░█▀▀░█▀▄░░█░░█░█░█▀█░░█░              ┃
┃   ░▀░░░░▀░░▀░░░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░              ┃
┃                                                 ┃
┃    Welcome to PyPRIMAT v{version} — Cyril Pitrou    ┃
┃                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""".format(version=__version__)


class PyPR:
    """
    Main PyPRIMAT class.

    Parameters
    ----------
    params : dict, optional
        Run-time parameters overriding defaults (see ``config.DEFAULT_PARAMS``).
    extra_rho : list of callable, optional
        Extra contributions to the total energy density entering the
        Friedmann equation.  Each element is a function
        ``rho(Tg) -> MeV^4`` of the photon temperature ``Tg`` [MeV],
        summed into ``rho_tot`` by :meth:`_Hubble`.  This is the generic
        plug-in point for "dark sector" components; Early Dark Energy
        (``cfg.fEDE > 0``) is implemented as the first such plug-in (see
        :meth:`_setup_EDE`) and is appended automatically -- callers do not
        need to include it here.

        Example: a constant extra radiation density of dRho [MeV^4],
            >>> PyPR({"network": "small"}, extra_rho=[lambda Tg: dRho])
    """

    def __init__(self, params=None, extra_rho=None):

        # ------------------------------------------------------------------
        # 1. Build configuration
        # ------------------------------------------------------------------
        self.cfg = PyPRConfig(params or {})
        cfg = self.cfg
        self.N = {name: NZ[0]           for name, NZ in cfg.Nuclides.items()}
        self.Z = {name: NZ[1]           for name, NZ in cfg.Nuclides.items()}
        self.A = {name: NZ[0] + NZ[1]   for name, NZ in cfg.Nuclides.items()}

        if cfg.verbose:
            print(_BANNER)
            for msg in cfg._init_messages:
                print(msg)
            self._t0 = time.time()

        # ------------------------------------------------------------------
        # 2. Initialise thermodynamics (loads QED/neutrino tables)
        # ------------------------------------------------------------------
        # A per-instance Plasma object (rather than the module-level default)
        # so that several PyPR instances coexisting in the same process
        # (e.g. QED_corrections=True/False comparisons, MC workers) each
        # carry their own QED/electron-thermo tables without overwriting
        # one another's state.
        self.plasma = PyPRthermo.Plasma(cfg)

        # ------------------------------------------------------------------
        # 3. Pluggable extra energy-density components
        # ------------------------------------------------------------------
        self._extra_rho = list(extra_rho) if extra_rho is not None else []
        self._setup_EDE()

        # ------------------------------------------------------------------
        # 4b. Initialize nuclear network (MT/LT eras)
        # ------------------------------------------------------------------
        from .nuclear import UpdateNuclearRates
        self.nucl = UpdateNuclearRates(cfg)

        # ------------------------------------------------------------------
        # 5. Compute or load thermal background + cosmological functions
        # ------------------------------------------------------------------
        self._setup_background_and_cosmo()
        self._setup_derived_cosmo()

        # ------------------------------------------------------------------
        # 6. Compute or load n <--> p weak rates
        # ------------------------------------------------------------------
        self._setup_weak_rates()
        self._results = None

        if cfg.verbose:
            print(f"[init]  Initialisation complete in {time.time()-self._t0:.1f} s")

    # ======================================================================
    # Private: Early Dark Energy setup
    # ======================================================================

    def _setup_EDE(self):
        """Build the EDE energy-density function from cfg.fEDE/zcEDE/wnEDE.

        If fEDE > 0, appends a ``rho_EDE(Tg) -> MeV^4`` callable to
        ``self._extra_rho`` (the generic extra-energy-density plug-in list);
        otherwise a no-op.  Must be called after self.plasma
        and self._extra_rho are set, since it evaluates rho_g and appends to
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

        self._extra_rho.append(rho_EDE)

    # ======================================================================
    # Private: background thermodynamics + cosmological setup
    # ======================================================================

    # Friedmann expansion rate
    def _Hubble(self, Tg, Tnue, Tnumu, Tnutau):
        cfg     = self.cfg
        thermo  = self.plasma
        rho_pl  = thermo.rho_g(Tg) + thermo.rho_e(Tg) - thermo.PQEDofT(Tg) + Tg * thermo.dPQEDdT(Tg)
        rho_3nu = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau)
        rho_tot = rho_pl + rho_3nu + thermo.rho_nu_extra(Tg)
        for rho_extra in self._extra_rho:
            rho_tot += rho_extra(Tg)
        # For analytic spectral distortions the neutrino phase-space distribution
        # is shifted from a perfect FD, adding extra energy density.  The NEVO
        # case needs no correction: the NEVO temperatures are defined as the
        # energy-equivalent FD temperature, so rho_nu already accounts for the
        # distortion.
        if getattr(self, '_rho_nu_SD', None) is not None:
            # Average T_ν: use the energy-weighted mean of the three flavours.
            Tnu_avg = ((Tnue**4 + Tnumu**4 + Tnutau**4) / 3.)**0.25
            rho_tot += self._rho_nu_SD(Tnu_avg)
        return cfg.MeV_to_secm1 * (rho_tot * 8. * np.pi / (3. * cfg.Mpl**2))**0.5

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
        ``_setup_derived_cosmo``, ``_setup_weak_rates``, and ``solve()``.
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

        # Spectral-distortion hooks consumed by _Hubble (extra ρ via
        # self._rho_nu_SD) and _setup_weak_rates (self._dFDneu_func).  Both are
        # None when there are no distortions.
        self._dFDneu_func = nh.dFDneu_func   # None means "no spectral distortions"
        self._rho_nu_SD   = nh.rho_nu_SD     # None means "no extra energy density"

        # ------------------------------------------------------------------
        # Step 2 – Solve a(T) ODE / invert to T(a)
        # ------------------------------------------------------------------
        def _sbar(T):
            return thermo.spl(T) / T**3   # dimensionless

        if not cfg.analytic_entropy_derivative:
            if cfg.numdiff_installed:
                from numdifftools import Derivative
                _dsbardT = Derivative(_sbar, n=1)
            else:
                def _dsbardT(T):
                    dToT = 1.e-3
                    return (_sbar((1. + dToT) * T) - _sbar((1. - dToT) * T)) / (2. * dToT * T)

        if cfg.analytic_entropy_derivative:
            def _dlnadlnT_NEVO(lnT, y):
                T = np.exp(lnT)
                s, ds_dT = thermo.spl_and_dspl_dT(T)
                sb     = s / T**3
                dsbdT  = ds_dT / T**3 - 3. * s / T**4
                N = float(N_NEVO_of_Tg(T))
                return [-(3. * sb + T * dsbdT) / (N + 3. * sb)]
        else:
            def _dlnadlnT_NEVO(lnT, y):
                T   = np.exp(lnT)
                sb  = _sbar(T)
                N   = float(N_NEVO_of_Tg(T))
                return [-(3. * sb + T * _dsbardT(T)) / (N + 3. * sb)]       

        z0   = cfg.T0CMB / cfg.MeV_to_Kelvin   # [MeV]
        zend = z0 / (_sbar(Tend) / cfg.s0bar) ** (1. / 3.)
        lna_end = np.log(zend / Tend)

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
            return self._Hubble(Tg, Tnue_of_Tg(Tg), Tnumu_of_Tg(Tg), Tnutau_of_Tg(Tg))

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

        self._t_vec      = t_vec
        self._Tg_vec     = Tg_vec
        self._Tnu_vec    = Tnu_avg_vec   # average, used by _setup_derived_cosmo and _setup_weak_rates
        self._Tnue_vec   = Tnue_vec
        self._Tnumu_vec  = Tnumu_vec
        self._Tnutau_vec = Tnutau_vec

        self._t_of_T = interp1d(Tg_vec, t_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._T_of_t = interp1d(t_vec, Tg_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._TnuofT = interp1d(Tg_vec, Tnu_avg_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._a_of_T = a_of_T   # already vectorised: np.exp(interp1d(log T))
        self._a_of_t = interp1d(t_vec, a_arr, bounds_error=False,
                                 fill_value=(a_arr[0], a_arr[-1]))
        self._N_NEVO_of_Tg = N_NEVO_of_Tg

    def _setup_derived_cosmo(self):
        """Build N_eff and relic-neutrino Omega functions from the stored background.

        Called after _setup_background_and_cosmo.
        Requires self._Tg_vec, self._Tnu_vec to be set.
        """
        cfg    = self.cfg
        thermo = self.plasma

        # N_eff
        def N_eff(Tg, Tnue, Tnumu, Tnutau):
            rho_g   = thermo.rho_g(Tg)
            rho_rad = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau) + rho_g + thermo.rho_nu_extra(Tg)
            return (rho_rad - rho_g) / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

        self._N_eff = N_eff

        # Relic neutrino abundances
        def Omeganuh2_relnu():
            Tnu0 = self._Tnu_vec[-1] / self._Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (7. * np.pi**2 / 120. * Tnu0**4) / cfg.rhocOverh2

        def Omeganuh2_nrnu():
            Tnu0 = self._Tnu_vec[-1] / self._Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (3. / 2. * zeta(3) / np.pi**2 * Tnu0**3) / cfg.rhocOverh2

        self._Omeganuh2_relnu = Omeganuh2_relnu
        self._Omeganuh2_nrnu  = Omeganuh2_nrnu

    # ======================================================================
    # Private: weak rates
    # ======================================================================

    def _setup_weak_rates(self):
        cfg = self.cfg
        _t_weak0 = time.time()
        # Single forward and backward n<->p interpolant over the whole BBN
        # temperature range (the rate is continuous, so one grid suffices).
        self._nTOp_frwrd, self._nTOp_bkwrd = \
            PyPRnTOp.RecomputeWeakRates([self._Tg_vec, self._Tnue_vec], cfg,
                                        dFDneu_func=self._dFDneu_func)
        if cfg.debug:
            # Wording is generic on purpose: RecomputeWeakRates may have either
            # recomputed the rates (~2 s) or loaded them from a fingerprinted
            # cache file (~0 s) -- see pyprimat.weak_rates.RecomputeWeakRates.
            print((f"[weak]  n <--> p weak rates ready in "
                   f"{time.time()-_t_weak0:.2f} s"), flush=True)

        # Normalisation factor
        _t_norm0 = time.time()
        if cfg.tau_n_flag:
            Fn = PyPRnTOp.ComputeFn(cfg)
            self._NormWeakRates = 1. / (Fn * cfg.tau_n)   # [s^-1]
        else:
            GFtilde2 = (cfg.GF * cfg.Vud)**2 * (1. + 3. * cfg.gA**2) / (2. * np.pi**3)
            self._NormWeakRates = cfg.MeV_to_secm1 * (GFtilde2 * cfg.me**5)

    # ======================================================================
    # solve(): integrate nuclear network ODEs
    # ======================================================================

    def solve(self):
        """
        Integrate the nuclear network over the three temperature eras and
        return a dict of BBN observables.
        """
        from .nuclear import SPECIES_MD   # noqa: F401 (used for default-zero filling)
        cfg       = self.cfg
        t_vec     = self._t_vec
        Tg_vec    = self._Tg_vec
        Tnu_vec   = self._Tnu_vec
        a_of_t    = self._a_of_t
        T_of_t    = self._T_of_t
        t_of_T    = self._t_of_T
        NormWR    = self._NormWeakRates
        nucl      = self.nucl

        # Refresh nuclear rates with current variation parameters (p_*, NP_delta_*)
        nucl.apply_variations(cfg)

        if cfg.verbose:
            _t0 = time.time()

        # ------------------------------------------------------------------
        # Temperature era boundaries [s]
        # ------------------------------------------------------------------
        t_start = t_of_T(cfg.T_start / cfg.MeV_to_Kelvin)
        t_weak  = t_of_T(cfg.T_weak  / cfg.MeV_to_Kelvin)
        t_nucl  = t_of_T(cfg.T_nucl  / cfg.MeV_to_Kelvin)
        t_end   = t_of_T(cfg.T_end   / cfg.MeV_to_Kelvin)

        # ------------------------------------------------------------------
        # Baryon density for the nuclear network
        # ------------------------------------------------------------------
        def nB(a):
            # Comoving baryon number density: n_B = n_B0 / a³  [MeV³].
            # n0CMB is the present-day CMB photon number density (MeV³),
            # eta0b = n_B/n_γ the baryon-to-photon ratio (from cfg.Omegabh2).
            return cfg.n0CMB * cfg.eta0b / a**3   # [MeV^3]

        def etab_of_T(T_K):
            # Effective baryon-to-photon ratio at photon temperature T_K (Kelvin).
            # η_b(T) = n_B(a(T)) / n_γ(T),  n_γ = (2ζ(3)/π²) T³.
            T_MeV = T_K / cfg.MeV_to_Kelvin
            ngCMB = (2. * zeta(3)) / np.pi**2 * T_MeV**3
            return nB(self._a_of_T(T_MeV)) / ngCMB

        def rhoB_BBN(a):
            # Baryon mass density at scale factor a  [g cm⁻³].
            # Used as the prefactor for nuclear reaction rates (rate ∝ ρ_B).
            n0B = cfg.n0CMB * cfg.eta0b
            return cfg.ma * n0B * cfg.MeV4_to_gcmm3 / a**3  # [g cm^-3]

        # ------------------------------------------------------------------
        # Local thermal equilibrium (Saha) abundance
        # ------------------------------------------------------------------
        def YA(name, Yn, Yp, T):
            """Saha equilibrium mass-fraction abundance of nuclide `name`.

            At high temperature each nuclide is maintained in Nuclear Statistical
            Equilibrium (NSE) with free neutrons and protons via photo-dissociation.
            The Saha formula gives (Phys. Rep. §V.A):

                Y_A = g_A ζ(3)^{A-1} π^{(1-A)/2} 2^{(3A-5)/2}
                      × (M_A / mₙ^N mₚ^Z)^{3/2}
                      × (kB T)^{3(A-1)/2} η_b^{A-1}
                      × Yₙ^N Yₚ^Z exp(B_A / kB T)

            where A=N+Z is the mass number, g_A=2J+1 the spin degeneracy,
            B_A the binding energy (keV), and η_b = n_B/n_γ the baryon-to-photon
            ratio.  Used to seed the MT and LT era initial conditions.

            Args:
                name : nuclide name string (key into cfg.Nuclides/NuclExcessMass).
                Yn   : free neutron mass fraction.
                Yp   : free proton mass fraction.
                T    : photon temperature in Kelvin.

            Returns:
                Y_A  : dimensionless mass fraction (≪ 1 at T ≫ BBN onset).
            """
            x     = cfg.Nuclides[name]
            A     = x[0] + x[1]
            Z     = x[1]
            N     = A - Z
            Mass  = (A * cfg.ma * cfg.MeV
                     + cfg.keV * cfg.NuclExcessMass[name]
                     - Z * cfg.me * cfg.MeV)
            BindE = (N * cfg.NuclExcessMass["n"]
                     + Z * cfg.NuclExcessMass["p"]
                     - cfg.NuclExcessMass[name])
            # (M_A / mₙ^N mₚ^Z)^{3/2}: ratio of nuclear to free-nucleon masses
            NormYA = (Mass / ((cfg.mn * cfg.MeV)**(A - Z)
                              * (cfg.mp * cfg.MeV)**Z))**(3. / 2.)
            return ((2 * cfg.NuclSpin[name] + 1)
                    * zeta(3)**(A - 1) * np.pi**((1 - A) / 2.)
                    * 2**((3 * A - 5) / 2.)
                    * NormYA
                    * (cfg.kB * T)**(3. / 2. * (A - 1))
                    * etab_of_T(T)**(A - 1)
                    * Yp**Z * Yn**(A - Z)
                    * np.exp(BindE * cfg.keV / (cfg.kB * T)))

        # ------------------------------------------------------------------
        # High-temperature (HT) era: only n and p
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl]  Solving neutron decoupling at high temperature era")

        nTOp_frwrd_HT = self._nTOp_frwrd
        nTOp_bkwrd_HT = self._nTOp_bkwrd

        def nTOp_frwrd_HT_norm(T): return NormWR * nTOp_frwrd_HT(T)
        def nTOp_bkwrd_HT_norm(T): return NormWR * nTOp_bkwrd_HT(T)

        def Yn_i_func(T):
            b = nTOp_bkwrd_HT_norm(T)
            return b / (b + nTOp_frwrd_HT_norm(T))

        def Y_prime_HT(t, Y):
            T_K = T_of_t(t) * cfg.MeV_to_Kelvin
            f   = nTOp_frwrd_HT_norm(T_K)
            b   = nTOp_bkwrd_HT_norm(T_K)
            return b * Y[1] - f * Y[0], f * Y[0] - b * Y[1]

        Yn_i = Yn_i_func(cfg.T_start)
        Yp_i = 1. - Yn_i
        _t_ht0 = time.time()
        sol_HT = solve_ivp(Y_prime_HT, [t_start, t_weak], [Yn_i, Yp_i],
                           method='LSODA', rtol=cfg.numerical_precision, atol=1e-10)
        if cfg.verbose:
            print((f"[nucl]  [HT] Finished solve_ivp in {time.time()-_t_ht0:.2f} s "
                   f"(status={sol_HT.status}, nfev={sol_HT.nfev})"), flush=True)
        Yn_HT_f, Yp_HT_f = sol_HT.y[0][-1], sol_HT.y[1][-1]

        # ------------------------------------------------------------------
        # ODE systems for the full network
        # ------------------------------------------------------------------
        def make_nTOp_pair(frwrd_raw, bkwrd_raw):
            # Wrap the raw rate interpolants with the NormWR normalisation factor.
            # NormWR = tau_n_ref / tau_n rescales K so the measured τ_n is used;
            # it equals 1 when tau_n_flag=False.
            def f(T): return NormWR * frwrd_raw(T)
            def b(T): return NormWR * bkwrd_raw(T)
            return f, b

        nTOp_f_MT, nTOp_b_MT = make_nTOp_pair(self._nTOp_frwrd, self._nTOp_bkwrd)
        nTOp_f_LT, nTOp_b_LT = make_nTOp_pair(self._nTOp_frwrd, self._nTOp_bkwrd)

        def Y_prime_MT(t, Y):
            rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.rhsMT(Y, T_K, rho, nTOp_f_MT, nTOp_b_MT)

        def Jacobian_MT(t, Y):
            rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.JacobianMT(Y, T_K, rho, nTOp_f_MT, nTOp_b_MT)

        def Y_prime_LT(t, Y):
            rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.rhsLT(Y, T_K, rho, nTOp_f_LT, nTOp_b_LT)

        def Jacobian_LT(t, Y):
            rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.JacobianLT(Y, T_K, rho, nTOp_f_LT, nTOp_b_LT)

        # ------------------------------------------------------------------
        # Mid-temperature (MT) era
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl]  Solving nuclear network at mid temperature era")

        # Saha (NSE) seed for all MT species except n and p, which come from
        # the HT solution.  The MT network's species list is determined by the
        # NetworkDefinition, so this loop is independent of the network size.
        mt_species = nucl._mt_net.species   # e.g. 8 for small, 12 for medium
        mt_saha = {"n": Yn_HT_f, "p": Yp_HT_f}
        for s in mt_species:
            if s not in mt_saha:
                mt_saha[s] = YA(s, Yn_HT_f, Yp_HT_f, cfg.T_weak)
        Yi_MT = [mt_saha[s] for s in mt_species]

        _t_mt0 = time.time()
        sol_MT = solve_ivp(Y_prime_MT, [t_weak, t_nucl], Yi_MT,
                           method='BDF', jac=Jacobian_MT,
                           rtol=cfg.numerical_precision, atol=1e-15)
        if cfg.verbose:
            print((f"[nucl]  [MT] Finished solve_ivp ({cfg.network} network, "
                   f"{len(mt_species)} species) in {time.time()-_t_mt0:.2f} s "
                   f"(status={sol_MT.status}, nfev={sol_MT.nfev})"), flush=True)
        # Extract MT final values by name — works for any network size.
        mt_final_raw = {s: sol_MT.y[i][-1] for i, s in enumerate(mt_species)}

        # ------------------------------------------------------------------
        # Low-temperature (LT) era
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl]  Solving nuclear network at low temperature era")

        # Seed the LT vector from MT final values, filling any extra species
        # (present in the LT but absent in MT) with 0.  By looking up by name,
        # this works for any MT and LT network sizes without hardcoding.
        species_L = nucl.species_large
        Yi_LT = [mt_final_raw.get(s, 0.0) for s in species_L]

        _t_lt0 = time.time()
        atol = cfg.atol_large_LT if cfg.is_large else 1e-15
        sol_LT = solve_ivp(Y_prime_LT, [t_nucl, t_end], Yi_LT,
                           method='BDF', jac=Jacobian_LT,
                           rtol=10.*cfg.numerical_precision, atol=atol)
        if cfg.verbose:
            print((f"[nucl]  [LT] Finished solve_ivp ({cfg.network} network, "
                   f"{len(species_L)} nuclides) in {time.time()-_t_lt0:.2f} s "
                   f"(status={sol_LT.status}, nfev={sol_LT.nfev})"), flush=True)
        # Build LT final abundances by name; fill in 0 for any standard light
        # species that the chosen network does not track (e.g. heavy-nuclide-only
        # networks that drop He6 — though in practice all three standard networks
        # include all SPECIES_MD members).
        finL = {s: sol_LT.y[i][-1] for i, s in enumerate(species_L)}
        for s in SPECIES_MD:
            finL.setdefault(s, 0.0)

        if cfg.verbose:
            print("-" * 50)
            print("Predicted primordial abundances at the end of BBN")
            print("-" * 50)
            for label, key in [("Yp", "p"), ("Yd", "H2"), ("Yt", "H3"),
                                ("YHe3", "He3"), ("Ya", "He4"),
                                ("YLi7", "Li7"), ("YBe7", "Be7")]:
                print(f"{label:<6}= {finL[key]}")

        # ------------------------------------------------------------------
        # Store final Y values for direct access (used by get_quantity)
        # ------------------------------------------------------------------
        # Use the LT species list as the canonical name list for any network.
        self._abundance_names = species_L
        self._Y_final = dict(finL)
        if not cfg.is_small:
            # Extend the N/Z/A maps to every network nuclide so callers
            # (e.g. the AbundanceEvolution notebook) can plot all species.
            for s, (N, Z) in nucl.large_NZ.items():
                self.N[s], self.Z[s], self.A[s] = N, Z, N + Z

        # ------------------------------------------------------------------
        # Build abundance interpolator (always, so __getitem__ works)
        # ------------------------------------------------------------------
        # Each era integrates a different (growing) set of species; embed every
        # era's solution into the common abundance-vector columns *by name*, so
        # eras with fewer species (HT: n,p; MT: 12) line up with the LT layout.
        names = self._abundance_names
        col = {s: i for i, s in enumerate(names)}
        HT_names = ["n", "p"]
        MT_names = nucl._mt_net.species

        def _embed(sol_y, era_names):
            out = np.zeros((sol_y.shape[1], len(names)))
            for j, nm in enumerate(era_names):
                out[:, col[nm]] = sol_y[j]
            return out

        _t_nuc = np.concatenate((sol_HT.t, sol_MT.t[1:], sol_LT.t[1:]))
        _Y_nuc = np.vstack((_embed(sol_HT.y, HT_names),
                            _embed(sol_MT.y, MT_names)[1:, :],
                            _embed(sol_LT.y, names)[1:, :]))
        self._Y_of_t = interp1d(_t_nuc, _Y_nuc, axis=0, bounds_error=False,
                                fill_value=(0, _Y_nuc[-1]))

        # ------------------------------------------------------------------
        # Optional output: full time evolution of background + abundances
        # ------------------------------------------------------------------
        if cfg.output_time_evolution:
            if cfg.is_large:
                # The .tsv writer recomputes per-reaction fluxes for the fixed
                # 8/12-species networks; it does not cover the large network.
                # The full time evolution is still available via __getitem__
                # (the abundance interpolator built above) for all ~59 nuclides.
                # Always announce so the user knows no file was written and why.
                print("[output] output_time_evolution not written for the large "
                      "network (unsupported); use the abundance interpolator "
                      "run[species](t) for all ~59 nuclides instead.")
            else:
                self._write_time_evolution(
                    sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                    nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                    nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                    nucl,
                )

        # ------------------------------------------------------------------
        # Optional output: two-column (nuclide, final abundance Y) table
        # ------------------------------------------------------------------
        if cfg.output_final_result:
            self._write_final_result()

        # ------------------------------------------------------------------
        # Final observables
        # ------------------------------------------------------------------
        Tg_last  = self._Tg_vec[-1]
        Tnu_last = self._Tnu_vec[-1]

        Neff = self._N_eff(Tg_last, Tnu_last, Tnu_last, Tnu_last)

        # Access final abundances by name from the dict built in the LT era.
        Yp_f  = finL["p"];    Yd_f  = finL["H2"]; Yt_f  = finL["H3"]
        YHe3_f = finL["He3"]; Ya_f  = finL["He4"]
        YLi7_f = finL["Li7"]; YBe7_f = finL["Be7"]

        YPBBN  = 4. * Ya_f
        YPCMB  = ((cfg.He4Overma / 4.) * YPBBN
                  / ((cfg.He4Overma / 4.) * YPBBN + cfg.HOverma * (1. - YPBBN)))

        self._results = {
            "Neff":            Neff,
            "Omeganurel":      self._Omeganuh2_relnu() * 1e+6,
            "OneOverOmeganunr": 1. / (self._Omeganuh2_nrnu() * 1e-6),
            "YPCMB":           YPCMB,
            "YPBBN":           YPBBN,
            "DoH":             Yd_f / Yp_f,
            "He3oH":           (Yt_f + YHe3_f) / Yp_f,
            "He3oHe4":         (Yt_f + YHe3_f) / Ya_f,
            "Li7oH":           (YLi7_f + YBe7_f) / Yp_f,
        }
        return self._results

    def _write_final_result(self):
        """Write a two-column ``nuclide  Y`` table of final abundances.

        Dumps every tracked nuclide of the active network and its final
        mass-fraction abundance ``Y`` at the end of BBN to
        ``cfg.output_final_file``.  ``Y`` is normalised so that
        ``sum_s A_s Y_s = 1`` (A = mass number), i.e. it is the per-baryon
        abundance weighted by A.  The rows are exactly the species of the
        chosen network: 8 for ``small``, 12 for ``medium``, ~59 for ``large``,
        in abundance-vector order (``n`` and ``p`` first).

        Enabled by ``output_final_result=True``; the destination is
        ``output_final_file`` (relative paths resolve against the current
        working directory, like ``output_file``).  Typical use -- get the
        full nuclide vector of a
        single run without going through ``get_quantity`` for each name::

            PyPR(params={'output_final_result': True,
                              'output_final_file': 'results/run_final.dat',
                              'network': 'large'}).solve()

        produces a file whose first lines read::

            # nuclide       Y
            n             4.032109e-16
            p             7.530243e-01
            H2            1.835287e-05
            ...
        """
        cfg  = self.cfg
        # Resolve relative paths against the current working directory (the
        # universal convention), same rule as output_file.
        path = os.path.abspath(cfg.output_final_file)
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        names = self._abundance_names
        with open(path, 'w') as f:
            f.write(f"# {'nuclide':<12}Y\n")
            for nm in names:
                f.write(f"{nm:<14}{self._Y_final[nm]:.6e}\n")
        # Always announce: this file is written only on explicit request
        # (output_final_result=True), so the user wants to know where it landed.
        print(f"[output] Final abundances ({len(names)} nuclides) written to {path}")

    def _write_time_evolution(self, sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                              nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                              nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                              nucl):
        cfg = self.cfg
        # Derive column names from the actual abundance names so custom networks
        # (which may have fewer or different nuclides than the standard 8 or 12)
        # produce a TSV with the correct number of columns.
        nuc_cols = ["Y" + s for s in self._abundance_names]
        n_nuc = len(nuc_cols)

        Y_of_t = self._Y_of_t

        # Uniform log-spaced output grid from T_start_cosmo to end of LT era
        t_cosmo = self._t_of_T(cfg.T_start_cosmo / cfg.MeV_to_Kelvin)
        t_end   = sol_LT.t[-1]
        t_out   = np.logspace(np.log10(t_cosmo), np.log10(t_end), cfg.output_n_points)

        a_out = self._a_of_t(t_out)
        T_out = self._T_of_t(t_out)

        Tnue_of_t   = interp1d(self._t_vec, self._Tnue_vec,   bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnumu_of_t  = interp1d(self._t_vec, self._Tnumu_vec,  bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnutau_of_t = interp1d(self._t_vec, self._Tnutau_vec, bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnue_out   = Tnue_of_t(t_out)
        Tnumu_out  = Tnumu_of_t(t_out)
        Tnutau_out = Tnutau_of_t(t_out)

        H_out = np.array([
            self._Hubble(T_out[i], Tnue_out[i], Tnumu_out[i], Tnutau_out[i])
            for i in range(t_out.size)
        ])

        # Weak rates: zero before nuclear network starts
        t_start = sol_HT.t[0]
        T_K_out = T_out * cfg.MeV_to_Kelvin
        weak_n_to_p_out = np.zeros_like(t_out)
        weak_p_to_n_out = np.zeros_like(t_out)

        mask_ht = (t_out >= t_start) & (t_out <= t_weak)
        mask_mt = (t_out >  t_weak)  & (t_out <= t_nucl)
        mask_lt =  t_out >  t_nucl

        weak_n_to_p_out[mask_ht] = nTOp_frwrd_HT_norm(T_K_out[mask_ht])
        weak_p_to_n_out[mask_ht] = nTOp_bkwrd_HT_norm(T_K_out[mask_ht])
        weak_n_to_p_out[mask_mt] = nTOp_f_MT(T_K_out[mask_mt])
        weak_p_to_n_out[mask_mt] = nTOp_b_MT(T_K_out[mask_mt])
        weak_n_to_p_out[mask_lt] = nTOp_f_LT(T_K_out[mask_lt])
        weak_p_to_n_out[mask_lt] = nTOp_b_LT(T_K_out[mask_lt])

        # Abundances: zero before nuclear network starts
        Y_out = np.zeros((len(t_out), n_nuc))
        mask_nuc = t_out >= t_start
        Y_out[mask_nuc] = Y_of_t(t_out[mask_nuc])

        if cfg.output_rates_time_evolution:
            rxn_rate_cols = sorted(
                name for name in dir(nucl)
                if name.endswith("_frwrd") and callable(getattr(nucl, name))
            )
            rxn_rate_out = np.zeros((len(t_out), len(rxn_rate_cols)))
            rxn_rate_out[mask_nuc] = np.column_stack([
                getattr(nucl, name)(T_K_out[mask_nuc]) for name in rxn_rate_cols
            ])
        else:
            rxn_rate_cols = []
            rxn_rate_out = np.empty((len(t_out), 0))

        Nheating_out = self._N_NEVO_of_Tg(T_out)

        # Resolve relative paths against the current working directory (the
        # universal convention), not the installed-package directory.
        out_path = os.path.abspath(cfg.output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_data = np.column_stack((a_out, T_out, t_out, H_out,
                                    Tnue_out, Tnumu_out, Tnutau_out, Nheating_out,
                                    Y_out,
                                    weak_n_to_p_out, weak_p_to_n_out, rxn_rate_out))
        out_header = "\t".join(["a", "T", "t", "H",
                                 "Tnue", "Tnumu", "Tnutau", "Nheating"]
                               + nuc_cols
                               + ["n_to_p_weak_rate", "p_to_n_weak_rate"] + rxn_rate_cols)
        np.savetxt(out_path, out_data, delimiter='\t', header=out_header, comments='')

        # Always announce: written only on explicit request (output_time_evolution=True).
        print(f"[output] Time-evolution data ({len(t_out)} rows) written to {out_path}")

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def T_of_t(self):
        """T_γ(t) interpolator [MeV], available after initialisation."""
        return self._T_of_t

    @property
    def t_of_T(self):
        """t(T_γ) interpolator [s], available after initialisation."""
        return self._t_of_T

    def __getitem__(self, species):
        """Return Y(t) for a species name (e.g. 'H2', 'He4', 'Li7').

        Calls solve() automatically if needed.
        """
        self._ensure_solved()
        if species not in self._abundance_names:
            raise KeyError(
                f"Unknown species '{species}'. Available: {self._abundance_names}"
            )
        idx = self._abundance_names.index(species)
        def fn(t):
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            vals  = self._Y_of_t(t_arr)[:, idx]
            return float(vals[0]) if np.ndim(t) == 0 else vals
        return fn

    def _ensure_solved(self):
        if self._results is None:
            self.solve()

    def PyPRresults(self):
        """Return the BBN result dict, running ``solve()`` first if needed."""
        self._ensure_solved()
        return self._results

    @property
    def abundance_names(self):
        """Tracked nuclide names, in abundance-vector order (solves if needed).

        For the large network this is the full ~59-nuclide list; accessing it
        also guarantees ``self.A``/``N``/``Z`` cover every species (handy for
        plotting ``A_i Y_i`` for all nuclides)."""
        self._ensure_solved()
        return self._abundance_names

    # Convenience accessors
    def Neff(self):          self._ensure_solved(); return self._results["Neff"]
    def Omeganurel(self):    self._ensure_solved(); return self._results["Omeganurel"]
    def Omeganunonrel(self): self._ensure_solved(); return 1. / self._results["OneOverOmeganunr"]
    def YPCMB(self):         self._ensure_solved(); return self._results["YPCMB"]
    def YPBBN(self):         self._ensure_solved(); return self._results["YPBBN"]
    def DoH(self):           self._ensure_solved(); return self._results["DoH"]
    def He3oH(self):         self._ensure_solved(); return self._results["He3oH"]
    def Li7oH(self):         self._ensure_solved(); return self._results["Li7oH"]

    def get_quantity(self, quantity):
        """Return a scalar BBN quantity by name.

        Accepts any key from the result dict ('YPBBN', 'DoH', 'He3oH',
        'Li7oH', 'Neff', 'YPCMB', ...) or a nuclide name from
        cfg.Nuclides ('H2', 'He4', 'Li7', ...) for the final mass fraction Y.
        """
        self._ensure_solved()
        if quantity in self._results:
            return self._results[quantity]
        if quantity in self._Y_final:
            return self._Y_final[quantity]
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            f"Valid result keys: {list(self._results.keys())}. "
            f"Valid nuclide names: {list(self._Y_final.keys())}."
        )


# ---------------------------------------------------------------------------
# MC result classes
# ---------------------------------------------------------------------------

class MCQuantityResult:
    """MC statistics for a single BBN quantity.

    Attributes
    ----------
    central : float
        Value at nominal rates (all p_* = 0).
    mean : float
        Mean of MC samples.
    std : float
        1σ standard deviation of MC samples.
    values : np.ndarray, shape (num_mc,)
        All individual MC sample values.
    """
    __slots__ = ('central', 'mean', 'std', 'values')

    def __init__(self, central, samples):
        self.central = float(central)
        self.values  = np.asarray(samples)
        self.mean    = float(np.mean(self.values))
        self.std     = float(np.std(self.values))

    def __repr__(self):
        return (f"MCQuantityResult(central={self.central:.6g}, "
                f"mean={self.mean:.6g}, std={self.std:.6g}, "
                f"n={len(self.values)})")


class MCResult:
    """MC results for one or more BBN quantities, indexed by name.

    Usage::

        mc = mc_uncertainty(100, ['YPBBN', 'DoH'], params=...)
        mc['YPBBN'].mean
        mc['YPBBN'].std
        mc['YPBBN'].values
        mc['DoH'].central
    """
    def __init__(self, data):
        self._data = data   # dict: str -> MCQuantityResult

    def __getitem__(self, quantity):
        return self._data[quantity]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        lines = [f"MCResult({len(self._data)} quantities):"]
        for k, v in self._data.items():
            lines.append(f"  {k}: central={v.central:.6g}, "
                         f"mean={v.mean:.6g}, std={v.std:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level MC worker (must be at module level for joblib pickling)
# ---------------------------------------------------------------------------

def _mc_run_batch(base_params, rate_keys, quantities, seeds):
    """Run a batch of MC samples in one process, reusing a single PyPR.

    The cosmological background and n<->p weak rates depend only on
    ``base_params`` — *not* on the nuclear-rate offsets ``p_*`` — so they are
    computed once when the instance is built and then reused for every sample
    in the batch.  Each sample only re-draws the nuclear rates and re-solves the
    nuclear network (the cheap part), which is the whole point of the speed-up.

    Drawing the rate vector from ``default_rng(seed)`` per seed makes the result
    for a given seed independent of how the seeds are batched, so the output is
    identical (and reproducible) regardless of ``n_jobs``.
    """
    inst = PyPR(params=base_params)
    cfg  = inst.cfg
    results = []
    for seed in seeds:
        rng    = np.random.default_rng(seed)
        p_vals = rng.standard_normal(len(rate_keys))
        for k, v in zip(rate_keys, p_vals):
            setattr(cfg, k, float(v))
        inst.solve()
        results.append([inst.get_quantity(q) for q in quantities])
    return results


def mc_uncertainty(num_mc, quantity, params=None, n_jobs=-1, seed=0):
    """Estimate nuclear rate uncertainties on BBN observables via Monte Carlo.

    Each MC sample draws all active nuclear rate offsets p_* independently from
    N(0,1) and runs a full PyPRIMAT solve.  By default all reactions in the 
    selected network are varied.

    Parameters
    ----------
    num_mc : int
        Number of MC samples.
    quantity : str or list of str
        A key from the result dict ('YPBBN', 'DoH', 'He3oH',
        'Li7oH', 'Neff', 'YPCMB', ...) or a nuclide name ('H2', 'He4',
        'Li7', ...) for the final mass fraction Y.  Pass a list to evaluate
        multiple quantities in one MC pass (more efficient than separate calls).
    params : dict, optional
        Base parameters for PyPR (e.g. Omegabh2, is_small, network).
    n_jobs : int
        Number of parallel workers passed to joblib.Parallel (-1 = all CPUs).
    seed : int
        Base random seed; sample i uses seed + i for reproducibility.
        When evaluating on a parameter grid (e.g. scanning Ω_b h²), use the
        **same seed at every grid point** so that sample i draws the same rate
        vector p_* everywhere.  This correlates the MC noise across the grid,
        making any finite-sample bias cancel when comparing predictions at
        different parameter values.

    Returns
    -------
    MCResult
        Dict-like object indexed by quantity name.  Each value is an
        ``MCQuantityResult`` with attributes ``central``, ``mean``, ``std``,
        and ``values``.

    Example
    -------
    >>> mc = mc_uncertainty(100, ['YPBBN', 'DoH'], params={'Omegabh2': 0.022})
    >>> mc['YPBBN'].central
    >>> mc['YPBBN'].std
    >>> mc['DoH'].values   # full sample array
    """
    from joblib import Parallel, delayed, effective_n_jobs
    from .nuclear import load_reaction_names

    quantities = [quantity] if isinstance(quantity, str) else list(quantity)

    base_params = dict(params or {})
    base_params.setdefault('verbose', False)
    base_params.setdefault('debug',   False)

    # Rate offsets to vary: all thermonuclear reactions in the selected network.
    # We construct a temporary config just to resolve the working directory
    # and selected network filename correctly.
    tmp_cfg = PyPRConfig(base_params)
    reactions = load_reaction_names(tmp_cfg)
    rate_keys = [f'p_{rxn}' for rxn in reactions]

    # Central value (all p_* = 0).
    central_inst = PyPR(params=base_params)
    central_inst.solve()
    centrals = [central_inst.get_quantity(q) for q in quantities]

    # Split the seeds into one chunk per worker so the expensive background +
    # weak-rate setup is paid once per worker instead of once per sample.
    seeds    = [seed + i for i in range(num_mc)]
    n_chunks = max(1, min(len(seeds), effective_n_jobs(n_jobs)))
    chunks   = [list(c) for c in np.array_split(seeds, n_chunks)]

    raw = Parallel(n_jobs=n_jobs)(
        delayed(_mc_run_batch)(base_params, rate_keys, quantities, chunk)
        for chunk in chunks
    )
    samples = np.array([row for chunk in raw for row in chunk])   # (num_mc, n_q)

    return MCResult({
        q: MCQuantityResult(centrals[j], samples[:, j])
        for j, q in enumerate(quantities)
    })
