# -*- coding: utf-8 -*-
"""
nuclear_network.py
==================
``NuclearNetwork`` (Class 2 of the PRIMAT split, see ``primat.background`` for
Class 1): the nuclear-reaction-network ODE integration across the HT/MT/LT
temperature eras.

Design
------
``NuclearNetwork`` is driven purely through the *minimal* public interface of
a ``primat.background.Background`` instance:

* ``T_of_t(t)`` / ``t_of_T(T)``  -- time <-> temperature
* ``rhoB_BBN(t)``                -- baryon mass density [g/cm^3] as a
  function of cosmic time (the prefactor for nuclear reaction rates)
* ``weak_nTOp_frwrd(T_K)`` / ``weak_nTOp_bkwrd(T_K)`` -- already-normalised
  n<->p weak rates [s^-1] at photon temperature ``T_K`` [Kelvin]

It knows nothing about *how* the background was constructed (NEVO table,
instantaneous decoupling, external background, scale factor, neutrino
sector, ...) -- this is exactly the seam that makes the background pluggable
(``primat.background.Background``).  In particular it does **not** use
``a_of_t``, ``Hubble``, the individual neutrino temperatures, or the NEVO
heating function: those are output-only quantities written directly by
``Background.write_time_evolution`` (see ``primat.background``), not
consumed by the nuclear solve.

``solve()`` integrates:

* **HT** (high temperature, T > T_weak ~ 1 MeV): n <-> p only.
* **MT** (mid temperature, T_weak -> T_nucl ~ 0.1 MeV): the fixed 18-reaction
  subset (n<->p + 17 reactions), regardless of network size.
* **LT** (low temperature, T_nucl -> T_end ~ 0.001 MeV): the chosen network
  (small/large, optionally amax-restricted).

and populates the public ``Y_final``, ``abundance_names`` and ``Y_of_t``
attributes consumed by ``PRIMAT``'s observable accessors (``get_quantity``,
``__getitem__``, ...) and by ``PRIMAT.solve()`` (which builds the BBN
observables dict -- ``Neff``, ``YPBBN``, ``YPCMB``, ``DoH``, ``He3oH``,
``He3oHe4``, ``Li7oH``, ``Omeganurel``, ``OneOverOmeganunr`` -- from
``Y_final`` and from ``background``'s optional neutrino-sector hooks).
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

from .evolution import EvolutionResult, dump_evolution

__all__ = ["NuclearNetwork"]


class NuclearNetwork:
    """The nuclear reaction network (Class 2): HT/MT/LT ODE integration.

    Parameters
    ----------
    cfg : primat.config.PRIMATConfig
        Run-time configuration (network choice, temperature-era boundaries,
        numerical tolerances, rate-variation parameters p_*, ...).
    nucl : primat.network_data.UpdateNuclearRates
        Compiled MT/LT reaction-rate kernels (RHS + Jacobian) for the chosen
        network.
    background : primat.background.Background
        The cosmological background (Class 1) supplying ``T_of_t``/``t_of_T``,
        ``rhoB_BBN(t)``, and the normalised n<->p weak rates
        ``weak_nTOp_frwrd``/``weak_nTOp_bkwrd`` (see the module docstring for
        the full minimal interface).

    Attributes (populated by :meth:`solve`)
    ----------------------------------------
    Y_final : dict or None
        Final mass-fraction abundance ``Y`` of every nuclide in
        ``abundance_names``.
    abundance_names : list of str or None
        Tracked nuclide names, in abundance-vector order (LT species list).
    Y_of_t : scipy.interpolate.interp1d or None
        Abundance-vector interpolator ``Y(t)`` -> shape ``(len(abundance_names),)``,
        spanning HT+MT+LT.
    """

    def __init__(self, cfg, nucl, background):
        self.cfg = cfg
        self.nucl = nucl
        self.background = background
        self.Y_final = None
        self.abundance_names = None
        self.Y_of_t = None
        self._t_end = None   # cosmic time [s] at end of LT era; set by solve()
        self.evolution = None   # EvolutionResult; set by solve() iff output_time_evolution=True

    # ======================================================================
    # solve(): integrate nuclear network ODEs
    # ======================================================================

    def solve(self):
        """
        Integrate the nuclear network over the three temperature eras.

        Populates ``self.Y_final``, ``self.abundance_names`` and
        ``self.Y_of_t`` and returns ``self.Y_final`` (the dict of final
        mass-fraction abundances by nuclide name).  The BBN observables dict
        (``Neff``, ``YPBBN``, ``DoH``, ...) is built by ``PRIMAT.solve()`` from
        ``self.Y_final`` and from ``background``'s optional neutrino-sector
        hooks -- it is no longer computed here.
        """
        from .network_data import SPECIES_MD   # noqa: F401 (used for default-zero filling)
        cfg       = self.cfg
        background = self.background
        T_of_t    = background.T_of_t
        t_of_T    = background.t_of_T
        rhoB_BBN  = background.rhoB_BBN
        nTOp_frwrd = background.weak_nTOp_frwrd
        nTOp_bkwrd = background.weak_nTOp_bkwrd
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
        self._t_end = t_end   # store for DT-era helpers and tests

        # ------------------------------------------------------------------
        # Baryon-to-photon ratio at T_weak, for the MT-era Saha (NSE) seed
        # ------------------------------------------------------------------
        # eta_b = n_B/n_gamma, evaluated once at T = T_weak from the two
        # compulsory Background primitives rhoB_BBN(t) and t_of_T(T): YA is
        # only ever called at T = cfg.T_weak (the MT seed below), so this
        # single value is exact -- no etab_of_T(T) interpolant is needed.
        T_weak_MeV  = cfg.T_weak / cfg.MeV_to_Kelvin
        nB_weak     = rhoB_BBN(t_weak) / (cfg.ma * cfg.MeV4_to_gcmm3)   # [MeV^3]
        ngamma_weak = (2. * zeta(3) / np.pi**2) * T_weak_MeV**3        # [MeV^3]
        eta_b_weak  = nB_weak / ngamma_weak

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
            ratio.  Used to seed the MT-era initial conditions at T = T_weak,
            where η_b = eta_b_weak (closed over from the enclosing scope, see
            above).

            Args:
                name : nuclide name string (key into cfg.Nuclides/NuclExcessMass).
                Yn   : free neutron mass fraction.
                Yp   : free proton mass fraction.
                T    : photon temperature in Kelvin (= cfg.T_weak at every
                       call site).

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
                    * eta_b_weak**(A - 1)
                    * Yp**Z * Yn**(A - Z)
                    * np.exp(BindE * cfg.keV / (cfg.kB * T)))

        # ------------------------------------------------------------------
        # High-temperature (HT) era: only n and p
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl-py]  Solving neutron decoupling at high temperature era")

        def Yn_i_func(T):
            b = nTOp_bkwrd(T)
            return b / (b + nTOp_frwrd(T))

        def Y_prime_HT(t, Y):
            T_K = T_of_t(t) * cfg.MeV_to_Kelvin
            f   = nTOp_frwrd(T_K)
            b   = nTOp_bkwrd(T_K)
            return b * Y[1] - f * Y[0], f * Y[0] - b * Y[1]

        Yn_i = Yn_i_func(cfg.T_start)
        Yp_i = 1. - Yn_i
        _t_ht0 = time.time()
        sol_HT = solve_ivp(Y_prime_HT, [t_start, t_weak], [Yn_i, Yp_i],
                           method='LSODA', rtol=cfg.numerical_precision, atol=1e-10)
        if cfg.verbose:
            print((f"[nucl-py]  [HT] Finished solve_ivp in {time.time()-_t_ht0:.2f} s "
                   f"(status={sol_HT.status}, nfev={sol_HT.nfev})"), flush=True)
        Yn_HT_f, Yp_HT_f = sol_HT.y[0][-1], sol_HT.y[1][-1]

        # ------------------------------------------------------------------
        # ODE systems for the full network
        # ------------------------------------------------------------------
        def Y_prime_MT(t, Y):
            rho = rhoB_BBN(t); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.rhsMT(Y, T_K, rho, nTOp_frwrd, nTOp_bkwrd)

        def Jacobian_MT(t, Y):
            rho = rhoB_BBN(t); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.JacobianMT(Y, T_K, rho, nTOp_frwrd, nTOp_bkwrd)

        def Y_prime_LT(t, Y):
            rho = rhoB_BBN(t); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.rhsLT(Y, T_K, rho, nTOp_frwrd, nTOp_bkwrd)

        def Jacobian_LT(t, Y):
            rho = rhoB_BBN(t); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
            return nucl.JacobianLT(Y, T_K, rho, nTOp_frwrd, nTOp_bkwrd)

        # ------------------------------------------------------------------
        # Mid-temperature (MT) era
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl-py]  Solving nuclear network at mid temperature era")

        # Saha (NSE) seed for all MT species except n and p, which come from
        # the HT solution.  The MT network's species list is determined by the
        # NetworkDefinition, so this loop is independent of the network size.
        mt_species = nucl._mt_net.species   # e.g. 8 for small, 12 for large/amax=8
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
            print((f"[nucl-py]  [MT] Finished solve_ivp ({cfg.network} network, "
                   f"{len(mt_species)} species) in {time.time()-_t_mt0:.2f} s "
                   f"(status={sol_MT.status}, nfev={sol_MT.nfev})"), flush=True)
        # Extract MT final values by name — works for any network size.
        mt_final_raw = {s: sol_MT.y[i][-1] for i, s in enumerate(mt_species)}

        # ------------------------------------------------------------------
        # Low-temperature (LT) era
        # ------------------------------------------------------------------
        if cfg.verbose:
            print("[nucl-py]  Solving nuclear network at low temperature era")

        # Seed the LT vector from MT final values, filling any extra species
        # (present in the LT but absent in MT) with 0.  By looking up by name,
        # this works for any MT and LT network sizes without hardcoding.
        species_L = nucl.species_large
        Yi_LT = [mt_final_raw.get(s, 0.0) for s in species_L]

        _t_lt0 = time.time()
        atol = cfg.atol_large_LT if cfg.is_large else 1e-20
        sol_LT = solve_ivp(Y_prime_LT, [t_nucl, t_end], Yi_LT,
                           method='BDF', jac=Jacobian_LT,
                           rtol=10.*cfg.numerical_precision, atol=atol)
        if cfg.verbose:
            print((f"[nucl-py]  [LT] Finished solve_ivp ({cfg.network} network, "
                   f"{len(species_L)} nuclides) in {time.time()-_t_lt0:.2f} s "
                   f"(status={sol_LT.status}, nfev={sol_LT.nfev})"), flush=True)
        # Build LT final abundances by name; fill in 0 for any standard light
        # species that the chosen network does not track (e.g. heavy-nuclide-only
        # networks that drop He6 — though in practice all three standard networks
        # include all SPECIES_MD members).
        # Clamp to 0: the BDF solver can leave near-extinct nuclides at a
        # tiny negative value (numerical noise around zero), which is
        # unphysical for an abundance and breaks log-scale displays/ratios.
        finL = {s: max(sol_LT.y[i][-1], 0.0) for i, s in enumerate(species_L)}
        for s in SPECIES_MD:
            finL.setdefault(s, 0.0)

        if cfg.verbose:
            # Full list of every nuclide that was integrated numerically in the
            # LT era (species_L is exactly the LT solver's state vector).  The
            # list grows with the chosen network (8 / 12 / ~59 nuclides for
            # small / large, optionally amax-restricted).
            print("-" * 50)
            print(f"Predicted primordial abundances at the end of BBN "
                  f"({len(species_L)} numerically solved nuclides)")
            print("-" * 50)
            for s in species_L:
                print(f"  Y{s:<5}= {finL[s]:.6e}")

        # ------------------------------------------------------------------
        # Store final Y values for direct access (used by get_quantity)
        # ------------------------------------------------------------------
        # Use the LT species list as the canonical name list for any network.
        self.abundance_names = species_L
        self.Y_final = dict(finL)

        # ------------------------------------------------------------------
        # Build abundance interpolator (always, so __getitem__ works)
        # ------------------------------------------------------------------
        # Each era integrates a different (growing) set of species; embed every
        # era's solution into the common abundance-vector columns *by name*, so
        # eras with fewer species (HT: n,p; MT: 12) line up with the LT layout.
        names = self.abundance_names
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
        self.Y_of_t = interp1d(_t_nuc, _Y_nuc, axis=0, bounds_error=False,
                                fill_value=(0, _Y_nuc[-1]))

        # ------------------------------------------------------------------
        # Optional output: full time evolution of abundances + weak rates
        # ------------------------------------------------------------------
        if cfg.output_time_evolution:
            self._write_time_evolution(sol_HT, sol_LT, nucl)

        # ------------------------------------------------------------------
        # Optional output: two-column (nuclide, final abundance Y) table
        # ------------------------------------------------------------------
        if cfg.output_final_result:
            self._write_final_result()

        # ------------------------------------------------------------------
        # Decay Time (DT) era (optional, large network only)
        # ------------------------------------------------------------------
        # After BBN ends at t_end, long-lived radioactive isotopes (C14, Be10,
        # Na22, …) continue to decay on timescales of years to millions of
        # years.  The DT era propagates the abundance vector forward in time
        # using only the constant decay matrix (no Hubble expansion, no
        # thermal production), via matrix exponentiation:
        #   Y(t) = exp(D × Δt) × Y(t_end)
        # where D is the (constant) decay-rate matrix assembled from the decay
        # reactions in the LT network (see _build_decay_matrix).
        if cfg.decay_era and cfg.is_large:
            Y0_DT = np.array([self.Y_final.get(s, 0.0) for s in self.abundance_names])
            D = self._build_decay_matrix(nucl._lt_net)
            t_decay_end = cfg.t_decay_end
            decay_n     = cfg.decay_n_points
            # Time grid log-spaced in the *elapsed* time Δt = t − t_end (not in
            # absolute t).  This is essential: the residual free neutron decays
            # with τ_n ≈ 880 s, a transient ~10 decades shorter than t_end
            # (~1.3×10⁶ s).  A grid log-spaced in absolute t would put its first
            # interior point ~10⁵ s past t_end, completely skipping the neutron
            # decay (linear interpolation between t_end and that point would
            # flatten it).  Spacing in Δt from Δt_min = 1 s gives dense sampling
            # immediately after t_end (resolving n, and any other fast residual)
            # while still reaching t_decay_end with coarse late-time sampling
            # for the slow decays (Na22, C14, Be10).
            t_DT = t_end + np.logspace(np.log10(1.0),
                                       np.log10(t_decay_end), decay_n)
            Y_DT = self._integrate_decay_era(D, Y0_DT, t_end, t_DT)
            if cfg.verbose:
                print(f"[nucl-py]  [DT] Decay era: {decay_n} time points from "
                      f"t={t_end:.3g} s to t={t_end + t_decay_end:.3g} s")
                for i, s in enumerate(self.abundance_names[:12]):
                    if Y_DT[-1, i] > 0:
                        print(f"  Y{s:<5}= {Y_DT[-1, i]:.6e}")

            # Extend the public Y(t) interpolator across the DT era so that
            # callers (``run[species](t)``, ``get_quantity(..., t=...)``) see a
            # single seamless history t_start … t_end+t_decay_end, exactly like
            # the HT→MT→LT concatenation above.  t_DT[0] = t_end+1 > t_end, so
            # the appended grid stays strictly increasing; Y_DT is already in
            # ``abundance_names`` column order (it is built from Y0_DT, which is
            # itself indexed by ``abundance_names``), matching _Y_nuc's layout.
            # The right-hand fill_value becomes the late-time DT value (the
            # fully-decayed state) instead of the LT endpoint.
            _t_nuc = np.concatenate((_t_nuc, t_DT))
            _Y_nuc = np.vstack((_Y_nuc, Y_DT))
            self.Y_of_t = interp1d(_t_nuc, _Y_nuc, axis=0, bounds_error=False,
                                    fill_value=(0, _Y_nuc[-1]))

            if cfg.output_decay_evolution:
                self._write_decay_evolution(t_DT, Y_DT)

        return self.Y_final

    def _lt_t_end_s(self):
        """Return the cosmic time [s] at the end of the LT era.

        Populated by :meth:`solve`.  Used by DT-era helpers and tests to anchor
        the Δt = t − t_end offset for the decay matrix exponentiation.

        Returns
        -------
        float
            Cosmic time at T_end [s]; e.g. ~1.3×10^6 s (≈ 15 days) for the
            default T_end_MeV = 0.001.

        Raises
        ------
        RuntimeError
            If called before :meth:`solve`.
        """
        if self._t_end is None:
            raise RuntimeError("_lt_t_end_s() called before solve()")
        return self._t_end

    def _write_final_result(self):
        """Write a two-column ``nuclide  Y`` table of final abundances.

        Dumps every tracked nuclide of the active network and its final
        mass-fraction abundance ``Y`` at the end of BBN to
        ``cfg.output_final_file``.  ``Y`` is normalised so that
        ``sum_s A_s Y_s = 1`` (A = mass number), i.e. it is the per-baryon
        abundance weighted by A.  The rows are exactly the species of the
        chosen network: 8 for ``small``, ~59 for ``large`` (fewer with an
        ``amax`` cutoff, e.g. 12 for ``large, amax=8``),
        in abundance-vector order (``n`` and ``p`` first).

        Enabled by ``output_final_result=True``; the destination is
        ``output_final_file`` (relative paths resolve against the current
        working directory, like ``output_file``).  Typical use -- get the
        full nuclide vector of a
        single run without going through ``get_quantity`` for each name::

            PRIMAT(params={'output_final_result': True,
                              'output_final_file': 'results/run_final.dat',
                              'network': 'large'}).solve()

        produces a file whose first lines read::

            # nuclide       Y
            n             4.032109e-16
            p             7.530243e-01
            H2            1.835287e-05
            ...
        """
        from .backend import dump_final_with_sigma

        cfg  = self.cfg
        # Resolve relative paths against the current working directory (the
        # universal convention), same rule as output_file.
        path = os.path.abspath(cfg.output_final_file)
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        names = self.abundance_names
        with open(path, 'w') as f:
            f.write(dump_final_with_sigma(names, self.Y_final))
        # Always announce: this file is written only on explicit request
        # (output_final_result=True), so the user wants to know where it landed.
        print(f"[output] Final abundances ({len(names)} nuclides) written to {path}")

    def _write_time_evolution(self, sol_HT, sol_LT, nucl):
        """Build the unified ``EvolutionResult`` (see :mod:`primat.evolution`,
        ``PRIMAT.md`` S7.2/S7.3) and write it to ``cfg.output_file``.

        Enabled by ``output_time_evolution=True``.  Always sets
        ``self.evolution`` to the in-memory result (no disk I/O required to
        get it -- e.g. ``PRIMAT(...).solve()["evolution"]``); additionally
        writes ``cfg.output_file`` as a convenience via
        :func:`primat.evolution.dump_evolution`.  Works for all three
        networks (``small``/``large``, optionally ``amax``-restricted) --
        the ``Y_<species>`` columns are derived from
        ``self.abundance_names`` (8 / 12 / ~59 nuclides).

        Columns (see :mod:`primat.evolution` for the exact schema): cosmic
        time ``t_s``, scale factor ``a``, photon temperature
        ``T_gamma_MeV``, the three flavour neutrino temperatures, and one
        ``Y_<species>`` column per tracked nuclide (mass-fraction
        abundance). ``a``/the neutrino temperatures come from
        ``self.background`` (``np.nan`` if it tracks no scale factor/
        neutrino sector, e.g. a minimal custom background).

        Before the nuclear network starts integrating a given species (the
        HT era for everything but n/p, and the time before ``T_start_cosmo``
        for n/p too), its ``Y_<species>`` column is **exactly 0** -- the
        value ``_embed``/``Y_of_t`` produce there.  A previous version of
        this method filled that region with the Nuclear Statistical
        Equilibrium (Saha) prediction ``YA(name, Yn, Yp, T)`` for a smoother
        log-log plot; this was removed because the fill is *often wrong*:
        NSE need not hold for every nuclide all the way down to
        ``T_start_cosmo`` (e.g. for non-standard backgrounds with extra
        entropy injection or a non-thermal neutrino sector), and a
        silently-injected equilibrium value is worse than an honest 0 that a
        plotting tool can choose to mask. Consumers that want a smooth
        pre-MT curve can compute the Saha value themselves from the ``t_s``/
        ``T_gamma_MeV`` columns.

        Per-reaction flux columns (``<reaction>_frwrd``,
        ``cfg.output_rates_time_evolution=True``) and the n<->p weak rates
        are deferred from this unified schema (``PRIMAT.md`` S7.2: the
        former is explicitly a v0.3.0-deferred "bonus column block" pending a
        C-side port; the latter is recoverable directly from
        ``run.background.weak_nTOp_frwrd``/``weak_nTOp_bkwrd``, evaluated at
        the ``T_gamma_MeV`` column, with no need to duplicate it on disk).
        The richer background-only TSV (``H``, ``Nheating``, energy
        densities, ...) is still written separately by
        ``background.write_time_evolution``/``time_evolution_text`` when
        ``cfg.output_background_evolution=True`` (see
        :mod:`primat.background`).
        """
        cfg = self.cfg
        background = self.background
        # Derive column names from the actual abundance names so custom networks
        # (which may have fewer or different nuclides than the standard 8 or 12)
        # produce a result with the correct number of columns.
        names = self.abundance_names

        Y_of_t = self.Y_of_t

        # Uniform log-spaced output grid from T_start_cosmo to end of LT era
        t_cosmo = background.t_of_T(cfg.T_start_cosmo / cfg.MeV_to_Kelvin)
        t_end   = sol_LT.t[-1]
        t_out   = np.logspace(np.log10(t_cosmo), np.log10(t_end), cfg.output_n_points)

        T_out = background.T_of_t(t_out)
        a_out = (background.a_of_t(t_out) if background.has_scale_factor
                 else np.full_like(t_out, np.nan))
        Tnu = background.Tnu_of_t(t_out)
        if Tnu is None:
            nan_col = np.full_like(t_out, np.nan)
            Tnu = {"e": nan_col, "mu": nan_col, "tau": nan_col}

        # Abundances: zero before nuclear network starts (Y_of_t's fill_value)
        t_start = sol_HT.t[0]
        Y_out = np.zeros((len(t_out), len(names)))
        mask_nuc = t_out >= t_start
        Y_out[mask_nuc] = Y_of_t(t_out[mask_nuc])
        Y = {s: Y_out[:, j] for j, s in enumerate(names)}

        if cfg.output_rates_time_evolution:
            print("[output] output_rates_time_evolution ignored: per-reaction "
                  "flux columns are deferred from the unified time-evolution "
                  "schema (PRIMAT.md S7.2). Use nucl.<reaction>_frwrd(T_K) "
                  "directly if reaction-level fluxes are needed.")

        self.evolution = EvolutionResult(t=t_out, a=a_out, T_gamma=T_out, T_nu=Tnu, Y=Y)

        # cfg.output_file=None is the in-memory-only escape hatch (e.g.
        # primat-gui's _solve, PRIMAT.md S7.5): self.evolution above is what
        # that caller actually wants, with no disk I/O at all -- this is the
        # only output_*=True flag in the package with that escape hatch,
        # since it is also the only one a hosted GUI needs to suppress.
        if cfg.output_file is None:
            return

        # Resolve relative paths against the current working directory (the
        # universal convention), not the installed-package directory.
        out_path = os.path.abspath(cfg.output_file)
        dump_evolution(self.evolution, out_path)

        # Always announce: written only on explicit request (output_time_evolution=True).
        print(f"[output] Time-evolution data ({len(t_out)} rows) written to {out_path}")

    # ======================================================================
    # Decay Time (DT) era helpers
    # ======================================================================

    def _build_decay_matrix(self, net):
        r"""Build the constant decay-rate matrix D for the DT era.

        In the DT era all thermonuclear reactions are frozen (T is too low for
        any thermal activation), so only radioactive decays remain.  The
        abundance vector Y (mass fractions Y_s = A_s n_s / n_B) evolves as:

            dY/dt = D · Y

        where D is a constant N×N matrix (N = number of nuclides in the LT
        network).  Each decay reaction ``X → Y + (Z) + B±`` contributes:

            D[X_idx, X_idx] -= rate_X         (loss term for the parent X)
            D[s_idx, X_idx] += rate_X × A_s / A_X  for each stable product s
                                                    (gain term, mass-fraction
                                                    weighted by A_s / A_X)

        The factor A_s / A_X converts the number-fraction decay flux into a
        mass-fraction flux: if X decays to Y with rate λ, then
        dY_Y/dt = λ × (A_Y / A_X) × Y_X (mass-fraction balance).

        Photons and leptons (Bm/Bp) are excluded from the ODE state vector; only
        nuclear species (those in ``net.species``) appear in D.

        In addition to the ``decays.txt`` reactions, the free-neutron β decay
        ``n → p`` is added explicitly with rate ``1/cfg.tau_n``: it is the
        T→0 limit of the thermal n↔p weak rate (which is handled by the
        background during HT/MT/LT, not stored as a decay table), so without it
        the residual free neutrons at ``t_end`` would never decay.

        Parameters
        ----------
        net : NetworkDefinition
            The LT network (``nucl._lt_net``); supplies species, N, Z,
            stoichiometry (``net.network``), decay-reaction flags
            (``net.weak_indices``), and rate tables (``net._fwd_median``).

        Returns
        -------
        D : np.ndarray, shape (N, N)
            Decay-rate matrix in [s^-1].  Off-diagonal entries are ≥ 0;
            diagonal entries are ≤ 0 (D is a generator matrix for a Markov
            chain, i.e. column sums are approximately 0 by mass-fraction
            conservation — approximately because small fractions go to photons
            and leptons).

        Notes
        -----
        Mass-fraction conservation: Σ_s A_s D_{s,X} = 0 for each parent X.
        Checking this is a useful consistency test (verified in the
        implementation by the mass-action stoichiometry).

        Example
        -------
        For a single decay C14 → N14 + Bm with rate λ (and A_C14 = A_N14 = 14):

            D[C14, C14] = -λ      (C14 is lost)
            D[N14, C14] = +λ × 14/14 = +λ    (N14 is gained)

        The sum Σ_s A_s D_{s,C14} = 14×λ + 14×(-λ) = 0 ✓ (mass conserved).
        """
        N = len(net.species)
        D = np.zeros((N, N))

        # The rate tables (_fwd_median) are indexed without the n__p slot:
        # names[0] = "n__p", names[1:] = thermonuclear reactions.
        # _fwd_median[i] corresponds to names[i+1], so we need offset by 1.
        # Mass numbers A_s for each species:
        A_s = (net.N + net.Z).astype(float)   # shape (N,)

        for rxn_idx in net.weak_indices:
            if rxn_idx == 0:
                continue   # n__p handled by the HT/MT/LT eras, not the DT era
            name = net.names[rxn_idx]
            # The decay rate is constant (T9-independent), stored as a
            # uniform array.  Read from _fwd_median at grid index 0.
            # rate_table_idx is rxn_idx - 1 because _fwd_median excludes n__p.
            rate = float(net._fwd_median[rxn_idx - 1, 0])   # [s^-1]
            if rate == 0.0:
                continue

            react, prod = net.network[rxn_idx]   # {species_idx: multiplicity}

            # Parent nuclide: the sole nuclear reactant (multiplicity 1 for all
            # beta/EC decays; multi-nucleon decays like Li8→α+α+Bm are handled
            # via the products dict below).
            for X_idx, X_mult in react.items():
                # Loss term for the parent X
                D[X_idx, X_idx] -= rate * X_mult
                A_X = A_s[X_idx]

                # Gain terms for nuclear products (the lepton Bm/Bp and photons
                # are excluded from the ODE state and are already absent from
                # net.network's index-based stoichiometry).
                for P_idx, P_mult in prod.items():
                    A_P = A_s[P_idx]
                    # Mass-fraction gain: dY_P/dt = rate × P_mult × A_P/A_X × Y_X
                    D[P_idx, X_idx] += rate * P_mult * A_P / A_X

        # ------------------------------------------------------------------
        # Free-neutron β decay  n → p + e⁻ + ν̄
        # ------------------------------------------------------------------
        # The n→p transition is *not* a decays.txt entry: during BBN it is the
        # thermal weak rate (n__p, rxn_idx 0, T-dependent, computed by the
        # background) and is therefore skipped above.  In the DT era T→0, so
        # that thermal rate reduces to the vacuum decay constant λ_n = 1/τ_n
        # (τ_n = cfg.tau_n, the neutron lifetime).  Without this term the
        # residual free neutrons surviving at t_end (Y_n ~ 4×10⁻¹⁶) would be
        # frozen for all of cosmic time instead of decaying to protons within
        # ~minutes; including it lets the DT era track n correctly.  A_n = A_p
        # = 1, so the mass-fraction gain factor A_p/A_n is unity.
        if "n" in net.species and "p" in net.species:
            n_idx = list(net.species).index("n")
            p_idx = list(net.species).index("p")
            lam_n = 1.0 / self.cfg.tau_n   # [s^-1]; τ_n = neutron lifetime
            D[n_idx, n_idx] -= lam_n
            D[p_idx, n_idx] += lam_n

        return D

    def _integrate_decay_era(self, D, Y0, t_end, t_grid):
        r"""Propagate abundances through the DT era via matrix exponentiation.

        The DT (Decay Time) ODE ``dY/dt = D · Y`` with constant coefficient
        matrix D is solved exactly by:

            Y(t) = exp(D × (t − t_end)) · Y_0

        We form the dense matrix exponential with ``scipy.linalg.expm`` (Padé
        approximation with *scaling-and-squaring*) and apply it to Y0:

            Y(t_i) = expm(D × Δt_i) @ Y0

        where Δt_i = t_i − t_end is the elapsed time since BBN end.

        **Why not ``scipy.sparse.linalg.expm_multiply``?**  The decay matrix has
        a colossal eigenvalue spread: the fastest decay (B15, T½ ≈ 10 ms) gives
        an eigenvalue ~70 s⁻¹, while Δt reaches ~1 Gyr ≈ 3×10¹⁶ s, so
        ‖D·Δt‖ ~ 10¹⁸.  ``expm_multiply`` selects its number of internal
        matrix–vector products *linearly* in ‖D·Δt‖ (Al-Mohy & Higham 2011,
        Eq. 3.6), so for this norm it attempts ~10¹⁸ products and effectively
        hangs.  ``scipy.linalg.expm`` instead uses scaling-and-squaring whose
        cost grows only *logarithmically* in ‖D·Δt‖ (≈ log₂‖D·Δt‖ ~ 60
        squarings), so it handles the full 16-decade spread in milliseconds.
        Since D is small (N ≤ 60), forming the dense exp(D·Δt) is cheap
        (~3 ms per time point, ~0.1 s for the default 200-point grid).

        Parameters
        ----------
        D : np.ndarray, shape (N, N)
            Decay-rate matrix from :meth:`_build_decay_matrix` [s^-1].
        Y0 : np.ndarray, shape (N,)
            Initial abundance vector at t = t_end (end of LT era).
        t_end : float
            Cosmic time at end of BBN / start of DT era [s].
        t_grid : np.ndarray, shape (M,)
            Output times [s], all > t_end; log-spaced from solve().

        Returns
        -------
        Y_t : np.ndarray, shape (M, N)
            Abundance vectors at each output time.  Row i is Y(t_grid[i]).

        Notes
        -----
        D's eigenvalues are the negative decay constants (≤ 0), so exp(D·Δt)
        is a contraction and the result is numerically stable for any
        positive Δt.

        References
        ----------
        Al-Mohy & Higham (2009), "A New Scaling and Squaring Algorithm for the
        Matrix Exponential", SIAM J. Matrix Anal. Appl. 31, 970–989 (the
        algorithm behind ``scipy.linalg.expm``).
        """
        from scipy.linalg import expm

        N_t = len(t_grid)
        N   = len(Y0)
        Y_t = np.zeros((N_t, N))

        for k, t_k in enumerate(t_grid):
            dt = t_k - t_end   # elapsed time since end of BBN [s]
            # expm(D*dt) @ Y0 computes the exact solution Y(t_k) of dY/dt = D·Y.
            Y_t[k] = expm(D * dt) @ Y0
            # Clip small negative values that arise from floating-point
            # cancellation (the matrix exp may produce tiny negatives for
            # species whose abundance is near zero).
            np.clip(Y_t[k], 0.0, None, out=Y_t[k])

        return Y_t

    def _write_decay_evolution(self, t_grid, Y_t):
        """Write the DT-era abundance time series to a TSV file.

        Enabled by ``cfg.output_decay_evolution=True``; the destination is
        ``cfg.output_decay_file`` (relative paths resolve against the current
        working directory).

        Columns: ``t`` [s], then one ``Y<species>`` column per tracked
        nuclide in ``self.abundance_names`` (in abundance-vector order).

        Parameters
        ----------
        t_grid : np.ndarray, shape (M,)
            Output times [s] (log-spaced from t_end to t_end + t_decay_end).
        Y_t : np.ndarray, shape (M, N)
            Abundance vectors at each time, from :meth:`_integrate_decay_era`.
        """
        cfg  = self.cfg
        path = os.path.abspath(cfg.output_decay_file)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        nuc_cols = ["Y" + s for s in self.abundance_names]
        out_data = np.column_stack([t_grid, Y_t])
        out_header = "\t".join(["t"] + nuc_cols)
        np.savetxt(path, out_data, delimiter='\t', header=out_header, comments='')
        print(f"[output] Decay-era evolution ({len(t_grid)} rows) written to {path}")
