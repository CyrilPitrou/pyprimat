# -*- coding: utf-8 -*-
"""
nuclear_network.py
==================
``NuclearNetwork`` (Class 2 of the PyPR split, see ``pyprimat.background`` for
Class 1): the nuclear-reaction-network ODE integration across the HT/MT/LT
temperature eras.

Design
------
``NuclearNetwork`` is driven purely through the *minimal* public interface of
a ``pyprimat.background.Background`` instance:

* ``T_of_t(t)`` / ``t_of_T(T)``  -- time <-> temperature
* ``rhoB_BBN(t)``                -- baryon mass density [g/cm^3] as a
  function of cosmic time (the prefactor for nuclear reaction rates)
* ``weak_nTOp_frwrd(T_K)`` / ``weak_nTOp_bkwrd(T_K)`` -- already-normalised
  n<->p weak rates [s^-1] at photon temperature ``T_K`` [Kelvin]

It knows nothing about *how* the background was constructed (NEVO table,
instantaneous decoupling, external background, scale factor, neutrino
sector, ...) -- this is exactly the seam that makes the background pluggable
(``pyprimat.background.Background``).  In particular it does **not** use
``a_of_t``, ``Hubble``, the individual neutrino temperatures, or the NEVO
heating function: those are output-only quantities written directly by
``Background.write_time_evolution`` (see ``pyprimat.background``), not
consumed by the nuclear solve.

``solve()`` integrates:

* **HT** (high temperature, T > T_weak ~ 1 MeV): n <-> p only.
* **MT** (mid temperature, T_weak -> T_nucl ~ 0.1 MeV): the fixed 18-reaction
  subset (n<->p + 17 reactions), regardless of network size.
* **LT** (low temperature, T_nucl -> T_end ~ 0.001 MeV): the chosen network
  (small/medium/large).

and populates the public ``Y_final``, ``abundance_names`` and ``Y_of_t``
attributes consumed by ``PyPR``'s observable accessors (``get_quantity``,
``__getitem__``, ...) and by ``PyPR.solve()`` (which builds the BBN
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

__all__ = ["NuclearNetwork"]


class NuclearNetwork:
    """The nuclear reaction network (Class 2): HT/MT/LT ODE integration.

    Parameters
    ----------
    cfg : pyprimat.config.PyPRConfig
        Run-time configuration (network choice, temperature-era boundaries,
        numerical tolerances, rate-variation parameters p_*, ...).
    nucl : pyprimat.network_data.UpdateNuclearRates
        Compiled MT/LT reaction-rate kernels (RHS + Jacobian) for the chosen
        network.
    background : pyprimat.background.Background
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

    # ======================================================================
    # solve(): integrate nuclear network ODEs
    # ======================================================================

    def solve(self):
        """
        Integrate the nuclear network over the three temperature eras.

        Populates ``self.Y_final``, ``self.abundance_names`` and
        ``self.Y_of_t`` and returns ``self.Y_final`` (the dict of final
        mass-fraction abundances by nuclide name).  The BBN observables dict
        (``Neff``, ``YPBBN``, ``DoH``, ...) is built by ``PyPR.solve()`` from
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
            print("[nucl]  Solving neutron decoupling at high temperature era")

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
            print((f"[nucl]  [HT] Finished solve_ivp in {time.time()-_t_ht0:.2f} s "
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
        atol = cfg.atol_large_LT if cfg.is_large else 1e-20
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
            # Full list of every nuclide that was integrated numerically in the
            # LT era (species_L is exactly the LT solver's state vector).  The
            # list grows with the chosen network (8 / 12 / ~59 nuclides for
            # small / medium / large).
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

        return self.Y_final

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
        names = self.abundance_names
        with open(path, 'w') as f:
            f.write(f"# {'nuclide':<12}Y\n")
            for nm in names:
                f.write(f"{nm:<14}{self.Y_final[nm]:.6e}\n")
        # Always announce: this file is written only on explicit request
        # (output_final_result=True), so the user wants to know where it landed.
        print(f"[output] Final abundances ({len(names)} nuclides) written to {path}")

    def _write_time_evolution(self, sol_HT, sol_LT, nucl):
        """Write the abundance + weak-rate time series to a TSV file.

        Enabled by ``output_time_evolution=True``; the destination is
        ``cfg.output_file``.  Works for all three networks
        (``small``/``medium``/``large``) -- ``Y<species>`` columns are
        derived from ``self.abundance_names`` (8 / 12 / ~59 nuclides).

        Columns, always present:
            ``T``, ``t``                     -- photon temperature [MeV],
                cosmic time [s];
            ``Y<species>``                   -- one column per tracked
                nuclide (mass-fraction abundance);
            ``n_to_p_weak_rate``, ``p_to_n_weak_rate`` -- n<->p weak rates
                [s^-1] (already normalised, see
                ``background.weak_nTOp_frwrd``/``weak_nTOp_bkwrd``; defined
                at every output time, including before the nuclear network
                starts, since they depend only on T).

        Before the nuclear network starts integrating a given species (the
        HT era for everything but n/p, and the time before ``T_start_cosmo``
        for n/p too), its ``Y<species>`` column is **exactly 0** -- the value
        ``_embed``/``Y_of_t`` produce there.  A previous version of this
        method filled that region with the Nuclear Statistical Equilibrium
        (Saha) prediction ``YA(name, Yn, Yp, T)`` for a smoother log-log
        plot; this was removed because the fill is *often wrong*: NSE need
        not hold for every nuclide all the way down to ``T_start_cosmo``
        (e.g. for non-standard backgrounds with extra entropy injection or a
        non-thermal neutrino sector), and a silently-injected equilibrium
        value is worse than an honest 0 that a plotting tool can choose to
        mask. Consumers that want a smooth pre-MT curve can compute the Saha
        value themselves from the ``T``/``t`` columns.

        Conditional columns:
            per-reaction flux columns (``<reaction>_frwrd``) -- included only
                when ``cfg.output_rates_time_evolution=True`` *and*
                ``network`` is ``small``/``medium``.  Omitted for
                ``network="large"`` (~433 reactions): use the
                ``run[species](t)`` abundance interpolators (and the
                tabulated rates on ``nucl``) directly if reaction-level
                fluxes are needed for the large network.

        The background-only columns (``a``, ``H``, ``Tnue``/``Tnumu``/
        ``Tnutau``, ``Nheating``, energy densities, ...) are no longer part
        of this file -- they are written separately by
        ``background.write_time_evolution`` when
        ``cfg.output_background_evolution=True`` (see
        :mod:`pyprimat.background`).
        """
        cfg = self.cfg
        background = self.background
        nTOp_frwrd = background.weak_nTOp_frwrd
        nTOp_bkwrd = background.weak_nTOp_bkwrd
        # Derive column names from the actual abundance names so custom networks
        # (which may have fewer or different nuclides than the standard 8 or 12)
        # produce a TSV with the correct number of columns.
        nuc_cols = ["Y" + s for s in self.abundance_names]
        n_nuc = len(nuc_cols)

        Y_of_t = self.Y_of_t

        # Uniform log-spaced output grid from T_start_cosmo to end of LT era
        t_cosmo = background.t_of_T(cfg.T_start_cosmo / cfg.MeV_to_Kelvin)
        t_end   = sol_LT.t[-1]
        t_out   = np.logspace(np.log10(t_cosmo), np.log10(t_end), cfg.output_n_points)

        T_out = background.T_of_t(t_out)

        # Weak rates: already normalised and defined for any T, including
        # before the nuclear network starts (mask_nuc below is for the
        # abundance columns only).
        T_K_out = T_out * cfg.MeV_to_Kelvin
        weak_n_to_p_out = nTOp_frwrd(T_K_out)
        weak_p_to_n_out = nTOp_bkwrd(T_K_out)

        # Abundances: zero before nuclear network starts (Y_of_t's fill_value)
        t_start = sol_HT.t[0]
        Y_out = np.zeros((len(t_out), n_nuc))
        mask_nuc = t_out >= t_start
        Y_out[mask_nuc] = Y_of_t(t_out[mask_nuc])

        if cfg.output_rates_time_evolution and not cfg.is_large:
            rxn_rate_cols = sorted(
                name for name in dir(nucl)
                if name.endswith("_frwrd") and callable(getattr(nucl, name))
            )
            rxn_rate_out = np.zeros((len(t_out), len(rxn_rate_cols)))
            rxn_rate_out[mask_nuc] = np.column_stack([
                getattr(nucl, name)(T_K_out[mask_nuc]) for name in rxn_rate_cols
            ])
        else:
            # Per-reaction flux columns are omitted for network="large"
            # (~433 reactions): use the run[species](t) abundance
            # interpolators instead if reaction-level fluxes are needed.
            if cfg.output_rates_time_evolution and cfg.is_large:
                print("[output] output_rates_time_evolution ignored for "
                      "network='large' (~433 reactions); per-reaction flux "
                      "columns are omitted from the TSV.")
            rxn_rate_cols = []
            rxn_rate_out = np.empty((len(t_out), 0))

        # Resolve relative paths against the current working directory (the
        # universal convention), not the installed-package directory.
        out_path = os.path.abspath(cfg.output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_data = np.column_stack((T_out, t_out, Y_out,
                                     weak_n_to_p_out, weak_p_to_n_out, rxn_rate_out))
        out_header = "\t".join(["T", "t"]
                               + nuc_cols
                               + ["n_to_p_weak_rate", "p_to_n_weak_rate"] + rxn_rate_cols)
        np.savetxt(out_path, out_data, delimiter='\t', header=out_header, comments='')

        # Always announce: written only on explicit request (output_time_evolution=True).
        print(f"[output] Time-evolution data ({len(t_out)} rows) written to {out_path}")
