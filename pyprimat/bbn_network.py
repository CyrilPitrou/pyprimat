# -*- coding: utf-8 -*-
"""
bbn_network.py
===============
``NuclearNetwork`` (Class 2 of the PyPR split, see ``pyprimat.background`` for
Class 1): the nuclear-reaction-network ODE integration across the HT/MT/LT
temperature eras.

Design
------
``NuclearNetwork`` is driven purely through the public interface of a
``pyprimat.background.Background`` instance: ``T_of_t``/``t_of_T``/``a_of_t``,
``rhoB_BBN``/``etab_of_T`` (baryon sector), the normalised n<->p weak rates
(``weak_nTOp_frwrd_raw``/``weak_nTOp_bkwrd_raw`` plus ``NormWeakRates``), and a
handful of background quantities needed only for *output*
(``Hubble``, ``Tnue_vec``/``Tnumu_vec``/``Tnutau_vec``, ``t_vec``,
``N_NEVO_of_Tg``/``has_heating_table``).  It knows nothing about *how* the
background was constructed (NEVO table, instantaneous decoupling, external
background, ...) -- this is exactly the seam that makes the background
pluggable (``pyprimat.background.Background``).

``solve()`` integrates:

* **HT** (high temperature, T > T_weak ~ 1 MeV): n <-> p only.
* **MT** (mid temperature, T_weak -> T_nucl ~ 0.1 MeV): the fixed 18-reaction
  subset (n<->p + 17 reactions), regardless of network size.
* **LT** (low temperature, T_nucl -> T_end ~ 0.001 MeV): the chosen network
  (small/medium/large).

and returns the BBN observables dict (``Neff``, ``YPBBN``, ``YPCMB``, ``DoH``,
``He3oH``, ``He3oHe4``, ``Li7oH``, ``Omeganurel``, ``OneOverOmeganunr``), while
also populating the public ``results``, ``Y_final``, ``abundance_names`` and
``Y_of_t`` attributes consumed by ``PyPR``'s observable accessors
(``get_quantity``, ``__getitem__``, ...).
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
    nucl : pyprimat.nuclear.UpdateNuclearRates
        Compiled MT/LT reaction-rate kernels (RHS + Jacobian) for the chosen
        network.
    background : pyprimat.background.Background
        The cosmological background (Class 1) supplying ``T_of_t``/``t_of_T``,
        ``a_of_t``, ``rhoB_BBN``/``etab_of_T``, the normalised n<->p weak
        rates, and the output-only quantities (``Hubble``, neutrino
        temperature vectors, ``N_NEVO_of_Tg``/``has_heating_table``).

    Attributes (populated by :meth:`solve`)
    ----------------------------------------
    results : dict or None
        The BBN observables dict (``None`` until :meth:`solve` has run).
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
        self.results = None
        self.Y_final = None
        self.abundance_names = None
        self.Y_of_t = None

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
        background = self.background
        t_vec     = background.t_vec
        Tg_vec    = background.Tg_vec
        Tnu_vec   = background.Tnu_vec
        a_of_t    = background.a_of_t
        T_of_t    = background.T_of_t
        t_of_T    = background.t_of_T
        NormWR    = background.NormWeakRates
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
        # Baryon density for the nuclear network (Background, see
        # pyprimat.background.StandardBackground.rhoB_BBN/etab_of_T)
        # ------------------------------------------------------------------
        rhoB_BBN  = background.rhoB_BBN
        etab_of_T = background.etab_of_T

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

        nTOp_frwrd_HT = background.weak_nTOp_frwrd_raw
        nTOp_bkwrd_HT = background.weak_nTOp_bkwrd_raw

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

        nTOp_f_MT, nTOp_b_MT = make_nTOp_pair(background.weak_nTOp_frwrd_raw, background.weak_nTOp_bkwrd_raw)
        nTOp_f_LT, nTOp_b_LT = make_nTOp_pair(background.weak_nTOp_frwrd_raw, background.weak_nTOp_bkwrd_raw)

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
        # Optional output: full time evolution of background + abundances
        # ------------------------------------------------------------------
        if cfg.output_time_evolution:
            self._write_time_evolution(
                sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                nucl, YA,
            )

        # ------------------------------------------------------------------
        # Optional output: two-column (nuclide, final abundance Y) table
        # ------------------------------------------------------------------
        if cfg.output_final_result:
            self._write_final_result()

        # ------------------------------------------------------------------
        # Final observables
        # ------------------------------------------------------------------
        Tg_last  = background.Tg_vec[-1]
        Tnu_last = background.Tnu_vec[-1]

        Neff = background.N_eff(Tg_last, Tnu_last, Tnu_last, Tnu_last)

        # Access final abundances by name from the dict built in the LT era.
        Yp_f  = finL["p"];    Yd_f  = finL["H2"]; Yt_f  = finL["H3"]
        YHe3_f = finL["He3"]; Ya_f  = finL["He4"]
        YLi7_f = finL["Li7"]; YBe7_f = finL["Be7"]

        YPBBN  = 4. * Ya_f
        YPCMB  = ((cfg.He4Overma / 4.) * YPBBN
                  / ((cfg.He4Overma / 4.) * YPBBN + cfg.HOverma * (1. - YPBBN)))

        # The results dict returned by solve()/PyPRresults() and used by the
        # get_quantity()/__getitem__/Neff/YPBBN/... accessors.  All nine keys
        # below are computed unconditionally from the background + nuclear
        # solve and are always present, regardless of which optional flags
        # (incomplete_decoupling, spectral_distortions, network, ...) are set
        # -- there are no flag-dependent placeholder entries here.  The only
        # other (file) output that does depend on a flag is the
        # _write_time_evolution TSV's "Nheating" column, see its docstring.
        self.results = {
            "Neff":            Neff,
            "Omeganurel":      background.Omeganuh2_relnu() * 1e+6,
            "OneOverOmeganunr": 1. / (background.Omeganuh2_nrnu() * 1e-6),
            "YPCMB":           YPCMB,
            "YPBBN":           YPBBN,
            "DoH":             Yd_f / Yp_f,
            "He3oH":           (Yt_f + YHe3_f) / Yp_f,
            "He3oHe4":         (Yt_f + YHe3_f) / Ya_f,
            "Li7oH":           (YLi7_f + YBe7_f) / Yp_f,
        }
        return self.results

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

    def _write_time_evolution(self, sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                              nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                              nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                              nucl, YA):
        """Write the full background + abundance time series to a TSV file.

        Enabled by ``output_time_evolution=True``; the destination is
        ``cfg.output_file``.  Works for all three networks
        (``small``/``medium``/``large``) -- ``Y<species>`` columns are
        derived from ``self.abundance_names`` (8 / 12 / ~59 nuclides).

        Columns, always present:
            ``a``, ``T``, ``t``, ``H``       -- scale factor, photon
                temperature [MeV], cosmic time [s], Hubble rate [s^-1];
            ``Tnue``, ``Tnumu``, ``Tnutau``  -- the three flavour neutrino
                temperatures [MeV];
            ``Y<species>``                   -- one column per tracked
                nuclide (mass-fraction abundance). During the HT era (and
                before ``T_start_cosmo``), where every nuclide but n/p is
                not yet integrated, its column holds the Nuclear Statistical
                Equilibrium (Saha) prediction ``YA(name, Yn, Yp, T)``
                (``solve()``'s local ``YA``, IDEAS2.md item 1) instead of a
                hard 0 -- giving a smooth, physically-motivated curve in
                log-log abundance plots instead of a gap. From the MT era
                onward the column is the actual integrated value (which may
                still be 0 or tiny for an untracked/negligible heavy
                nuclide -- the Saha formula is not applied there, as NSE no
                longer holds and ``BindE/(kB T)`` would overflow at low T9);
            ``n_to_p_weak_rate``, ``p_to_n_weak_rate`` -- n<->p weak rates
                [s^-1] (zero before the nuclear network starts).

        Conditional columns:
            ``Nheating`` -- the NEVO heating function N(T_gamma) driving the
                a(T_gamma) ODE (see
                :meth:`pyprimat.background.StandardBackground._setup_background_and_cosmo`).
                Included only when ``background.has_heating_table`` (i.e.
                ``cfg.incomplete_decoupling=True``); under
                ``InstantaneousDecoupling`` it would just be a column of
                zeros, so it is omitted entirely.
            per-reaction flux columns (``<reaction>_frwrd``) -- included only
                when ``cfg.output_rates_time_evolution=True`` *and*
                ``network`` is ``small``/``medium``.  Omitted for
                ``network="large"`` (~433 reactions): use the
                ``run[species](t)`` abundance interpolators (and the
                tabulated rates on ``nucl``) directly if reaction-level
                fluxes are needed for the large network.
        """
        cfg = self.cfg
        background = self.background
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

        a_out = background.a_of_t(t_out)
        T_out = background.T_of_t(t_out)

        Tnue_of_t   = interp1d(background.t_vec, background.Tnue_vec,   bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnumu_of_t  = interp1d(background.t_vec, background.Tnumu_vec,  bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnutau_of_t = interp1d(background.t_vec, background.Tnutau_vec, bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnue_out   = Tnue_of_t(t_out)
        Tnumu_out  = Tnumu_of_t(t_out)
        Tnutau_out = Tnutau_of_t(t_out)

        H_out = np.array([
            background.Hubble(T_out[i], Tnue_out[i], Tnumu_out[i], Tnutau_out[i])
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

        # ------------------------------------------------------------------
        # Fill not-yet-tracked abundances with their NSE (Saha) prediction
        # ------------------------------------------------------------------
        # _embed (in solve()) zero-fills any species not yet integrated in a
        # given era -- every nuclide but n/p during the HT era, and every
        # species before t_start (only the cosmological background has been
        # solved there). Those exact 0s are a hard discontinuity in a
        # log-log abundance plot (pyprimat.gui.panels.render_evolution_panel
        # masks out y<=0 entirely). Replace each such 0 with the Nuclear
        # Statistical Equilibrium value YA(name, Yn, Yp, T) (IDEAS2.md item
        # 1): at these early, hot times every nuclide *is* in NSE, and YA's
        # eta_b^(A-1) suppression already makes it negligibly small there --
        # so this gives a smooth, physical curve down to T_start_cosmo
        # instead of a gap.
        i_n = self.abundance_names.index("n")
        i_p = self.abundance_names.index("p")
        Yn_for_NSE = Y_out[:, i_n].copy()
        Yp_for_NSE = Y_out[:, i_p].copy()
        mask_pre = ~mask_nuc
        if mask_pre.any():
            # Before t_start, n and p are not yet integrated either; use the
            # n<->p weak-equilibrium fraction at T_out (Yn_i_func in solve(),
            # b/(b+f) with b, f the backward/forward HT weak rates).
            b_pre = nTOp_bkwrd_HT_norm(T_K_out[mask_pre])
            f_pre = nTOp_frwrd_HT_norm(T_K_out[mask_pre])
            Yn_eq = b_pre / (b_pre + f_pre)
            Yn_for_NSE[mask_pre] = Yn_eq
            Yp_for_NSE[mask_pre] = 1. - Yn_eq
            Y_out[mask_pre, i_n] = Yn_eq
            Y_out[mask_pre, i_p] = 1. - Yn_eq

        # Restrict the replacement to the HT era (t_out < t_weak, which also
        # covers t_out < t_start): this is the only region where _embed is
        # guaranteed to zero-fill *every* non-n/p species (only n,p are
        # integrated there), and where T is high enough (>= T_weak ~ 1 MeV)
        # that BindE*keV/(kB*T) stays of order a few hundred at most --
        # exp() cannot overflow. For t_out >= t_weak, a column that is
        # exactly 0 reflects the MT/LT solution itself (a genuinely tiny or
        # untracked heavy nuclide); applying the Saha formula there would be
        # both physically wrong (NSE has long broken down) and numerically
        # unsafe (BindE/(kB T) -> large at the low T9 of the LT era, so
        # exp() overflows to inf, and inf*0 from the eta_b^(A-1)/Y^... prefactors
        # yields nan).
        mask_ht_or_pre = t_out < t_weak
        for j, name in enumerate(self.abundance_names):
            if name in ("n", "p"):
                continue
            zero = mask_ht_or_pre & (Y_out[:, j] == 0.)
            if zero.any():
                Y_out[zero, j] = YA(name, Yn_for_NSE[zero], Yp_for_NSE[zero],
                                     T_K_out[zero])

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

        # The "Nheating" column is the NEVO heating function N(T_gamma) that
        # drives the a(T_gamma) ODE (see
        # StandardBackground._setup_background_and_cosmo).  It is only
        # physically meaningful when a real NEVO table was loaded
        # (cfg.incomplete_decoupling=True); under InstantaneousDecoupling,
        # background.N_NEVO_of_Tg is the N=0 stub used to close the ODE under
        # plain EM entropy conservation, so writing it out would just be a
        # column of zeros masquerading as data.  Include the column only when
        # it carries real information (background.has_heating_table).
        heating_cols = ["Nheating"] if background.has_heating_table else []
        if background.has_heating_table:
            heating_out = (background.N_NEVO_of_Tg(T_out),)
        else:
            heating_out = ()

        # Resolve relative paths against the current working directory (the
        # universal convention), not the installed-package directory.
        out_path = os.path.abspath(cfg.output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_data = np.column_stack((a_out, T_out, t_out, H_out,
                                    Tnue_out, Tnumu_out, Tnutau_out)
                                    + heating_out
                                    + (Y_out,
                                       weak_n_to_p_out, weak_p_to_n_out, rxn_rate_out))
        out_header = "\t".join(["a", "T", "t", "H",
                                 "Tnue", "Tnumu", "Tnutau"]
                               + heating_cols
                               + nuc_cols
                               + ["n_to_p_weak_rate", "p_to_n_weak_rate"] + rxn_rate_cols)
        np.savetxt(out_path, out_data, delimiter='\t', header=out_header, comments='')

        # Always announce: written only on explicit request (output_time_evolution=True).
        print(f"[output] Time-evolution data ({len(t_out)} rows) written to {out_path}")
