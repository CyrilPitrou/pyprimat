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

import re
import time
import numpy as np

__all__ = ['PyPR', 'mc_uncertainty']

from .config       import PyPRConfig
from . import plasma      as PyPRthermo
from .background   import StandardBackground
from .bbn_network  import NuclearNetwork


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
        summed into ``rho_tot`` by :meth:`pyprimat.background.StandardBackground.Hubble`.
        This is the generic plug-in point for "dark sector" components; Early
        Dark Energy (``cfg.fEDE > 0``) is implemented as the first such
        plug-in (see :meth:`pyprimat.background.StandardBackground._setup_EDE`)
        and is appended automatically -- callers do not need to include it
        here.

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
        # 3. Initialize nuclear network (MT/LT eras)
        # ------------------------------------------------------------------
        from .nuclear import UpdateNuclearRates
        self.nucl = UpdateNuclearRates(cfg)

        # ------------------------------------------------------------------
        # 4. Build the cosmological background (Class 1): a<->t<->T relations,
        #    rho_B(t), n<->p weak rates, Neff/Omega_nu -- everything the
        #    nuclear network needs about the expanding Universe.  Early Dark
        #    Energy (cfg.fEDE > 0) is appended to extra_rho automatically by
        #    StandardBackground; see pyprimat.background.
        # ------------------------------------------------------------------
        self.background = StandardBackground(cfg, self.plasma, extra_rho)

        # ------------------------------------------------------------------
        # 5. Build the nuclear network (Class 2): the HT/MT/LT ODE
        #    integration, driven by the Background's T(t)/rho_B(t)/weak
        #    rates -- see pyprimat.bbn_network.NuclearNetwork.
        # ------------------------------------------------------------------
        self.nuclear = NuclearNetwork(cfg, self.nucl, self.background)

        if cfg.verbose:
            print(f"[init]  Initialisation complete in {time.time()-self._t0:.1f} s")

    # ======================================================================
    # solve(): integrate nuclear network ODEs
    # ======================================================================

    def solve(self):
        """
        Integrate the nuclear network over the three temperature eras and
        return a dict of BBN observables.

        Delegates to :meth:`pyprimat.bbn_network.NuclearNetwork.solve`
        (Class 2), which is driven by ``self.background`` (Class 1, see
        :mod:`pyprimat.background`).  After this call, ``self.nuclear.results``,
        ``self.nuclear.Y_final``, ``self.nuclear.abundance_names`` and
        ``self.nuclear.Y_of_t`` are populated.
        """
        results = self.nuclear.solve()

        # For the large network, NuclearNetwork.solve() discovers nuclides
        # beyond the small/medium set (e.g. B10, C12, ...).  Extend the
        # N/Z/A maps (built in __init__ from cfg.Nuclides) so callers (e.g.
        # the AbundanceEvolution notebook, pyprimat.gui.panels) can look up
        # A[name]/Z[name]/N[name] for every species returned by
        # abundance_names.
        if not self.cfg.is_small:
            for s, (N, Z) in self.nucl.large_NZ.items():
                self.N[s], self.Z[s], self.A[s] = N, Z, N + Z

        return results

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def T_of_t(self):
        """T_γ(t) interpolator [MeV], available after initialisation."""
        return self.background.T_of_t

    @property
    def t_of_T(self):
        """t(T_γ) interpolator [s], available after initialisation."""
        return self.background.t_of_T

    @property
    def a_of_T(self):
        """Scale factor a(T_γ), available after initialisation.

        ``a`` follows the same normalisation as the internal a(T) ODE
        (:meth:`pyprimat.background.StandardBackground._setup_background_and_cosmo`):
        entropy conservation ``a^3 * spl(T) = const`` fixed so that
        ``a * T -> T0CMB`` [MeV] as ``T -> 0``, i.e. ``a = 1`` today up to the
        small entropy-injection correction from e+e- annihilation encoded in
        ``spl(T)``.

        Example
        -------
        >>> p.a_of_T(1.0)   # scale factor at T_gamma = 1 MeV
        """
        return self.background.a_of_T

    @property
    def T_of_a(self):
        """T_γ(a) interpolator [MeV], available after initialisation.

        Inverse of :attr:`a_of_T`; same normalisation convention for ``a``.
        """
        return self.background.T_of_a

    @property
    def a_of_t(self):
        """Scale factor a(t), available after initialisation.

        Same normalisation as :attr:`a_of_T`; ``t`` is the cosmic time [s]
        used by :attr:`T_of_t`/:attr:`t_of_T`.
        """
        return self.background.a_of_t

    @property
    def t_of_a(self):
        """Cosmic time t(a) [s], available after initialisation.

        Inverse of :attr:`a_of_t`; same normalisation convention for ``a``.
        """
        return self.background.t_of_a

    def __getitem__(self, species):
        """Return Y(t) for a species name (e.g. 'H2', 'He4', 'Li7').

        Calls solve() automatically if needed.
        """
        self._ensure_solved()
        names = self.nuclear.abundance_names
        if species not in names:
            raise KeyError(
                f"Unknown species '{species}'. Available: {names}"
            )
        idx = names.index(species)
        def fn(t):
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            vals  = self.nuclear.Y_of_t(t_arr)[:, idx]
            return float(vals[0]) if np.ndim(t) == 0 else vals
        return fn

    def _ensure_solved(self):
        if self.nuclear.results is None:
            self.solve()

    def PyPRresults(self):
        """Return the BBN result dict, running ``solve()`` first if needed."""
        self._ensure_solved()
        return self.nuclear.results

    @property
    def abundance_names(self):
        """Tracked nuclide names, in abundance-vector order (solves if needed).

        For the large network this is the full ~59-nuclide list; accessing it
        also guarantees ``self.A``/``N``/``Z`` cover every species (handy for
        plotting ``A_i Y_i`` for all nuclides)."""
        self._ensure_solved()
        return self.nuclear.abundance_names

    # Convenience accessors
    def Neff(self):          self._ensure_solved(); return self.nuclear.results["Neff"]
    def Omeganurel(self):    self._ensure_solved(); return self.nuclear.results["Omeganurel"]
    def Omeganunonrel(self): self._ensure_solved(); return 1. / self.nuclear.results["OneOverOmeganunr"]
    def YPCMB(self):         self._ensure_solved(); return self.nuclear.results["YPCMB"]
    def YPBBN(self):         self._ensure_solved(); return self.nuclear.results["YPBBN"]
    def DoH(self):           self._ensure_solved(); return self.nuclear.results["DoH"]
    def He3oH(self):         self._ensure_solved(); return self.nuclear.results["He3oH"]
    def Li7oH(self):         self._ensure_solved(); return self.nuclear.results["Li7oH"]

    def get_quantity(self, quantity):
        """Return a scalar BBN quantity by name.

        Accepts any key from the result dict ('YPBBN', 'DoH', 'He3oH',
        'Li7oH', 'Neff', 'YPCMB', ...) or a nuclide name from
        cfg.Nuclides ('H2', 'He4', 'Li7', ...) for the final mass fraction Y.
        """
        self._ensure_solved()
        results = self.nuclear.results
        Y_final = self.nuclear.Y_final
        if quantity in results:
            return results[quantity]
        if quantity in Y_final:
            return Y_final[quantity]
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            f"Valid result keys: {list(results.keys())}. "
            f"Valid nuclide names: {list(Y_final.keys())}."
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

    Attributes
    ----------
    seed : int or None
        The base random seed used to generate the samples (sample ``i`` used
        ``seed + i``).  Stored so that :func:`mc_uncertainty` can *extend* an
        existing result with more samples without recomputing the ones it
        already has: a previous result is only reused when its ``seed`` and its
        set/order of quantities match the new request (see ``prev`` there).
    """
    def __init__(self, data, seed=None):
        self._data = data   # dict: str -> MCQuantityResult
        self.seed = seed

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

    In addition to the ``len(rate_keys)`` nuclear-rate offsets, each sample
    draws one further ``standard_normal()`` from the *same* per-sample
    ``Generator`` (after the rate offsets, so the RNG stream order does not
    depend on ``len(rate_keys)``) and uses it to perturb the neutron lifetime:
    ``tau_n_sample = tau_n_central + std_tau_n * randn()``.  When
    ``cfg.tau_n_flag`` is True this rescales the weak-rate normalisation
    ``background.NormWeakRates = 1/(Fn * tau_n)``
    (``StandardBackground._setup_weak_rates``) without recomputing the ``Fn``
    integral -- ``Fn`` does not depend on ``tau_n``, so
    ``background.NormWeakRates * tau_n`` is invariant and is precomputed once
    before the loop.  When ``cfg.tau_n_flag`` is False, ``tau_n`` does not
    enter the normalisation and the draw is a harmless no-op (kept so the RNG
    stream is identical either way).
    """
    inst = PyPR(params=base_params)
    cfg  = inst.cfg
    tau_n_central = cfg.tau_n
    # background.NormWeakRates = 1/(Fn * tau_n) (cfg.tau_n_flag=True case of
    # StandardBackground._setup_weak_rates), so this product is the
    # tau_n-independent 1/Fn -- rescaling by it for each sampled tau_n avoids
    # recomputing Fn.
    norm_times_tau_n = inst.background.NormWeakRates * tau_n_central
    results = []
    for seed in seeds:
        rng    = np.random.default_rng(seed)
        p_vals = rng.standard_normal(len(rate_keys))
        for k, v in zip(rate_keys, p_vals):
            setattr(cfg, k, float(v))
        tau_n_sample = tau_n_central + cfg.std_tau_n * rng.standard_normal()
        if cfg.tau_n_flag:
            cfg.tau_n = tau_n_sample
            inst.background.NormWeakRates = norm_times_tau_n / tau_n_sample
        inst.solve()
        results.append([inst.get_quantity(q) for q in quantities])
    return results


def _mc_collect_samples(base_params, rate_keys, quantities, seeds, n_jobs):
    """Run :func:`_mc_run_batch` for a list of seeds and stack the results.

    Splits ``seeds`` into one chunk per worker so the expensive cosmological
    background + n<->p weak-rate setup (which does *not* depend on the sampled
    nuclear rates) is paid once per worker instead of once per sample, then
    returns the ``(len(seeds), len(quantities))`` array of sampled quantity
    values.  Because every sample draws its rate vector from
    ``default_rng(seed)`` (see ``_mc_run_batch``), the row for a given seed is
    independent of how the seeds are chunked -- so callers can safely build the
    full sample set incrementally by collecting disjoint seed ranges and
    stacking them (this is what the ``prev`` reuse in :func:`mc_uncertainty`
    relies on).

    An empty ``seeds`` list returns an empty ``(0, len(quantities))`` array so
    the result can always be ``np.vstack``-ed with an existing sample block.
    """
    from joblib import Parallel, delayed, effective_n_jobs

    if not seeds:
        return np.empty((0, len(quantities)))
    n_chunks = max(1, min(len(seeds), effective_n_jobs(n_jobs)))
    chunks   = [list(c) for c in np.array_split(seeds, n_chunks)]
    raw = Parallel(n_jobs=n_jobs)(
        delayed(_mc_run_batch)(base_params, rate_keys, quantities, chunk)
        for chunk in chunks
    )
    return np.array([row for chunk in raw for row in chunk])


def mc_uncertainty(num_mc, quantity, params=None, n_jobs=-1, seed=0, prev=None):
    """Estimate nuclear-rate and neutron-lifetime uncertainties on BBN
    observables via Monte Carlo.

    Each MC sample draws all active nuclear rate offsets p_* independently from
    N(0,1), plus the neutron lifetime ``tau_n ~ N(cfg.tau_n, cfg.std_tau_n)``
    (used when ``cfg.tau_n_flag=True``, the default), and runs a full PyPRIMAT
    solve.  By default all reactions in the selected network are varied.

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
    prev : MCResult, optional
        A previously computed result to *extend* rather than recompute from
        scratch.  Because sample ``i`` is fully determined by ``seed + i``, the
        first ``min(len(prev), num_mc)`` samples are identical to ``prev`` as
        long as the seed and the set/order of quantities match, so only the
        missing samples (``seed + n_prev .. seed + num_mc - 1``) are actually
        solved.  This makes it cheap to refine an estimate -- e.g. going from 30
        to 50 samples only runs the 20 new ones.  ``prev`` is silently ignored
        (full recompute) if its ``seed`` or its quantities differ from this
        call; if ``num_mc`` is *smaller* than ``len(prev)``, the result is just
        ``prev`` truncated to ``num_mc`` samples (nothing is solved).

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
    
    # Each entry is "bare_name" or "bare_name, filename.txt".
    # Extract only the bare_name for rate variation.
    bare_reactions = []
    for line in reactions:
        parts = re.split(r'[, ]+', line, maxsplit=1)
        bare_reactions.append(parts[0])
        
    rate_keys = [f'p_{rxn}' for rxn in bare_reactions]

    # Reuse a previous result only when it is sample-for-sample compatible with
    # this call: same base seed and same quantities (in the same order, so the
    # stacked sample columns line up).  ``list(prev)`` iterates the quantity
    # names in their stored order (MCResult wraps an insertion-ordered dict).
    reuse = (prev is not None
             and getattr(prev, 'seed', None) == seed
             and list(prev) == quantities)

    if reuse:
        # The central value (all p_* = 0) does not depend on num_mc, so take it
        # straight from prev instead of re-solving it.
        centrals = [prev[q].central for q in quantities]
        # prev's per-quantity value arrays are the columns of the sample
        # matrix; transpose back to (n_prev, n_q) and keep at most num_mc rows
        # (a smaller num_mc simply truncates, solving nothing new).
        prev_samples = np.column_stack([prev[q].values for q in quantities])
        n_prev = min(prev_samples.shape[0], num_mc)
        prev_samples = prev_samples[:n_prev]
    else:
        # Central value (all p_* = 0).
        central_inst = PyPR(params=base_params)
        central_inst.solve()
        centrals = [central_inst.get_quantity(q) for q in quantities]
        prev_samples = np.empty((0, len(quantities)))
        n_prev = 0

    # Only the samples beyond the reused prefix need solving.
    new_seeds   = [seed + i for i in range(n_prev, num_mc)]
    new_samples = _mc_collect_samples(base_params, rate_keys, quantities,
                                      new_seeds, n_jobs)
    samples = np.vstack([prev_samples, new_samples])   # (num_mc, n_q)

    return MCResult({
        q: MCQuantityResult(centrals[j], samples[:, j])
        for j, q in enumerate(quantities)
    }, seed=seed)
