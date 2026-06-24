# -*- coding: utf-8 -*-
"""
primat.backend
================

Dispatch layer choosing between the compiled C extension
(``primat._primat_c``, wrapping ``primat-c``'s ``cprimat_run``) and the
pure-Python solver (``primat.main.PRIMAT``), per ``PRIMAT.md`` S5.4.

``HAS_C_BACKEND`` is probed once at import time (``True`` iff the extension
built successfully -- see ``setup.py``'s ``optional_build_ext``, which lets
``pip install`` succeed even without a C compiler). :func:`run_bbn` is the
single dispatch entry point; everything else in this module supports it.

Feature gaps (C side does not implement these -- mirrors ``cprimat/api.h``'s
own "out of scope" notes):

* ``extra_rho``, ``custom_network``, ``background=`` (the Python-only
  ``PRIMAT.__init__`` constructor extensions) -- always force the Python
  backend.

``output_time_evolution=True`` *is* supported on both backends (PRIMAT.md
S7.3/S7.6): the C extension's ``cprimat_run`` populates ``CPRResults``'s
``evol_*`` in-memory arrays (``primat-c/include/cprimat/api.h``) and
``primat/_primat_c/_wrapper.c`` hands them back as an ``"evolution"`` dict
key (plain Python lists, no numpy C-API dependency in the extension); this
module assembles the same :class:`primat.evolution.EvolutionResult` shape
the Python backend produces, with no disk I/O on either backend's part.

``rates_dir``/``user_rates_dir`` (the ``rates/`` overlay, see CLAUDE.md's
"Rates directory resolution" section) *are* supported on both backends as of
``primat-c``'s ``cpr_config_resolve_rates_path`` (``primat-c/src/config.c``):
both apply the same lookup order (``rates_dir`` full takeover ->
``user_rates_dir`` additive overlay -> shipped default) to the network-file
path and each reaction's rate-table file. They are ordinary ``params`` dict
keys, applied generically via ``cpr_config_set_by_name``, so no special-casing
is needed here.

:func:`run_mc` is the MC counterpart of :func:`run_bbn`: it dispatches between
``primat._primat_c``'s ``run_mc`` (wrapping ``primat-c/src/mc.c``'s threaded
``cpr_mc_uncertainty``) and ``primat.main.mc_uncertainty`` (joblib), returning
the same :class:`primat.main.MCResult` shape either way -- the "common
language" the two backends share for MC results (CLAUDE.md's backend-parity
mandate). The C path uses a pthread/xoshiro256** RNG, *not* NumPy's
``default_rng``, so individual samples are not bit-for-bit comparable across
backends (only statistically, mean/std convergence -- see ``mc.h``). ``prev``
(incremental sample reuse) and ``custom_network`` have no C-side equivalent
and always force the Python path (mirroring ``extra_rho``/``custom_network``/
``background=`` above for :func:`run_bbn`).
"""
import os

__all__ = ["HAS_C_BACKEND", "run_bbn", "run_mc", "dump_mc_samples", "dump_final_with_sigma"]

# Observables included by default (alongside every tracked nuclide's final Y)
# when run_mc's `quantities` argument is omitted -- the same six ratios the
# CLI's plain-text summary prints (primat.cli.main).
_DEFAULT_MC_OBSERVABLES = ("Neff", "YPBBN", "YPCMB", "DoH", "He3oH", "Li7oH")

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from . import _primat_c as _c_ext
    HAS_C_BACKEND = True
except ImportError:
    _c_ext = None
    HAS_C_BACKEND = False


def _python_solve(params, extra_rho, custom_network, background):
    """Run the pure-Python backend and return PRIMAT.solve()'s result dict."""
    from .main import PRIMAT
    return PRIMAT(params=params, extra_rho=extra_rho,
                  custom_network=custom_network, background=background).solve()


def run_bbn(params=None, force_backend=None, extra_rho=None,
            custom_network=None, background=None):
    """Run one BBN computation, dispatching to the C or Python backend.

    This mirrors ``PRIMAT(params=params, ...).solve()``'s result dict (same
    keys: ``YPBBN``, ``DoH``, ``Neff``, ... -- see ``primat.main.PRIMAT.solve``
    and ``tests/test_backend_parity.py``), so callers can switch backends
    transparently.

    Args:
        params: dict, optional. Same ``PRIMATConfig`` overrides accepted by
            ``PRIMAT(params=...)``.
        force_backend: ``{None, "auto", "c", "python"}``. ``None``/``"auto"``
            (default) picks the C extension when it is available and the
            request has no C-unsupported feature (see module docstring),
            otherwise the Python backend. ``"c"``/``"python"`` force that
            backend, raising ``RuntimeError``/``ValueError`` respectively if
            the C backend is unavailable or the request uses a C-unsupported
            feature.
        extra_rho, custom_network, background: forwarded to ``PRIMAT.__init__``
            verbatim; Python-only (see module docstring), so any non-``None``
            value forces the Python backend regardless of ``force_backend``
            (except ``force_backend="c"``, which raises instead).

    Returns:
        dict: the BBN result dict (``YPBBN``, ``DoH``, ``Neff``, ..., plus a
        ``Y_final`` sub-dict of every tracked nuclide's final mass fraction).

    Example:
        >>> run_bbn({"network": "small"})["YPBBN"]
        0.24700...
        >>> run_bbn({"network": "small"}, force_backend="python")["YPBBN"]
        0.24699...
    """
    if force_backend not in (None, "auto", "c", "python"):
        raise ValueError(f"force_backend must be one of None/'auto'/'c'/'python', "
                          f"got {force_backend!r}")

    params = params or {}

    # Validate params the same way regardless of backend (PRIMATConfig's
    # __init__ does all the checking -- e.g. an unknown --network name --
    # so a bad request raises the same ValueError whether or not the C
    # backend ends up being used; the resulting cfg itself is discarded for
    # the "c" path, which re-derives its own CPRConfig from params instead).
    from .config import PRIMATConfig
    PRIMATConfig(params)

    python_only_feature = (extra_rho is not None or custom_network is not None
                            or background is not None)

    if force_backend == "python":
        return _python_solve(params, extra_rho, custom_network, background)

    if force_backend == "c":
        if not HAS_C_BACKEND:
            raise RuntimeError(
                "force_backend='c' requested but primat._primat_c is not "
                "available (the C extension failed to build or was not "
                "compiled -- see setup.py)."
            )
        if extra_rho is not None or custom_network is not None or background is not None:
            raise ValueError(
                "force_backend='c' is incompatible with extra_rho/"
                "custom_network/background (Python-only features, no C-side "
                "equivalent)."
            )
        return _assemble_c_result(_c_ext.run_bbn(params, _PACKAGE_DIR))

    # force_backend in (None, "auto"): use the C backend opportunistically,
    # falling back to Python for anything it cannot express.
    if HAS_C_BACKEND and not python_only_feature:
        return _assemble_c_result(_c_ext.run_bbn(params, _PACKAGE_DIR))
    return _python_solve(params, extra_rho, custom_network, background)


def _assemble_c_result(result):
    """Replaces the C extension's plain-list ``"evolution"`` dict (see
    ``primat/_primat_c/_wrapper.c``'s ``evolution_to_dict``) with the same
    :class:`primat.evolution.EvolutionResult` the Python backend attaches
    under ``result["evolution"]`` -- so callers can switch backends
    transparently (``PRIMAT.md`` S7.3). No-op if ``output_time_evolution``
    wasn't requested (no ``"evolution"`` key at all)."""
    evo = result.get("evolution")
    if evo is None:
        return result
    import numpy as np
    from .evolution import EvolutionResult
    result["evolution"] = EvolutionResult(
        t=np.asarray(evo["t"]), a=np.asarray(evo["a"]), T_gamma=np.asarray(evo["T_gamma"]),
        T_nu={"e": np.asarray(evo["T_nue"]), "mu": np.asarray(evo["T_numu"]),
              "tau": np.asarray(evo["T_nutau"])},
        Y={name: np.asarray(arr) for name, arr in evo["Y"].items()},
    )
    return result


def _default_mc_quantities(params):
    """Every tracked nuclide's final-Y name plus the standard observables.

    Resolved from one ordinary :func:`run_bbn` call (cheap relative to an
    ``num_mc``-sample MC run) rather than re-deriving the network's nuclide
    list from scratch, so this always matches exactly what the chosen
    ``network``/``amax``/``custom_network`` would track -- no duplicated
    network-introspection logic between here and ``NuclearNetwork``/
    ``cpr_nuclear_network``.
    """
    central = run_bbn(params)
    names = list(central["Y_final"].keys())
    names += [q for q in _DEFAULT_MC_OBSERVABLES if q in central]
    return names


def _assemble_c_mc_result(raw, quantities, seed, params):
    """Converts the C extension's ``run_mc`` dict (``{name: {central, mean,
    std, values}}``, see ``_wrapper.c``) into the same
    :class:`primat.main.MCResult` :func:`primat.main.mc_uncertainty` returns,
    so callers can switch backends transparently. Mean/std are recomputed
    from ``values`` via :class:`primat.main.MCQuantityResult` (rather than
    trusting the C side's own mean/std fields) so both backends' MCResult
    objects are built by the exact same code, with only the sample source
    differing.
    """
    from .main import MCQuantityResult, MCResult
    data = {q: MCQuantityResult(raw[q]["central"], raw[q]["values"]) for q in quantities}
    return MCResult(data, seed=seed, params=params, custom_network=None)


def run_mc(num_mc, quantities=None, params=None, force_backend=None, seed=0,
           n_jobs=-1, prev=None, custom_network=None):
    """Run an MC nuclear-rate/tau_n uncertainty propagation, dispatching to
    the C or Python backend (the MC counterpart of :func:`run_bbn`).

    This mirrors :func:`primat.main.mc_uncertainty`'s return value (an
    :class:`primat.main.MCResult`, indexed by quantity name -- same
    ``.central``/``.mean``/``.std``/``.values`` per quantity), so callers can
    switch backends transparently; see this module's docstring for the
    RNG caveat (C samples are statistically, not bit-for-bit, comparable to
    Python's).

    Args:
        num_mc: int. Number of MC samples.
        quantities: str or list of str, optional. A result-dict key
            (``'YPBBN'``, ``'DoH'``, ...) or nuclide name, or a list of
            either. ``None`` (default) uses every tracked nuclide's final Y
            plus ``Neff``/``YPBBN``/``YPCMB``/``DoH``/``He3oH``/``Li7oH``
            (see :func:`_default_mc_quantities`).
        params, seed, n_jobs: forwarded verbatim; see
            ``primat.main.mc_uncertainty``'s docstring.
        force_backend: ``{None, "auto", "c", "python"}``, same semantics as
            :func:`run_bbn`.
        prev, custom_network: Python-only (no C-side equivalent -- ``mc.c``
            has no incremental-reuse or custom-network support); any
            non-``None`` value forces the Python backend regardless of
            ``force_backend`` (except ``force_backend="c"``, which raises).

    Returns:
        primat.main.MCResult

    Example:
        >>> run_mc(50, ['YPBBN', 'DoH'], params={'network': 'small'})['YPBBN'].std
        >>> run_mc(50, force_backend='python')['DoH'].mean
    """
    if force_backend not in (None, "auto", "c", "python"):
        raise ValueError(f"force_backend must be one of None/'auto'/'c'/'python', "
                          f"got {force_backend!r}")

    params = params or {}
    from .config import PRIMATConfig
    PRIMATConfig(params)  # validate params the same way regardless of backend

    if quantities is None:
        quantities = _default_mc_quantities(params)
    quantities = [quantities] if isinstance(quantities, str) else list(quantities)

    python_only_feature = prev is not None or custom_network is not None

    def _python_mc():
        from .main import mc_uncertainty
        return mc_uncertainty(num_mc, quantities, params=params, n_jobs=n_jobs,
                               seed=seed, prev=prev, custom_network=custom_network)

    if force_backend == "python":
        return _python_mc()

    if force_backend == "c":
        if not HAS_C_BACKEND:
            raise RuntimeError(
                "force_backend='c' requested but primat._primat_c is not "
                "available (the C extension failed to build or was not "
                "compiled -- see setup.py)."
            )
        if python_only_feature:
            raise ValueError(
                "force_backend='c' is incompatible with prev/custom_network "
                "(Python-only features, no C-side equivalent)."
            )
        raw = _c_ext.run_mc(params, _PACKAGE_DIR, num_mc, quantities, seed, n_jobs)
        return _assemble_c_mc_result(raw, quantities, seed, params)

    # force_backend in (None, "auto"): use the C backend opportunistically,
    # falling back to Python for anything it cannot express.
    if HAS_C_BACKEND and not python_only_feature:
        raw = _c_ext.run_mc(params, _PACKAGE_DIR, num_mc, quantities, seed, n_jobs)
        return _assemble_c_mc_result(raw, quantities, seed, params)
    return _python_mc()


def dump_mc_samples(mc):
    """Serialise an :class:`primat.main.MCResult` to TSV text: one column per
    quantity (header = quantity names, in their original order), one row per
    MC sample -- the on-disk "common language" for MC results shared by both
    backends (CLAUDE.md's backend-parity mandate), and the same shape
    written to ``cfg.output_mc_file`` when ``output_mc_samples=True``.

    Args:
        mc: primat.main.MCResult.

    Returns:
        str: TSV text, with a trailing newline.
    """
    names = mc.quantity_names()
    samples = mc.samples_array()
    lines = ["\t".join(names)]
    lines += ["\t".join(f"{v:.10e}" for v in row) for row in samples]
    return "\n".join(lines) + "\n"


def dump_final_with_sigma(names, Y, sigma=None, num_mc=None):
    """Render the ``output_final.dat``-format final-abundances text.

    Two columns (``# nuclide  Y``) when ``sigma`` is ``None`` -- identical to
    the plain single-run format written by
    ``NuclearNetwork._write_final_result``. Three columns (``# nuclide  Y
    sigma_N<num_mc>``) when an MC ``sigma`` dict is supplied, so the sample
    count backing the uncertainty estimate is recorded directly in the
    header rather than only in the (separate) MC-samples file.

    Args:
        names: list of str. Nuclide names, in the order to write them.
        Y: dict, name -> final mass-fraction abundance.
        sigma: dict, name -> 1-sigma MC uncertainty on ``Y[name]``, optional.
        num_mc: int, required when ``sigma`` is given -- the MC sample count,
            recorded in the header (e.g. ``sigma_N50``).

    Returns:
        str: the file text, with a trailing newline.
    """
    if sigma is None:
        lines = [f"# {'nuclide':<12}Y"]
        lines += [f"{nm:<14}{Y[nm]:.6e}" for nm in names]
    else:
        if num_mc is None:
            raise ValueError("num_mc is required when sigma is given")
        lines = [f"# {'nuclide':<12}{'Y':<14}sigma_N{num_mc}"]
        lines += [f"{nm:<14}{Y[nm]:<14.6e}{sigma[nm]:.6e}" for nm in names]
    return "\n".join(lines) + "\n"
