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
* ``output_time_evolution=True`` -- forces the Python backend under
  ``force_backend="auto"``/``None``; raises under ``force_backend="c"``. The
  unified ``EvolutionResult``/``run.evolution`` (``primat.evolution``,
  ``PRIMAT.md`` S7.3) is Python-only so far: the C extension's
  ``cprimat_run`` does not yet return per-step arrays to Python (it still
  writes its own *legacy*, non-unified TSV to ``cfg.output_file`` when this
  flag is set, which a caller expecting the unified schema would
  misinterpret) -- a known Phase D follow-up, not yet ported.

``rates_dir``/``user_rates_dir`` (the ``rates/`` overlay, see CLAUDE.md's
"Rates directory resolution" section) *are* supported on both backends as of
``primat-c``'s ``cpr_config_resolve_rates_path`` (``primat-c/src/config.c``):
both apply the same lookup order (``rates_dir`` full takeover ->
``user_rates_dir`` additive overlay -> shipped default) to the network-file
path and each reaction's rate-table file. They are ordinary ``params`` dict
keys, applied generically via ``cpr_config_set_by_name``, so no special-casing
is needed here.
"""
import os

__all__ = ["HAS_C_BACKEND", "run_bbn"]

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
                            or background is not None
                            or params.get("output_time_evolution"))

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
        if params.get("output_time_evolution"):
            raise ValueError(
                "force_backend='c' is incompatible with output_time_evolution=True "
                "(the unified EvolutionResult has no C-side equivalent yet, "
                "see module docstring)."
            )
        return _c_ext.run_bbn(params, _PACKAGE_DIR)

    # force_backend in (None, "auto"): use the C backend opportunistically,
    # falling back to Python for anything it cannot express.
    if HAS_C_BACKEND and not python_only_feature:
        return _c_ext.run_bbn(params, _PACKAGE_DIR)
    return _python_solve(params, extra_rho, custom_network, background)
