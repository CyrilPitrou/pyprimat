# -*- coding: utf-8 -*-
"""
pyprimat.cli
============

Command-line entry point for PyPRIMAT.

This wraps the same "build a params dict and call PyPR" pattern used by
``runfiles/PyPRIMAT_run.py``, exposing the handful of options most users
need (baryon density, extra relativistic species, network choice) so a
``pip install``-ed user can get BBN abundances without writing any Python::

    pyprimat --Omegabh2 0.02242 --network large --amax 8

Anything not exposed as a flag here can still be reached by writing a short
script that builds a ``params`` dict (see ``runfiles/PyPRIMAT_run.py`` for
the full set of ``PyPRConfig`` keys) and calling ``pyprimat.PyPR`` directly.
"""
import argparse
import json
import sys
import time

from . import PyPR, __version__


def _build_parser():
    """Build the ``argparse.ArgumentParser`` for the ``pyprimat`` CLI.

    Only the most commonly varied ``PyPRConfig`` keys are exposed as flags;
    each flag's ``dest`` matches the corresponding config key so that
    ``main()`` can forward it to ``PyPR(params=...)`` unchanged.  Flags
    default to ``None`` (rather than duplicating ``PyPRConfig``'s defaults)
    so that only options the user actually passed override the config.
    """
    parser = argparse.ArgumentParser(
        prog="pyprimat",
        description="Run a Big Bang Nucleosynthesis computation with "
                     "PyPRIMAT and print the resulting Neff/abundances.",
    )
    # `version` action prints the string and exits before any computation;
    # the version itself comes from the installed distribution metadata via
    # pyprimat.__version__ (single source of truth in pyproject.toml).
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
        help="Print the PyPRIMAT version and exit.",
    )
    parser.add_argument(
        "--Omegabh2", type=float, default=None, metavar="VALUE",
        help="Baryon density Omega_b h^2 (PyPRConfig default: 0.022425).",
    )
    parser.add_argument(
        "--DeltaNeff", type=float, default=None, metavar="VALUE",
        help="Extra relativistic degrees of freedom on top of the SM "
             "neutrino sector (PyPRConfig default: 0).",
    )
    parser.add_argument(
        "--network", default=None, metavar="NAME",
        help="Nuclear reaction network used in the LT era "
             "(PyPRConfig default: small). Built-in choices are 'small', "
             "'small_parthenope' and 'large', but any name for which "
             "rates/nuclear/networks/<NAME>.txt exists is accepted; "
             "PyPRConfig raises a ValueError if no such file is found.",
    )
    parser.add_argument(
        "--amax", type=int, default=None, metavar="A",
        help="Drop reactions involving any nuclide with mass number > A "
             "(must be a positive integer); applies to any --network, not "
             "just 'large'. E.g. --network large --amax 8 reproduces the "
             "old 'medium' network's 68 reactions.",
    )
    parser.add_argument(
        "--numerical_precision", type=float, default=None, metavar="RTOL",
        help="Relative tolerance passed to solve_ivp (PyPRConfig default: 1e-7).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full results dict as JSON instead of a short summary.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable PyPRIMAT's internal progress messages (timings, cache hits, ...).",
    )
    return parser


def main(argv=None):
    """Entry point for the ``pyprimat`` console script.

    Parses command-line arguments into a ``PyPRConfig`` ``params`` dict,
    runs ``PyPR(params).PyPRresults()``, and prints either a short
    human-readable summary (default) or the full results dict as JSON
    (``--json``).

    Parameters
    ----------
    argv : list of str, optional
        Argument vector to parse; defaults to ``sys.argv[1:]``. Exposed as a
        parameter so the CLI can be invoked programmatically (e.g. in tests)
        without spawning a subprocess.

    Returns
    -------
    int
        Process exit code, always ``0`` on success (argparse itself exits
        with code 2 on a bad argument).

    Example
    -------
        $ pyprimat --Omegabh2 0.02242 --network large --amax 8
        Neff       = 3.04397730
        YP (BBN)   = 0.24691900
        ...
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Only forward options the user actually set, so unset flags fall back
    # to PyPRConfig's own defaults rather than a value duplicated here.
    params = {}
    for key in ("Omegabh2", "DeltaNeff", "network", "amax", "numerical_precision"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.verbose:
        params["verbose"] = True

    start_time = time.time()
    results = PyPR(params=params).PyPRresults()
    elapsed = time.time() - start_time

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Neff       = {results['Neff']:.8f}")
        print(f"YP (BBN)   = {results['YPBBN']:.8f}")
        print(f"YP (CMB)   = {results['YPCMB']:.8f}")
        print(f"D/H        = {results['DoH']:.7e}")
        print(f"He3/H      = {results['He3oH']:.7e}")
        print(f"He3/He4    = {results['He3oHe4']:.7e}")
        print(f"Li7/H      = {results['Li7oH']:.6e}")
        if "Li6oLi7" in results:
            print(f"Li6/Li7    = {results['Li6oLi7']:.6e}")
        if "YCNO" in results:
            print(f"CNO (mass) = {results['YCNO']:.6e}")
        print(f"--- running time: {elapsed:.2f} seconds ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
