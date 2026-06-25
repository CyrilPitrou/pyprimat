# -*- coding: utf-8 -*-
"""
primat.cli
============

Command-line entry point for primat.

This wraps the same "build a params dict and call PRIMAT" pattern used by
``runfiles/PyPRIMAT_run.py``, exposing the handful of options most users
need (baryon density, extra relativistic species, network choice) so a
``pip install``-ed user can get BBN abundances without writing any Python::

    primat --Omegabh2 0.02242 --network large --amax 8

Anything not exposed as a named flag here can still be set without writing a
script, via the (intentionally undocumented in ``--help``, to keep the
printed help short) ``--set KEY=VALUE`` escape hatch, repeatable for any
``PRIMATConfig`` key (including ``p_<reaction>``/``NP_delta_<reaction>``
rate-variation keys), e.g.::

    primat --set T_end_MeV=1e-4 --set decay_era=True --set network=large

Values are parsed with ``ast.literal_eval`` (so ``True``/``False``/``None``,
numbers, and quoted strings all work); anything that fails to parse as a
Python literal is kept as a plain string (e.g. ``--set network=large``).
"""
import argparse
import ast
import json
import os
import sys
import time

from . import PRIMAT, __version__
from .backend import HAS_C_BACKEND, dump_mc_samples, run_bbn, run_mc
from .cache_utils import clear_weak_cache, list_weak_cache_files
from .config import PRIMATConfig


def _parse_set_value(raw: str):
    """Parse the value half of a ``--set KEY=VALUE`` CLI argument.

    Tries ``ast.literal_eval`` first, so numeric, boolean, ``None``, and
    quoted-string values are converted to the right Python type (e.g.
    ``"True"`` -> ``True``, ``"1e-4"`` -> ``1e-4``). Falls back to the raw
    string unchanged when it is not a valid Python literal (e.g. an
    unquoted network name like ``large``), since ``PRIMATConfig`` string
    parameters (``network``, ``custom_background``, ...) are passed this way.

    Example
    -------
        >>> _parse_set_value("1e-4")
        0.0001
        >>> _parse_set_value("large")
        'large'
    """
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _build_parser():
    """Build the ``argparse.ArgumentParser`` for the ``primat`` CLI.

    Only the most commonly varied ``PRIMATConfig`` keys are exposed as flags;
    each flag's ``dest`` matches the corresponding config key so that
    ``main()`` can forward it to ``PRIMAT(params=...)`` unchanged.  Flags
    default to ``None`` (rather than duplicating ``PRIMATConfig``'s defaults)
    so that only options the user actually passed override the config.
    """
    parser = argparse.ArgumentParser(
        prog="primat",
        description="Run a Big Bang Nucleosynthesis computation with "
                     "primat and print the resulting Neff/abundances.",
        epilog="Any other PRIMATConfig parameter (including p_<reaction>/"
               "NP_delta_<reaction> rate variations) can be set with "
               "repeated --set KEY=VALUE, e.g. --set T_end_MeV=1e-4.",
    )
    # `version` action prints the string and exits before any computation;
    # the version itself comes from the installed distribution metadata via
    # primat.__version__ (single source of truth in pyproject.toml).
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
        help="Print the primat version and exit.",
    )
    parser.add_argument(
        "--Omegabh2", type=float, default=None, metavar="VALUE",
        help="Baryon density Omega_b h^2 (PRIMATConfig default: 0.022425).",
    )
    parser.add_argument(
        "--DeltaNeff", type=float, default=None, metavar="VALUE",
        help="Extra relativistic degrees of freedom on top of the SM "
             "neutrino sector (PRIMATConfig default: 0).",
    )
    parser.add_argument(
        "--network", default=None, metavar="NAME",
        help="Nuclear reaction network used in the LT era "
             "(PRIMATConfig default: small). Built-in choices are 'small', "
             "'small_parthenope' and 'large', but any name for which "
             "rates/nuclear/networks/<NAME>.txt exists is accepted; "
             "PRIMATConfig raises a ValueError if no such file is found.",
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
        help="Relative tolerance passed to solve_ivp (PRIMATConfig default: 1e-7).",
    )
    parser.add_argument(
        "--munuOverTnu", type=float, default=None, metavar="XI",
        help="Reduced neutrino chemical potential mu/T, same for all flavours "
             "(PRIMATConfig default: 0).",
    )
    # Boolean PRIMATConfig flags exposed as --flag/--no-flag pairs (argparse's
    # BooleanOptionalAction), so they don't need the --set escape hatch.
    for flag_name, cfg_default, help_text in (
        ("QED_corrections", True,
         "QED interaction corrections to the EM plasma equation of state."),
        ("nuclear_qed_corrections", True,
         "QED corrections to radiative-capture nuclear reaction rates "
         "(Pitrou & Pospelov 2020)."),
        ("radiative_corrections", True,
         "Coulomb + T=0 resummed radiative corrections to n<->p (CCR); "
         "if False, use the Born approximation instead."),
        ("finite_mass_corrections", True,
         "Finite-nucleon-mass (Fokker-Planck) correction to n<->p."),
        ("thermal_corrections", True,
         "Finite-temperature radiative corrections to n<->p (CCRTh; "
         "Brown & Sawyer 2001)."),
        ("spectral_distortions", True,
         "Correct n<->p rates for non-Fermi-Dirac neutrino distributions."),
        ("output_time_evolution", False,
         "Write the full time-evolution series (in-memory always; to disk "
         "if output_file is set)."),
        ("output_final_result", False,
         "Write the final results dict to output_file."),
        ("output_background_evolution", False,
         "Write the cosmological background time series to disk."),
        ("output_mc_samples", False,
         "Write --mc samples to output_mc_file (overridden by --mc-output "
         "when that flag is given)."),
    ):
        parser.add_argument(
            f"--{flag_name}", action=argparse.BooleanOptionalAction, default=None,
            help=f"{help_text} (PRIMATConfig default: {cfg_default}).",
        )
    parser.add_argument(
        "--backend", choices=("auto", "c", "python"), default="auto",
        help="Which solver implementation to use: 'auto' (default) picks the "
             "compiled C extension when available, 'c' forces it (error if "
             "unavailable), 'python' forces the pure-Python implementation.",
    )
    parser.add_argument(
        "--mc", type=int, default=None, metavar="N",
        help="Also run an N-sample Monte-Carlo nuclear-rate/tau_n uncertainty "
             "propagation (primat.backend.run_mc) and print each observable "
             "as 'value +/- sigma' instead of a bare value. Uses the C "
             "backend when available (--backend auto/c), else joblib in "
             "pure Python; see primat.backend's docstring for the RNG caveat.",
    )
    parser.add_argument(
        "--mc-seed", type=int, default=0, metavar="SEED",
        help="Base RNG seed for --mc (default: 0); sample i uses seed+i.",
    )
    parser.add_argument(
        "--mc-jobs", type=int, default=-1, metavar="N",
        help="Parallel worker count for --mc (default: -1, all CPUs; Python "
             "backend only -- the C backend always uses one pthread per sample).",
    )
    parser.add_argument(
        "--mc-output", default=None, metavar="FILE",
        help="Write the --mc samples (one column per quantity, one row per "
             "sample) to FILE as TSV. Without this flag, the samples file is "
             "only written when output_mc_samples=True (--set "
             "output_mc_samples=True), to output_mc_file.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full results dict as JSON instead of a short summary.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable primat's internal progress messages (timings, cache hits, ...).",
    )
    parser.add_argument(
        "--cache-info", action="store_true",
        help="Print the number of cached n<->p weak-rate files "
             "(primat/data/weak/nTOp_*.txt) and exit, without running a solve.",
    )
    parser.add_argument(
        "--cache-clear", action="store_true",
        help="Delete every cached n<->p weak-rate file and exit, without "
             "running a solve. The cache is always safely regenerable: a "
             "later run just pays the one-time recompute cost again.",
    )
    # Generic escape hatch: lets any PRIMATConfig key (including p_<reaction>/
    # NP_delta_<reaction>) be set from the CLI without a dedicated flag.
    # help=SUPPRESS keeps it out of --help, per the handful of named flags
    # above being the only ones intended to show there; see the module
    # docstring for usage.
    parser.add_argument(
        "--set", action="append", dest="set_params", metavar="KEY=VALUE",
        default=[], help=argparse.SUPPRESS,
    )
    return parser


def main(argv=None):
    """Entry point for the ``primat`` console script.

    Parses command-line arguments into a ``PRIMATConfig`` ``params`` dict,
    runs ``PRIMAT(params).primat_results()``, and prints either a short
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
        $ primat --Omegabh2 0.02242 --network large --amax 8
        Neff       = 3.04397730
        YP (BBN)   = 0.24691900
        ...
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cache_info or args.cache_clear:
        cfg = PRIMATConfig({})
        if args.cache_clear:
            n = clear_weak_cache(cfg)
            print(f"Removed {n} cached weak-rate file(s) from {cfg.data_dir}/data/weak/.")
        else:
            n = len(list_weak_cache_files(cfg))
            print(f"{n} cached weak-rate file(s) in {cfg.data_dir}/data/weak/.")
        return 0

    # Only forward options the user actually set, so unset flags fall back
    # to PRIMATConfig's own defaults rather than a value duplicated here.
    params = {}
    for key in (
        "Omegabh2", "DeltaNeff", "network", "amax", "numerical_precision",
        "munuOverTnu", "QED_corrections", "nuclear_qed_corrections",
        "radiative_corrections", "finite_mass_corrections", "thermal_corrections",
        "spectral_distortions", "output_time_evolution", "output_final_result",
        "output_background_evolution", "output_mc_samples",
    ):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    if args.verbose:
        params["verbose"] = True
    for entry in args.set_params:
        if "=" not in entry:
            parser.error(f"--set {entry!r}: expected KEY=VALUE")
        key, _, raw_value = entry.partition("=")
        params[key] = _parse_set_value(raw_value)

    start_time = time.time()
    results = run_bbn(params=params, force_backend=args.backend)
    mc = None
    if args.mc is not None:
        mc = run_mc(args.mc, params=params, force_backend=args.backend,
                    seed=args.mc_seed, n_jobs=args.mc_jobs)
    elapsed = time.time() - start_time

    if args.json:
        out = dict(results)
        if mc is not None:
            out["mc"] = {q: {"central": mc[q].central, "mean": mc[q].mean,
                              "std": mc[q].std, "values": list(mc[q].values)}
                         for q in mc.quantity_names()}
        print(json.dumps(out, indent=2))
    else:
        print(f"Neff       = {results['Neff']:.8f}" +
              (f" +/- {mc['Neff'].std:.8f}" if mc is not None and "Neff" in mc.quantity_names() else ""))
        print(f"YP (BBN)   = {results['YPBBN']:.8f}" +
              (f" +/- {mc['YPBBN'].std:.8f}" if mc is not None and "YPBBN" in mc.quantity_names() else ""))
        print(f"YP (CMB)   = {results['YPCMB']:.8f}" +
              (f" +/- {mc['YPCMB'].std:.8f}" if mc is not None and "YPCMB" in mc.quantity_names() else ""))
        print(f"D/H        = {results['DoH']:.7e}" +
              (f" +/- {mc['DoH'].std:.7e}" if mc is not None and "DoH" in mc.quantity_names() else ""))
        print(f"He3/H      = {results['He3oH']:.7e}" +
              (f" +/- {mc['He3oH'].std:.7e}" if mc is not None and "He3oH" in mc.quantity_names() else ""))
        print(f"He3/He4    = {results['He3oHe4']:.7e}")
        print(f"Li7/H      = {results['Li7oH']:.6e}" +
              (f" +/- {mc['Li7oH'].std:.6e}" if mc is not None and "Li7oH" in mc.quantity_names() else ""))
        if "Li6oLi7" in results:
            print(f"Li6/Li7    = {results['Li6oLi7']:.6e}")
        if "YCNO" in results:
            print(f"CNO (mass) = {results['YCNO']:.6e}")
        if mc is not None:
            print(f"--- Monte-Carlo: {args.mc} samples ---")
        print(f"--- running time: {elapsed:.2f} seconds ---")

    if mc is not None:
        mc_output_path = args.mc_output
        if mc_output_path is None:
            cfg_check = PRIMATConfig(params)
            if cfg_check.output_mc_samples:
                mc_output_path = cfg_check.output_mc_file
        if mc_output_path:
            out_dir = os.path.dirname(mc_output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(mc_output_path, "w") as f:
                f.write(dump_mc_samples(mc))
            print(f"MC samples written to {mc_output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
