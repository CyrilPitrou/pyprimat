# -*- coding: utf-8 -*-
"""
profile_solve.py
=================
FUTURE.md P4: measure where a default PyPR run actually spends its time,
before attempting any speed optimisation.

Runs ``PyPR(params).solve()`` once with ``verbose=True, debug=True`` so that
every stage already prints its own wall-clock time (HT/MT/LT solves in
``nuclear_network.py``, the a(T)/t(a) ODE solves and the n<->p weak-rate
setup in ``background.py``).  Rather than re-instrumenting those stages a
second time, this script captures that stdout, parses the per-stage timings
back out with a few regexes, and prints a single attribution table:

    init total
      ├─ rates: network load/compile      (UpdateNuclearRates -- remainder,
      │                                     not separately timed upstream)
      ├─ background: a(T) solve
      ├─ background: t(a) solve
      └─ background: n<->p weak rates      (cache hit ~0 s, miss ~seconds)
    solve total
      ├─ HT era (n<->p only)
      ├─ MT era (18-reaction subset)
      └─ LT era (network-dependent)

Usage
-----
    python studies/profile_solve.py                       # large, amax=8 (default)
    python studies/profile_solve.py --network large
    python studies/profile_solve.py --network small --profile

``--profile`` additionally runs the whole init+solve under ``cProfile`` and
prints the top 20 functions by cumulative time, to attribute the
"rate loading/compile" remainder (and anything else not already covered by
the verbose/debug stage timers) to actual code.
"""

import argparse
import contextlib
import cProfile
import io
import os
import pstats
import re
import sys
import time

_pyprimat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pyprimat_path not in sys.path:
    sys.path.insert(0, _pyprimat_path)

from pyprimat import PyPR


# Stage label -> regex with one float-capturing group (seconds), matched
# against the captured verbose+debug stdout of a single PyPR(...).solve() run.
_STAGE_PATTERNS = {
    "background: a(T) solve":     r"Finished a\(T\) solve in ([\d.]+) s",
    "background: t(a) solve":     r"Finished t\(a\) solve in ([\d.]+) s",
    "background: weak rates":     r"n <--> p weak rates ready in ([\d.]+) s",
    "HT era (n<->p)":             r"\[HT\] Finished solve_ivp in ([\d.]+) s",
    "MT era":                     r"\[MT\] Finished solve_ivp .*? in ([\d.]+) s",
    "LT era":                     r"\[LT\] Finished solve_ivp .*? in ([\d.]+) s",
}


def _grab(pattern, log):
    m = re.search(pattern, log)
    return float(m.group(1)) if m else 0.0


def profile_once(params):
    """Run one PyPR(params).solve(), return (stage_times, init_s, solve_s)."""
    params = dict(params)
    params["verbose"] = True
    params["debug"] = True

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t0 = time.perf_counter()
        p = PyPR(params)
        t1 = time.perf_counter()
        p.solve()
        t2 = time.perf_counter()
    log = buf.getvalue()

    stage_times = {label: _grab(pat, log) for label, pat in _STAGE_PATTERNS.items()}
    return stage_times, t1 - t0, t2 - t1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="large", help="network name (default: large)")
    ap.add_argument("--amax", type=int, default=8,
                    help="amax filter (default: 8, the 'medium'-equivalent network; "
                         "pass --amax 0 to disable and run the full network)")
    ap.add_argument("--profile", action="store_true",
                    help="also run under cProfile and print the top 20 functions "
                         "by cumulative time")
    args = ap.parse_args()

    params = {"network": args.network}
    if args.amax:
        params["amax"] = args.amax

    stage_times, init_s, solve_s = profile_once(params)

    background_s = sum(v for k, v in stage_times.items() if k.startswith("background"))
    rates_s = max(init_s - background_s, 0.0)
    era_s = sum(v for k, v in stage_times.items() if "era" in k)
    solve_other_s = max(solve_s - era_s, 0.0)

    print(f"Network: {args.network!r}, amax={args.amax or None}\n")
    print(f"{'stage':38s} {'seconds':>10s}")
    print("-" * 49)
    print(f"{'init total':38s} {init_s:10.3f}")
    print(f"{'  rates: network load/compile':38s} {rates_s:10.3f}")
    for label in ("background: a(T) solve", "background: t(a) solve",
                  "background: weak rates"):
        print(f"{'  ' + label:38s} {stage_times[label]:10.3f}")
    print(f"{'solve total':38s} {solve_s:10.3f}")
    for label in ("HT era (n<->p)", "MT era", "LT era"):
        print(f"{'  ' + label:38s} {stage_times[label]:10.3f}")
    print(f"{'  result assembly (remainder)':38s} {solve_other_s:10.3f}")
    print("-" * 49)
    print(f"{'TOTAL (init + solve)':38s} {init_s + solve_s:10.3f}")

    if args.profile:
        print("\ncProfile top 20 by cumulative time:\n")
        profiler = cProfile.Profile()
        profiler.enable()
        p = PyPR(dict(params))
        p.solve()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
